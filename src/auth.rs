//! Authentication manager for GCP Vertex AI access tokens.
//!
//! Key features:
//! - Proactive background token refresh (refreshes before expiry)
//! - Uses actual `expires_in` from GCP metadata response
//! - Force-refresh on 401 errors (for retry scenarios)
//! - Refresh coalescing (prevents thundering herd)
//! - Token age tracking for diagnostics

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

/// How many seconds before actual expiry we should proactively refresh.
/// GCP tokens typically last 3600s (1 hour). We refresh 10 minutes early.
const REFRESH_BUFFER_SECS: u64 = 600;

/// Maximum time a token is kept regardless of upstream TTL.
/// This prevents stale tokens if the clock drifts.
const MAX_TOKEN_AGE_SECS: u64 = 3300; // 55 minutes

#[derive(Clone)]
pub struct AuthManager {
    token: Arc<RwLock<CachedToken>>,
    /// Tracks whether a refresh is currently in-flight to coalesce requests.
    refreshing: Arc<tokio::sync::Notify>,
    /// Atomic flag: 0 = not refreshing, 1 = refreshing in progress
    is_refreshing: Arc<std::sync::atomic::AtomicBool>,
    client: reqwest::Client,
    /// When the current token was obtained (epoch seconds) — for diagnostics
    token_obtained_at: Arc<AtomicU64>,
    /// Number of times token has been refreshed (diagnostics)
    refresh_count: Arc<AtomicU64>,
}

struct CachedToken {
    value: String,
    project_id: String,
    /// Absolute epoch timestamp when the token expires
    expires_at: u64,
    /// When this token was obtained (epoch secs)
    obtained_at: u64,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl AuthManager {
    pub fn new(client: reqwest::Client) -> Self {
        Self {
            token: Arc::new(RwLock::new(CachedToken {
                value: String::new(),
                project_id: String::new(),
                expires_at: 0,
                obtained_at: 0,
            })),
            refreshing: Arc::new(tokio::sync::Notify::new()),
            is_refreshing: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            client,
            token_obtained_at: Arc::new(AtomicU64::new(0)),
            refresh_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get a valid token, refreshing if necessary.
    /// This is the primary method used by request handlers.
    pub async fn get_token(&self) -> Result<(String, String), String> {
        // Fast path: read lock, check cache
        {
            let cached = self.token.read().await;
            if self.is_token_valid(&cached) {
                return Ok((cached.value.clone(), cached.project_id.clone()));
            }
        }

        // Slow path: refresh the token
        self.refresh_token().await
    }

    /// Force-refresh the token, ignoring the current cache.
    /// Used when a 401 is received from upstream (the token may have been
    /// revoked or expired server-side even if we think it's still valid).
    pub async fn force_refresh(&self) -> Result<(String, String), String> {
        // Invalidate the current token first
        {
            let mut cached = self.token.write().await;
            cached.expires_at = 0; // Force expiry
        }
        self.refresh_token().await
    }

    /// Internal method: refresh the token with coalescing.
    /// If another task is already refreshing, wait for it to complete
    /// and then return the new token (avoids thundering herd).
    async fn refresh_token(&self) -> Result<(String, String), String> {
        // Try to become the refresher
        if self
            .is_refreshing
            .compare_exchange(
                false,
                true,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
        {
            // We are the refresher — fetch new token
            let result = self.do_fetch_token().await;

            // Reset flag and notify waiters
            self.is_refreshing.store(false, Ordering::SeqCst);
            self.refreshing.notify_waiters();

            result
        } else {
            // Another task is already refreshing — wait for it
            // But first, do a quick check: maybe it finished already
            loop {
                // Wait for notification or timeout
                tokio::select! {
                    _ = self.refreshing.notified() => {
                        // The refresh completed — check if token is now valid
                        let cached = self.token.read().await;
                        if self.is_token_valid(&cached) {
                            return Ok((cached.value.clone(), cached.project_id.clone()));
                        }
                        // Token still not valid? Maybe refresh failed.
                        // Try to become the refresher ourselves.
                        if self.is_refreshing.compare_exchange(
                            false, true, Ordering::SeqCst, Ordering::SeqCst
                        ).is_ok() {
                            let result = self.do_fetch_token().await;
                            self.is_refreshing.store(false, Ordering::SeqCst);
                            self.refreshing.notify_waiters();
                            return result;
                        }
                        // Another task is refreshing again — loop
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
                        // Timeout — try ourselves
                        tracing::warn!("Token refresh wait timed out, attempting own refresh");
                        if self.is_refreshing.compare_exchange(
                            false, true, Ordering::SeqCst, Ordering::SeqCst
                        ).is_ok() {
                            let result = self.do_fetch_token().await;
                            self.is_refreshing.store(false, Ordering::SeqCst);
                            self.refreshing.notify_waiters();
                            return result;
                        }
                        // Still can't become refresher — read whatever we have
                        let cached = self.token.read().await;
                        if !cached.value.is_empty() {
                            return Ok((cached.value.clone(), cached.project_id.clone()));
                        }
                        return Err("Token refresh timed out and no cached token available".to_string());
                    }
                }
            }
        }
    }

    /// Actually fetch a new token from the GCP metadata server.
    async fn do_fetch_token(&self) -> Result<(String, String), String> {
        let now = now_secs();

        match fetch_metadata_token(&self.client).await {
            Ok((token, project_id, expires_in)) => {
                // Use the actual expires_in from GCP, but cap it to MAX_TOKEN_AGE
                let effective_ttl = expires_in.min(MAX_TOKEN_AGE_SECS);
                // Refresh early: subtract the buffer
                let refresh_at = now + effective_ttl.saturating_sub(REFRESH_BUFFER_SECS);

                {
                    let mut cached = self.token.write().await;
                    cached.value = token.clone();
                    cached.project_id = project_id.clone();
                    cached.expires_at = refresh_at;
                    cached.obtained_at = now;
                }

                self.token_obtained_at.store(now, Ordering::Relaxed);
                self.refresh_count.fetch_add(1, Ordering::Relaxed);

                tracing::info!(
                    "Token refreshed successfully. TTL={}s, refresh_at={} (in {}s), refresh_count={}",
                    expires_in,
                    refresh_at,
                    refresh_at.saturating_sub(now),
                    self.refresh_count.load(Ordering::Relaxed),
                );

                Ok((token, project_id))
            }
            Err(e) => {
                tracing::error!("Failed to refresh token: {e}");
                Err(e)
            }
        }
    }

    /// Check if the cached token is still valid.
    fn is_token_valid(&self, cached: &CachedToken) -> bool {
        let now = now_secs();
        // Token is valid if:
        // 1. It's not empty
        // 2. It hasn't reached its refresh time
        // 3. It hasn't exceeded max age since obtained
        if cached.value.is_empty() {
            return false;
        }
        if now >= cached.expires_at {
            return false;
        }
        // Safety check: if token is older than MAX_TOKEN_AGE, consider it stale
        if now.saturating_sub(cached.obtained_at) > MAX_TOKEN_AGE_SECS {
            tracing::warn!("Token exceeded max age ({}s), marking as stale", MAX_TOKEN_AGE_SECS);
            return false;
        }
        true
    }

    /// Get diagnostic info about the current token state.
    pub async fn token_info(&self) -> TokenInfo {
        let cached = self.token.read().await;
        let now = now_secs();
        let obtained_at = self.token_obtained_at.load(Ordering::Relaxed);
        let refresh_count = self.refresh_count.load(Ordering::Relaxed);

        TokenInfo {
            has_token: !cached.value.is_empty(),
            expires_at: cached.expires_at,
            obtained_at: cached.obtained_at,
            seconds_until_refresh: cached.expires_at.saturating_sub(now),
            token_age_secs: now.saturating_sub(obtained_at),
            refresh_count,
            is_refreshing: self.is_refreshing.load(Ordering::Relaxed),
        }
    }

    /// Spawn a background task that proactively refreshes the token
    /// before it expires. This ensures requests never have to wait
    /// for a synchronous token refresh.
    pub fn spawn_background_refresh(self: &AuthManager) {
        let manager = self.clone();
        tokio::spawn(async move {
            // Initial delay before first check
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;

            loop {
                let info = manager.token_info().await;

                if info.seconds_until_refresh <= 60 || !info.has_token {
                    // Token is about to expire or doesn't exist — refresh now
                    tracing::info!(
                        "Background token refresh triggered (seconds_until_refresh={}, has_token={})",
                        info.seconds_until_refresh,
                        info.has_token,
                    );
                    match manager.refresh_token().await {
                        Ok(_) => {}
                        Err(e) => {
                            tracing::error!("Background token refresh failed: {e}");
                            // Retry sooner after failure
                            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                            continue;
                        }
                    }
                }

                // Check every 2 minutes
                let sleep_duration = std::time::Duration::from_secs(120);
                tokio::time::sleep(sleep_duration).await;
            }
        });
    }
}

/// Diagnostic info about the current token state.
#[derive(Debug, serde::Serialize)]
pub struct TokenInfo {
    pub has_token: bool,
    pub expires_at: u64,
    pub obtained_at: u64,
    pub seconds_until_refresh: u64,
    pub token_age_secs: u64,
    pub refresh_count: u64,
    pub is_refreshing: bool,
}

/// Fetch a new access token and project ID from the GCP metadata server.
/// Returns (access_token, project_id, expires_in_seconds).
async fn fetch_metadata_token(
    client: &reqwest::Client,
) -> Result<(String, String, u64), String> {
    // Get access token from GCP metadata server
    let token_resp: serde_json::Value = client
        .get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
        .header("Metadata-Flavor", "Google")
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("Metadata server error: {e}"))?
        .json()
        .await
        .map_err(|e| format!("Token parse error: {e}"))?;

    let token = token_resp["access_token"]
        .as_str()
        .ok_or("No access_token in metadata response")?
        .to_string();

    // Parse actual expires_in (default 3600 if not present)
    let expires_in = token_resp["expires_in"]
        .as_u64()
        .unwrap_or(3600);

    // Get project ID
    let project_id = client
        .get("http://metadata.google.internal/computeMetadata/v1/project/project-id")
        .header("Metadata-Flavor", "Google")
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("Metadata project error: {e}"))?
        .text()
        .await
        .map_err(|e| format!("Project ID parse error: {e}"))?;

    Ok((token, project_id, expires_in))
}
