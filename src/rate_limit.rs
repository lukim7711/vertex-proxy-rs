//! Sliding window rate limiter for vertex-proxy-rs.
//!
//! Tracks request counts per key (API key), per IP, and globally using
//! a sliding window of timestamped events. Old entries outside the window
//! are lazily cleaned on each check, keeping memory usage bounded.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;

use crate::config::RateLimitConfig;

/// Result of a rate limit check.
#[derive(Debug, Clone)]
pub struct RateLimitResult {
    /// Whether the request is allowed
    pub allowed: bool,
    /// Limit for the scope that was checked
    pub limit: u32,
    /// Remaining requests in the current window
    pub remaining: u32,
    /// Unix timestamp when the window resets
    pub reset_at: u64,
    /// If not allowed, how many seconds until the client should retry
    pub retry_after_secs: Option<u32>,
    /// Which scope triggered the rate limit (for error messages)
    pub scope: RateLimitScope,
}

/// Which scope triggered the rate limit.
#[derive(Debug, Clone, PartialEq)]
pub enum RateLimitScope {
    Global,
    PerKey,
    PerIp,
    None,
}

/// Internal state for a single sliding window counter.
#[derive(Debug, Clone, Default)]
struct WindowState {
    /// Timestamps of requests within the current window
    timestamps: Vec<Instant>,
}

impl WindowState {
    /// Remove entries older than `window_secs` seconds ago and return the
    /// count of remaining (active) entries.
    fn cleanup_and_count(&mut self, window_secs: u64) -> usize {
        let cutoff = Instant::now() - std::time::Duration::from_secs(window_secs);
        self.timestamps.retain(|ts| *ts > cutoff);
        self.timestamps.len()
    }

    /// Count active entries without modifying the collection.
    /// This is a read-only count that checks which timestamps are still
    /// within the window. Less precise than cleanup_and_count but
    /// doesn't require &mut self.
    fn count_active(&self, window_secs: u64) -> usize {
        let cutoff = Instant::now() - std::time::Duration::from_secs(window_secs);
        self.timestamps.iter().filter(|ts| **ts > cutoff).count()
    }

    /// Record a new request timestamp.
    fn record(&mut self) {
        self.timestamps.push(Instant::now());
    }
}

/// Thread-safe, async-compatible sliding window rate limiter.
#[derive(Clone)]
pub struct RateLimiter {
    /// Global request window
    global: Arc<RwLock<WindowState>>,
    /// Per API key windows
    by_key: Arc<RwLock<HashMap<String, WindowState>>>,
    /// Per IP address windows
    by_ip: Arc<RwLock<HashMap<String, WindowState>>>,
    /// Configuration
    config: RateLimitConfig,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            global: Arc::new(RwLock::new(WindowState::default())),
            by_key: Arc::new(RwLock::new(HashMap::new())),
            by_ip: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Check whether a request is allowed, and if so, record it.
    ///
    /// Returns a `RateLimitResult` with details about the current rate
    /// limit status. If the request is allowed, it is automatically
    /// recorded in all applicable windows.
    pub async fn check_and_record(
        &self,
        api_key: Option<&str>,
        ip: Option<&str>,
    ) -> RateLimitResult {
        if !self.config.enabled {
            return RateLimitResult {
                allowed: true,
                limit: 0,
                remaining: u32::MAX,
                reset_at: 0,
                retry_after_secs: None,
                scope: RateLimitScope::None,
            };
        }

        let window_secs: u64 = 60; // 1 minute sliding window

        // ── 1. Check global limit ──────────────────────────────────────
        {
            let mut global = self.global.write().await;
            let count = global.cleanup_and_count(window_secs);
            let effective_limit = self.config.requests_per_minute + self.config.burst_size;
            if count as u32 >= effective_limit {
                let remaining_secs = Self::estimate_retry_after(&global.timestamps, window_secs);
                return RateLimitResult {
                    allowed: false,
                    limit: self.config.requests_per_minute,
                    remaining: 0,
                    reset_at: Self::current_epoch() + remaining_secs as u64,
                    retry_after_secs: Some(remaining_secs),
                    scope: RateLimitScope::Global,
                };
            }
        }

        // ── 2. Check per-key limit ─────────────────────────────────────
        if let Some(key) = api_key {
            let mut by_key = self.by_key.write().await;
            let entry = by_key.entry(key.to_string()).or_default();
            let count = entry.cleanup_and_count(window_secs);
            let effective_limit = self.config.requests_per_minute_per_key + self.config.burst_size;
            if count as u32 >= effective_limit {
                let remaining_secs = Self::estimate_retry_after(&entry.timestamps, window_secs);
                return RateLimitResult {
                    allowed: false,
                    limit: self.config.requests_per_minute_per_key,
                    remaining: 0,
                    reset_at: Self::current_epoch() + remaining_secs as u64,
                    retry_after_secs: Some(remaining_secs),
                    scope: RateLimitScope::PerKey,
                };
            }
        }

        // ── 3. Check per-IP limit ──────────────────────────────────────
        if let Some(ip_addr) = ip {
            let mut by_ip = self.by_ip.write().await;
            let entry = by_ip.entry(ip_addr.to_string()).or_default();
            let count = entry.cleanup_and_count(window_secs);
            let effective_limit = self.config.requests_per_minute_per_ip + self.config.burst_size;
            if count as u32 >= effective_limit {
                let remaining_secs = Self::estimate_retry_after(&entry.timestamps, window_secs);
                return RateLimitResult {
                    allowed: false,
                    limit: self.config.requests_per_minute_per_ip,
                    remaining: 0,
                    reset_at: Self::current_epoch() + remaining_secs as u64,
                    retry_after_secs: Some(remaining_secs),
                    scope: RateLimitScope::PerIp,
                };
            }
        }

        // ── 4. All checks passed — record the request ──────────────────
        // We record in each window and track counts for computing remaining.
        let g_count: u32;
        {
            let mut global = self.global.write().await;
            global.record();
            g_count = global.count_active(window_secs) as u32;
        }

        let mut k_count: u32 = 0;
        let mut k_limit: u32 = 0;
        if let Some(key) = api_key {
            let mut by_key = self.by_key.write().await;
            by_key.entry(key.to_string()).or_default().record();
            k_count = by_key.get(key).map(|e| e.count_active(window_secs) as u32).unwrap_or(0);
            k_limit = self.config.requests_per_minute_per_key;
        }

        let mut i_count: u32 = 0;
        let mut i_limit: u32 = 0;
        if let Some(ip_addr) = ip {
            let mut by_ip = self.by_ip.write().await;
            by_ip.entry(ip_addr.to_string()).or_default().record();
            i_count = by_ip.get(ip_addr).map(|e| e.count_active(window_secs) as u32).unwrap_or(0);
            i_limit = self.config.requests_per_minute_per_ip;
        }

        // ── 5. Compute remaining for the tightest scope ────────────────
        let g_limit = self.config.requests_per_minute;
        let g_remaining = g_limit.saturating_sub(g_count);
        let k_remaining = if k_limit > 0 { k_limit.saturating_sub(k_count) } else { u32::MAX };
        let i_remaining = if i_limit > 0 { i_limit.saturating_sub(i_count) } else { u32::MAX };

        let (remaining, limit, reset_at) = if k_remaining <= g_remaining && k_remaining <= i_remaining {
            (k_remaining, k_limit, Self::current_epoch() + 60)
        } else if g_remaining <= i_remaining {
            (g_remaining, g_limit, Self::current_epoch() + 60)
        } else {
            (i_remaining, i_limit, Self::current_epoch() + 60)
        };

        RateLimitResult {
            allowed: true,
            limit,
            remaining,
            reset_at,
            retry_after_secs: None,
            scope: RateLimitScope::None,
        }
    }

    /// Get the current rate limit status (without recording a request).
    /// Used for the admin endpoint.
    pub async fn get_status(&self) -> RateLimitStatus {
        let window_secs: u64 = 60;

        let global_used = {
            let mut global = self.global.write().await;
            global.cleanup_and_count(window_secs) as u32
        };

        let by_key_status = {
            let mut by_key = self.by_key.write().await;
            let mut map = HashMap::new();
            for (key, state) in by_key.iter_mut() {
                let count = state.cleanup_and_count(window_secs) as u32;
                map.insert(key.clone(), WindowStatus {
                    used: count,
                    limit: self.config.requests_per_minute_per_key,
                });
            }
            map
        };

        let by_ip_status = {
            let mut by_ip = self.by_ip.write().await;
            let mut map = HashMap::new();
            for (ip, state) in by_ip.iter_mut() {
                let count = state.cleanup_and_count(window_secs) as u32;
                map.insert(ip.clone(), WindowStatus {
                    used: count,
                    limit: self.config.requests_per_minute_per_ip,
                });
            }
            map
        };

        RateLimitStatus {
            global: WindowStatus {
                used: global_used,
                limit: self.config.requests_per_minute,
            },
            by_key: by_key_status,
            by_ip: by_ip_status,
            config: self.config.clone(),
        }
    }

    /// Periodic cleanup task to prune stale entries from the IP and key maps.
    /// Should be spawned as a background task.
    pub async fn cleanup_task(&self) {
        let window_secs: u64 = 60;
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;

            // Clean global
            {
                let mut global = self.global.write().await;
                global.cleanup_and_count(window_secs);
            }

            // Clean by_key — remove entries with no active timestamps
            {
                let mut by_key = self.by_key.write().await;
                by_key.retain(|_, state| {
                    state.cleanup_and_count(window_secs) > 0
                });
            }

            // Clean by_ip — remove entries with no active timestamps
            {
                let mut by_ip = self.by_ip.write().await;
                by_ip.retain(|_, state| {
                    state.cleanup_and_count(window_secs) > 0
                });
            }

            tracing::debug!("Rate limiter periodic cleanup completed");
        }
    }

    /// Estimate how many seconds until the oldest request in the window
    /// expires, which is approximately when a slot will free up.
    fn estimate_retry_after(timestamps: &[Instant], window_secs: u64) -> u32 {
        if timestamps.is_empty() {
            return 1;
        }
        // Find the oldest timestamp
        let oldest = timestamps.iter().min().unwrap();
        let elapsed = Instant::now().duration_since(*oldest);
        let remaining = std::time::Duration::from_secs(window_secs).saturating_sub(elapsed);
        // Add 1 second buffer to avoid immediate retry hitting the same window
        (remaining.as_secs() as u32).saturating_add(1).min(60)
    }

    fn current_epoch() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

/// Status snapshot for the admin endpoint.
#[derive(Debug, serde::Serialize)]
pub struct RateLimitStatus {
    pub global: WindowStatus,
    pub by_key: HashMap<String, WindowStatus>,
    pub by_ip: HashMap<String, WindowStatus>,
    pub config: RateLimitConfig,
}

#[derive(Debug, serde::Serialize)]
pub struct WindowStatus {
    pub used: u32,
    pub limit: u32,
}
