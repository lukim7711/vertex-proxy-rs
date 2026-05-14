use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AuthManager {
    token: Arc<RwLock<CachedToken>>,
    client: reqwest::Client,
}

struct CachedToken {
    value: String,
    project_id: String,
    expires_at: u64,
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
            })),
            client,
        }
    }

    pub async fn get_token(&self) -> Result<(String, String), String> {
        // Fast path: read lock, check cache
        {
            let cached = self.token.read().await;
            if now_secs() < cached.expires_at && !cached.value.is_empty() {
                return Ok((cached.value.clone(), cached.project_id.clone()));
            }
        }

        // Slow path: acquire write lock, double-check, then fetch
        let mut cached = self.token.write().await;

        // Double-check: another task may have refreshed while we waited for the lock
        if now_secs() < cached.expires_at && !cached.value.is_empty() {
            return Ok((cached.value.clone(), cached.project_id.clone()));
        }

        let (token, project_id) = fetch_metadata_token(&self.client).await?;

        cached.value = token.clone();
        cached.project_id = project_id.clone();
        cached.expires_at = now_secs() + 3000; // ~50 min buffer (tokens last 60 min)

        Ok((token, project_id))
    }
}

async fn fetch_metadata_token(client: &reqwest::Client) -> Result<(String, String), String> {
    // Get access token from GCP metadata server
    let token_resp: serde_json::Value = client
        .get("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
        .header("Metadata-Flavor", "Google")
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

    // Get project ID
    let project_id = client
        .get("http://metadata.google.internal/computeMetadata/v1/project/project-id")
        .header("Metadata-Flavor", "Google")
        .send()
        .await
        .map_err(|e| format!("Metadata project error: {e}"))?
        .text()
        .await
        .map_err(|e| format!("Project ID parse error: {e}"))?;

    Ok((token, project_id))
}
