use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub master_key: String,
    pub default_model: String,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
    #[serde(default)]
    pub rate_limit: RateLimitConfig,
    #[serde(default)]
    pub retry: RetryConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RateLimitConfig {
    /// Whether rate limiting is enabled
    #[serde(default = "default_false")]
    pub enabled: bool,
    /// Global requests per minute limit (total across all clients)
    #[serde(default = "default_rpm")]
    pub requests_per_minute: u32,
    /// Per API key requests per minute limit
    #[serde(default = "default_rpm_per_key")]
    pub requests_per_minute_per_key: u32,
    /// Per IP address requests per minute limit
    #[serde(default = "default_rpm_per_ip")]
    pub requests_per_minute_per_ip: u32,
    /// Burst size — allows short bursts above the steady rate
    #[serde(default = "default_burst")]
    pub burst_size: u32,
}

// Rate limit defaults — Claude Code friendly (disabled by default, high limits)
fn default_false() -> bool { false }
fn default_rpm() -> u32 { 300 }
fn default_rpm_per_key() -> u32 { 120 }
fn default_rpm_per_ip() -> u32 { 120 }
fn default_burst() -> u32 { 50 }

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: default_false(),
            requests_per_minute: default_rpm(),
            requests_per_minute_per_key: default_rpm_per_key(),
            requests_per_minute_per_ip: default_rpm_per_ip(),
            burst_size: default_burst(),
        }
    }
}

/// Configuration for automatic retry on upstream errors.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RetryConfig {
    /// Whether automatic retry is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Maximum number of retry attempts (0 = no retry, 1 = retry once, etc.)
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Initial delay in milliseconds before first retry
    #[serde(default = "default_initial_delay_ms")]
    pub initial_delay_ms: u64,
    /// Maximum delay in milliseconds between retries (cap for exponential backoff)
    #[serde(default = "default_max_delay_ms")]
    pub max_delay_ms: u64,
    /// Multiplier for exponential backoff (e.g., 2.0 means delay doubles each retry)
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f64,
    /// HTTP status codes that should trigger a retry
    /// Default: 401 (token expired), 429 (rate limited), 500, 502, 503, 504
    #[serde(default = "default_retry_on_statuses")]
    pub retry_on_statuses: Vec<u16>,
}

fn default_true() -> bool { true }
fn default_max_retries() -> u32 { 3 }
fn default_initial_delay_ms() -> u64 { 500 }
fn default_max_delay_ms() -> u64 { 10000 }
fn default_backoff_multiplier() -> f64 { 2.0 }
fn default_retry_on_statuses() -> Vec<u16> { vec![401, 429, 500, 502, 503, 504] }

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            max_retries: default_max_retries(),
            initial_delay_ms: default_initial_delay_ms(),
            max_delay_ms: default_max_delay_ms(),
            backoff_multiplier: default_backoff_multiplier(),
            retry_on_statuses: default_retry_on_statuses(),
        }
    }
}

impl RetryConfig {
    /// Calculate the delay for a given retry attempt (0-indexed).
    /// Uses exponential backoff with jitter.
    pub fn delay_for_attempt(&self, attempt: u32) -> std::time::Duration {
        let base_delay = self.initial_delay_ms as f64
            * self.backoff_multiplier.powi(attempt as i32);
        let capped_delay = base_delay.min(self.max_delay_ms as f64);

        // Add jitter: randomize between 50%-100% of the calculated delay
        // This prevents thundering herd when multiple requests retry simultaneously
        let jittered_delay = capped_delay * (0.5 + rand_jitter() * 0.5);

        std::time::Duration::from_millis(jittered_delay as u64)
    }

    /// Check if a given HTTP status code should trigger a retry.
    pub fn should_retry_status(&self, status: u16) -> bool {
        self.retry_on_statuses.contains(&status)
    }
}

/// Simple deterministic pseudo-random jitter based on thread id and time.
/// Avoids adding `rand` crate as a dependency.
fn rand_jitter() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    // Map to 0.0..1.0 using simple hash
    let hash = (nanos.wrapping_mul(2654435761)) % 1_000_000;
    hash as f64 / 1_000_000.0
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub name: String,
    pub vertex_name: String,
    pub publisher: String, // "google", "openapi", "anthropic"
    pub region: String,
}

impl Config {
    pub fn load(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path).map_err(|e| format!("Cannot read config: {e}"))?;
        serde_yaml::from_str(&content).map_err(|e| format!("Invalid config: {e}"))
    }

    pub fn find_model(&self, name: &str) -> Option<&ModelConfig> {
        self.models.iter().find(|m| m.name == name)
    }
}
