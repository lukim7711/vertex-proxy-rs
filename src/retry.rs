//! Retry utilities for upstream API calls.
//!
//! Provides automatic retry with exponential backoff for transient errors.
//! When a 401 is received, forces a token refresh before retrying.

use crate::config::RetryConfig;
use crate::AppState;

/// Result of a retried request attempt.
#[derive(Debug)]
#[allow(dead_code)]
pub struct RetryAttempt {
    /// Which attempt number (0 = first, 1 = first retry, etc.)
    pub attempt: u32,
    /// The HTTP status code received
    pub status: u16,
    /// Whether this was a retry (attempt > 0)
    pub was_retried: bool,
}

/// Execute an async operation with automatic retry on transient errors.
///
/// The operation receives the current attempt number and should return:
/// - `Ok(T)` on success
/// - `Err(RetryableError)` on a retryable error
///
/// On 401 errors, the token is force-refreshed before retrying.
pub async fn with_retry<F, Fut, T>(
    state: &AppState,
    operation: F,
) -> Result<T, RetryableError>
where
    F: Fn(u32) -> Fut,
    Fut: std::future::Future<Output = Result<T, RetryableError>>,
{
    let retry_config = &state.config.retry;

    if !retry_config.enabled {
        return operation(0).await;
    }

    let max_retries = retry_config.max_retries;
    let mut last_error: Option<RetryableError> = None;

    for attempt in 0..=max_retries {
        match operation(attempt).await {
            Ok(result) => {
                if attempt > 0 {
                    tracing::info!("Request succeeded on attempt {}", attempt);
                }
                return Ok(result);
            }
            Err(e) => {
                let should_retry = e.retryable && attempt < max_retries;

                if should_retry {
                    let delay = retry_config.delay_for_attempt(attempt);
                    tracing::warn!(
                        "Request failed (attempt {}/{}): {}. Retrying in {:?}...",
                        attempt + 1,
                        max_retries + 1,
                        e.message,
                        delay,
                    );

                    // If 401, force-refresh the token before retrying
                    if e.status_code == Some(401) {
                        tracing::info!("401 received, force-refreshing auth token before retry");
                        match state.auth.force_refresh().await {
                            Ok(_) => tracing::info!("Token force-refreshed successfully"),
                            Err(refresh_err) => {
                                tracing::error!("Token force-refresh failed: {refresh_err}");
                                // Still retry — the old token might work or the issue might be transient
                            }
                        }
                    }

                    tokio::time::sleep(delay).await;
                    last_error = Some(e);
                } else {
                    tracing::error!(
                        "Request failed permanently (attempt {}/{}): {}",
                        attempt + 1,
                        max_retries + 1,
                        e.message,
                    );
                    return Err(e);
                }
            }
        }
    }

    // All retries exhausted
    Err(last_error.unwrap_or_else(|| RetryableError {
        message: "All retry attempts exhausted".to_string(),
        retryable: false,
        status_code: None,
    }))
}

/// An error that may be retryable.
#[derive(Debug)]
pub struct RetryableError {
    /// Human-readable error message
    pub message: String,
    /// Whether this error is worth retrying
    pub retryable: bool,
    /// HTTP status code, if available
    pub status_code: Option<u16>,
}

impl RetryableError {
    /// Create a retryable error from an HTTP status code and message.
    pub fn from_status(status: u16, message: String, retry_config: &RetryConfig) -> Self {
        Self {
            message,
            retryable: retry_config.should_retry_status(status),
            status_code: Some(status),
        }
    }

    /// Create a non-retryable error.
    #[allow(dead_code)]
    pub fn permanent(message: String) -> Self {
        Self {
            message,
            retryable: false,
            status_code: None,
        }
    }

    /// Create a retryable error for connection/transport failures.
    pub fn transport(message: String) -> Self {
        Self {
            message,
            retryable: true,
            status_code: None,
        }
    }
}

/// Check if an upstream HTTP response status should trigger a retry.
/// This is a simple helper for use outside the full retry loop.
#[allow(dead_code)]
pub fn is_retryable_status(status: u16, retry_config: &RetryConfig) -> bool {
    retry_config.should_retry_status(status)
}
