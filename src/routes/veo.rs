use axum::{extract::State, http::{HeaderMap, StatusCode}, Json};
use serde_json::{json, Value};

use crate::models::{build_veo_predict_url, build_veo_fetch_url};
use crate::rate_limit::RateLimitScope;
use crate::retry::{with_retry, RetryableError};
use crate::AppState;

/// Default VEO model if not specified
const DEFAULT_VEO_MODEL: &str = "veo-3.1-fast-generate-001";
const DEFAULT_VEO_REGION: &str = "us-central1";

// ---------------------------------------------------------------------------
// Auth helper (shared with other routes)
// ---------------------------------------------------------------------------

fn check_veo_auth(headers: &HeaderMap, master_key: &str) -> Result<(), (StatusCode, Json<Value>)> {
    let key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            headers
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
        });

    if key != Some(master_key) {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": {"type": "authentication_error", "message": "Invalid API key"}})),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// POST /v1/veo/generate — Submit video generation job
// ---------------------------------------------------------------------------

pub async fn generate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    check_veo_auth(&headers, &state.config.master_key)?;

    // Rate limiting check
    if let Some(rate_limit_err) = check_rate_limit(&state, &headers).await {
        return Err(rate_limit_err);
    }

    // Validate prompt
    let prompt = body
        .get("prompt")
        .and_then(|p| p.as_str())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": "prompt is required", "type": "invalid_request_error"}})),
            )
        })?;

    if prompt.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"message": "prompt cannot be empty", "type": "invalid_request_error"}})),
        ));
    }

    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(DEFAULT_VEO_MODEL);

    let region = body
        .get("region")
        .and_then(|r| r.as_str())
        .unwrap_or(DEFAULT_VEO_REGION);

    // Get auth token
    let (token, project_id) = state.auth.get_token().await.map_err(|e| {
        tracing::error!("Auth error: {e}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"message": format!("Failed to obtain auth token: {e}"), "type": "server_error"}})),
        )
    })?;

    // Build VEO predictLongRunning request
    let mut instance = json!({
        "prompt": prompt
    });

    // Optional parameters
    if let Some(neg) = body.get("negativePrompt").and_then(|v| v.as_str()) {
        instance["negativePrompt"] = json!(neg);
    }
    if let Some(dur) = body.get("duration").and_then(|v| v.as_u64()) {
        instance["duration"] = json!(dur);
    }
    if let Some(res) = body.get("resolution").and_then(|v| v.as_str()) {
        instance["resolution"] = json!(res);
    }
    if let Some(ar) = body.get("aspectRatio").and_then(|v| v.as_str()) {
        instance["aspectRatio"] = json!(ar);
    }
    if let Some(seed) = body.get("seed").and_then(|v| v.as_u64()) {
        instance["seed"] = json!(seed);
    }
    if let Some(nv) = body.get("numVideos").and_then(|v| v.as_u64()) {
        instance["numVideos"] = json!(nv);
    }

    let sample_count = body
        .get("sampleCount")
        .and_then(|v| v.as_u64())
        .unwrap_or(1);

    let veo_body = json!({
        "instances": [instance],
        "parameters": {
            "sampleCount": sample_count
        }
    });

    let url = build_veo_predict_url(&project_id, region, model);

    tracing::info!(model = model, region = region, prompt_len = prompt.len(), "VEO generate request");

    // Send with retry support
    let retry_config = state.config.retry.clone();
    let resp = with_retry(&state, |attempt| {
        let url = url.clone();
        let token = token.clone();
        let veo_body = veo_body.clone();
        let client = state.client.clone();
        let retry_config = retry_config.clone();

        async move {
            let resp = client
                .post(&url)
                .header("Authorization", format!("Bearer {token}"))
                .json(&veo_body)
                .send()
                .await
                .map_err(|e| RetryableError::transport(format!("VEO upstream request failed: {e}")))?;

            let status = resp.status();
            if status.is_success() {
                Ok(resp)
            } else {
                let status_code = status.as_u16();
                let resp_body: Value = resp.json().await.unwrap_or(json!({"error": "Unknown error"}));
                let error_msg = resp_body
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .or_else(|| resp_body.get("error").and_then(|e| e.as_str()))
                    .unwrap_or("Unknown upstream error");

                tracing::warn!("VEO upstream returned {} (attempt {}): {}", status_code, attempt, error_msg);

                Err(RetryableError::from_status(
                    status_code,
                    format!("{} {}", status_code, error_msg),
                    &retry_config,
                ))
            }
        }
    })
    .await
    .map_err(|e| {
        tracing::error!("VEO generate failed after retries: {}", e.message);
        (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": {"message": e.message, "type": "upstream_error"}})),
        )
    })?;

    let resp_body: Value = resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": {"message": format!("Failed to parse upstream response: {e}"), "type": "upstream_error"}})),
        )
    })?;

    // Extract operation name from response
    let operation_name = resp_body
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("unknown");

    tracing::info!(operation = operation_name, "VEO job submitted");

    Ok(Json(json!({
        "id": operation_name,
        "model": model,
        "status": "submitted",
        "operationName": operation_name,
        "message": "Video generation job submitted. Poll /v1/veo/result with operationName to check status."
    })))
}

// ---------------------------------------------------------------------------
// POST /v1/veo/result — Poll for video generation result
// ---------------------------------------------------------------------------

pub async fn fetch_result(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    check_veo_auth(&headers, &state.config.master_key)?;

    // Rate limiting check
    if let Some(rate_limit_err) = check_rate_limit(&state, &headers).await {
        return Err(rate_limit_err);
    }

    let operation_name = body
        .get("operationName")
        .and_then(|o| o.as_str())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": "operationName is required", "type": "invalid_request_error"}})),
            )
        })?;

    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(DEFAULT_VEO_MODEL);

    let region = body
        .get("region")
        .and_then(|r| r.as_str())
        .unwrap_or(DEFAULT_VEO_REGION);

    // Get auth token
    let (token, project_id) = state.auth.get_token().await.map_err(|e| {
        tracing::error!("Auth error: {e}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"message": format!("Failed to obtain auth token: {e}"), "type": "server_error"}})),
        )
    })?;

    // Build fetchPredictOperation request
    let fetch_body = json!({
        "operationName": operation_name
    });

    let url = build_veo_fetch_url(&project_id, region, model);

    tracing::debug!(operation = operation_name, "VEO fetch result");

    // Send with retry support
    let retry_config = state.config.retry.clone();
    let resp = with_retry(&state, |attempt| {
        let url = url.clone();
        let token = token.clone();
        let fetch_body = fetch_body.clone();
        let client = state.client.clone();
        let retry_config = retry_config.clone();

        async move {
            let resp = client
                .post(&url)
                .header("Authorization", format!("Bearer {token}"))
                .json(&fetch_body)
                .send()
                .await
                .map_err(|e| RetryableError::transport(format!("VEO fetch upstream request failed: {e}")))?;

            let status = resp.status();
            if status.is_success() {
                Ok(resp)
            } else {
                let status_code = status.as_u16();
                let resp_body: Value = resp.json().await.unwrap_or(json!({"error": "Unknown error"}));
                let error_msg = resp_body
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .or_else(|| resp_body.get("error").and_then(|e| e.as_str()))
                    .unwrap_or("Unknown upstream error");

                tracing::warn!("VEO fetch upstream returned {} (attempt {}): {}", status_code, attempt, error_msg);

                Err(RetryableError::from_status(
                    status_code,
                    format!("{} {}", status_code, error_msg),
                    &retry_config,
                ))
            }
        }
    })
    .await
    .map_err(|e| {
        tracing::error!("VEO fetch failed after retries: {}", e.message);
        (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": {"message": e.message, "type": "upstream_error"}})),
        )
    })?;

    let resp_body: Value = resp.json().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": {"message": format!("Failed to parse upstream response: {e}"), "type": "upstream_error"}})),
        )
    })?;

    // Check if operation is done
    let done = resp_body
        .get("done")
        .and_then(|d| d.as_bool())
        .unwrap_or(false);

    if done {
        tracing::info!(operation = operation_name, "VEO job completed");
        Ok(Json(json!({
            "status": "completed",
            "operationName": operation_name,
            "done": true,
            "result": resp_body.get("response").cloned().unwrap_or(json!({}))
        })))
    } else {
        tracing::debug!(operation = operation_name, "VEO job still processing");
        Ok(Json(json!({
            "status": "processing",
            "operationName": operation_name,
            "done": false,
            "message": "Video generation in progress. Poll again later."
        })))
    }
}

// ---------------------------------------------------------------------------
// POST /v1/veo/generate-sync — Submit and auto-poll until complete
// ---------------------------------------------------------------------------

pub async fn generate_sync(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    check_veo_auth(&headers, &state.config.master_key)?;

    // Rate limiting check
    if let Some(rate_limit_err) = check_rate_limit(&state, &headers).await {
        return Err(rate_limit_err);
    }

    let poll_interval_secs = body
        .get("pollInterval")
        .and_then(|v| v.as_u64())
        .unwrap_or(5);

    let max_poll_secs = body
        .get("maxPollTime")
        .and_then(|v| v.as_u64())
        .unwrap_or(300);

    // Step 1: Submit the job
    let submit_result = generate(
        State(state.clone()),
        headers.clone(),
        Json(body.clone()),
    )
    .await
    .map_err(|(code, json_val)| (code, json_val))?;

    let operation_name = submit_result
        .get("operationName")
        .and_then(|o| o.as_str())
        .ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": {"message": "No operationName in submit response", "type": "server_error"}})),
            )
        })?
        .to_string();

    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(DEFAULT_VEO_MODEL)
        .to_string();

    let region = body
        .get("region")
        .and_then(|r| r.as_str())
        .unwrap_or(DEFAULT_VEO_REGION)
        .to_string();

    // Step 2: Poll until done or timeout
    let start = std::time::Instant::now();
    let max_duration = std::time::Duration::from_secs(max_poll_secs);
    let poll_interval = std::time::Duration::from_secs(poll_interval_secs.min(30).max(2));

    loop {
        tokio::time::sleep(poll_interval).await;

        if start.elapsed() >= max_duration {
            return Ok(Json(json!({
                "status": "timeout",
                "operationName": operation_name,
                "done": false,
                "message": format!("Video generation still in progress after {}s. Use /v1/veo/result to continue polling.", max_poll_secs)
            })));
        }

        let poll_body = json!({
            "operationName": operation_name,
            "model": model,
            "region": region
        });

        let poll_result = fetch_result(
            State(state.clone()),
            headers.clone(),
            Json(poll_body),
        )
        .await
        .map_err(|(code, json_val)| (code, json_val))?;

        let done = poll_result.get("done").and_then(|d| d.as_bool()).unwrap_or(false);

        if done {
            return Ok(poll_result);
        }

        tracing::debug!(
            operation = %operation_name,
            elapsed = ?start.elapsed(),
            "VEO still processing, polling again"
        );
    }
}

// ---------------------------------------------------------------------------
// Rate limiting helper for VEO endpoints
// ---------------------------------------------------------------------------

async fn check_rate_limit(state: &AppState, headers: &HeaderMap) -> Option<(StatusCode, Json<Value>)> {
    let api_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            headers
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
        });

    let ip = headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .map(|s| s.trim());

    let rl_result = state.rate_limiter.check_and_record(api_key, ip).await;

    if !rl_result.allowed {
        let retry_after = rl_result.retry_after_secs.unwrap_or(30);
        let scope_msg = match rl_result.scope {
            RateLimitScope::Global => "Global rate limit exceeded",
            RateLimitScope::PerKey => "Per-key rate limit exceeded",
            RateLimitScope::PerIp => "Per-IP rate limit exceeded",
            RateLimitScope::None => "Rate limit exceeded",
        };
        Some((
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({
                "error": {
                    "type": "rate_limit_error",
                    "message": format!("{}. Retry after {} seconds.", scope_msg, retry_after)
                }
            })),
        ))
    } else {
        None
    }
}
