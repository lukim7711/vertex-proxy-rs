use axum::{extract::State, http::{HeaderMap, HeaderValue, StatusCode}, Json, response::IntoResponse};
use serde_json::{json, Value};

use crate::models::{build_vertex_url, resolve_model, Publisher};
use crate::rate_limit::RateLimitScope;
use crate::retry::{with_retry, RetryableError};
use crate::AppState;

pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    // Auth check
    let key = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .or_else(|| {
            headers
                .get("x-api-key")
                .and_then(|v| v.to_str().ok())
        });

    if key != Some(&state.config.master_key) {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": {"message": "Invalid API key", "type": "authentication_error"}})),
        ));
    }

    // ── Rate limiting check ──────────────────────────────────────────
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

        let mut response = axum::response::Response::new(
            serde_json::to_string(&json!({
                "error": {
                    "type": "rate_limit_error",
                    "message": format!("{}. Retry after {} seconds.", scope_msg, retry_after)
                }
            }))
            .unwrap_or_default()
            .into(),
        );
        *response.status_mut() = StatusCode::TOO_MANY_REQUESTS;
        response.headers_mut().insert("Retry-After", HeaderValue::from_str(&retry_after.to_string()).unwrap_or_else(|_| HeaderValue::from_static("30")));
        response.headers_mut().insert("X-RateLimit-Limit", HeaderValue::from_str(&rl_result.limit.to_string()).unwrap_or_else(|_| HeaderValue::from_static("0")));
        response.headers_mut().insert("X-RateLimit-Remaining", HeaderValue::from_str(&rl_result.remaining.to_string()).unwrap_or_else(|_| HeaderValue::from_static("0")));
        response.headers_mut().insert("X-RateLimit-Reset", HeaderValue::from_str(&rl_result.reset_at.to_string()).unwrap_or_else(|_| HeaderValue::from_static("0")));
        response.headers_mut().insert("Content-Type", HeaderValue::from_static("application/json"));
        return Ok(response);
    }

    let rl_limit = rl_result.limit;
    let rl_remaining = rl_result.remaining;
    let rl_reset = rl_result.reset_at;

    let model_name = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(&state.config.default_model);

    let resolved = resolve_model(&state.config, &state.dynamic_models, model_name).await.map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": e, "type": "invalid_request_error"}})))
    })?;

    let (token, project_id) = state.auth.get_token().await.map_err(|e| {
        tracing::error!("Auth error: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": {"message": format!("Failed to obtain auth token: {e}"), "type": "server_error"}})))
    })?;

    let url = build_vertex_url(
        &project_id,
        &resolved.region,
        &resolved.publisher,
        &resolved.vertex_name,
    );

    let result_json = match resolved.publisher {
        Publisher::OpenApi => {
            // Direct passthrough — already in OpenAI format
            let mut oai_body = body.clone();
            oai_body["model"] = json!(resolved.vertex_name);

            let retry_config = state.config.retry.clone();
            let resp = with_retry(&state, |attempt| {
                let url = url.clone();
                let token = token.clone();
                let oai_body = oai_body.clone();
                let client = state.client.clone();
                let retry_config = retry_config.clone();

                async move {
                    let resp = client
                        .post(&url)
                        .header("Authorization", format!("Bearer {token}"))
                        .json(&oai_body)
                        .send()
                        .await
                        .map_err(|e| RetryableError::transport(format!("Upstream request failed: {e}")))?;

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
                            .unwrap_or("Unknown upstream error");

                        tracing::warn!("OpenAPI upstream returned {} (attempt {}): {}", status_code, attempt, error_msg);

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
                (StatusCode::BAD_GATEWAY, Json(json!({"error": {"message": e.message, "type": "upstream_error"}})))
            })?;

            let resp_body = resp.json::<Value>().await.map_err(|e| {
                (StatusCode::BAD_GATEWAY, Json(json!({"error": {"message": e.to_string(), "type": "upstream_error"}})))
            })?;

            resp_body
        }
        Publisher::Google => {
            // Convert OpenAI messages → Gemini contents
            let messages = body
                .get("messages")
                .and_then(|m| m.as_array())
                .ok_or_else(|| {
                    (StatusCode::BAD_REQUEST, Json(json!({"error": {"message": "messages required", "type": "invalid_request_error"}})))
                })?;
            let max_tokens = body
                .get("max_tokens")
                .and_then(|t| t.as_u64())
                .unwrap_or(4096);

            // Extract system message and build contents
            let mut system_text: Option<String> = None;
            let mut contents: Vec<Value> = Vec::new();

            for m in messages {
                let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                let text = m.get("content").and_then(|c| c.as_str()).unwrap_or("");

                if role == "system" {
                    system_text = Some(text.to_string());
                    continue;
                }

                let gemini_role = if role == "user" { "user" } else { "model" };

                // Merge consecutive same-role turns
                if let Some(last) = contents.last_mut() {
                    if last.get("role").and_then(|r| r.as_str()) == Some(gemini_role) {
                        if let Some(parts) = last.get_mut("parts").and_then(|p| p.as_array_mut()) {
                            parts.push(json!({"text": text}));
                            continue;
                        }
                    }
                }

                contents.push(json!({"role": gemini_role, "parts": [{"text": text}]}));
            }

            let mut gemini_body = json!({
                "contents": contents,
                "generationConfig": {"maxOutputTokens": max_tokens}
            });

            if let Some(sys) = system_text {
                gemini_body["systemInstruction"] = json!({"parts": [{"text": sys}]});
            }

            let retry_config = state.config.retry.clone();
            let resp = with_retry(&state, |attempt| {
                let url = url.clone();
                let token = token.clone();
                let gemini_body = gemini_body.clone();
                let client = state.client.clone();
                let retry_config = retry_config.clone();

                async move {
                    let resp = client
                        .post(&url)
                        .header("Authorization", format!("Bearer {token}"))
                        .json(&gemini_body)
                        .send()
                        .await
                        .map_err(|e| RetryableError::transport(format!("Upstream request failed: {e}")))?;

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
                            .unwrap_or("Unknown upstream error");

                        tracing::warn!("Gemini upstream returned {} (attempt {}): {}", status_code, attempt, error_msg);

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
                (StatusCode::BAD_GATEWAY, Json(json!({"error": {"message": e.message, "type": "upstream_error"}})))
            })?;

            let resp_body = resp.json::<Value>().await.map_err(|e| {
                (StatusCode::BAD_GATEWAY, Json(json!({"error": {"message": e.to_string(), "type": "upstream_error"}})))
            })?;

            // Convert Gemini response → OpenAI format
            let text = resp_body
                .get("candidates")
                .and_then(|c| c.as_array())
                .and_then(|c| c.first())
                .and_then(|c| c.get("content"))
                .and_then(|c| c.get("parts"))
                .and_then(|p| p.as_array())
                .and_then(|p| p.first())
                .and_then(|p| p.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("");

            let usage = resp_body.get("usageMetadata");
            json!({
                "id": format!("chatcmpl-{}", uuid::Uuid::new_v4().to_string().replace('-', "")[..24].to_string()),
                "object": "chat.completion",
                "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": usage.and_then(|u| u.get("promptTokenCount")).and_then(|t| t.as_u64()).unwrap_or(0),
                    "completion_tokens": usage.and_then(|u| u.get("candidatesTokenCount")).and_then(|t| t.as_u64()).unwrap_or(0),
                    "total_tokens": usage.and_then(|u| u.get("totalTokenCount")).and_then(|t| t.as_u64()).unwrap_or(0),
                }
            })
        }
        Publisher::Anthropic => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"message": "Use /v1/messages for Anthropic models", "type": "invalid_request_error"}})),
            ));
        }
    };

    let mut resp = Json(result_json).into_response();
    resp.headers_mut().insert("X-RateLimit-Limit", HeaderValue::from_str(&rl_limit.to_string()).unwrap_or_else(|_| HeaderValue::from_static("0")));
    resp.headers_mut().insert("X-RateLimit-Remaining", HeaderValue::from_str(&rl_remaining.to_string()).unwrap_or_else(|_| HeaderValue::from_static("0")));
    resp.headers_mut().insert("X-RateLimit-Reset", HeaderValue::from_str(&rl_reset.to_string()).unwrap_or_else(|_| HeaderValue::from_static("0")));
    Ok(resp)
}
