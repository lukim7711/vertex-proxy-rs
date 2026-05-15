use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use serde_json::{json, Value};
use std::convert::Infallible;

use crate::config::Config;
use crate::models::{build_vertex_streaming_url, build_vertex_url, resolve_model, Publisher};
use crate::rate_limit::RateLimitScope;
use crate::retry::{with_retry, RetryableError};
use crate::transform::{
    anthropic_to_gemini, anthropic_to_openai, gemini_stream::GeminiStreamState,
    openai_stream::OpenAiStreamState, sse_parser::SseParser,
};
use crate::SignatureCache;
use crate::AppState;

// ---------------------------------------------------------------------------
// Fake streaming — wraps a complete response into SSE events (fallback)
// ---------------------------------------------------------------------------

/// Wrap a complete Anthropic response into SSE stream events.
///
/// This "fakes" streaming by chunking a complete response into SSE events
/// that match the Anthropic streaming protocol. Used as fallback when
/// real streaming is not available (e.g., Anthropic passthrough publisher).
fn response_to_sse(response: Value) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let msg_id = response["id"]
        .as_str()
        .unwrap_or("msg_unknown")
        .to_string();
    let model = response["model"]
        .as_str()
        .unwrap_or("unknown")
        .to_string();
    let stop_reason = response["stop_reason"]
        .as_str()
        .unwrap_or("end_turn")
        .to_string();
    let usage = response.get("usage").cloned().unwrap_or(json!({"input_tokens": 0, "output_tokens": 0}));

    let mut events: Vec<Result<Event, Infallible>> = vec![Ok(Event::default()
        .event("message_start")
        .data(
            json!({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": usage
                }
            })
            .to_string(),
        ))];

    if let Some(blocks) = response["content"].as_array() {
        for (i, block) in blocks.iter().enumerate() {
            let idx = i as u64;
            match block["type"].as_str() {
                Some("thinking") => {
                    let thinking_text = block["thinking"].as_str().unwrap_or("");
                    events.push(Ok(Event::default().event("content_block_start").data(
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "thinking", "thinking": ""}
                        })
                        .to_string(),
                    )));
                    if !thinking_text.is_empty() {
                        for chunk in thinking_text.as_bytes().chunks(256) {
                            let partial = String::from_utf8_lossy(chunk).to_string();
                            events.push(Ok(Event::default().event("content_block_delta").data(
                                json!({
                                    "type": "content_block_delta",
                                    "index": idx,
                                    "delta": {"type": "thinking_delta", "thinking": partial}
                                })
                                .to_string(),
                            )));
                        }
                    }
                    events.push(Ok(Event::default().event("content_block_stop").data(
                        json!({"type": "content_block_stop", "index": idx}).to_string(),
                    )));
                }
                Some("text") => {
                    let text = block["text"].as_str().unwrap_or("");
                    events.push(Ok(Event::default().event("content_block_start").data(
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "text", "text": ""}
                        })
                        .to_string(),
                    )));
                    if !text.is_empty() {
                        for chunk in text.as_bytes().chunks(256) {
                            let partial = String::from_utf8_lossy(chunk).to_string();
                            events.push(Ok(Event::default().event("content_block_delta").data(
                                json!({
                                    "type": "content_block_delta",
                                    "index": idx,
                                    "delta": {"type": "text_delta", "text": partial}
                                })
                                .to_string(),
                            )));
                        }
                    }
                    events.push(Ok(Event::default().event("content_block_stop").data(
                        json!({"type": "content_block_stop", "index": idx}).to_string(),
                    )));
                }
                Some("tool_use") => {
                    let tool_id = block["id"]
                        .as_str()
                        .unwrap_or("toolu_unknown")
                        .to_string();
                    let tool_name = block["name"]
                        .as_str()
                        .unwrap_or("unknown")
                        .to_string();
                    // Preserve _thought_signature for round-trip through Claude Code
                    let mut content_block = json!({
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": {}
                    });
                    if let Some(sig) = block.get("_thought_signature") {
                        content_block["_thought_signature"] = sig.clone();
                    }
                    events.push(Ok(Event::default().event("content_block_start").data(
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": content_block
                        })
                        .to_string(),
                    )));
                    let input_str = serde_json::to_string(&block["input"]).unwrap_or_default();
                    for chunk in input_str.as_bytes().chunks(64) {
                        let partial = String::from_utf8_lossy(chunk).to_string();
                        events.push(Ok(Event::default().event("content_block_delta").data(
                            json!({
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "input_json_delta", "partial_json": partial}
                            })
                            .to_string(),
                        )));
                    }
                    events.push(Ok(Event::default().event("content_block_stop").data(
                        json!({"type": "content_block_stop", "index": idx}).to_string(),
                    )));
                }
                _ => {}
            }
        }
    }

    let output_tokens = usage.get("output_tokens").and_then(|t| t.as_u64()).unwrap_or(0);

    events.push(Ok(Event::default().event("message_delta").data(
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        })
        .to_string(),
    )));

    events.push(
        Ok(Event::default()
            .event("message_stop")
            .data(json!({
                "type": "message_stop",
                "usage": usage
            }).to_string())),
    );

    Sse::new(futures::stream::iter(events))
}

// ---------------------------------------------------------------------------
// Real streaming — reads SSE from Vertex AI and transforms on-the-fly
// ---------------------------------------------------------------------------

/// Receiver-to-Stream adapter for tokio::sync::mpsc::Receiver.
/// This avoids adding tokio-stream as a dependency.
struct ReceiverStream<T> {
    rx: tokio::sync::mpsc::Receiver<T>,
}

impl<T> futures::Stream for ReceiverStream<T> {
    type Item = T;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

/// Stream Gemini SSE chunks from Vertex AI and transform to Anthropic SSE on-the-fly.
fn stream_gemini(
    response: reqwest::Response,
    model: String,
    signatures: SignatureCache,
    thinking_enabled: bool,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        let mut parser = SseParser::new();
        let mut state = GeminiStreamState::new(model);
        state.set_thinking_enabled(thinking_enabled);
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(bytes) => {
                    let data_events = parser.push_bytes(&bytes);
                    for data in data_events {
                        // Parse JSON
                        let chunk: Value = match serde_json::from_str(&data) {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::warn!("Failed to parse Gemini streaming chunk: {e}");
                                continue;
                            }
                        };

                        // Transform to Anthropic SSE events
                        let events = state.process_chunk(&chunk);
                        for event in events {
                            if tx.send(Ok(event)).await.is_err() {
                                // Client disconnected
                                return;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Gemini streaming error: {e}");
                    // Send finish events to gracefully close the stream
                    let finish_events = state.finish(None);
                    for event in finish_events {
                        if tx.send(Ok(event)).await.is_err() {
                            return;
                        }
                    }
                    // Still store any signatures we collected before the error
                    let tool_sigs = state.take_tool_thought_signatures();
                    let text_sigs = state.take_text_thought_signatures();
                    signatures.store_tool_signatures_batch(tool_sigs).await;
                    signatures.store_text_signatures_batch(text_sigs).await;
                    return;
                }
            }
        }

        // Stream ended — if we haven't sent finish events yet, send them now
        let finish_events = state.finish(None);
        for event in finish_events {
            if tx.send(Ok(event)).await.is_err() {
                return;
            }
        }

        // Store thought signatures in server-side cache for round-trip
        let tool_sigs = state.take_tool_thought_signatures();
        let text_sigs = state.take_text_thought_signatures();
        signatures.store_tool_signatures_batch(tool_sigs).await;
        signatures.store_text_signatures_batch(text_sigs).await;
    });

    Sse::new(ReceiverStream { rx })
}

/// Stream OpenAI SSE chunks from Vertex AI and transform to Anthropic SSE on-the-fly.
fn stream_openai(
    response: reqwest::Response,
    model: String,
    thinking_enabled: bool,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        let mut parser = SseParser::new();
        let mut state = OpenAiStreamState::new(model);
        state.set_thinking_enabled(thinking_enabled);
        let mut stream = response.bytes_stream();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(bytes) => {
                    let data_events = parser.push_bytes(&bytes);
                    for data in data_events {
                        let chunk: Value = match serde_json::from_str(&data) {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::warn!("Failed to parse OpenAI streaming chunk: {e}");
                                continue;
                            }
                        };

                        let events = state.process_chunk(&chunk);
                        for event in events {
                            if tx.send(Ok(event)).await.is_err() {
                                return;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("OpenAI streaming error: {e}");
                    let finish_events = state.finish(None);
                    for event in finish_events {
                        if tx.send(Ok(event)).await.is_err() {
                            return;
                        }
                    }
                    return;
                }
            }
        }

        // Stream ended without explicit finish — close gracefully
        let finish_events = state.finish(None);
        for event in finish_events {
            if tx.send(Ok(event)).await.is_err() {
                return;
            }
        }
    });

    Sse::new(ReceiverStream { rx })
}

// ---------------------------------------------------------------------------
// Upstream request with retry support
// ---------------------------------------------------------------------------

/// Send a request to Vertex AI with automatic retry on transient errors.
/// Returns the response on success, or a RetryableError on failure.
async fn send_with_retry(
    state: &AppState,
    url: &str,
    token: &str,
    body: &Value,
) -> Result<reqwest::Response, RetryableError> {
    let retry_config = &state.config.retry;

    let result = with_retry(state, |attempt| {
        let url = url.to_string();
        let token = token.to_string();
        let body = body.clone();
        let client = state.client.clone();
        let retry_config = retry_config.clone();

        async move {
            let resp = client
                .post(&url)
                .header("Authorization", format!("Bearer {token}"))
                .json(&body)
                .send()
                .await
                .map_err(|e| RetryableError::transport(format!("Upstream request failed: {e}")))?;

            let status = resp.status();
            if status.is_success() {
                Ok(resp)
            } else {
                let status_code = status.as_u16();
                // Read the error body for logging
                let resp_body: Value = resp.json().await.unwrap_or(json!({"error": "Unknown error"}));
                let error_msg = resp_body
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .or_else(|| {
                        resp_body
                            .get("error")
                            .and_then(|e| e.as_str())
                    })
                    .unwrap_or("Unknown upstream error");

                tracing::warn!(
                    "Upstream returned {} (attempt {}): {}",
                    status_code,
                    attempt,
                    error_msg,
                );

                Err(RetryableError::from_status(
                    status_code,
                    format!("{} {}", status_code, error_msg),
                    &retry_config,
                ))
            }
        }
    })
    .await;

    result
}

/// Extract error message from an upstream response body.
#[allow(dead_code)]
fn extract_error_msg(resp_body: &Value) -> String {
    resp_body
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .or_else(|| {
            resp_body
                .get("error")
                .and_then(|e| e.as_str())
        })
        .unwrap_or("Unknown upstream error")
        .to_string()
}

// ---------------------------------------------------------------------------
// Main handler
// ---------------------------------------------------------------------------

pub async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    check_auth(&headers, &state.config)?;

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

    // Store rate limit info for adding to response headers later
    let rl_limit = rl_result.limit;
    let rl_remaining = rl_result.remaining;
    let rl_reset = rl_result.reset_at;

    let model_name = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(&state.config.default_model);

    let is_streaming = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);

    // ── Extract thinking parameter ────────────────────────────────────
    let thinking_param = body.get("thinking");

    tracing::info!(
        "Request model={model_name} stream={is_streaming} max_tokens={:?} thinking={:?}",
        body.get("max_tokens").and_then(|t| t.as_u64()),
        thinking_param.is_some(),
    );

    let resolved = resolve_model(&state.config, &state.dynamic_models, model_name).await.map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(json!({"error": e})))
    })?;

    let (token, project_id) = state.auth.get_token().await.map_err(|e| {
        tracing::error!("Auth error: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": {"type": "authentication_error", "message": format!("Failed to obtain auth token: {e}")}})))
    })?;

    let max_tokens = body
        .get("max_tokens")
        .and_then(|t| t.as_u64())
        .unwrap_or(4096);
    let system = body.get("system");
    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": "messages field is required"})),
            )
        })?;

    // =======================================================================
    // REAL STREAMING PATH — Gemini and OpenAPI publishers
    // =======================================================================
    if is_streaming && matches!(resolved.publisher, Publisher::Google | Publisher::OpenApi) {
        let url = build_vertex_streaming_url(
            &project_id,
            &resolved.region,
            &resolved.publisher,
            &resolved.vertex_name,
        );

        match resolved.publisher {
            Publisher::Google => {
                let tools = body
                    .get("tools")
                    .and_then(|t| t.as_array())
                    .map(|t| t.as_slice())
                    .unwrap_or(&[]);

                let gemini_contents = anthropic_to_gemini::messages_to_gemini(messages, &state.signatures).await;

                // Extract thinking config for Gemini
                let (thinking_config, thinking_enabled) =
                    anthropic_to_gemini::extract_thinking_config(thinking_param);

                let mut gemini_body = json!({
                    "contents": gemini_contents,
                    "generationConfig": {"maxOutputTokens": max_tokens},
                });

                if let Some(sys_text) = anthropic_to_gemini::extract_system_text(system) {
                    gemini_body["systemInstruction"] =
                        json!({"parts": [{"text": sys_text}]});
                }

                if !tools.is_empty() {
                    gemini_body["tools"] =
                        json!(anthropic_to_gemini::tools_to_gemini(tools));
                }

                // Add thinkingConfig if thinking mode is enabled
                if let Some(tc) = thinking_config {
                    gemini_body["generationConfig"]["thinkingConfig"] = tc;
                }

                tracing::debug!("Gemini streaming request to {url}");
                let resp = send_with_retry(&state, &url, &token, &gemini_body).await.map_err(|e| {
                    tracing::error!("Gemini streaming request failed after retries: {}", e.message);
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({"error": {"type": "upstream_error", "message": e.message}})),
                    )
                })?;

                let model_name_owned = model_name.to_string();
                let sse = stream_gemini(resp, model_name_owned, state.signatures.clone(), thinking_enabled);
                let mut resp = sse.into_response();
                add_rate_limit_headers(resp.headers_mut(), rl_limit, rl_remaining, rl_reset);
                return Ok(resp);
            }

            Publisher::OpenApi => {
                let tools = body
                    .get("tools")
                    .and_then(|t| t.as_array())
                    .map(|t| t.as_slice())
                    .unwrap_or(&[]);

                // Extract thinking config for OpenAPI
                let (thinking_config, thinking_enabled) =
                    anthropic_to_openai::extract_thinking_config(thinking_param);

                let mut openai_body = json!({
                    "model": resolved.vertex_name,
                    "messages": anthropic_to_openai::messages_to_openai(system, messages),
                    "max_tokens": max_tokens,
                    "stream": true,
                });

                if !tools.is_empty() {
                    openai_body["tools"] =
                        json!(anthropic_to_openai::tools_to_openai(tools));
                }

                // Merge thinking config into body (e.g., reasoning_effort)
                if let Some(tc) = thinking_config {
                    if let Value::Object(map) = tc {
                        for (k, v) in map {
                            openai_body[k] = v;
                        }
                    }
                }

                tracing::debug!("OpenAPI streaming request to {url}");
                let resp = send_with_retry(&state, &url, &token, &openai_body).await.map_err(|e| {
                    tracing::error!("OpenAPI streaming request failed after retries: {}", e.message);
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({"error": {"type": "upstream_error", "message": e.message}})),
                    )
                })?;

                let model_name_owned = model_name.to_string();
                let sse = stream_openai(resp, model_name_owned, thinking_enabled);
                let mut resp = sse.into_response();
                add_rate_limit_headers(resp.headers_mut(), rl_limit, rl_remaining, rl_reset);
                return Ok(resp);
            }

            _ => unreachable!(),
        }
    }

    // =======================================================================
    // NON-STREAMING PATH (or Anthropic passthrough with fake streaming)
    // =======================================================================
    let url = build_vertex_url(
        &project_id,
        &resolved.region,
        &resolved.publisher,
        &resolved.vertex_name,
    );

    // ── Parse thinking config once for non-streaming path ──────────
    let (gemini_thinking_config, gemini_thinking_enabled) =
        anthropic_to_gemini::extract_thinking_config(thinking_param);
    let (openai_thinking_config, openai_thinking_enabled) =
        anthropic_to_openai::extract_thinking_config(thinking_param);

    let result = match resolved.publisher {
        Publisher::OpenApi => {
            let tools = body
                .get("tools")
                .and_then(|t| t.as_array())
                .map(|t| t.as_slice())
                .unwrap_or(&[]);
            let mut openai_body = json!({
                "model": resolved.vertex_name,
                "messages": anthropic_to_openai::messages_to_openai(system, messages),
                "max_tokens": max_tokens,
            });
            if !tools.is_empty() {
                openai_body["tools"] = json!(anthropic_to_openai::tools_to_openai(tools));
            }

            // Merge thinking config into body (e.g., reasoning_effort)
            if let Some(tc) = openai_thinking_config {
                if let Value::Object(map) = tc {
                    for (k, v) in map {
                        openai_body[k] = v;
                    }
                }
            }

            tracing::debug!("OpenAPI request to {url}");
            let resp = send_with_retry(&state, &url, &token, &openai_body).await.map_err(|e| {
                (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": {"type": "upstream_error", "message": e.message}})),
                )
            })?;

            let resp_body = resp.json::<Value>().await.map_err(|e| {
                (StatusCode::BAD_GATEWAY, Json(json!({"error": {"type": "upstream_error", "message": format!("Failed to parse upstream response: {e}")}})))
            })?;

            tracing::debug!("OpenAPI raw response: {resp_body}");

            anthropic_to_openai::response_to_anthropic(&resp_body, model_name, openai_thinking_enabled)
        }

        Publisher::Google => {
            let tools = body
                .get("tools")
                .and_then(|t| t.as_array())
                .map(|t| t.as_slice())
                .unwrap_or(&[]);

            let gemini_contents = anthropic_to_gemini::messages_to_gemini(messages, &state.signatures).await;

            let mut gemini_body = json!({
                "contents": gemini_contents,
                "generationConfig": {"maxOutputTokens": max_tokens},
            });

            if let Some(sys_text) = anthropic_to_gemini::extract_system_text(system) {
                gemini_body["systemInstruction"] = json!({"parts": [{"text": sys_text}]});
            }

            if !tools.is_empty() {
                gemini_body["tools"] = json!(anthropic_to_gemini::tools_to_gemini(tools));
            }

            // Add thinkingConfig if thinking mode is enabled
            if let Some(tc) = gemini_thinking_config {
                gemini_body["generationConfig"]["thinkingConfig"] = tc;
            }

            tracing::debug!("Gemini request to {url}");
            let resp = send_with_retry(&state, &url, &token, &gemini_body).await.map_err(|e| {
                (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": {"type": "upstream_error", "message": e.message}})),
                )
            })?;

            let resp_body = resp.json::<Value>().await.map_err(|e| {
                (StatusCode::BAD_GATEWAY, Json(json!({"error": {"type": "upstream_error", "message": format!("Failed to parse upstream response: {e}")}})))
            })?;

            anthropic_to_gemini::response_to_anthropic(&resp_body, model_name, &state.signatures, gemini_thinking_enabled).await
        }

        Publisher::Anthropic => {
            let mut vertex_body = body.clone();
            vertex_body["anthropic_version"] = json!("vertex-2023-10-16");
            // Remove stream field — Anthropic on Vertex doesn't support it the same way
            vertex_body.as_object_mut().map(|m| m.remove("stream"));
            // Keep "thinking" parameter — Anthropic on Vertex AI natively supports it

            tracing::debug!("Anthropic passthrough to {url}");
            let resp = send_with_retry(&state, &url, &token, &vertex_body).await.map_err(|e| {
                (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": {"type": "upstream_error", "message": e.message}})),
                )
            })?;

            let resp_body = resp.json::<Value>().await.map_err(|e| {
                (StatusCode::BAD_GATEWAY, Json(json!({"error": {"type": "upstream_error", "message": format!("Failed to parse upstream response: {e}")}})))
            })?;

            resp_body
        }
    };

    if is_streaming {
        // Fake streaming fallback for Anthropic passthrough
        let mut resp = response_to_sse(result).into_response();
        add_rate_limit_headers(resp.headers_mut(), rl_limit, rl_remaining, rl_reset);
        Ok(resp)
    } else {
        let mut resp = Json(result).into_response();
        add_rate_limit_headers(resp.headers_mut(), rl_limit, rl_remaining, rl_reset);
        Ok(resp)
    }
}

/// Add rate limit headers to a response.
fn add_rate_limit_headers(
    headers: &mut axum::http::HeaderMap,
    limit: u32,
    remaining: u32,
    reset_at: u64,
) {
    headers.insert("X-RateLimit-Limit", HeaderValue::from_str(&limit.to_string()).unwrap_or_else(|_| HeaderValue::from_static("0")));
    headers.insert("X-RateLimit-Remaining", HeaderValue::from_str(&remaining.to_string()).unwrap_or_else(|_| HeaderValue::from_static("0")));
    headers.insert("X-RateLimit-Reset", HeaderValue::from_str(&reset_at.to_string()).unwrap_or_else(|_| HeaderValue::from_static("0")));
}

fn check_auth(
    headers: &axum::http::HeaderMap,
    config: &Config,
) -> Result<(), (StatusCode, Json<Value>)> {
    let key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            headers
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
        });

    if key != Some(&config.master_key) {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": {"type": "authentication_error", "message": "Invalid API key"}})),
        ));
    }
    Ok(())
}
