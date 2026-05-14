use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use serde_json::{json, Value};
use std::convert::Infallible;

use crate::config::Config;
use crate::models::{build_vertex_streaming_url, build_vertex_url, resolve_model, Publisher};
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
                    "usage": response.get("usage").cloned().unwrap_or(json!({"input_tokens": 0, "output_tokens": 0}))
                }
            })
            .to_string(),
        ))];

    if let Some(blocks) = response["content"].as_array() {
        for (i, block) in blocks.iter().enumerate() {
            let idx = i as u64;
            match block["type"].as_str() {
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

    events.push(Ok(Event::default().event("message_delta").data(
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": response["usage"]["output_tokens"]}
        })
        .to_string(),
    )));
    events.push(
        Ok(Event::default()
            .event("message_stop")
            .data(json!({"type": "message_stop"}).to_string())),
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
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        let mut parser = SseParser::new();
        let mut state = GeminiStreamState::new(model);
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
                    let sigs = state.take_thought_signatures();
                    for (tool_id, sig_val) in sigs {
                        signatures.store_tool_signature(tool_id, sig_val).await;
                    }
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
        let sigs = state.take_thought_signatures();
        for (tool_id, sig_val) in sigs {
            signatures.store_tool_signature(tool_id, sig_val).await;
        }
    });

    Sse::new(ReceiverStream { rx })
}

/// Stream OpenAI SSE chunks from Vertex AI and transform to Anthropic SSE on-the-fly.
fn stream_openai(
    response: reqwest::Response,
    model: String,
) -> Sse<ReceiverStream<Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        let mut parser = SseParser::new();
        let mut state = OpenAiStreamState::new(model);
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
// Main handler
// ---------------------------------------------------------------------------

pub async fn messages(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    check_auth(&headers, &state.config)?;

    let model_name = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(&state.config.default_model);

    let is_streaming = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);

    tracing::info!(
        "Request model={model_name} stream={is_streaming} max_tokens={:?}",
        body.get("max_tokens").and_then(|t| t.as_u64())
    );

    let resolved = resolve_model(&state.config, model_name).map_err(|e| {
        (StatusCode::BAD_REQUEST, Json(json!({"error": e})))
    })?;

    let (token, project_id) = state.auth.get_token().await.map_err(|e| {
        tracing::error!("Auth error: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e})))
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

                tracing::debug!("Gemini streaming request to {url}");
                let resp = state
                    .client
                    .post(&url)
                    .header("Authorization", format!("Bearer {token}"))
                    .json(&gemini_body)
                    .send()
                    .await
                    .map_err(|e| {
                        tracing::error!("Vertex AI streaming request failed: {e}");
                        (
                            StatusCode::BAD_GATEWAY,
                            Json(json!({"error": format!("Upstream request failed: {e}")})),
                        )
                    })?;

                let status = resp.status();
                if !status.is_success() {
                    // Error response is not SSE — parse as regular JSON
                    let status_code =
                        StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
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
                    tracing::error!("Vertex AI streaming error {status}: {error_msg}");
                    return Err((
                        status_code,
                        Json(json!({"error": {"type": "upstream_error", "message": error_msg}})),
                    ));
                }

                let model_name_owned = model_name.to_string();
                let sse = stream_gemini(resp, model_name_owned, state.signatures.clone());
                return Ok(sse.into_response());
            }

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
                    "stream": true,
                });

                if !tools.is_empty() {
                    openai_body["tools"] =
                        json!(anthropic_to_openai::tools_to_openai(tools));
                }

                tracing::debug!("OpenAPI streaming request to {url}");
                let resp = state
                    .client
                    .post(&url)
                    .header("Authorization", format!("Bearer {token}"))
                    .json(&openai_body)
                    .send()
                    .await
                    .map_err(|e| {
                        tracing::error!("Vertex AI streaming request failed: {e}");
                        (
                            StatusCode::BAD_GATEWAY,
                            Json(json!({"error": format!("Upstream request failed: {e}")})),
                        )
                    })?;

                let status = resp.status();
                if !status.is_success() {
                    let status_code =
                        StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
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
                    tracing::error!("Vertex AI streaming error {status}: {error_msg}");
                    return Err((
                        status_code,
                        Json(json!({"error": {"type": "upstream_error", "message": error_msg}})),
                    ));
                }

                let model_name_owned = model_name.to_string();
                let sse = stream_openai(resp, model_name_owned);
                return Ok(sse.into_response());
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

            tracing::debug!("OpenAPI request to {url}");
            let resp = state
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {token}"))
                .json(&openai_body)
                .send()
                .await
                .map_err(|e| {
                    tracing::error!("Vertex AI request failed: {e}");
                    (StatusCode::BAD_GATEWAY, Json(json!({"error": format!("Upstream request failed: {e}")})))
                })?;

            let status = resp.status();
            let resp_body = resp.json::<Value>().await.map_err(|e| {
                (StatusCode::BAD_GATEWAY, Json(json!({"error": format!("Failed to parse upstream response: {e}")})))
            })?;

            if !status.is_success() {
                let error_msg = resp_body
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .or_else(|| resp_body.get("error").and_then(|e| e.as_str()))
                    .unwrap_or("Unknown upstream error");
                tracing::error!("Vertex AI error {status}: {error_msg}");
                return Err((
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                    Json(json!({"error": {"type": "upstream_error", "message": error_msg}})),
                ));
            }

            tracing::debug!("OpenAPI raw response: {resp_body}");

            anthropic_to_openai::response_to_anthropic(&resp_body, model_name)
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

            tracing::debug!("Gemini request to {url}");
            let resp = state
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {token}"))
                .json(&gemini_body)
                .send()
                .await
                .map_err(|e| {
                    tracing::error!("Vertex AI request failed: {e}");
                    (StatusCode::BAD_GATEWAY, Json(json!({"error": format!("Upstream request failed: {e}")})))
                })?;

            let status = resp.status();
            let resp_body = resp.json::<Value>().await.map_err(|e| {
                (StatusCode::BAD_GATEWAY, Json(json!({"error": format!("Failed to parse upstream response: {e}")})))
            })?;

            if !status.is_success() {
                let error_msg = resp_body
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .or_else(|| resp_body.get("error").and_then(|e| e.as_str()))
                    .unwrap_or("Unknown upstream error");
                tracing::error!("Vertex AI error {status}: {error_msg}");
                return Err((
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                    Json(json!({"error": {"type": "upstream_error", "message": error_msg}})),
                ));
            }

            anthropic_to_gemini::response_to_anthropic(&resp_body, model_name, &state.signatures)
        }

        Publisher::Anthropic => {
            let mut vertex_body = body.clone();
            vertex_body["anthropic_version"] = json!("vertex-2023-10-16");
            // Remove stream field — Anthropic on Vertex doesn't support it the same way
            vertex_body.as_object_mut().map(|m| m.remove("stream"));

            tracing::debug!("Anthropic passthrough to {url}");
            let resp = state
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {token}"))
                .json(&vertex_body)
                .send()
                .await
                .map_err(|e| {
                    tracing::error!("Vertex AI request failed: {e}");
                    (StatusCode::BAD_GATEWAY, Json(json!({"error": format!("Upstream request failed: {e}")})))
                })?;

            let status = resp.status();
            let resp_body = resp.json::<Value>().await.map_err(|e| {
                (StatusCode::BAD_GATEWAY, Json(json!({"error": format!("Failed to parse upstream response: {e}")})))
            })?;

            if !status.is_success() {
                let error_msg = resp_body
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("Unknown upstream error");
                tracing::error!("Vertex AI error {status}: {error_msg}");
                return Err((
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                    Json(json!({"error": {"type": "upstream_error", "message": error_msg}})),
                ));
            }

            resp_body
        }
    };

    if is_streaming {
        // Fake streaming fallback for Anthropic passthrough
        Ok(response_to_sse(result).into_response())
    } else {
        Ok(Json(result).into_response())
    }
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
