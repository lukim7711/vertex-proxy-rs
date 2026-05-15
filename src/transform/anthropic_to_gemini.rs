use serde_json::{json, Value};
use uuid::Uuid;

use crate::SignatureCache;
use crate::transform::schema_clean::clean;

/// Anthropic tools → Gemini functionDeclarations
pub fn tools_to_gemini(tools: &[Value]) -> Vec<Value> {
    let declarations: Vec<Value> = tools
        .iter()
        .filter_map(|tool| {
            let name = tool.get("name")?.as_str()?;
            let description = tool.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let schema = clean(tool.get("input_schema").unwrap_or(&Value::Null));
            Some(json!({
                "name": name,
                "description": description,
                "parameters": schema,
            }))
        })
        .collect();

    if declarations.is_empty() {
        vec![]
    } else {
        vec![json!({"functionDeclarations": declarations})]
    }
}

/// Extract system prompt text from Anthropic system field.
/// Returns the text for use in Gemini's systemInstruction.
pub fn extract_system_text(system: Option<&Value>) -> Option<String> {
    match system {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Value::Array(arr)) => {
            let text: String = arr
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() { None } else { Some(text) }
        }
        _ => None,
    }
}

/// Anthropic messages → Gemini contents.
///
/// System prompt is NOT included here — it should be passed via
/// `systemInstruction` in the Gemini request body.
///
/// Handles tool_use/tool_result cycles and ensures role alternation
/// (Gemini requires strict user/model alternation).
///
/// Uses SignatureCache to retrieve thought signatures that were stored
/// when the proxy received Gemini responses. Also checks inline
/// `_thought_signature` fields as fallback.
pub async fn messages_to_gemini(messages: &[Value], signatures: &SignatureCache) -> Vec<Value> {
    // First pass: build tool name registry from assistant messages
    // so we can map tool_use_id → tool name for functionResponse
    let mut tool_names: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    for msg in messages {
        if let Some(Value::Array(blocks)) = msg.get("content") {
            for block in blocks {
                if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                    if let (Some(id), Some(name)) = (
                        block.get("id").and_then(|i| i.as_str()),
                        block.get("name").and_then(|n| n.as_str()),
                    ) {
                        tool_names.insert(id.to_string(), name.to_string());
                    }
                }
            }
        }
    }

    // Second pass: convert messages to Gemini contents
    let mut contents: Vec<Value> = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content");

        let parts = content_to_gemini_parts(content, &tool_names, signatures).await;

        if parts.is_empty() {
            continue;
        }

        let gemini_role = if role == "user" { "user" } else { "model" };

        // Gemini requires strict role alternation.
        // If the last message has the same role, merge parts into it.
        if let Some(last) = contents.last_mut() {
            if last.get("role").and_then(|r| r.as_str()) == Some(gemini_role) {
                if let Some(existing_parts) = last.get_mut("parts").and_then(|p| p.as_array_mut()) {
                    existing_parts.extend(parts);
                    continue;
                }
            }
        }

        contents.push(json!({"role": gemini_role, "parts": parts}));
    }

    contents
}

async fn content_to_gemini_parts(
    content: Option<&Value>,
    tool_names: &std::collections::HashMap<String, String>,
    signatures: &SignatureCache,
) -> Vec<Value> {
    match content {
        Some(Value::String(text)) => {
            let mut part = json!({"text": text});
            // Look up text signature from cache
            let text_hash = SignatureCache::hash_text(text);
            if let Some(sig) = signatures.get_text_signature(&text_hash).await {
                part["thoughtSignature"] = sig;
            }
            vec![part]
        }
        Some(Value::Array(blocks)) => {
            let mut parts: Vec<Value> = Vec::new();
            for block in blocks {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            if !text.is_empty() {
                                let mut part = json!({"text": text});
                                // Look up text signature from cache (thinking models require this)
                                let text_hash = SignatureCache::hash_text(text);
                                if let Some(sig) = signatures.get_text_signature(&text_hash).await {
                                    part["thoughtSignature"] = sig;
                                }
                                parts.push(part);
                            }
                        }
                    }
                    Some("tool_use") => {
                        let tool_id = block.get("id").and_then(|i| i.as_str()).unwrap_or("");
                        let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                        let input = block.get("input").unwrap_or(&Value::Null);
                        let mut part = json!({
                            "functionCall": {"name": name, "args": input}
                        });
                        // Look up thoughtSignature from server-side cache (primary)
                        if let Some(sig) = signatures.get_tool_signature(tool_id).await {
                            part["thoughtSignature"] = sig;
                        }
                        // Also check if the client preserved it inline (fallback)
                        else if let Some(sig) = block.get("_thought_signature") {
                            part["thoughtSignature"] = sig.clone();
                        }
                        parts.push(part);
                    }
                    Some("tool_result") => {
                        // Lookup tool name by tool_use_id
                        let tool_use_id = block
                            .get("tool_use_id")
                            .and_then(|i| i.as_str())
                            .unwrap_or("");
                        let name = tool_names
                            .get(tool_use_id)
                            .map(|s| s.as_str())
                            .unwrap_or("unknown");

                        let text = extract_tool_result_text(block.get("content"));

                        // Handle error tool results
                        let is_error = block.get("is_error").and_then(|e| e.as_bool()).unwrap_or(false);
                        let response_text = if is_error {
                            format!("ERROR: {text}")
                        } else {
                            text
                        };

                        let mut part = json!({
                            "functionResponse": {
                                "name": name,
                                "response": {"content": response_text}
                            }
                        });
                        // Look up thoughtSignature from server-side cache (primary)
                        if let Some(sig) = signatures.get_tool_signature(tool_use_id).await {
                            part["thoughtSignature"] = sig;
                        }
                        // FIX BUG 6: Also check inline _thought_signature (fallback)
                        else if let Some(sig) = block.get("_thought_signature") {
                            part["thoughtSignature"] = sig.clone();
                        }
                        parts.push(part);
                    }
                    Some("image") => {
                        if let Some(src) = block.get("source") {
                            if src.get("type").and_then(|t| t.as_str()) == Some("base64") {
                                let mime = src.get("media_type").and_then(|t| t.as_str()).unwrap_or("image/png");
                                let data = src.get("data").and_then(|d| d.as_str()).unwrap_or("");
                                parts.push(json!({
                                    "inlineData": {"mimeType": mime, "data": data}
                                }));
                            }
                        }
                    }
                    _ => {}
                }
            }
            parts
        }
        _ => vec![],
    }
}

/// Extract text content from a tool_result's content field
fn extract_tool_result_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Extract thinking parameter from Anthropic request body and convert to
/// Gemini's thinkingConfig format.
///
/// Anthropic thinking modes:
/// - `"enabled"` or `{"type": "enabled", "budget_tokens": N}` → enable thinking
/// - `"adaptive"` → enable adaptive thinking
///
/// Gemini thinkingConfig:
/// - `{"thinkingConfig": {"thinkingBudget": N}}` where 0 = disabled, >0 = enabled
/// - If no budget specified, uses a default of 8192 tokens
pub fn extract_thinking_config(thinking: Option<&Value>) -> (Option<Value>, bool) {
    match thinking {
        None => (None, false),
        Some(Value::Null) => (None, false),
        Some(Value::String(s)) if s == "enabled" || s == "adaptive" => {
            let budget = 8192u64; // Default budget when no specific value given
            (Some(json!({"thinkingBudget": budget})), true)
        }
        Some(Value::Object(obj)) => {
            let thinking_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if thinking_type == "enabled" || thinking_type == "adaptive" {
                let budget = obj.get("budget_tokens")
                    .and_then(|b| b.as_u64())
                    .unwrap_or(8192);
                (Some(json!({"thinkingBudget": budget})), true)
            } else {
                (None, false)
            }
        }
        Some(other) => {
            // Unexpected format — try to interpret as enabled
            tracing::warn!(?other, "Unexpected thinking parameter format, ignoring");
            (None, false)
        }
    }
}

/// Gemini API response → Anthropic format.
/// Also stores thought signatures in the server-side cache for round-trip.
///
/// Supports thinking mode: Gemini parts with `"thought": true` are exposed
/// as Anthropic `thinking` content blocks when thinking mode was requested.
///
/// FIX BUG 1: Now includes `_thought_signature` in tool_use blocks.
/// FIX BUG 2: Now stores signatures synchronously (no tokio::spawn).
/// FIX BUG 4: Now stores text part signatures too.
pub async fn response_to_anthropic(
    response: &Value,
    model: &str,
    signatures: &SignatureCache,
    thinking_enabled: bool,
) -> Value {
    let msg_id = format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]);

    let candidates = response.get("candidates").and_then(|c| c.as_array());
    let mut content_blocks: Vec<Value> = Vec::new();
    let mut has_tool_use = false;
    let mut finish_reason = "end_turn";
    let mut input_tokens = 0u64;
    let mut output_tokens = 0u64;

    // Collect signatures for batch storage
    let mut tool_sigs_to_store: Vec<(String, serde_json::Value)> = Vec::new();
    let mut text_sigs_to_store: Vec<(String, serde_json::Value)> = Vec::new();

    if let Some(candidates) = candidates {
        if let Some(first) = candidates.first() {
            let gemini_finish = first
                .get("finishReason")
                .and_then(|f| f.as_str())
                .unwrap_or("STOP");

            finish_reason = match gemini_finish {
                "MAX_TOKENS" => "max_tokens",
                _ => "end_turn", // will be overridden if tool_use detected
            };

            if let Some(content) = first.get("content") {
                let empty_vec = vec![];
                let parts = content
                    .get("parts")
                    .and_then(|p| p.as_array())
                    .unwrap_or(&empty_vec);

                for part in parts {
                    let is_thought = part.get("thought").and_then(|t| t.as_bool()) == Some(true);

                    // ── Thinking part ──────────────────────────────────────
                    if is_thought && thinking_enabled {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            if !text.is_empty() {
                                content_blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": text
                                }));
                            }
                        }
                        continue;
                    }

                    // Skip thought parts when thinking mode is NOT enabled
                    if is_thought {
                        continue;
                    }

                    // Read thoughtSignature from Gemini response (camelCase primary, snake_case fallback)
                    let thought_sig = part.get("thoughtSignature")
                        .or_else(|| part.get("thought_signature"))
                        .cloned();

                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        if !text.is_empty() {
                            // FIX BUG 4: Store text part signature for round-trip
                            if let Some(ref sig) = thought_sig {
                                let text_hash = SignatureCache::hash_text(text);
                                text_sigs_to_store.push((text_hash, sig.clone()));
                            }
                            content_blocks.push(json!({"type": "text", "text": text}));
                        }
                    }
                    if let Some(fc) = part.get("functionCall") {
                        has_tool_use = true;
                        let tool_id = format!("toolu_{}", &Uuid::new_v4().to_string().replace('-', "")[..16]);

                        // FIX BUG 1: Include _thought_signature in the tool_use block
                        let mut tool_block = json!({
                            "type": "tool_use",
                            "id": tool_id.clone(),
                            "name": fc.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                            "input": fc.get("args").unwrap_or(&Value::Null),
                        });

                        if let Some(sig) = thought_sig {
                            tool_block["_thought_signature"] = sig.clone();
                            // FIX BUG 2: Collect for synchronous batch storage (no tokio::spawn)
                            tool_sigs_to_store.push((tool_id.clone(), sig));
                        }

                        content_blocks.push(tool_block);
                    }
                }
            }
        }
    }

    // FIX BUG 2: Store signatures synchronously (batch write)
    if !tool_sigs_to_store.is_empty() || !text_sigs_to_store.is_empty() {
        signatures.store_tool_signatures_batch(tool_sigs_to_store).await;
        signatures.store_text_signatures_batch(text_sigs_to_store).await;
    }

    // Gemini returns STOP even when there are function calls.
    // Anthropic clients (Claude Code) need "tool_use" to trigger tool execution.
    let stop_reason = if has_tool_use {
        "tool_use"
    } else {
        finish_reason
    };

    // Extract token usage including thinking tokens from Gemini
    let cache_creation_input_tokens = 0u64;
    let mut cache_read_input_tokens = 0u64;

    if let Some(um) = response.get("usageMetadata") {
        input_tokens = um.get("promptTokenCount").and_then(|t| t.as_u64()).unwrap_or(0);
        output_tokens = um.get("candidatesTokenCount").and_then(|t| t.as_u64()).unwrap_or(0);
        // Gemini may provide cachedContentTokenCount for cache read tokens
        cache_read_input_tokens = um.get("cachedContentTokenCount").and_then(|t| t.as_u64()).unwrap_or(0);
    }

    json!({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        }
    })
}
