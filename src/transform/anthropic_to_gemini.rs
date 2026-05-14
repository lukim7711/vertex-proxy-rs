use serde_json::{json, Value};
use uuid::Uuid;

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
pub fn messages_to_gemini(messages: &[Value]) -> Vec<Value> {
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

        let parts = content_to_gemini_parts(content, &tool_names);

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

fn content_to_gemini_parts(
    content: Option<&Value>,
    tool_names: &std::collections::HashMap<String, String>,
) -> Vec<Value> {
    match content {
        Some(Value::String(text)) => vec![json!({"text": text})],
        Some(Value::Array(blocks)) => {
            let mut parts: Vec<Value> = Vec::new();
            for block in blocks {
                match block.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                            if !text.is_empty() {
                                parts.push(json!({"text": text}));
                            }
                        }
                    }
                    Some("tool_use") => {
                        let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                        let input = block.get("input").unwrap_or(&Value::Null);
                        parts.push(json!({
                            "functionCall": {"name": name, "args": input}
                        }));
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

                        parts.push(json!({
                            "functionResponse": {
                                "name": name,
                                "response": {"content": response_text}
                            }
                        }));
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

/// Gemini API response → Anthropic format
pub fn response_to_anthropic(response: &Value, model: &str) -> Value {
    let msg_id = format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]);

    let candidates = response.get("candidates").and_then(|c| c.as_array());
    let mut content_blocks: Vec<Value> = Vec::new();
    let mut has_tool_use = false;
    let mut finish_reason = "end_turn";
    let mut input_tokens = 0u64;
    let mut output_tokens = 0u64;

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
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        if !text.is_empty() {
                            content_blocks.push(json!({"type": "text", "text": text}));
                        }
                    }
                    if let Some(fc) = part.get("functionCall") {
                        has_tool_use = true;
                        content_blocks.push(json!({
                            "type": "tool_use",
                            "id": format!("toolu_{}", &Uuid::new_v4().to_string().replace('-', "")[..16]),
                            "name": fc.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                            "input": fc.get("args").unwrap_or(&Value::Null),
                        }));
                    }
                }
            }
        }
    }

    // Gemini returns STOP even when there are function calls.
    // Anthropic clients (Claude Code) need "tool_use" to trigger tool execution.
    let stop_reason = if has_tool_use {
        "tool_use"
    } else {
        finish_reason
    };

    if let Some(um) = response.get("usageMetadata") {
        input_tokens = um.get("promptTokenCount").and_then(|t| t.as_u64()).unwrap_or(0);
        output_tokens = um.get("candidatesTokenCount").and_then(|t| t.as_u64()).unwrap_or(0);
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
        }
    })
}
