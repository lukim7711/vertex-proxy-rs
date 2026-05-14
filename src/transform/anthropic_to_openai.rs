use serde_json::{json, Value};
use uuid::Uuid;

use crate::transform::schema_clean::clean;

/// Anthropic tools → OpenAI function tools
pub fn tools_to_openai(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter_map(|tool| {
            let name = tool.get("name")?.as_str()?;
            let description = tool.get("description").and_then(|v| v.as_str()).unwrap_or("");
            let schema = clean(tool.get("input_schema").unwrap_or(&Value::Null));
            Some(json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": schema,
                }
            }))
        })
        .collect()
}

/// Anthropic messages → OpenAI messages
///
/// Handles the structural differences:
/// - Anthropic: system is a top-level field, content can be string or array of blocks
/// - OpenAI: system is a message with role "system", content is string
/// - Anthropic tool_result (role: user) → OpenAI tool message
/// - Anthropic tool_use (role: assistant) → OpenAI assistant with tool_calls
pub fn messages_to_openai(system: Option<&Value>, messages: &[Value]) -> Vec<Value> {
    let mut openai_msgs: Vec<Value> = Vec::new();

    // System prompt
    if let Some(sys) = system {
        let sys_text = match sys {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n"),
            other => other.to_string(),
        };
        if !sys_text.is_empty() {
            openai_msgs.push(json!({"role": "system", "content": sys_text}));
        }
    }

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content");

        match content {
            Some(Value::String(text)) => {
                openai_msgs.push(json!({"role": role, "content": text}));
            }
            Some(Value::Array(blocks)) => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<Value> = Vec::new();
                let mut tool_results: Vec<Value> = Vec::new();

                for block in blocks {
                    match block.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                                if !t.is_empty() {
                                    text_parts.push(t.to_string());
                                }
                            }
                        }
                        Some("tool_use") => {
                            let args = serde_json::to_string(
                                block.get("input").unwrap_or(&Value::Null),
                            )
                            .unwrap_or_default();
                            tool_calls.push(json!({
                                "id": block.get("id").and_then(|i| i.as_str()).unwrap_or("call_0"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                                    "arguments": args,
                                }
                            }));
                        }
                        Some("tool_result") => {
                            let result_text = extract_tool_result_text(block.get("content"));
                            tool_results.push(json!({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id").and_then(|i| i.as_str()).unwrap_or(""),
                                "content": result_text,
                            }));
                        }
                        _ => {}
                    }
                }

                // Emit assistant message with tool_calls if present
                if !tool_calls.is_empty() {
                    let text = if text_parts.is_empty() {
                        Value::Null
                    } else {
                        Value::String(text_parts.join("\n"))
                    };
                    openai_msgs.push(json!({
                        "role": "assistant",
                        "content": text,
                        "tool_calls": tool_calls,
                    }));
                } else if !text_parts.is_empty() {
                    // Plain text message (user or assistant)
                    openai_msgs.push(json!({"role": role, "content": text_parts.join("\n")}));
                }

                // Emit tool result messages (always separate, after assistant)
                for tr in tool_results {
                    openai_msgs.push(tr);
                }
            }
            _ => {}
        }
    }

    openai_msgs
}

/// Extract text from tool_result content field
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

/// OpenAI response → Anthropic format
///
/// Handles model quirks:
/// - GLM-5 may return `content: null` instead of a string
/// - GLM-5 may return `tool_calls` with empty function names (not real tool calls)
/// - Some models return `finish_reason: "tool_calls"` without valid tool_calls
pub fn response_to_anthropic(response: &Value, model: &str) -> Value {
    let msg_id = format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]);
    let choice = response
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first());

    let msg = choice.and_then(|c| c.get("message"));
    let mut content_blocks: Vec<Value> = Vec::new();

    // Text content — handle both string and null
    if let Some(content_val) = msg.and_then(|m| m.get("content")) {
        if let Some(text) = content_val.as_str() {
            if !text.is_empty() {
                content_blocks.push(json!({"type": "text", "text": text}));
            }
        }
        // content: null is normal when model returns only tool_calls — skip silently
    }

    // Tool calls — validate each one before including
    if let Some(tool_calls) = msg.and_then(|m| m.get("tool_calls")).and_then(|t| t.as_array()) {
        for tc in tool_calls {
            let func = tc.get("function");
            let name = func
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("");

            // Skip tool calls with empty function name — these are model hallucinations
            if name.is_empty() {
                continue;
            }

            let tc_id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");

            // Skip tool calls with empty ID
            if tc_id.is_empty() {
                continue;
            }

            let args: Value = func
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or(Value::Object(serde_json::Map::new()));

            content_blocks.push(json!({
                "type": "tool_use",
                "id": tc_id,
                "name": name,
                "input": args,
            }));
        }
    }

    // Determine stop_reason based on what's actually in content_blocks,
    // NOT based on the model's finish_reason (which can lie, e.g. GLM-5)
    let has_valid_tool_use = content_blocks
        .iter()
        .any(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"));

    let stop_reason = if has_valid_tool_use {
        "tool_use"
    } else if choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|f| f.as_str())
        == Some("length")
    {
        "max_tokens"
    } else {
        "end_turn"
    };

    // If content is empty but we have text from finish_reason, ensure at least empty response
    // This prevents Claude Code from seeing stop_reason: end_turn with empty content
    if content_blocks.is_empty() && stop_reason == "end_turn" {
        content_blocks.push(json!({"type": "text", "text": ""}));
    }

    let usage = response.get("usage").unwrap_or(&Value::Null);

    json!({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.get("prompt_tokens").and_then(|t| t.as_u64()).unwrap_or(0),
            "output_tokens": usage.get("completion_tokens").and_then(|t| t.as_u64()).unwrap_or(0),
        }
    })
}
