use axum::response::sse::Event;
use serde_json::{json, Value};
use std::collections::HashMap;
use uuid::Uuid;

/// State machine for transforming OpenAI SSE streaming chunks
/// into Anthropic SSE streaming events.
///
/// OpenAI streaming sends `chat.completion.chunk` objects with incremental
/// `delta` fields. Tool calls come with incremental `function.arguments`.
///
/// Handles quirks:
/// - GLM-5 may return `tool_calls` with empty function names (skipped as hallucinations)
/// - Some models return `finish_reason: "tool_calls"` without valid tool_calls
/// - The `stop_reason` is determined by actual content, not model's finish_reason
pub struct OpenAiStreamState {
    model: String,
    msg_id: String,
    block_index: u64,
    text_block_open: bool,
    /// Tracks which OpenAI tool_call index maps to which Anthropic block
    /// and stores (tool_id, tool_name)
    active_tools: HashMap<u64, (String, String)>,
    /// The OpenAI tool_call index currently being streamed
    current_tool_index: Option<u64>,
    has_tool_use: bool,
    input_tokens: u64,
    output_tokens: u64,
    started: bool,
    finished: bool,
}

impl OpenAiStreamState {
    pub fn new(model: String) -> Self {
        Self {
            model,
            msg_id: format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]),
            block_index: 0,
            text_block_open: false,
            active_tools: HashMap::new(),
            current_tool_index: None,
            has_tool_use: false,
            input_tokens: 0,
            output_tokens: 0,
            started: false,
            finished: false,
        }
    }

    /// Process a parsed OpenAI streaming chunk and return zero or more Anthropic SSE events.
    pub fn process_chunk(&mut self, chunk: &Value) -> Vec<Event> {
        let mut events = Vec::new();

        if self.finished {
            return events;
        }

        // Emit message_start on first chunk
        if !self.started {
            events.push(self.message_start_event());
            self.started = true;
        }

        // Update usage if present (some providers include usage in the final chunk)
        if let Some(usage) = chunk.get("usage") {
            if !usage.is_null() {
                self.input_tokens = usage
                    .get("prompt_tokens")
                    .and_then(|t| t.as_u64())
                    .unwrap_or(self.input_tokens);
                self.output_tokens = usage
                    .get("completion_tokens")
                    .and_then(|t| t.as_u64())
                    .unwrap_or(self.output_tokens);
            }
        }

        // Process choices
        if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
            if let Some(choice) = choices.first() {
                let delta = choice.get("delta");
                let finish_reason = choice.get("finish_reason");

                if let Some(delta) = delta {
                    // Text content delta
                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                        if !content.is_empty() {
                            if !self.text_block_open {
                                events.push(self.open_text_block());
                                self.text_block_open = true;
                            }
                            events.push(self.text_delta(content));
                        }
                    }

                    // Tool calls delta
                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                        for tc in tool_calls {
                            let tc_index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                            let tc_id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");
                            let func = tc.get("function");
                            let func_name = func
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .unwrap_or("");
                            let func_args = func
                                .and_then(|f| f.get("arguments"))
                                .and_then(|a| a.as_str())
                                .unwrap_or("");

                            // New tool call starts (has id or name)
                            if !tc_id.is_empty() || !func_name.is_empty() {
                                // Skip tool calls with empty name and empty id (GLM-5 quirk)
                                if func_name.is_empty() && tc_id.is_empty() {
                                    continue;
                                }

                                // Close text block if open
                                if self.text_block_open {
                                    events.push(self.close_block());
                                    self.text_block_open = false;
                                }

                                // Close previous tool block if switching to a different tool
                                if self.current_tool_index.is_some()
                                    && self.current_tool_index != Some(tc_index)
                                {
                                    events.push(self.close_block());
                                    self.current_tool_index = None;
                                }

                                let tool_id = if tc_id.is_empty() {
                                    format!(
                                        "toolu_{}",
                                        &Uuid::new_v4().to_string().replace('-', "")[..16]
                                    )
                                } else {
                                    tc_id.to_string()
                                };

                                if !func_name.is_empty() {
                                    self.active_tools
                                        .insert(tc_index, (tool_id.clone(), func_name.to_string()));

                                    events.push(self.open_tool_block(&tool_id, func_name));
                                    self.current_tool_index = Some(tc_index);
                                    self.has_tool_use = true;

                                    // If arguments come in the same chunk, stream them
                                    if !func_args.is_empty() {
                                        events.push(self.input_json_delta(func_args));
                                    }
                                }
                            } else if !func_args.is_empty() {
                                // Continuation of existing tool call arguments
                                events.push(self.input_json_delta(func_args));
                            }
                        }
                    }
                }

                // Check finish_reason (not null, not "null" string)
                if let Some(fr) = finish_reason {
                    if !fr.is_null() {
                        let fr_str = fr.as_str().unwrap_or("");
                        if !fr_str.is_empty() && fr_str != "null" {
                            events.extend(self.finish(Some(fr_str)));
                        }
                    }
                }
            }
        }

        events
    }

    /// Generate finish events.
    pub fn finish(&mut self, finish_reason: Option<&str>) -> Vec<Event> {
        let mut events = Vec::new();

        if self.finished {
            return events;
        }
        self.finished = true;

        // Close text block if open
        if self.text_block_open {
            events.push(self.close_block());
            self.text_block_open = false;
        }

        // Close any open tool block
        if self.current_tool_index.is_some() {
            events.push(self.close_block());
            self.current_tool_index = None;
        }

        // Determine stop_reason based on actual content blocks, not model's finish_reason
        let stop_reason = if self.has_tool_use {
            "tool_use"
        } else {
            match finish_reason {
                Some("length") => "max_tokens",
                _ => "end_turn",
            }
        };

        events.push(
            Event::default()
                .event("message_delta")
                .data(
                    json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                        "usage": {"output_tokens": self.output_tokens}
                    })
                    .to_string(),
                ),
        );

        events.push(
            Event::default()
                .event("message_stop")
                .data(json!({"type": "message_stop"}).to_string()),
        );

        events
    }

    fn message_start_event(&self) -> Event {
        Event::default()
            .event("message_start")
            .data(
                json!({
                    "type": "message_start",
                    "message": {
                        "id": self.msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": null,
                        "stop_sequence": null,
                        "usage": {"input_tokens": 0, "output_tokens": 0}
                    }
                })
                .to_string(),
            )
    }

    fn open_text_block(&mut self) -> Event {
        let idx = self.block_index;
        Event::default()
            .event("content_block_start")
            .data(
                json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""}
                })
                .to_string(),
            )
    }

    fn text_delta(&self, text: &str) -> Event {
        Event::default()
            .event("content_block_delta")
            .data(
                json!({
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "text_delta", "text": text}
                })
                .to_string(),
            )
    }

    fn open_tool_block(&mut self, tool_id: &str, name: &str) -> Event {
        let idx = self.block_index;
        Event::default()
            .event("content_block_start")
            .data(
                json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": {}
                    }
                })
                .to_string(),
            )
    }

    fn input_json_delta(&self, partial: &str) -> Event {
        Event::default()
            .event("content_block_delta")
            .data(
                json!({
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "input_json_delta", "partial_json": partial}
                })
                .to_string(),
            )
    }

    fn close_block(&mut self) -> Event {
        let idx = self.block_index;
        self.block_index += 1;
        Event::default()
            .event("content_block_stop")
            .data(json!({"type": "content_block_stop", "index": idx}).to_string())
    }
}
