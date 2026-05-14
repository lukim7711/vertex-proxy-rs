use axum::response::sse::Event;
use serde_json::{json, Value};
use uuid::Uuid;

/// State machine for transforming Gemini SSE streaming chunks
/// into Anthropic SSE streaming events.
///
/// Gemini streaming sends incremental `data:` events with partial content.
/// Each chunk contains a complete JSON object with `candidates[0].content.parts[]`.
///
/// This state machine tracks:
/// - Whether the `message_start` event has been sent
/// - Whether a text content block is currently open
/// - The current block index for multi-block responses
/// - Tool use detection for proper `stop_reason`
pub struct GeminiStreamState {
    model: String,
    msg_id: String,
    block_index: u64,
    text_block_open: bool,
    has_tool_use: bool,
    input_tokens: u64,
    output_tokens: u64,
    started: bool,
    finished: bool,
}

impl GeminiStreamState {
    pub fn new(model: String) -> Self {
        Self {
            model,
            msg_id: format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]),
            block_index: 0,
            text_block_open: false,
            has_tool_use: false,
            input_tokens: 0,
            output_tokens: 0,
            started: false,
            finished: false,
        }
    }

    /// Process a parsed Gemini JSON chunk and return zero or more Anthropic SSE events.
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

        // Update usage metadata if present
        if let Some(um) = chunk.get("usageMetadata") {
            self.input_tokens = um
                .get("promptTokenCount")
                .and_then(|t| t.as_u64())
                .unwrap_or(self.input_tokens);
            self.output_tokens = um
                .get("candidatesTokenCount")
                .and_then(|t| t.as_u64())
                .unwrap_or(self.output_tokens);
        }

        // Check for error in streaming response
        if chunk.get("error").is_some() {
            tracing::error!("Gemini streaming error: {:?}", chunk["error"]);
            events.extend(self.finish(None));
            return events;
        }

        // Process candidates
        if let Some(candidates) = chunk.get("candidates").and_then(|c| c.as_array()) {
            if let Some(first) = candidates.first() {
                let finish_reason = first.get("finishReason").and_then(|f| f.as_str());

                if let Some(content) = first.get("content") {
                    if let Some(parts) = content.get("parts").and_then(|p| p.as_array()) {
                        for part in parts {
                            // Text content
                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                if !text.is_empty() {
                                    if !self.text_block_open {
                                        events.push(self.open_text_block());
                                        self.text_block_open = true;
                                    }
                                    events.push(self.text_delta(text));
                                }
                            }

                            // Function call
                            if let Some(fc) = part.get("functionCall") {
                                // Close text block if open
                                if self.text_block_open {
                                    events.push(self.close_block());
                                    self.text_block_open = false;
                                }

                                let tool_id = format!(
                                    "toolu_{}",
                                    &Uuid::new_v4().to_string().replace('-', "")[..16]
                                );
                                let name = fc
                                    .get("name")
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("");
                                let input = fc.get("args").unwrap_or(&Value::Null);
                                let input_str = serde_json::to_string(input).unwrap_or_default();

                                events.push(self.open_tool_block(&tool_id, name));

                                // Stream input JSON incrementally
                                for chunk in input_str.as_bytes().chunks(64) {
                                    let partial = String::from_utf8_lossy(chunk).to_string();
                                    events.push(self.input_json_delta(&partial));
                                }

                                events.push(self.close_block());
                                self.has_tool_use = true;
                            }
                        }
                    }
                }

                // Final chunk has finishReason
                if finish_reason.is_some() {
                    events.extend(self.finish(finish_reason));
                }
            }
        }

        events
    }

    /// Generate finish events: close any open block, then message_delta + message_stop.
    pub fn finish(&mut self, finish_reason: Option<&str>) -> Vec<Event> {
        let mut events = Vec::new();

        if self.finished {
            return events;
        }
        self.finished = true;

        if self.text_block_open {
            events.push(self.close_block());
            self.text_block_open = false;
        }

        let stop_reason = if self.has_tool_use {
            "tool_use"
        } else {
            match finish_reason {
                Some("MAX_TOKENS") => "max_tokens",
                Some("RECITATION") => "end_turn",
                Some("SAFETY") => "end_turn",
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
