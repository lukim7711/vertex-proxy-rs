use axum::response::sse::Event;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::SignatureCache;

/// State machine for transforming Gemini SSE streaming chunks
/// into Anthropic SSE streaming events.
///
/// Gemini streaming sends incremental `data:` events with partial content.
/// Each chunk contains a complete JSON object with `candidates[0].content.parts[]`.
///
/// This state machine tracks:
/// - Whether the `message_start` event has been sent
/// - Whether a text/thinking content block is currently open
/// - The current block index for multi-block responses
/// - Tool use detection for proper `stop_reason`
/// - Thought signatures for both functionCall AND text parts
/// - Thinking mode: exposes Gemini's `thought: true` parts as thinking blocks
/// - Token reporting: tracks input/output tokens for usage events
pub struct GeminiStreamState {
    model: String,
    msg_id: String,
    block_index: u64,
    text_block_open: bool,
    thinking_block_open: bool,
    has_tool_use: bool,
    input_tokens: u64,
    output_tokens: u64,
    /// Token count consumed by thinking/reasoning (Gemini: thoughtsTokenCount)
    thinking_tokens: u64,
    started: bool,
    finished: bool,
    /// Whether thinking mode is enabled (client requested it)
    thinking_enabled: bool,
    /// Accumulated thought_signature values from functionCall parts.
    /// (tool_id, signature)
    tool_thought_signatures: Vec<(String, Value)>,
    /// Accumulated thought_signature values from text parts.
    /// (text_hash, signature)
    text_thought_signatures: Vec<(String, Value)>,
}

impl GeminiStreamState {
    pub fn new(model: String) -> Self {
        Self {
            model,
            msg_id: format!("msg_{}", &Uuid::new_v4().to_string().replace('-', "")[..24]),
            block_index: 0,
            text_block_open: false,
            thinking_block_open: false,
            has_tool_use: false,
            input_tokens: 0,
            output_tokens: 0,
            thinking_tokens: 0,
            started: false,
            finished: false,
            thinking_enabled: false,
            tool_thought_signatures: Vec::new(),
            text_thought_signatures: Vec::new(),
        }
    }

    /// Set whether thinking mode is enabled for this stream.
    pub fn set_thinking_enabled(&mut self, enabled: bool) {
        self.thinking_enabled = enabled;
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
            // Track thinking tokens from Gemini's thoughtsTokenCount
            self.thinking_tokens = um
                .get("thoughtsTokenCount")
                .and_then(|t| t.as_u64())
                .unwrap_or(self.thinking_tokens);
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
                            let is_thought = part.get("thought").and_then(|t| t.as_bool()) == Some(true);

                            // Read thoughtSignature from Gemini part
                            let thought_sig = part.get("thoughtSignature")
                                .or_else(|| part.get("thought_signature"))
                                .cloned();

                            // ── Thinking part ──────────────────────────────────
                            if is_thought && self.thinking_enabled {
                                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                    if !text.is_empty() {
                                        // Close text block if open (thinking comes before text)
                                        if self.text_block_open {
                                            events.push(self.close_block());
                                            self.text_block_open = false;
                                        }

                                        if !self.thinking_block_open {
                                            events.push(self.open_thinking_block());
                                            self.thinking_block_open = true;
                                        }
                                        events.push(self.thinking_delta(text));
                                    }
                                }
                                continue;
                            }

                            // Skip thought parts when thinking mode is NOT enabled
                            if is_thought {
                                continue;
                            }

                            // ── Text content (non-thought) ────────────────────
                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                if !text.is_empty() {
                                    // Close thinking block if open (text comes after thinking)
                                    if self.thinking_block_open {
                                        events.push(self.close_block());
                                        self.thinking_block_open = false;
                                    }

                                    if !self.text_block_open {
                                        events.push(self.open_text_block());
                                        self.text_block_open = true;
                                    }
                                    events.push(self.text_delta(text));

                                    // FIX BUG 4: Store text part signature for round-trip
                                    if let Some(ref sig) = thought_sig {
                                        let text_hash = SignatureCache::hash_text(text);
                                        self.text_thought_signatures.push((text_hash, sig.clone()));
                                    }
                                }
                            }

                            // ── Function call ─────────────────────────────────
                            if let Some(fc) = part.get("functionCall") {
                                // Close any open block
                                if self.thinking_block_open {
                                    events.push(self.close_block());
                                    self.thinking_block_open = false;
                                }
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

                                // Preserve thoughtSignature for round-trip
                                events.push(self.open_tool_block(&tool_id, name, thought_sig.as_ref()));

                                // Stream input JSON incrementally
                                for chunk in input_str.as_bytes().chunks(64) {
                                    let partial = String::from_utf8_lossy(chunk).to_string();
                                    events.push(self.input_json_delta(&partial));
                                }

                                events.push(self.close_block());
                                self.has_tool_use = true;

                                // Store signature for later cache storage
                                if let Some(sig_val) = thought_sig {
                                    self.tool_thought_signatures.push((tool_id, sig_val));
                                }
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

    /// Take the accumulated tool thought signatures, clearing them from this state.
    pub fn take_tool_thought_signatures(&mut self) -> Vec<(String, Value)> {
        std::mem::take(&mut self.tool_thought_signatures)
    }

    /// Take the accumulated text thought signatures, clearing them from this state.
    pub fn take_text_thought_signatures(&mut self) -> Vec<(String, Value)> {
        std::mem::take(&mut self.text_thought_signatures)
    }

    /// Generate finish events: close any open block, then message_delta + message_stop.
    pub fn finish(&mut self, finish_reason: Option<&str>) -> Vec<Event> {
        let mut events = Vec::new();

        if self.finished {
            return events;
        }
        self.finished = true;

        if self.thinking_block_open {
            events.push(self.close_block());
            self.thinking_block_open = false;
        }
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
                .data(
                    json!({
                        "type": "message_stop",
                        "usage": {
                            "input_tokens": self.input_tokens,
                            "output_tokens": self.output_tokens,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0
                        }
                    })
                    .to_string(),
                ),
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
                        "usage": {"input_tokens": self.input_tokens, "output_tokens": 0}
                    }
                })
                .to_string(),
            )
    }

    fn open_thinking_block(&mut self) -> Event {
        let idx = self.block_index;
        Event::default()
            .event("content_block_start")
            .data(
                json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "thinking", "thinking": ""}
                })
                .to_string(),
            )
    }

    fn thinking_delta(&self, text: &str) -> Event {
        Event::default()
            .event("content_block_delta")
            .data(
                json!({
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "thinking_delta", "thinking": text}
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

    fn open_tool_block(&mut self, tool_id: &str, name: &str, thought_signature: Option<&Value>) -> Event {
        let idx = self.block_index;
        let mut content_block = json!({
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": {}
        });
        // Preserve thought_signature as _thought_signature for round-trip
        if let Some(sig) = thought_signature {
            content_block["_thought_signature"] = sig.clone();
        }
        Event::default()
            .event("content_block_start")
            .data(
                json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": content_block
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
