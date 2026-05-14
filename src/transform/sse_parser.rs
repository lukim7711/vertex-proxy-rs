use bytes::Bytes;

/// Incremental SSE (Server-Sent Events) parser.
///
/// Bytes arrive in arbitrary chunks from the HTTP stream. This parser
/// buffers them and extracts complete `data:` lines as they become available.
pub struct SseParser {
    buffer: String,
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Push new bytes into the buffer and return any complete SSE data values found.
    ///
    /// SSE format:
    /// ```text
    /// data: {"json": "payload"}\n\n
    /// data: {"more": "data"}\n\n
    /// data: [DONE]\n\n
    /// ```
    pub fn push_bytes(&mut self, bytes: &Bytes) -> Vec<String> {
        let text = String::from_utf8_lossy(bytes);
        // Normalize CRLF to LF for consistent parsing
        self.buffer.push_str(&text.replace("\r\n", "\n"));
        let mut results = Vec::new();

        while let Some(pos) = self.buffer.find("\n\n") {
            let block: String = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos + 2..].to_string();

            for line in block.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    let data = data.trim();
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }
                    results.push(data.to_string());
                }
            }
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_complete_event() {
        let mut parser = SseParser::new();
        let bytes = Bytes::from("data: {\"hello\": \"world\"}\n\n");
        let results = parser.push_bytes(&bytes);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], "{\"hello\": \"world\"}");
    }

    #[test]
    fn test_split_across_chunks() {
        let mut parser = SseParser::new();
        let chunk1 = Bytes::from("data: {\"hel");
        let chunk2 = Bytes::from("lo\": \"world\"}\n\n");
        assert!(parser.push_bytes(&chunk1).is_empty());
        let results = parser.push_bytes(&chunk2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], "{\"hello\": \"world\"}");
    }

    #[test]
    fn test_multiple_events_in_one_chunk() {
        let mut parser = SseParser::new();
        let bytes = Bytes::from("data: {\"a\":1}\n\ndata: {\"b\":2}\n\n");
        let results = parser.push_bytes(&bytes);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "{\"a\":1}");
        assert_eq!(results[1], "{\"b\":2}");
    }

    #[test]
    fn test_skip_done_marker() {
        let mut parser = SseParser::new();
        let bytes = Bytes::from("data: {\"a\":1}\n\ndata: [DONE]\n\n");
        let results = parser.push_bytes(&bytes);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_crlf_line_endings() {
        let mut parser = SseParser::new();
        let bytes = Bytes::from("data: {\"a\":1}\r\n\r\n");
        let results = parser.push_bytes(&bytes);
        assert_eq!(results.len(), 1);
    }
}
