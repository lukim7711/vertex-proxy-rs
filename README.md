# Vertex AI Proxy (Rust)

A lightweight, high-performance proxy that translates **Anthropic Messages API** and **OpenAI Chat Completions API** to **Google Vertex AI**, enabling tools like Claude Code to use Gemini, Grok, GLM, Claude, and 35+ other models on Vertex AI without paid API keys — leveraging GCP Free Tier $300 credits via ADC authentication.

**Binary ~3MB | Memory ~1.4MB idle | Zero runtime dependencies | Static linked**

---

## Architecture

```
Client (Claude Code, OpenAI SDK, etc.)     Proxy (Rust/Axum)                          Vertex AI
──────────────────────────────────         ──────────────────                         ──────────
POST /v1/messages ──────────────>   Anthropic → Gemini contents           ──>  Gemini (Google)
(Anthropic API format)               Anthropic → OpenAI chat               ──>  Grok, GLM, Mistral...
                                     Anthropic passthrough                 ──>  Claude (Anthropic)

POST /v1/chat/completions ──────>   OpenAI → Gemini contents              ──>  Gemini (Google)
(OpenAI API format)                   OpenAI passthrough                   ──>  Grok, GLM, Mistral...

POST /v1/veo/generate ──────────>   VEO predictLongRunning                ──>  VEO Video Generation
POST /v1/veo/result ────────────>   VEO fetchPredictOperation             ──>  VEO Result Polling

GET  /admin/models ─────────────>   Dynamic model management              ──>  Runtime config
POST /admin/models ─────────────>   Add model at runtime                  ──>  No restart needed
DELETE /admin/models/{name} ─────>   Remove dynamic model                  ──>  No restart needed
```

The proxy reads the model name from the request, resolves it against the config (static, dynamic, or auto-resolved), then routes to the appropriate Vertex AI endpoint. Three publisher paths:

| Publisher | Models | Vertex AI Endpoint | Transformation |
|---|---|---|---|
| `google` | Gemini, VEO | `publishers/google/models/{m}:generateContent` | Anthropic/OpenAI → Gemini |
| `openapi` | Grok, GLM, Mistral, Llama, DeepSeek, Qwen, etc. | `endpoints/openapi/chat/completions` | Anthropic → OpenAI |
| `anthropic` | Claude | `publishers/anthropic/models/{m}:rawPredict` | Passthrough |

---

## Features

### Core
- **Dual API**: Anthropic Messages API (`/v1/messages`) and OpenAI Chat Completions API (`/v1/chat/completions`)
- **35+ Models**: Gemini 3.1/3/2.5/2.0, Grok 4, GLM-5, Claude Opus/Sonnet/Haiku, Mistral, Llama 4, DeepSeek, Qwen 3, and more
- **Tool Use**: Full translation of Anthropic/OpenAI tools to Vertex AI format and back — supports multi-turn tool calling
- **Real Streaming (SSE)**: True token-by-token streaming for Google (Gemini) and OpenAPI publishers; fake streaming fallback for Anthropic passthrough
- **Thinking Model Support**: Handles Gemini thinking mode with server-side `thoughtSignature` cache that persists to disk, ensuring proper round-trip conversations with thinking models
- **Image Input**: Supports base64 image blocks in Anthropic format → Gemini `inlineData` conversion

### Dynamic Model Resolution
- **Auto-Resolve**: Models not in config are automatically resolved by name pattern:
  - `gemini-*` → Google publisher, us-central1
  - `veo-*` → Google publisher (VEO video), us-central1
  - `claude-*` → Anthropic publisher, us-central1
  - `*/*` (e.g., `xai/grok-4`) → OpenAPI publisher, global
- **Admin API**: Add/remove models at runtime without proxy restart
- **Priority**: Static config > Dynamic models > Auto-resolve

### VEO Video Generation
- **Async API**: Submit video generation jobs with `POST /v1/veo/generate`
- **Result Polling**: Check job status with `POST /v1/veo/result`
- **Sync Convenience**: Auto-poll with `POST /v1/veo/generate-sync` (blocks until done or timeout)
- **5 VEO Models**: veo-3.1-fast, veo-3.1, veo-3.1-lite, veo-3, veo-2

### Security & Auth
- **ADC Auth**: Authenticates to Vertex AI via GCP metadata server (Application Default Credentials) — no service account key needed
- **Token Caching**: Access tokens cached for ~50 minutes with double-check locking and auto-refresh
- **API Key Auth**: Protects proxy with `x-api-key` or `Authorization: Bearer` header validation
- **CORS**: Permissive CORS layer for cross-origin access

### Robustness
- **Streaming Error Handling**: Upstream errors during streaming are returned as properly formatted SSE error events (not raw JSON), so Claude Code can parse them correctly
- **Model Quirks**: Handles GLM-5 empty `content: null`, hallucinated tool calls with empty names, and other edge cases
- **Schema Cleaning**: Automatically strips unsupported JSON Schema keywords (`$ref`, `allOf`, `additionalProperties`, etc.) before sending to Gemini
- **Role Alternation**: Gemini requires strict user/model alternation — the proxy merges consecutive same-role turns automatically
- **Signature Persistence**: Thought signatures are saved to `signature_cache.json` with atomic writes (temp + rename), surviving proxy restarts

---

## Supported Models

### Google Gemini

| Model | Publisher | Region | Notes |
|---|---|---|---|
| `gemini-3.1-pro-preview` | Google | global | Latest, strongest reasoning |
| `gemini-3.1-flash-lite` | Google | global | Fast & lightweight |
| `gemini-3.1-flash-image` | Google | global | Image generation |
| `gemini-3-pro` | Google | us-central1 | Stable |
| `gemini-3-flash` | Google | us-central1 | Fast |
| `gemini-3-flash-image` | Google | us-central1 | Image generation |
| `gemini-2.5-pro` | Google | us-central1 | Generally available |
| `gemini-2.5-flash` | Google | us-central1 | Generally available |
| `gemini-2.5-flash-image` | Google | us-central1 | Image generation |
| `gemini-2.5-flash-lite` | Google | us-central1 | Lightweight |
| `gemini-2.0-flash` | Google | us-central1 | Stable |
| `gemini-2.0-flash-lite` | Google | us-central1 | Lightweight |

### Partner Models (OpenAPI)

| Model | Publisher | Region |
|---|---|---|
| `xai/grok-4.20-reasoning` | xAI | global |
| `xai/grok-4.1-fast` | xAI | global |
| `zai-org/glm-5-maas` | Zhipu AI | global |
| `zai-org/glm-4.7-maas` | Zhipu AI | global |
| `mistral/mistral-medium-3` | Mistral AI | global |
| `mistral/mistral-small-3.1` | Mistral AI | global |
| `mistral/codestral-2` | Mistral AI | global |
| `meta/llama-4-maverick` | Meta | global |
| `meta/llama-4-scout` | Meta | global |
| `meta/llama-3.3` | Meta | global |
| `deepseek/deepseek-v3.2` | DeepSeek | global |
| `deepseek/deepseek-v3.1` | DeepSeek | global |
| `deepseek/deepseek-r1-0528` | DeepSeek | global |
| `qwen/qwen3-next-instruct-80b` | Qwen | global |
| `qwen/qwen3-next-thinking-80b` | Qwen | global |
| `qwen/qwen3-coder` | Qwen | global |
| `qwen/qwen3-235b` | Qwen | global |
| `minimax/minimax-m2` | MiniMax | global |
| `kimi/kimi-k2-thinking` | Moonshot | global |
| `openai/gpt-oss-120b` | OpenAI | global |
| `openai/gpt-oss-20b` | OpenAI | global |

### Anthropic Claude (rawPredict)

| Model | Publisher | Region |
|---|---|---|
| `claude-opus-4-7` | Anthropic | us-central1 |
| `claude-sonnet-4-6` | Anthropic | us-central1 |
| `claude-opus-4-6` | Anthropic | us-central1 |
| `claude-opus-4-5` | Anthropic | us-central1 |
| `claude-sonnet-4-5` | Anthropic | us-central1 |
| `claude-haiku-4-5` | Anthropic | us-central1 |
| `claude-opus-4` | Anthropic | us-central1 |
| `claude-sonnet-4` | Anthropic | us-central1 |

### VEO Video Generation

| Model | Publisher | Region |
|---|---|---|
| `veo-3.1-fast-generate-001` | Google | us-central1 |
| `veo-3.1-generate-001` | Google | us-central1 |
| `veo-3.1-lite-generate-001` | Google | us-central1 |
| `veo-3-generate-001` | Google | us-central1 |
| `veo-2-generate-001` | Google | us-central1 |

> **Note**: The model list is configured in `config.yaml`. Models not in the config are auto-resolved by name pattern, so you can use any Vertex AI model without registering it.

---

## Quick Start

### Prerequisites

- Rust toolchain (1.75+) — only needed for building from source
- A GCP VM with Vertex AI API enabled
- Service account with `Vertex AI User` role (or appropriate IAM permissions)
- The pre-built binary works without Rust installed

### Build from Source

```bash
cargo build --release
```

Binary at `target/release/vertex-proxy-rs` (~3MB with LTO + strip).

### Configuration

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml`:

```yaml
master_key: "sk-your-secret-key"
default_model: "gemini-2.5-flash"

models:
  - name: "gemini-2.5-flash"
    vertex_name: "gemini-2.5-flash"
    publisher: "google"
    region: "us-central1"
  - name: "xai/grok-4.1-fast"
    vertex_name: "xai/grok-4.1-fast"
    publisher: "openapi"
    region: "global"
```

Each model entry has 4 fields:
- `name` — The name clients use (can be an alias)
- `vertex_name` — The actual model name in Vertex AI
- `publisher` — One of: `google`, `openapi`, `anthropic`
- `region` — Vertex AI region (`us-central1`, `global`, etc.)

### Run

```bash
./vertex-proxy-rs
# => Vertex AI Proxy (Rust) starting on 0.0.0.0:8000
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port |
| `CONFIG_PATH` | `config.yaml` | Path to config file |
| `SIGNATURE_CACHE_PATH` | `signature_cache.json` | Path to thought signature cache |

---

## API Endpoints

### POST `/v1/messages` — Anthropic Messages API

Used by Claude Code. Supports streaming (`"stream": true`).

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "max_tokens": 4096,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

Streaming:
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "max_tokens": 4096,
    "stream": true,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

With tools:
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "max_tokens": 4096,
    "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]
  }'
```

### POST `/v1/chat/completions` — OpenAI Chat Completions API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xai/grok-4.1-fast",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### VEO Video Generation

#### POST `/v1/veo/generate` — Submit video generation job

```bash
curl -X POST http://localhost:8000/v1/veo/generate \
  -H "x-api-key: sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "veo-3.1-fast-generate-001",
    "prompt": "A cat walking on a beach at sunset",
    "duration": 8,
    "resolution": "1080p",
    "aspectRatio": "16:9"
  }'
```

Response:
```json
{
  "id": "projects/.../operations/abc123",
  "model": "veo-3.1-fast-generate-001",
  "status": "submitted",
  "operationName": "projects/.../operations/abc123",
  "message": "Video generation job submitted. Poll /v1/veo/result with operationName to check status."
}
```

#### POST `/v1/veo/result` — Poll for result

```bash
curl -X POST http://localhost:8000/v1/veo/result \
  -H "x-api-key: sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "operationName": "projects/.../operations/abc123",
    "model": "veo-3.1-fast-generate-001"
  }'
```

#### POST `/v1/veo/generate-sync` — Submit and auto-poll (blocking)

```bash
curl -X POST http://localhost:8000/v1/veo/generate-sync \
  -H "x-api-key: sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "veo-3.1-fast-generate-001",
    "prompt": "A cat walking on a beach at sunset",
    "pollInterval": 5,
    "maxPollTime": 300
  }'
```

### Admin API

#### GET `/admin/models` — List all models with source info

```bash
curl -H "x-api-key: sk-your-secret-key" http://localhost:8000/admin/models
```

Response:
```json
{
  "total": 38,
  "models": [
    {"name": "gemini-2.5-flash", "vertex_name": "gemini-2.5-flash", "publisher": "google", "region": "us-central1", "source": "config"},
    {"name": "my-custom-model", "vertex_name": "my-custom-model", "publisher": "openapi", "region": "global", "source": "dynamic"}
  ],
  "auto_resolve_patterns": {
    "gemini-*": {"publisher": "google", "region": "us-central1"},
    "claude-*": {"publisher": "anthropic", "region": "us-central1"},
    "*/*": {"publisher": "openapi", "region": "global"}
  }
}
```

#### POST `/admin/models` — Add model at runtime

```bash
curl -X POST http://localhost:8000/admin/models \
  -H "x-api-key: sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-custom-model",
    "publisher": "openapi",
    "region": "global"
  }'
```

#### DELETE `/admin/models/{name}` — Remove dynamic model

```bash
curl -X DELETE http://localhost:8000/admin/models/my-custom-model \
  -H "x-api-key: sk-your-secret-key"
```

### GET `/v1/models` — List models (OpenAI-compatible)

```bash
curl http://localhost:8000/v1/models
```

### GET `/health` — Health check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "static_models": 35,
  "dynamic_models": 2,
  "auto_resolve_enabled": true
}
```

---

## Authentication

### Client → Proxy

Send the API key via either:
- Header `x-api-key: sk-your-secret-key`
- Header `Authorization: Bearer sk-your-secret-key`

The key is validated against `master_key` in `config.yaml`.

### Proxy → Vertex AI

The proxy uses the **GCP metadata server** to obtain an access token (Application Default Credentials). Tokens are cached for ~50 minutes with double-check locking (read lock for fast path, write lock with re-check for refresh). No service account key file needed — just run on a GCP VM with the appropriate IAM role.

---

## How Transformations Work

### Publisher: `google` (Gemini)

**Request:**
- System prompt → `systemInstruction.parts[].text`
- Messages → `contents[]` (role: user/model, parts: [{text}])
- Tools → `tools[].functionDeclarations` (with JSON Schema cleanup)
- Image blocks → `inlineData` with base64 data
- Thought signatures → `thoughtSignature` field on parts (from server-side cache)

**Response:**
- `candidates[0].content.parts[]` → Anthropic content blocks / OpenAI choices
- Thought parts (`"thought": true`) → skipped in output, signatures cached for round-trip
- `functionCall` parts → `tool_use` blocks with `_thought_signature`
- `usageMetadata` → token counts

**Streaming:**
- Uses `streamGenerateContent?alt=sse` for real SSE streaming
- State machine (`GeminiStreamState`) transforms chunks on-the-fly
- Thought signatures are collected during streaming and batch-stored to cache at stream end

### Publisher: `openapi` (Grok, GLM, Mistral, etc.)

**Request:**
- Anthropic tools → OpenAI `tools[].function`
- Anthropic messages → OpenAI messages (system/user/assistant/tool)
- Anthropic `tool_use` → OpenAI `tool_calls`
- Anthropic `tool_result` → OpenAI `tool` role message

**Response:**
- OpenAI format → Anthropic format
- Handles model quirks (GLM-5 `content: null`, empty tool call names, etc.)

**Streaming:**
- Same endpoint as non-streaming; streaming triggered by `"stream": true` in body
- State machine (`OpenAiStreamState`) transforms chunks on-the-fly

### Publisher: `anthropic` (Claude)

- Passthrough with `anthropic_version: vertex-2023-10-16` added
- Streaming: uses fake streaming (waits for complete response, then chunks into SSE events)
- Not recommended for production — use Anthropic API directly for better latency

---

## Thought Signature Handling

Gemini models with thinking mode (e.g., `gemini-2.5-flash` with extended thinking) attach a `thoughtSignature` to parts in their responses. This signature **must** be sent back with subsequent requests that reference those parts, otherwise the API returns an error.

The proxy handles this transparently using a **server-side signature cache** (`signature_cache.json`):

1. **Inbound**: When Gemini responds with `thoughtSignature` on `functionCall` or text parts, the proxy stores it in the cache keyed by tool_use_id or text content hash
2. **Outbound**: When converting Anthropic messages back to Gemini format, the proxy looks up stored signatures and attaches them to the corresponding parts
3. **Persistence**: The cache is written to disk with atomic writes (temp file + rename), so signatures survive proxy restarts
4. **Streaming**: Signatures are collected during streaming and batch-stored at stream completion

This approach works even when clients (like Claude Code) do not preserve custom fields like `_thought_signature` in their conversation history.

---

## Source Structure

```
src/
├── main.rs                              # Server startup, Axum router, SignatureCache, AppState
├── config.rs                            # YAML config loader, ModelConfig
├── auth.rs                              # GCP metadata server ADC token (cached, double-check locking)
├── models.rs                            # Model resolution (static → dynamic → auto), URL builders
├── routes/
│   ├── mod.rs                           # Routes module root
│   ├── anthropic.rs                     # POST /v1/messages + real/fake SSE streaming
│   ├── openai.rs                        # POST /v1/chat/completions
│   ├── health.rs                        # GET /health, /v1/models
│   ├── admin.rs                         # GET/POST/DELETE /admin/models
│   └── veo.rs                           # VEO video generation (generate, result, generate-sync)
└── transform/
    ├── mod.rs                           # Transform module root
    ├── anthropic_to_gemini.rs           # Anthropic ↔ Gemini format + thought signature handling
    ├── anthropic_to_openai.rs           # Anthropic ↔ OpenAI format + model quirk handling
    ├── gemini_stream.rs                 # Gemini SSE → Anthropic SSE state machine
    ├── openai_stream.rs                 # OpenAI SSE → Anthropic SSE state machine
    ├── sse_parser.rs                    # Incremental SSE parser (handles split chunks)
    └── schema_clean.rs                  # Strip unsupported JSON Schema keywords
```

---

## Deployment (systemd)

```ini
# /etc/systemd/system/vertex-proxy-rs.service
[Unit]
Description=Vertex AI Proxy (Rust)
After=network.target

[Service]
Type=simple
User=bod
WorkingDirectory=/home/bod
ExecStart=/home/bod/vertex-proxy-rs
Restart=always
RestartSec=5
Environment=PORT=8000
Environment=CONFIG_PATH=/home/bod/config.yaml
Environment=SIGNATURE_CACHE_PATH=/home/bod/signature_cache.json

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now vertex-proxy-rs
sudo systemctl status vertex-proxy-rs
```

---

## Claude Code Configuration

Set these environment variables to use the proxy with Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://your-gcp-vm-ip:8000
export ANTHROPIC_API_KEY=sk-your-secret-key
```

Then run Claude Code normally — it will use Gemini models through Vertex AI via the proxy.

---

## Performance

| Metric | Value |
|---|---|
| Binary size | ~3 MB |
| Memory (idle) | ~1.4 MB |
| Memory (under load) | ~5-8 MB |
| Startup time | <100ms |
| Runtime dependencies | 0 (static linked) |
| Streaming latency | Real-time (Gemini/OpenAPI) |

---

## Credits

Based on [OrionStarAI/claudecode-vertex-proxy](https://github.com/OrionStarAI/claudecode-vertex-proxy) (Python), rewritten in Rust with:
- ADC support (replaces service account key)
- Multi-publisher routing (Google, OpenAPI, Anthropic)
- Dual API (Anthropic + OpenAI)
- Real SSE streaming with state machines
- Thought signature server-side cache with persistence
- Dynamic model resolution with auto-resolve patterns
- Admin API for runtime model management
- VEO video generation support
- Streaming error handling (SSE-formatted errors)
- Model quirk handling (GLM-5, empty tool calls, etc.)

## License

MIT
