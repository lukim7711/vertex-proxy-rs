# Vertex AI Proxy (Rust)

A lightweight proxy that translates **Anthropic Messages API** and **OpenAI Chat Completions API** to **Google Vertex AI**, enabling tools like Claude Code to use models on Vertex AI (Gemini, Grok, GLM) without a Claude API key.

**Binary 2.7MB | Memory ~1.4MB | Zero runtime dependencies**

## Architecture

```
Claude Code / OpenAI Client          Proxy (Rust)                        Vertex AI
─────────────────────────            ─────────────────                   ──────────
POST /v1/messages ──────────>  Anthropic → Gemini contents        ──>  Gemini
(Anthropic API format)         Anthropic → OpenAI chat            ──>  Grok / GLM
                               Anthropic passthrough              ──>  Claude

POST /v1/chat/completions ───>  OpenAI → Gemini contents           ──>  Gemini
(OpenAI API format)             OpenAI passthrough                 ──>  Grok / GLM
```

The proxy reads the model from the request, matches it against the config, then routes to the appropriate Vertex AI endpoint. Three publisher paths:

| Publisher | Model | Vertex AI Endpoint | Transformation |
|---|---|---|---|
| `google` | Gemini | `publishers/google/models/{m}:generateContent` | Anthropic/OpenAI → Gemini |
| `openapi` | Grok, GLM | `endpoints/openapi/chat/completions` | Anthropic → OpenAI |
| `anthropic` | Claude | `publishers/anthropic/models/{m}:rawPredict` | Passthrough |

## Features

- **Dual API**: Anthropic Messages API (`/v1/messages`) and OpenAI Chat Completions API (`/v1/chat/completions`)
- **Multi-model**: Gemini, Grok (xAI), GLM (Zhipu AI), Claude — all through one proxy
- **Tool use**: Translates Anthropic/OpenAI tools to Vertex AI format and back
- **Streaming (SSE)**: Response chunked as Server-Sent Events (Anthropic streaming format)
- **ADC Auth**: Authenticates to Vertex AI via GCP metadata server (Application Default Credentials)
- **API key auth**: Protects the proxy with `x-api-key` or `Authorization: Bearer`

## Supported Models

| Model | Publisher | Region |
|---|---|---|
| `gemini-2.5-flash` | Google | us-central1 |
| `gemini-2.5-pro` | Google | us-central1 |
| `gemini-3.1-pro-preview` | Google | global |
| `gemini-3.1-flash-lite` | Google | global |
| `xai/grok-4.1-fast-reasoning` | xAI (OpenAPI) | global |
| `xai/grok-4.20-reasoning` | xAI (OpenAPI) | global |
| `zai-org/glm-5-maas` | Zhipu AI (OpenAPI) | global |

The model list is configured in `config.yaml` — add or remove as needed.

## Quick Start

### Prerequisites

- Rust toolchain (1.75+)
- Running on a GCP VM with Vertex AI API enabled
- Service account with `Vertex AI User` role

### Build

```bash
cargo build --release
```

Binary at `target/release/vertex-proxy-rs` (~2.7MB with LTO + strip).

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
```

### Run

```bash
./vertex-proxy-rs
# => Vertex AI Proxy (Rust) starting on 0.0.0.0:8000
```

Environment variables:
- `PORT` — port (default: `8000`)
- `CONFIG_PATH` — path to config (default: `config.yaml`)

## API Endpoints

### POST `/v1/messages` — Anthropic Messages API

Used by Claude Code. Supports streaming (`"stream": true`).

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### POST `/v1/chat/completions` — OpenAI Chat Completions API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xai/grok-4.1-fast-reasoning",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### GET `/v1/models` — List models

```bash
curl http://localhost:8000/v1/models
```

### GET `/health` — Health check

```bash
curl http://localhost:8000/health
```

## Authentication

### Client → Proxy

Send the API key via either:
- Header `x-api-key: sk-your-secret-key`
- Header `Authorization: Bearer sk-your-secret-key`

The key is validated against `master_key` in `config.yaml`.

### Proxy → Vertex AI

The proxy uses the **GCP metadata server** to obtain an access token (Application Default Credentials). Tokens are cached for ~50 minutes, auto-refreshed. No service account key file needed.

## Deployment (systemd)

```ini
# /etc/systemd/system/vertex-proxy-rust.service
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

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now vertex-proxy-rust
sudo systemctl status vertex-proxy-rust
```

## Source Structure

```
src/
├── main.rs       # Server startup, Axum router
├── config.rs     # YAML config loader
├── auth.rs       # GCP metadata server ADC token (cached)
├── models.rs     # Model registry + Vertex URL builder
├── routes/
│   ├── anthropic.rs   # POST /v1/messages + SSE streaming
│   ├── openai.rs      # POST /v1/chat/completions
│   └── health.rs      # GET /health, /v1/models
└── transform/
    ├── anthropic_to_gemini.rs   # Anthropic ↔ Gemini format
    ├── anthropic_to_openai.rs   # Anthropic ↔ OpenAI format
    └── schema_clean.rs          # Strip unsupported JSON Schema keys
```

## How Transformations Work

### Publisher: `google` (Gemini)

- System prompt → `systemInstruction.parts[].text`
- Messages → `contents[]` (role: user/model, parts: [{text}])
- Tools → `tools[].functionDeclarations`
- Response: `candidates[0].content.parts[]` → Anthropic content blocks / OpenAI choices

### Publisher: `openapi` (Grok, GLM)

- Anthropic tools → OpenAI `tools[].function`
- Anthropic messages → OpenAI messages (system/user/assistant/tool)
- Anthropic `tool_use` ↔ OpenAI `tool_calls`
- Anthropic `tool_result` ↔ OpenAI `tool` role message

### Publisher: `anthropic` (Claude)

- Passthrough with `anthropic_version: vertex-2023-10-16`

## Performance

| Metric | Value |
|---|---|
| Binary size | 2.7 MB |
| Memory (idle) | ~1.4 MB |
| Memory (load) | ~5-8 MB |
| Startup time | <100ms |
| Runtime dependencies | 0 (static linked) |

## Credits

Based on [OrionStarAI/claudecode-vertex-proxy](https://github.com/OrionStarAI/claudecode-vertex-proxy) (Python), rewritten in Rust with:
- ADC support (replaces service account key)
- Multi-publisher routing (Google, OpenAPI, Anthropic)
- Dual API (Anthropic + OpenAI)
- Proper SSE streaming

## License

MIT
