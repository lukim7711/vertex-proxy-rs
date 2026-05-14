# Vertex AI Proxy (Rust)

Proxy ringan yang menerjemahkan **Anthropic Messages API** dan **OpenAI Chat Completions API** ke **Google Vertex AI**, memungkinkan tools seperti Claude Code menggunakan model-model di Vertex AI (Gemini, Grok, GLM) tanpa Claude API key.

**Binary 2.7MB | Memory ~1.4MB | Zero runtime dependencies**

## Arsitektur

```
Claude Code / OpenAI Client          Proxy (Rust)                        Vertex AI
─────────────────────────            ─────────────────                   ──────────
POST /v1/messages ──────────>  Anthropic → Gemini contents        ──>  Gemini
(Anthropic API format)         Anthropic → OpenAI chat            ──>  Grok / GLM
                               Anthropic passthrough              ──>  Claude

POST /v1/chat/completions ───>  OpenAI → Gemini contents           ──>  Gemini
(OpenAI API format)             OpenAI passthrough                 ──>  Grok / GLM
```

Proxy membaca model dari request, mencocokkan dengan config, lalu merutekan ke Vertex AI endpoint yang sesuai. Tiga jalur publisher:

| Publisher | Model | Endpoint Vertex AI | Transformasi |
|---|---|---|---|
| `google` | Gemini | `publishers/google/models/{m}:generateContent` | Anthropic/OpenAI → Gemini |
| `openapi` | Grok, GLM | `endpoints/openapi/chat/completions` | Anthropic → OpenAI |
| `anthropic` | Claude | `publishers/anthropic/models/{m}:rawPredict` | Passthrough |

## Fitur

- **Dual API**: Anthropic Messages API (`/v1/messages`) dan OpenAI Chat Completions API (`/v1/chat/completions`)
- **Multi-model**: Gemini, Grok (xAI), GLM (Zhipu AI), Claude — semua lewat satu proxy
- **Tool use**: Menerjemahkan Anthropic/OpenAI tools ke format Vertex AI dan sebaliknya
- **Streaming (SSE)**: Response di-chunk sebagai Server-Sent Events (format Anthropic streaming)
- **ADC Auth**: Autentikasi ke Vertex AI via GCP metadata server (Application Default Credentials)
- **API key auth**: Melindungi proxy dengan `x-api-key` atau `Authorization: Bearer`

## Model yang Didukung

| Model | Publisher | Region |
|---|---|---|
| `gemini-2.5-flash` | Google | us-central1 |
| `gemini-2.5-pro` | Google | us-central1 |
| `gemini-3.1-pro-preview` | Google | global |
| `gemini-3.1-flash-lite` | Google | global |
| `xai/grok-4.1-fast-reasoning` | xAI (OpenAPI) | global |
| `xai/grok-4.20-reasoning` | xAI (OpenAPI) | global |
| `zai-org/glm-5-maas` | Zhipu AI (OpenAPI) | global |

Daftar model dikonfigurasi di `config.yaml` — bisa ditambah/kurangi sesuai kebutuhan.

## Quick Start

### Prasyarat

- Rust toolchain (1.75+)
- Berjalan di GCP VM dengan Vertex AI API enabled
- Service account dengan role `Vertex AI User`

### Build

```bash
cargo build --release
```

Binary ada di `target/release/vertex-proxy-rs` (~2.7MB dengan LTO + strip).

### Konfigurasi

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml`:

```yaml
master_key: "sk-kunci-rahasia-anda"
default_model: "gemini-2.5-flash"

models:
  - name: "gemini-2.5-flash"
    vertex_name: "gemini-2.5-flash"
    publisher: "google"
    region: "us-central1"
```

### Jalankan

```bash
./vertex-proxy-rs
# => Vertex AI Proxy (Rust) starting on 0.0.0.0:8000
```

Environment variables:
- `PORT` — port (default: `8000`)
- `CONFIG_PATH` — path ke config (default: `config.yaml`)

## API Endpoints

### POST `/v1/messages` — Anthropic Messages API

Digunakan oleh Claude Code. Mendukung streaming (`"stream": true`).

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: sk-kunci-rahasia-anda" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Halo"}]
  }'
```

### POST `/v1/chat/completions` — OpenAI Chat Completions API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-kunci-rahasia-anda" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xai/grok-4.1-fast-reasoning",
    "messages": [{"role": "user", "content": "Halo"}]
  }'
```

### GET `/v1/models` — Daftar model

```bash
curl http://localhost:8000/v1/models
```

### GET `/health` — Health check

```bash
curl http://localhost:8000/health
```

## Autentikasi

### Client → Proxy

Kirim API key via salah satu:
- Header `x-api-key: sk-kunci-rahasia-anda`
- Header `Authorization: Bearer sk-kunci-rahasia-anda`

Key divalidasi terhadap `master_key` di `config.yaml`.

### Proxy → Vertex AI

Proxy menggunakan **GCP metadata server** untuk mendapatkan access token (Application Default Credentials). Token di-cache selama ~50 menit, auto-refresh. Tidak perlu service account key file.

## Deploy (systemd)

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

## Struktur Source Code

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
    └── schema_clean.rs          # Strip kunci JSON Schema tidak didukung
```

## Cara Kerja Transformasi

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

- Passthrough dengan menambahkan `anthropic_version: vertex-2023-10-16`

## Performa

| Metrik | Nilai |
|---|---|
| Ukuran binary | 2.7 MB |
| Memory (idle) | ~1.4 MB |
| Memory (beban) | ~5-8 MB |
| Waktu startup | <100ms |
| Dependensi runtime | 0 (static linked) |

## Kredit

Berdasarkan [OrionStarAI/claudecode-vertex-proxy](https://github.com/OrionStarAI/claudecode-vertex-proxy) (Python), ditulis ulang dalam Rust dengan:
- ADC support (menggantikan service account key)
- Multi-publisher routing (Google, OpenAPI, Anthropic)
- Dual API (Anthropic + OpenAI)
- SSE streaming yang sebenarnya

## Lisensi

MIT
