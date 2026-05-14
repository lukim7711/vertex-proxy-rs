# Vertex AI Proxy (Rust)

Proxy ringan dan berkinerja tinggi yang menerjemahkan **Anthropic Messages API** dan **OpenAI Chat Completions API** ke **Google Vertex AI**, memungkinkan tools seperti Claude Code menggunakan Gemini, Grok, GLM, Claude, dan 35+ model lainnya di Vertex AI tanpa API key berbayar — memanfaatkan GCP Free Tier kredit $300 melalui autentikasi ADC.

**Binary ~3MB | Memory ~1.4MB idle | Zero runtime dependencies | Static linked**

---

## Arsitektur

```
Client (Claude Code, OpenAI SDK, dll)     Proxy (Rust/Axum)                          Vertex AI
──────────────────────────────────         ──────────────────                         ──────────
POST /v1/messages ──────────────>   Anthropic → Gemini contents           ──>  Gemini (Google)
(Anthropic API format)               Anthropic → OpenAI chat               ──>  Grok, GLM, Mistral...
                                     Anthropic passthrough                 ──>  Claude (Anthropic)

POST /v1/chat/completions ──────>   OpenAI → Gemini contents              ──>  Gemini (Google)
(OpenAI API format)                   OpenAI passthrough                   ──>  Grok, GLM, Mistral...

POST /v1/veo/generate ──────────>   VEO predictLongRunning                ──>  VEO Video Generation
POST /v1/veo/result ────────────>   VEO fetchPredictOperation             ──>  VEO Result Polling

GET  /admin/models ─────────────>   Dynamic model management              ──>  Runtime config
POST /admin/models ─────────────>   Tambah model saat runtime             ──>  Tanpa restart
DELETE /admin/models/{name} ─────>   Hapus model dinamis                  ──>  Tanpa restart
```

Proxy membaca nama model dari request, me-resolve-nya terhadap config (statis, dinamis, atau auto-resolve), lalu merutekan ke endpoint Vertex AI yang sesuai. Tiga jalur publisher:

| Publisher | Model | Endpoint Vertex AI | Transformasi |
|---|---|---|---|
| `google` | Gemini, VEO | `publishers/google/models/{m}:generateContent` | Anthropic/OpenAI → Gemini |
| `openapi` | Grok, GLM, Mistral, Llama, DeepSeek, Qwen, dll. | `endpoints/openapi/chat/completions` | Anthropic → OpenAI |
| `anthropic` | Claude | `publishers/anthropic/models/{m}:rawPredict` | Passthrough |

---

## Fitur

### Inti
- **Dual API**: Anthropic Messages API (`/v1/messages`) dan OpenAI Chat Completions API (`/v1/chat/completions`)
- **35+ Model**: Gemini 3.1/3/2.5/2.0, Grok 4, GLM-5, Claude Opus/Sonnet/Haiku, Mistral, Llama 4, DeepSeek, Qwen 3, dan lainnya
- **Tool Use**: Translasi lengkap Anthropic/OpenAI tools ke format Vertex AI dan sebaliknya — mendukung multi-turn tool calling
- **Real Streaming (SSE)**: Streaming token-per-token sesungguhnya untuk publisher Google (Gemini) dan OpenAPI; fake streaming fallback untuk Anthropic passthrough
- **Dukungan Thinking Model**: Menangani mode thinking Gemini dengan cache `thoughtSignature` server-side yang persisten ke disk, memastikan percakapan round-trip yang benar dengan model thinking
- **Input Gambar**: Mendukung blok gambar base64 dalam format Anthropic → konversi ke `inlineData` Gemini

### Dynamic Model Resolution
- **Auto-Resolve**: Model yang tidak ada di config otomatis di-resolve berdasarkan pola nama:
  - `gemini-*` → Publisher Google, us-central1
  - `veo-*` → Publisher Google (VEO video), us-central1
  - `claude-*` → Publisher Anthropic, us-central1
  - `*/*` (misalnya `xai/grok-4`) → Publisher OpenAPI, global
- **Admin API**: Tambah/hapus model saat runtime tanpa restart proxy
- **Prioritas**: Static config > Dynamic models > Auto-resolve

### VEO Video Generation
- **Async API**: Submit job generasi video dengan `POST /v1/veo/generate`
- **Result Polling**: Cek status job dengan `POST /v1/veo/result`
- **Sync Convenience**: Auto-poll dengan `POST /v1/veo/generate-sync` (blocking sampai selesai atau timeout)
- **5 Model VEO**: veo-3.1-fast, veo-3.1, veo-3.1-lite, veo-3, veo-2

### Keamanan & Autentikasi
- **ADC Auth**: Autentikasi ke Vertex AI via GCP metadata server (Application Default Credentials) — tidak perlu service account key
- **Token Caching**: Access token di-cache selama ~50 menit dengan double-check locking dan auto-refresh
- **API Key Auth**: Melindungi proxy dengan validasi header `x-api-key` atau `Authorization: Bearer`
- **CORS**: CORS layer permissive untuk akses cross-origin

### Ketahanan
- **Streaming Error Handling**: Error upstream saat streaming dikembalikan sebagai SSE error event yang terformat benar (bukan raw JSON), sehingga Claude Code bisa mem-parse-nya dengan benar
- **Model Quirks**: Menangani GLM-5 `content: null`, tool call halusinasi dengan nama kosong, dan edge case lainnya
- **Schema Cleaning**: Otomatis menghapus keyword JSON Schema yang tidak didukung (`$ref`, `allOf`, `additionalProperties`, dll.) sebelum mengirim ke Gemini
- **Role Alternation**: Gemini membutuhkan alternasi user/model yang ketat — proxy otomatis menggabungkan turn dengan role yang sama berturut-turut
- **Signature Persistence**: Thought signature disimpan ke `signature_cache.json` dengan atomic writes (temp + rename), bertahan dari restart proxy

---

## Model yang Didukung

### Google Gemini

| Model | Publisher | Region | Keterangan |
|---|---|---|---|
| `gemini-3.1-pro-preview` | Google | global | Terbaru, reasoning terkuat |
| `gemini-3.1-flash-lite` | Google | global | Cepat & ringan |
| `gemini-3.1-flash-image` | Google | global | Generasi gambar |
| `gemini-3-pro` | Google | us-central1 | Stabil |
| `gemini-3-flash` | Google | us-central1 | Cepat |
| `gemini-3-flash-image` | Google | us-central1 | Generasi gambar |
| `gemini-2.5-pro` | Google | us-central1 | Generally available |
| `gemini-2.5-flash` | Google | us-central1 | Generally available |
| `gemini-2.5-flash-image` | Google | us-central1 | Generasi gambar |
| `gemini-2.5-flash-lite` | Google | us-central1 | Ringan |
| `gemini-2.0-flash` | Google | us-central1 | Stabil |
| `gemini-2.0-flash-lite` | Google | us-central1 | Ringan |

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

> **Catatan**: Daftar model dikonfigurasi di `config.yaml`. Model yang tidak ada di config akan otomatis di-resolve berdasarkan pola nama, jadi Anda bisa menggunakan model Vertex AI apapun tanpa mendaftarkannya.

---

## Quick Start

### Prasyarat

- Rust toolchain (1.75+) — hanya diperlukan untuk build dari source
- GCP VM dengan Vertex AI API aktif
- Service account dengan role `Vertex AI User` (atau permission IAM yang sesuai)
- Binary pre-built bisa berjalan tanpa Rust terinstal

### Build dari Source

```bash
cargo build --release
```

Binary ada di `target/release/vertex-proxy-rs` (~3MB dengan LTO + strip).

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
  - name: "xai/grok-4.1-fast"
    vertex_name: "xai/grok-4.1-fast"
    publisher: "openapi"
    region: "global"
```

Setiap entri model punya 4 field:
- `name` — Nama yang digunakan client (bisa alias)
- `vertex_name` — Nama model sebenarnya di Vertex AI
- `publisher` — Salah satu: `google`, `openapi`, `anthropic`
- `region` — Region Vertex AI (`us-central1`, `global`, dll.)

### Jalankan

```bash
./vertex-proxy-rs
# => Vertex AI Proxy (Rust) starting on 0.0.0.0:8000
```

Environment variables:

| Variabel | Default | Deskripsi |
|---|---|---|
| `PORT` | `8000` | Port server |
| `CONFIG_PATH` | `config.yaml` | Path ke file config |
| `SIGNATURE_CACHE_PATH` | `signature_cache.json` | Path ke cache thought signature |

---

## API Endpoints

### POST `/v1/messages` — Anthropic Messages API

Digunakan oleh Claude Code. Mendukung streaming (`"stream": true`).

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: sk-kunci-rahasia-anda" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "max_tokens": 4096,
    "messages": [{"role": "user", "content": "Halo"}]
  }'
```

Streaming:
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: sk-kunci-rahasia-anda" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "max_tokens": 4096,
    "stream": true,
    "messages": [{"role": "user", "content": "Halo"}]
  }'
```

Dengan tools:
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "x-api-key: sk-kunci-rahasia-anda" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.5-flash",
    "max_tokens": 4096,
    "tools": [{"name": "get_weather", "description": "Ambil cuaca", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
    "messages": [{"role": "user", "content": "Bagaimana cuaca di Jakarta?"}]
  }'
```

### POST `/v1/chat/completions` — OpenAI Chat Completions API

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-kunci-rahasia-anda" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "xai/grok-4.1-fast",
    "messages": [{"role": "user", "content": "Halo"}]
  }'
```

### VEO Video Generation

#### POST `/v1/veo/generate` — Submit job generasi video

```bash
curl -X POST http://localhost:8000/v1/veo/generate \
  -H "x-api-key: sk-kunci-rahasia-anda" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "veo-3.1-fast-generate-001",
    "prompt": "Kucing berjalan di pantai saat matahari terbenam",
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

#### POST `/v1/veo/result` — Poll hasil

```bash
curl -X POST http://localhost:8000/v1/veo/result \
  -H "x-api-key: sk-kunci-rahasia-anda" \
  -H "Content-Type: application/json" \
  -d '{
    "operationName": "projects/.../operations/abc123",
    "model": "veo-3.1-fast-generate-001"
  }'
```

#### POST `/v1/veo/generate-sync` — Submit dan auto-poll (blocking)

```bash
curl -X POST http://localhost:8000/v1/veo/generate-sync \
  -H "x-api-key: sk-kunci-rahasia-anda" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "veo-3.1-fast-generate-001",
    "prompt": "Kucing berjalan di pantai saat matahari terbenam",
    "pollInterval": 5,
    "maxPollTime": 300
  }'
```

### Admin API

#### GET `/admin/models` — Daftar semua model dengan info source

```bash
curl -H "x-api-key: sk-kunci-rahasia-anda" http://localhost:8000/admin/models
```

Response:
```json
{
  "total": 38,
  "models": [
    {"name": "gemini-2.5-flash", "vertex_name": "gemini-2.5-flash", "publisher": "google", "region": "us-central1", "source": "config"},
    {"name": "model-custom-saya", "vertex_name": "model-custom-saya", "publisher": "openapi", "region": "global", "source": "dynamic"}
  ],
  "auto_resolve_patterns": {
    "gemini-*": {"publisher": "google", "region": "us-central1"},
    "claude-*": {"publisher": "anthropic", "region": "us-central1"},
    "*/*": {"publisher": "openapi", "region": "global"}
  }
}
```

#### POST `/admin/models` — Tambah model saat runtime

```bash
curl -X POST http://localhost:8000/admin/models \
  -H "x-api-key: sk-kunci-rahasia-anda" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "model-custom-saya",
    "publisher": "openapi",
    "region": "global"
  }'
```

#### DELETE `/admin/models/{name}` — Hapus model dinamis

```bash
curl -X DELETE http://localhost:8000/admin/models/model-custom-saya \
  -H "x-api-key: sk-kunci-rahasia-anda"
```

### GET `/v1/models` — Daftar model (kompatibel OpenAI)

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

## Autentikasi

### Client → Proxy

Kirim API key via salah satu:
- Header `x-api-key: sk-kunci-rahasia-anda`
- Header `Authorization: Bearer sk-kunci-rahasia-anda`

Key divalidasi terhadap `master_key` di `config.yaml`.

### Proxy → Vertex AI

Proxy menggunakan **GCP metadata server** untuk mendapatkan access token (Application Default Credentials). Token di-cache selama ~50 menit dengan double-check locking (read lock untuk fast path, write lock dengan re-check untuk refresh). Tidak perlu service account key file — cukup jalankan di GCP VM dengan role IAM yang sesuai.

---

## Cara Kerja Transformasi

### Publisher: `google` (Gemini)

**Request:**
- System prompt → `systemInstruction.parts[].text`
- Messages → `contents[]` (role: user/model, parts: [{text}])
- Tools → `tools[].functionDeclarations` (dengan pembersihan JSON Schema)
- Blok gambar → `inlineData` dengan data base64
- Thought signature → field `thoughtSignature` pada parts (dari cache server-side)

**Response:**
- `candidates[0].content.parts[]` → Anthropic content blocks / OpenAI choices
- Thought parts (`"thought": true`) → dilewati di output, signature di-cache untuk round-trip
- `functionCall` parts → blok `tool_use` dengan `_thought_signature`
- `usageMetadata` → jumlah token

**Streaming:**
- Menggunakan `streamGenerateContent?alt=sse` untuk SSE streaming sesungguhnya
- State machine (`GeminiStreamState`) mentransformasi chunk secara on-the-fly
- Thought signature dikumpulkan selama streaming dan disimpan batch ke cache saat stream berakhir

### Publisher: `openapi` (Grok, GLM, Mistral, dll.)

**Request:**
- Anthropic tools → OpenAI `tools[].function`
- Anthropic messages → OpenAI messages (system/user/assistant/tool)
- Anthropic `tool_use` → OpenAI `tool_calls`
- Anthropic `tool_result` → OpenAI pesan role `tool`

**Response:**
- Format OpenAI → format Anthropic
- Menangani model quirks (GLM-5 `content: null`, nama tool call kosong, dll.)

**Streaming:**
- Endpoint yang sama dengan non-streaming; streaming dipicu oleh `"stream": true` di body
- State machine (`OpenAiStreamState`) mentransformasi chunk secara on-the-fly

### Publisher: `anthropic` (Claude)

- Passthrough dengan penambahan `anthropic_version: vertex-2023-10-16`
- Streaming: menggunakan fake streaming (menunggu respons lengkap, lalu membagi ke SSE events)
- Tidak direkomendasikan untuk produksi — gunakan Anthropic API langsung untuk latensi yang lebih baik

---

## Penanganan Thought Signature

Model Gemini dengan mode thinking (misalnya, `gemini-2.5-flash` dengan extended thinking) melampirkan `thoughtSignature` pada parts di respons mereka. Signature ini **harus** dikirim kembali bersama request berikutnya yang mereferensikan parts tersebut, jika tidak API akan mengembalikan error.

Proxy menangani ini secara transparan menggunakan **cache signature server-side** (`signature_cache.json`):

1. **Inbound**: Saat Gemini merespons dengan `thoughtSignature` pada `functionCall` atau text parts, proxy menyimpannya di cache dengan key tool_use_id atau hash konten teks
2. **Outbound**: Saat mengkonversi pesan Anthropic kembali ke format Gemini, proxy mencari signature yang tersimpan dan melampirkannya ke parts yang sesuai
3. **Persistensi**: Cache ditulis ke disk dengan atomic writes (temp file + rename), sehingga signature bertahan dari restart proxy
4. **Streaming**: Signature dikumpulkan selama streaming dan disimpan batch saat stream selesai

Pendekatan ini berhasil bahkan ketika client (seperti Claude Code) tidak mempertahankan field custom seperti `_thought_signature` di riwayat percakapan mereka.

---

## Struktur Source Code

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
    ├── anthropic_to_openai.rs           # Anthropic ↔ OpenAI format + penanganan model quirk
    ├── gemini_stream.rs                 # Gemini SSE → Anthropic SSE state machine
    ├── openai_stream.rs                 # OpenAI SSE → Anthropic SSE state machine
    ├── sse_parser.rs                    # Incremental SSE parser (menangani chunk terpisah)
    └── schema_clean.rs                  # Hapus keyword JSON Schema yang tidak didukung
```

---

## Deploy (systemd)

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

## Konfigurasi Claude Code

Atur environment variables berikut untuk menggunakan proxy dengan Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://ip-vm-gcp-anda:8000
export ANTHROPIC_API_KEY=sk-kunci-rahasia-anda
```

Kemudian jalankan Claude Code secara normal — ia akan menggunakan model Gemini melalui Vertex AI via proxy.

---

## Performa

| Metrik | Nilai |
|---|---|
| Ukuran binary | ~3 MB |
| Memory (idle) | ~1.4 MB |
| Memory (beban) | ~5-8 MB |
| Waktu startup | <100ms |
| Dependensi runtime | 0 (static linked) |
| Latensi streaming | Real-time (Gemini/OpenAPI) |

---

## Kredit

Berdasarkan [OrionStarAI/claudecode-vertex-proxy](https://github.com/OrionStarAI/claudecode-vertex-proxy) (Python), ditulis ulang dalam Rust dengan:
- Dukungan ADC (menggantikan service account key)
- Multi-publisher routing (Google, OpenAPI, Anthropic)
- Dual API (Anthropic + OpenAI)
- SSE streaming sesungguhnya dengan state machine
- Cache thought signature server-side dengan persistensi
- Dynamic model resolution dengan pola auto-resolve
- Admin API untuk manajemen model saat runtime
- Dukungan VEO video generation
- Streaming error handling (error berformat SSE)
- Penanganan model quirk (GLM-5, tool call kosong, dll.)

## Lisensi

MIT
