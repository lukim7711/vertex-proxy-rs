# VEO Model Support — Implementation Report for GLM-5

## Problem

VEO video generation models use a **different API pattern** than LLM models. The current proxy only supports `:generateContent` / `:streamGenerateContent` / `:rawPredict` endpoints. VEO requires an **async operation-based** API.

## Current Architecture (LLM Models)

```
POST ...:generateContent
POST ...:streamGenerateContent?alt=sse
POST ...:rawPredict

Request → Process → Immediate Response (text/tool_use)
```

## VEO Architecture (Async Operation)

```
Step 1: POST ...:predictLongRunning → { "name": "operations/UUID" }
Step 2: POST ...:fetchPredictOperation → { "done": true, "response": {...} }
        (polling loop until "done": true)
```

## VEO Endpoints

| Operation | Endpoint |
|---|---|
| Submit job | `POST .../publishers/google/models/{model}:predictLongRunning` |
| Poll result | `POST .../publishers/google/models/{model}:fetchPredictOperation` |

## VEO Request Format (predictLongRunning)

```json
{
  "instances": [
    {
      "prompt": "A cat walking on a beach",
      "negativePrompt": "",
      "duration": 8,
      "resolution": "1080p",
      "aspectRatio": "16:9",
      "seed": 42,
      "numVideos": 1
    }
  ],
  "parameters": {
    "sampleCount": 1
  }
}
```

## VEO Response Flow

### Step 1 — predictLongRunning

```json
{
  "name": "projects/PROJECT_ID/locations/us-central1/publishers/google/models/veo-3.1-fast-generate-001/operations/abc123"
}
```

### Step 2 — fetchPredictOperation (polling)

Request:
```json
{
  "operationName": "projects/PROJECT_ID/locations/us-central1/publishers/google/models/veo-3.1-fast-generate-001/operations/abc123"
}
```

Response (done):
```json
{
  "done": true,
  "response": {
    "video": "gs://vertex-ai-video/abc123.mp4",
    "duration": 8,
    "resolution": "1080p"
  }
}
```

## Files to Modify

### `src/models.rs`

Add VEO publisher type and URL builder for `predictLongRunning` and `fetchPredictOperation`:

```rust
// New publisher variant (or handle as Google sub-type)
// VEO uses predictLongRunning/fetchPredictOperation instead of generateContent

pub fn build_veo_url(project_id: &str, region: &str, model: &str) -> String {
    format!("https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:predictLongRunning")
}

pub fn build_veo_poll_url(project_id: &str, region: &str, model: &str) -> String {
    format!("https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model}:fetchPredictOperation")
}
```

### `src/routes/mod.rs`

```rust
pub mod veo;  // add
```

### `src/routes/veo.rs` (NEW)

VEO-specific route handler:

```rust
// POST /v1/veo/generate — Submit video generation job
// POST /v1/veo/result — Poll for completed video
// GET /v1/veo/result/{operation_id} — Get result by operation ID (convenience)

pub async fn generate(...) -> Json<Value> {
    // 1. Accept VEO generation params (prompt, duration, resolution, etc.)
    // 2. Build request to Vertex AI :predictLongRunning
    // 3. Return operation ID
}

pub async fn fetch_result(...) -> Json<Value> {
    // 1. Accept operationName
    // 2. Poll Vertex AI :fetchPredictOperation
    // 3. Return video URI/details when done
}
```

### `src/main.rs`

Add routes:

```rust
.route("/v1/veo/generate", post(routes::veo::generate))
.route("/v1/veo/result", post(routes::veo::fetch_result))
```

### `src/config.rs`

Add VEO-specific config (if needed for different default region):

```rust
// Optional: VEO-specific default region or parameters
```

## VEO Models (Verified Available on Free Tier)

| Model | Endpoint |
|---|---|
| `veo-3.1-lite-generate-001` | `predictLongRunning` |
| `veo-3.1-generate-001` | `predictLongRunning` |
| `veo-3.1-fast-generate-001` | `predictLongRunning` |

All available at: `us-central1`

## Key Design Decisions Needed

1. **Proxy API format**: OpenAI video generation format or Anthropic-compatible?
2. **Polling**: Auto-poll with timeout, or client-side polling via operation ID?
3. **Auth**: Same API key auth (`x-api-key`) as LLM endpoints
4. **Response caching**: VEO responses are large — consider gs:// URI passthrough vs downloading

## Summary

VEO requires a **separate route module** because:
- Different endpoint suffix (`predictLongRunning` vs `generateContent`)
- Async operation pattern (submit → poll → result)
- Different request/response schema
- Not compatible with existing Anthropic/OpenAI message format transforms
