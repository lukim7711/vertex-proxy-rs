use crate::config::{Config, ModelConfig};

#[derive(Debug, Clone, PartialEq)]
pub enum Publisher {
    Google,
    OpenApi,
    Anthropic,
}

impl Publisher {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "google" => Ok(Publisher::Google),
            "openapi" => Ok(Publisher::OpenApi),
            "anthropic" => Ok(Publisher::Anthropic),
            other => Err(format!("Unknown publisher: {other}. Expected: google, openapi, anthropic")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Publisher::Google => "google",
            Publisher::OpenApi => "openapi",
            Publisher::Anthropic => "anthropic",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub vertex_name: String,
    pub publisher: Publisher,
    pub region: String,
    /// Whether this model was auto-resolved from name pattern (not in config)
    pub auto_resolved: bool,
}

/// Resolution source: where the model config came from
#[derive(Debug, Clone, PartialEq)]
pub enum ModelSource {
    /// From static config.yaml
    Config,
    /// Added at runtime via admin API
    Dynamic,
    /// Auto-resolved from model name pattern
    AutoResolved,
}

/// Resolve a model by name, checking: config → dynamic models → auto-resolve
pub async fn resolve_model(
    config: &Config,
    dynamic_models: &tokio::sync::RwLock<Vec<ModelConfig>>,
    model_name: &str,
) -> Result<ResolvedModel, String> {
    // 1. Check static config first
    if let Some(m) = config.find_model(model_name) {
        let publisher = Publisher::parse(&m.publisher)?;
        return Ok(ResolvedModel {
            vertex_name: m.vertex_name.clone(),
            publisher,
            region: m.region.clone(),
            auto_resolved: false,
        });
    }

    // 2. Check dynamic (runtime-added) models
    {
        let dyn_models = dynamic_models.read().await;
        if let Some(m) = dyn_models.iter().find(|m| m.name == model_name) {
            let publisher = Publisher::parse(&m.publisher)?;
            return Ok(ResolvedModel {
                vertex_name: m.vertex_name.clone(),
                publisher,
                region: m.region.clone(),
                auto_resolved: false,
            });
        }
    }

    // 3. Auto-resolve from model name pattern
    auto_resolve_model(model_name)
}

/// Auto-resolve a model based on naming conventions.
/// This allows using any Vertex AI model without explicit config entry.
///
/// Pattern rules:
/// - `gemini-*`         → Google publisher, us-central1
/// - `claude-*`         → Anthropic publisher, us-central1
/// - `xai/*`            → OpenAPI publisher, global
/// - `zai-org/*`        → OpenAPI publisher, global
/// - `*/*` (other)      → OpenAPI publisher, global (partner models)
pub fn auto_resolve_model(model_name: &str) -> Result<ResolvedModel, String> {
    let (publisher, region) = if model_name.starts_with("gemini-") {
        (Publisher::Google, "us-central1".to_string())
    } else if model_name.starts_with("claude-") {
        (Publisher::Anthropic, "us-central1".to_string())
    } else if model_name.starts_with("veo-") {
        // VEO video generation models
        (Publisher::Google, "us-central1".to_string())
    } else if model_name.contains('/') {
        // Partner models with publisher/name format (e.g., xai/grok-4, zai-org/glm-5)
        (Publisher::OpenApi, "global".to_string())
    } else {
        return Err(format!(
            "Unknown model: '{model_name}'. Not found in config and cannot auto-resolve. \
             Auto-resolve patterns: gemini-* → Google, veo-* → Google (VEO), claude-* → Anthropic, */* → OpenAPI. \
             Add it to config.yaml or via POST /admin/models"
        ));
    };

    tracing::info!(
        model = model_name,
        publisher = publisher.as_str(),
        region = %region,
        "Auto-resolved model from name pattern"
    );

    Ok(ResolvedModel {
        vertex_name: model_name.to_string(),
        publisher,
        region,
        auto_resolved: true,
    })
}

/// Collect all known models: static config + dynamic, with source info
pub async fn collect_all_models(
    config: &Config,
    dynamic_models: &tokio::sync::RwLock<Vec<ModelConfig>>,
) -> Vec<(ModelConfig, ModelSource)> {
    let mut result: Vec<(ModelConfig, ModelSource)> = config
        .models
        .iter()
        .cloned()
        .map(|m| (m, ModelSource::Config))
        .collect();

    let dyn_models = dynamic_models.read().await;
    for m in dyn_models.iter() {
        result.push((m.clone(), ModelSource::Dynamic));
    }

    result
}

/// Build the Vertex AI URL for non-streaming requests.
pub fn build_vertex_url(
    project_id: &str,
    region: &str,
    publisher: &Publisher,
    vertex_model: &str,
) -> String {
    let base = vertex_base_url(region);
    let location = region;

    match publisher {
        Publisher::OpenApi => {
            format!("{base}/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions")
        }
        Publisher::Google => {
            format!("{base}/projects/{project_id}/locations/{location}/publishers/google/models/{vertex_model}:generateContent")
        }
        Publisher::Anthropic => {
            format!("{base}/projects/{project_id}/locations/{location}/publishers/anthropic/models/{vertex_model}:rawPredict")
        }
    }
}

/// Build the Vertex AI URL for streaming requests.
/// - Google (Gemini): streamGenerateContent with ?alt=sse for SSE streaming
/// - OpenAPI (Grok, GLM): same endpoint as non-streaming, streaming via "stream": true in body
/// - Anthropic: same endpoint, not supported for real streaming (falls back to fake streaming)
pub fn build_vertex_streaming_url(
    project_id: &str,
    region: &str,
    publisher: &Publisher,
    vertex_model: &str,
) -> String {
    let base = vertex_base_url(region);
    let location = region;

    match publisher {
        Publisher::OpenApi => {
            // Same endpoint; streaming is triggered by "stream": true in the request body
            format!("{base}/projects/{project_id}/locations/{location}/endpoints/openapi/chat/completions")
        }
        Publisher::Google => {
            // Use streamGenerateContent with alt=sse for Server-Sent Events
            format!("{base}/projects/{project_id}/locations/{location}/publishers/google/models/{vertex_model}:streamGenerateContent?alt=sse")
        }
        Publisher::Anthropic => {
            // Anthropic passthrough — not used for real streaming
            format!("{base}/projects/{project_id}/locations/{location}/publishers/anthropic/models/{vertex_model}:rawPredict")
        }
    }
}

fn vertex_base_url(region: &str) -> String {
    if region == "global" {
        "https://aiplatform.googleapis.com/v1".to_string()
    } else {
        format!("https://{region}-aiplatform.googleapis.com/v1")
    }
}

// =============================================================================
// VEO Video Generation URL Builders
// =============================================================================

/// Build the Vertex AI URL for VEO predictLongRunning (submit video generation job)
pub fn build_veo_predict_url(project_id: &str, region: &str, model: &str) -> String {
    let base = vertex_base_url(region);
    format!("{base}/projects/{project_id}/locations/{region}/publishers/google/models/{model}:predictLongRunning")
}

/// Build the Vertex AI URL for VEO fetchPredictOperation (poll for result)
pub fn build_veo_fetch_url(project_id: &str, region: &str, model: &str) -> String {
    let base = vertex_base_url(region);
    format!("{base}/projects/{project_id}/locations/{region}/publishers/google/models/{model}:fetchPredictOperation")
}
