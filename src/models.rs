use crate::config::Config;

#[derive(Debug, Clone)]
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
}

pub struct ResolvedModel {
    pub vertex_name: String,
    pub publisher: Publisher,
    pub region: String,
}

pub fn resolve_model(config: &Config, model_name: &str) -> Result<ResolvedModel, String> {
    let m = config
        .find_model(model_name)
        .ok_or_else(|| {
            let available: Vec<&str> = config.models.iter().map(|m| m.name.as_str()).collect();
            format!("Unknown model: {model_name}. Available: {available:?}")
        })?;

    let publisher = Publisher::parse(&m.publisher)?;

    Ok(ResolvedModel {
        vertex_name: m.vertex_name.clone(),
        publisher,
        region: m.region.clone(),
    })
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
