use axum::{extract::{Path, State}, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::AppState;
use crate::models::{ModelSource, collect_all_models};

#[derive(Debug, Deserialize)]
pub struct AddModelRequest {
    pub name: String,
    #[serde(default)]
    pub vertex_name: Option<String>,
    pub publisher: String,
    #[serde(default)]
    pub region: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub vertex_name: String,
    pub publisher: String,
    pub region: String,
    pub source: String,
}

/// GET /admin/models — List all models with full details including source
pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let all = collect_all_models(&state.config, &state.dynamic_models).await;

    let models: Vec<ModelInfo> = all
        .iter()
        .map(|(m, source)| ModelInfo {
            name: m.name.clone(),
            vertex_name: m.vertex_name.clone(),
            publisher: m.publisher.clone(),
            region: m.region.clone(),
            source: match source {
                ModelSource::Config => "config".to_string(),
                ModelSource::Dynamic => "dynamic".to_string(),
                ModelSource::AutoResolved => "auto".to_string(),
            },
        })
        .collect();

    Json(json!({
        "total": models.len(),
        "models": models,
        "auto_resolve_patterns": {
            "gemini-*": {"publisher": "google", "region": "us-central1"},
            "claude-*": {"publisher": "anthropic", "region": "us-central1"},
            "*/*": {"publisher": "openapi", "region": "global"}
        }
    }))
}

/// POST /admin/models — Add a model at runtime
pub async fn add_model(
    State(state): State<AppState>,
    Json(body): Json<AddModelRequest>,
) -> Result<Json<Value>, (axum::http::StatusCode, Json<Value>)> {
    // Validate publisher
    if crate::models::Publisher::parse(&body.publisher).is_err() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!("Invalid publisher: '{}'. Must be: google, openapi, or anthropic", body.publisher)
            })),
        ));
    }

    // Check if model already exists in static config
    if state.config.find_model(&body.name).is_some() {
        return Err((
            axum::http::StatusCode::CONFLICT,
            Json(json!({
                "error": format!("Model '{}' already exists in static config", body.name)
            })),
        ));
    }

    let vertex_name = body.vertex_name.unwrap_or_else(|| body.name.clone());
    let region = body.region.unwrap_or_else(|| {
        // Default region based on publisher
        match body.publisher.as_str() {
            "google" => "us-central1".to_string(),
            "anthropic" => "us-central1".to_string(),
            _ => "global".to_string(),
        }
    });

    let model_config = crate::config::ModelConfig {
        name: body.name.clone(),
        vertex_name,
        publisher: body.publisher.clone(),
        region,
    };

    // Check if already in dynamic models
    {
        let mut dyn_models = state.dynamic_models.write().await;
        if dyn_models.iter().any(|m| m.name == body.name) {
            return Err((
                axum::http::StatusCode::CONFLICT,
                Json(json!({
                    "error": format!("Model '{}' already exists in dynamic models", body.name)
                })),
            ));
        }
        dyn_models.push(model_config.clone());
    }

    tracing::info!(model = %body.name, publisher = %body.publisher, "Dynamic model added via admin API");

    Ok(Json(json!({
        "message": format!("Model '{}' added successfully", body.name),
        "model": {
            "name": model_config.name,
            "vertex_name": model_config.vertex_name,
            "publisher": model_config.publisher,
            "region": model_config.region,
            "source": "dynamic"
        }
    })))
}

/// DELETE /admin/models/{name} — Remove a dynamic model
pub async fn delete_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<Json<Value>, (axum::http::StatusCode, Json<Value>)> {
    // Cannot delete from static config
    if state.config.find_model(&name).is_some() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!("Model '{}' is in static config, cannot delete at runtime. Remove from config.yaml instead.", name)
            })),
        ));
    }

    let mut dyn_models = state.dynamic_models.write().await;
    let original_len = dyn_models.len();
    dyn_models.retain(|m| m.name != name);

    if dyn_models.len() == original_len {
        return Err((
            axum::http::StatusCode::NOT_FOUND,
            Json(json!({
                "error": format!("Dynamic model '{}' not found", name)
            })),
        ));
    }

    tracing::info!(model = %name, "Dynamic model removed via admin API");

    Ok(Json(json!({
        "message": format!("Model '{}' removed successfully", name)
    })))
}
