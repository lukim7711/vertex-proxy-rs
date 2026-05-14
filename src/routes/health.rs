use axum::{extract::State, Json};
use serde_json::{json, Value};

use crate::AppState;

pub async fn health(State(state): State<AppState>) -> Json<Value> {
    let models: Vec<&str> = state.config.models.iter().map(|m| m.name.as_str()).collect();
    Json(json!({"status": "ok", "models": models}))
}

pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": state.config.models.iter().map(|m| json!({
            "id": m.name,
            "object": "model",
            "owned_by": m.publisher,
        })).collect::<Vec<_>>()
    }))
}
