use axum::{extract::State, Json};
use serde_json::{json, Value};

use crate::AppState;
use crate::models::collect_all_models;

pub async fn health(State(state): State<AppState>) -> Json<Value> {
    let static_count = state.config.models.len();
    let dynamic_count = state.dynamic_models.read().await.len();
    let token_info = state.auth.token_info().await;

    Json(json!({
        "status": "ok",
        "static_models": static_count,
        "dynamic_models": dynamic_count,
        "auto_resolve_enabled": true,
        "rate_limit_enabled": state.config.rate_limit.enabled,
        "retry_enabled": state.config.retry.enabled,
        "auth": {
            "has_token": token_info.has_token,
            "seconds_until_refresh": token_info.seconds_until_refresh,
            "token_age_secs": token_info.token_age_secs,
            "refresh_count": token_info.refresh_count,
            "is_refreshing": token_info.is_refreshing,
        }
    }))
}

pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let all = collect_all_models(&state.config, &state.dynamic_models).await;

    Json(json!({
        "object": "list",
        "data": all.iter().map(|(m, source)| json!({
            "id": m.name,
            "object": "model",
            "owned_by": m.publisher,
            "vertex_name": m.vertex_name,
            "region": m.region,
            "source": match source {
                crate::models::ModelSource::Config => "config",
                crate::models::ModelSource::Dynamic => "dynamic",
                crate::models::ModelSource::AutoResolved => "auto",
            },
        })).collect::<Vec<_>>()
    }))
}
