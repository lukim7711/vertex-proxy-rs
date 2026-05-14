mod config;
mod auth;
mod models;
mod routes;
mod transform;

use std::sync::Arc;
use axum::{Router, routing::{get, post}};
use tower_http::cors::CorsLayer;

use config::Config;
use auth::AuthManager;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub auth: AuthManager,
    pub client: reqwest::Client,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "config.yaml".to_string());
    let config = Config::load(&config_path).expect("Failed to load config");
    let config = Arc::new(config);

    // Shared HTTP client with connection pool and default timeout
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .pool_max_idle_per_host(10)
        .build()
        .expect("Failed to create HTTP client");

    let state = AppState {
        config: config.clone(),
        auth: AuthManager::new(client.clone()),
        client,
    };

    let app = Router::new()
        .route("/health", get(routes::health::health))
        .route("/v1/models", get(routes::health::list_models))
        .route("/v1/messages", post(routes::anthropic::messages))
        .route("/v1/chat/completions", post(routes::openai::chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8000".to_string());
    let addr = format!("0.0.0.0:{port}");

    tracing::info!("Vertex AI Proxy (Rust) starting on {addr}");
    tracing::info!("Models loaded: {}", config.models.len());

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
