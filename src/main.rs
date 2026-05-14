mod config;
mod auth;
mod models;
mod routes;
mod transform;

use std::sync::Arc;
use std::collections::HashMap;
use axum::{Router, routing::{get, post}};
use tower_http::cors::CorsLayer;
use tokio::sync::RwLock;

use config::Config;
use auth::AuthManager;

/// Server-side cache for Gemini thought signatures.
/// Maps tool_use_id → thoughtSignature value.
/// This is needed because Claude Code (and other Anthropic clients)
/// don't preserve custom fields like _thought_signature when sending
/// conversation history back. The proxy must store signatures server-side
/// and re-inject them when converting back to Gemini format.
#[derive(Clone, Default)]
pub struct SignatureCache {
    /// tool_use_id → thoughtSignature (from Gemini functionCall part)
    tool_signatures: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    /// msg_id → list of (part_index, thoughtSignature) for non-functionCall parts
    /// (text parts with thoughtSignature)
    text_signatures: Arc<RwLock<HashMap<String, Vec<(usize, serde_json::Value)>>>>,
}

impl SignatureCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a thought signature for a tool_use_id
    pub async fn store_tool_signature(&self, tool_use_id: String, signature: serde_json::Value) {
        if !signature.is_null() {
            self.tool_signatures.write().await.insert(tool_use_id, signature);
        }
    }

    /// Retrieve and remove a thought signature for a tool_use_id
    pub async fn take_tool_signature(&self, tool_use_id: &str) -> Option<serde_json::Value> {
        self.tool_signatures.write().await.remove(tool_use_id)
    }

    /// Peek at a thought signature without removing it
    pub async fn get_tool_signature(&self, tool_use_id: &str) -> Option<serde_json::Value> {
        self.tool_signatures.read().await.get(tool_use_id).cloned()
    }
}

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub auth: AuthManager,
    pub client: reqwest::Client,
    pub signatures: SignatureCache,
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
        signatures: SignatureCache::new(),
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
