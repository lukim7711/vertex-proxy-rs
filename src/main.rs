mod config;
mod auth;
mod models;
mod routes;
mod transform;

use std::sync::Arc;
use std::collections::HashMap;
use axum::{Router, routing::{get, post, delete}};
use tower_http::cors::CorsLayer;
use tokio::sync::RwLock;

use config::{Config, ModelConfig};
use auth::AuthManager;

/// Signature cache directory and file
const SIGNATURE_CACHE_FILE: &str = "signature_cache.json";

/// Server-side cache for Gemini thought signatures.
/// Persists to disk so signatures survive proxy restarts.
///
/// Two storage layers:
/// 1. tool_signatures: tool_use_id → thoughtSignature (for functionCall parts)
/// 2. text_signatures: text_content_hash → thoughtSignature (for text parts from thinking models)
///
/// Both are persisted to a JSON file on every write.
#[derive(Clone)]
pub struct SignatureCache {
    /// tool_use_id → thoughtSignature (from Gemini functionCall part)
    tool_signatures: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    /// text_content_hash → thoughtSignature (from Gemini text parts in thinking model responses)
    text_signatures: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    /// Path to the persistence file
    persist_path: String,
}

impl SignatureCache {
    pub fn new(persist_path: &str) -> Self {
        let (tool_sigs, text_sigs) = Self::load_from_file(persist_path);
        Self {
            tool_signatures: Arc::new(RwLock::new(tool_sigs)),
            text_signatures: Arc::new(RwLock::new(text_sigs)),
            persist_path: persist_path.to_string(),
        }
    }

    /// Store a thought signature for a tool_use_id (synchronous — no spawn)
    pub async fn store_tool_signature(&self, tool_use_id: String, signature: serde_json::Value) {
        if !signature.is_null() && !tool_use_id.is_empty() {
            self.tool_signatures.write().await.insert(tool_use_id.clone(), signature);
            self.persist_to_file().await;
        }
    }

    /// Store a thought signature for a text content hash
    pub async fn store_text_signature(&self, text_hash: String, signature: serde_json::Value) {
        if !signature.is_null() && !text_hash.is_empty() {
            self.text_signatures.write().await.insert(text_hash.clone(), signature);
            self.persist_to_file().await;
        }
    }

    /// Retrieve a thought signature for a tool_use_id (peek, does not remove)
    pub async fn get_tool_signature(&self, tool_use_id: &str) -> Option<serde_json::Value> {
        self.tool_signatures.read().await.get(tool_use_id).cloned()
    }

    /// Retrieve a thought signature for a text content hash (peek, does not remove)
    pub async fn get_text_signature(&self, text_hash: &str) -> Option<serde_json::Value> {
        self.text_signatures.read().await.get(text_hash).cloned()
    }

    /// Retrieve and remove a thought signature for a tool_use_id
    pub async fn take_tool_signature(&self, tool_use_id: &str) -> Option<serde_json::Value> {
        let val = self.tool_signatures.write().await.remove(tool_use_id);
        if val.is_some() {
            self.persist_to_file().await;
        }
        val
    }

    /// Simple hash for text content — used as key for text part signatures
    pub fn hash_text(text: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Load signatures from the persistence file
    fn load_from_file(path: &str) -> (HashMap<String, serde_json::Value>, HashMap<String, serde_json::Value>) {
        let empty = (HashMap::new(), HashMap::new());
        match std::fs::read_to_string(path) {
            Ok(content) => {
                match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(data) => {
                        let tool_sigs: HashMap<String, serde_json::Value> = data.get("tool_signatures")
                            .and_then(|v| serde_json::from_value(v.clone()).ok())
                            .unwrap_or_default();
                        let text_sigs: HashMap<String, serde_json::Value> = data.get("text_signatures")
                            .and_then(|v| serde_json::from_value(v.clone()).ok())
                            .unwrap_or_default();
                        tracing::info!("Loaded {} tool signatures and {} text signatures from cache file", 
                            tool_sigs.len(), text_sigs.len());
                        (tool_sigs, text_sigs)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse signature cache file: {e}");
                        empty
                    }
                }
            }
            Err(_) => {
                tracing::info!("No signature cache file found, starting fresh");
                empty
            }
        }
    }

    /// Persist current signatures to file
    async fn persist_to_file(&self) {
        let tool_sigs = self.tool_signatures.read().await;
        let text_sigs = self.text_signatures.read().await;

        let data = serde_json::json!({
            "tool_signatures": *tool_sigs,
            "text_signatures": *text_sigs,
        });

        let json_str = match serde_json::to_string_pretty(&data) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Failed to serialize signature cache: {e}");
                return;
            }
        };

        // Write to temp file first, then rename (atomic write)
        let tmp_path = format!("{}.tmp", self.persist_path);
        if let Err(e) = std::fs::write(&tmp_path, &json_str) {
            tracing::error!("Failed to write signature cache tmp file: {e}");
            return;
        }
        if let Err(e) = std::fs::rename(&tmp_path, &self.persist_path) {
            tracing::error!("Failed to rename signature cache file: {e}");
        }
    }

    /// Batch store multiple tool signatures at once (for streaming path)
    pub async fn store_tool_signatures_batch(&self, signatures: Vec<(String, serde_json::Value)>) {
        if signatures.is_empty() {
            return;
        }
        let mut tool_sigs = self.tool_signatures.write().await;
        for (tool_id, sig_val) in signatures {
            if !sig_val.is_null() && !tool_id.is_empty() {
                tool_sigs.insert(tool_id, sig_val);
            }
        }
        drop(tool_sigs); // release lock before IO
        self.persist_to_file().await;
    }

    /// Batch store multiple text signatures at once
    pub async fn store_text_signatures_batch(&self, signatures: Vec<(String, serde_json::Value)>) {
        if signatures.is_empty() {
            return;
        }
        let mut text_sigs = self.text_signatures.write().await;
        for (text_hash, sig_val) in signatures {
            if !sig_val.is_null() && !text_hash.is_empty() {
                text_sigs.insert(text_hash, sig_val);
            }
        }
        drop(text_sigs);
        self.persist_to_file().await;
    }
}

/// Runtime model registry — models can be added/removed via admin API without restart.
/// Initialized empty at startup; config models are resolved separately.
pub type DynamicModels = Arc<RwLock<Vec<ModelConfig>>>;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub auth: AuthManager,
    pub client: reqwest::Client,
    pub signatures: SignatureCache,
    pub dynamic_models: DynamicModels,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "config.yaml".to_string());
    let config = Config::load(&config_path).expect("Failed to load config");
    let config = Arc::new(config);

    let sig_cache_path = std::env::var("SIGNATURE_CACHE_PATH")
        .unwrap_or_else(|_| SIGNATURE_CACHE_FILE.to_string());

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
        signatures: SignatureCache::new(&sig_cache_path),
        dynamic_models: Arc::new(RwLock::new(Vec::new())),
    };

    let app = Router::new()
        .route("/health", get(routes::health::health))
        .route("/v1/models", get(routes::health::list_models))
        .route("/v1/messages", post(routes::anthropic::messages))
        .route("/v1/chat/completions", post(routes::openai::chat_completions))
        // Admin API for dynamic model management
        .route("/admin/models", get(routes::admin::list_models))
        .route("/admin/models", post(routes::admin::add_model))
        .route("/admin/models/{name}", delete(routes::admin::delete_model))
        // VEO Video Generation API
        .route("/v1/veo/generate", post(routes::veo::generate))
        .route("/v1/veo/result", post(routes::veo::fetch_result))
        .route("/v1/veo/generate-sync", post(routes::veo::generate_sync))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let port = std::env::var("PORT").unwrap_or_else(|_| "8000".to_string());
    let addr = format!("0.0.0.0:{port}");

    tracing::info!("Vertex AI Proxy (Rust) starting on {addr}");
    tracing::info!("Static models loaded: {}", config.models.len());
    tracing::info!("Signature cache: {}", sig_cache_path);
    tracing::info!("Auto-resolve patterns: gemini-* → Google, veo-* → Google (VEO), claude-* → Anthropic, */* → OpenAPI");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
