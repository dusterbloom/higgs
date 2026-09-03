use std::collections::{HashMap, HashSet};
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use regex::Regex;
use tracing::warn;

use crate::capacity::CapacityRegistry;
use crate::config::{ApiFormat, GenerationDefaults, HiggsConfig};
use crate::state::Engine;

/// How a model name was resolved to its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingMethod {
    /// Direct lookup by model name in local engines (no route matched).
    Direct,
    /// Matched a regex pattern route.
    Pattern,
    /// Selected by the auto-router AI classifier.
    Auto,
    /// Fell through to the default provider.
    Default,
}

/// Outcome of resolving a model name through the routing table.
pub enum ResolvedRoute {
    /// Serve locally via a loaded MLX engine.
    Higgs {
        engine: Arc<Engine>,
        model_name: String,
        generation_defaults: GenerationDefaults,
        routing_method: RoutingMethod,
    },
    /// Forward to a remote provider.
    Remote {
        provider_name: String,
        provider_url: String,
        provider_format: ApiFormat,
        model_rewrite: Option<String>,
        strip_auth: bool,
        api_key: Option<String>,
        stub_count_tokens: bool,
        routing_method: RoutingMethod,
    },
}

/// A named route candidate for auto-routing classification.
#[derive(Clone)]
pub struct RouteCandidate {
    pub name: String,
    pub description: String,
}

// -- Internal types --------------------------------------------------------

#[derive(Clone)]
enum RouteTarget {
    Higgs {
        model_rewrite: Option<String>,
    },
    Remote {
        provider_name: String,
        provider_url: String,
        provider_format: ApiFormat,
        model_rewrite: Option<String>,
        strip_auth: bool,
        api_key: Option<String>,
        stub_count_tokens: bool,
    },
}

struct CompiledRoute {
    pattern: Regex,
    target: RouteTarget,
}

struct AutoRouteEntry {
    name: String,
    target: RouteTarget,
}

#[derive(Clone)]
struct LocalEngineEntry {
    engine: Arc<Engine>,
    generation_defaults: GenerationDefaults,
    capacity_ready: bool,
}

/// Routes model names to local engines or remote providers.
///
/// Resolution order:
/// 1. If `model == "auto"`, try auto-routing classification
/// 2. Direct engine lookup by model name
/// 3. Pattern matching (first match wins)
/// 4. Default provider fallback
pub struct Router {
    /// Loaded local engines, mutable at runtime via the load/unload endpoints.
    /// Guarded by a `RwLock`: `resolve`/`list` take a read lock and clone the
    /// `Arc` out, so an in-flight request is never tied to map membership.
    local_engines: RwLock<HashMap<String, LocalEngineEntry>>,
    cache_policy: tokio::sync::Mutex<()>,
    compiled_routes: Vec<CompiledRoute>,
    auto_routes: Vec<AutoRouteEntry>,
    auto_candidates: Vec<RouteCandidate>,
    auto_router_engine: Option<Arc<Engine>>,
    /// Map key bound to the auto-router model, if enabled. Unloading it is
    /// refused because `auto_router_engine` keeps a separate `Arc` alive.
    auto_router_model_name: Option<String>,
    auto_router_force: bool,
    auto_router_timeout_ms: u64,
    default_target: RouteTarget,
}

struct DisabledRouteGuard<'a> {
    router: &'a Router,
    name: String,
    armed: bool,
}

impl DisabledRouteGuard<'_> {
    fn remove(mut self) -> Arc<Engine> {
        self.armed = false;
        self.router
            .remove_engine(&self.name)
            .expect("disabled route entry exists")
    }

    fn publish(mut self) {
        self.armed = false;
    }
}

impl Drop for DisabledRouteGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.router.remove_engine(&self.name);
        }
    }
}

impl Router {
    /// Build a router from the unified config and loaded local engines.
    pub fn from_config(
        config: &HiggsConfig,
        engines: HashMap<String, Arc<Engine>>,
    ) -> Result<Self, String> {
        let mut compiled_routes = Vec::new();
        let mut auto_routes = Vec::new();
        let mut auto_candidates = Vec::new();
        let mut seen_names = HashSet::new();
        let mut local_generation_defaults = HashMap::new();

        for model in &config.models {
            if let Some(name) = &model.name {
                local_generation_defaults.insert(name.clone(), model.generation_defaults.clone());
            }
        }

        for route in &config.routes {
            if route.pattern.is_none() && route.description.is_none() {
                return Err(format!(
                    "route for provider '{}' has neither pattern nor description",
                    route.provider
                ));
            }

            if route.description.is_some() && route.name.is_none() {
                return Err(format!(
                    "route for provider '{}' has description but no name",
                    route.provider
                ));
            }

            let target = build_route_target(&route.provider, route.model.clone(), config)?;

            if let Some(ref pattern_str) = route.pattern {
                let pattern = Regex::new(pattern_str)
                    .map_err(|e| format!("invalid regex '{pattern_str}': {e}"))?;
                compiled_routes.push(CompiledRoute {
                    pattern,
                    target: target.clone(),
                });
            }

            if let (Some(name), Some(description)) = (&route.name, &route.description) {
                if !seen_names.insert(name.clone()) {
                    return Err(format!("duplicate route name '{name}'"));
                }
                auto_routes.push(AutoRouteEntry {
                    name: name.clone(),
                    target,
                });
                auto_candidates.push(RouteCandidate {
                    name: name.clone(),
                    description: description.clone(),
                });
            }
        }

        let local_engines = engines
            .into_iter()
            .map(|(name, engine)| {
                let generation_defaults =
                    local_generation_defaults.remove(&name).unwrap_or_default();
                (
                    name,
                    LocalEngineEntry {
                        engine,
                        generation_defaults,
                        capacity_ready: true,
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        let (auto_router_engine, auto_router_model_name) = if config.auto_router.enabled {
            if config.auto_router.model.is_empty() {
                return Err("auto_router.enabled is true but model is empty".to_owned());
            }
            if auto_candidates.is_empty() {
                warn!("auto_router is enabled but no routes have descriptions");
            }
            let auto_model = &config.auto_router.model;
            let engine = local_engines
                .get(auto_model)
                .map(|entry| Arc::clone(&entry.engine))
                .ok_or_else(|| {
                    format!("auto_router model '{auto_model}' not found among loaded models")
                })?;
            (Some(engine), Some(auto_model.clone()))
        } else {
            (None, None)
        };

        let default_target = build_route_target(&config.default.provider, None, config)?;

        Ok(Self {
            local_engines: RwLock::new(local_engines),
            cache_policy: tokio::sync::Mutex::new(()),
            compiled_routes,
            auto_routes,
            auto_candidates,
            auto_router_engine,
            auto_router_model_name,
            auto_router_force: config.auto_router.force,
            auto_router_timeout_ms: config.auto_router.timeout_ms,
            default_target,
        })
    }

    /// Resolve a model name to a route.
    ///
    /// Pass `messages` for auto-routing support. Used when `model == "auto"` or
    /// when `force` mode is enabled.
    pub async fn resolve(
        &self,
        model: &str,
        messages: Option<&[serde_json::Value]>,
    ) -> Result<ResolvedRoute, String> {
        if self.auto_router_force || model == "auto" {
            if let Some(resolved) = self.try_auto_route(messages).await {
                return Ok(resolved);
            }
            if model == "auto" {
                return self.resolve_target(&self.default_target, model, RoutingMethod::Default);
            }
            // force mode: auto-routing returned nothing, fall through to normal resolution
        }

        // Direct engine lookup. Clone the Arc out and drop the read guard
        // immediately -- the in-flight request owns the engine independent of
        // map membership, so a concurrent unload can never free it mid-request.
        let direct = self
            .engines_read()
            .get(model)
            .filter(|entry| entry.capacity_ready)
            .cloned();
        if let Some(entry) = direct {
            return Ok(ResolvedRoute::Higgs {
                engine: entry.engine,
                model_name: model.to_owned(),
                generation_defaults: entry.generation_defaults,
                routing_method: RoutingMethod::Direct,
            });
        }

        // Pattern matching (first match wins)
        for route in &self.compiled_routes {
            if route.pattern.is_match(model) {
                return self.resolve_target(&route.target, model, RoutingMethod::Pattern);
            }
        }

        // Default fallback
        self.resolve_target(&self.default_target, model, RoutingMethod::Default)
    }

    /// Sorted names of all loaded local engines (snapshot under a read lock).
    pub fn local_model_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .engines_read()
            .iter()
            .filter(|(_, entry)| entry.capacity_ready)
            .map(|(name, _)| name.clone())
            .collect();
        names.sort_unstable();
        names
    }

    /// Sorted `(name, is_vlm)` for all loaded local engines (snapshot under a
    /// read lock). Used by `GET /v1/models` to advertise image-input support.
    pub fn local_models_with_vlm(&self) -> Vec<(String, bool)> {
        let mut models: Vec<(String, bool)> = self
            .engines_read()
            .iter()
            .filter(|(_, entry)| entry.capacity_ready)
            .map(|(name, entry)| (name.clone(), entry.engine.is_vlm()))
            .collect();
        models.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        models
    }

    /// Whether a local engine is currently registered under `name`.
    pub fn contains_engine(&self, name: &str) -> bool {
        self.engines_read()
            .get(name)
            .is_some_and(|entry| entry.capacity_ready)
    }

    /// Register a freshly-loaded engine. Returns `Err(name)` if the name is
    /// already taken (checked under the write lock, so it is race-free against
    /// concurrent loads).
    pub fn insert_engine(&self, name: String, engine: Arc<Engine>) -> Result<(), String> {
        self.insert_engine_with_defaults(name, engine, GenerationDefaults::default())
            .map_err(|(name, _engine)| name)
    }

    /// Register a freshly-loaded engine with per-model request defaults.
    pub fn insert_engine_with_defaults(
        &self,
        name: String,
        engine: Arc<Engine>,
        generation_defaults: GenerationDefaults,
    ) -> Result<(), (String, Arc<Engine>)> {
        let mut engines = self.engines_write();
        if engines.contains_key(&name) {
            return Err((name, engine));
        }
        engines.insert(
            name,
            LocalEngineEntry {
                engine,
                generation_defaults,
                capacity_ready: true,
            },
        );
        Ok(())
    }

    /// Remove an engine from the routing table, returning it if present. The
    /// caller is responsible for dropping it once no request still holds a clone.
    pub fn remove_engine(&self, name: &str) -> Option<Arc<Engine>> {
        self.engines_write().remove(name).map(|entry| entry.engine)
    }

    /// Map key bound to the auto-router model, if the auto-router is enabled.
    pub fn auto_router_model_name(&self) -> Option<&str> {
        self.auto_router_model_name.as_deref()
    }

    /// Snapshot of all local engines for cache aggregation and session reset.
    pub fn local_engines(&self) -> Vec<Arc<Engine>> {
        self.engines_read()
            .values()
            .map(|entry| Arc::clone(&entry.engine))
            .collect()
    }

    /// Apply one coherent registry allocation snapshot without holding the
    /// router lock while a batch worker acknowledges eviction.
    pub async fn apply_capacity_cache_allocations(
        &self,
        capacity: &CapacityRegistry,
    ) -> Result<(), String> {
        let _serialized = self.cache_policy.lock().await;
        loop {
            let plan = capacity.cache_allocation_plan();
            self.apply_cache_plan(&plan, None).await?;
            if capacity.publish_cache_allocation_revision(plan.revision) {
                return Ok(());
            }
        }
    }

    /// Atomically apply one current cache policy to the loaded set and a
    /// provisional engine, then expose that engine before a newer publisher
    /// can overtake it.
    pub async fn insert_engine_with_capacity(
        &self,
        name: String,
        engine: Arc<Engine>,
        generation_defaults: GenerationDefaults,
        capacity: &CapacityRegistry,
    ) -> Result<(), (String, Arc<Engine>, String)> {
        let _serialized = self.cache_policy.lock().await;
        let mut inserted = false;
        let mut disabled_route: Option<DisabledRouteGuard<'_>> = None;
        let mut pending_engine = Some(engine);
        let mut pending_defaults = Some(generation_defaults);
        loop {
            let plan = capacity.cache_allocation_plan();
            let provisional = pending_engine.as_ref().map(|engine| (&*name, engine));
            if let Err(error) = self.apply_cache_plan(&plan, provisional).await {
                let engine = if inserted {
                    disabled_route
                        .take()
                        .expect("disabled route guard exists")
                        .remove()
                } else {
                    pending_engine.take().expect("provisional engine exists")
                };
                return Err((name, engine, error));
            }
            if !capacity.publish_cache_allocation_revision(plan.revision) {
                continue;
            }
            if !inserted {
                {
                    let mut engines = self.engines_write();
                    if engines.contains_key(&name) {
                        return Err((
                            name,
                            pending_engine.take().expect("provisional engine exists"),
                            "model name is already loaded".to_owned(),
                        ));
                    }
                    engines.insert(
                        name.clone(),
                        LocalEngineEntry {
                            engine: pending_engine.take().expect("provisional engine exists"),
                            generation_defaults: pending_defaults
                                .take()
                                .expect("provisional defaults exist"),
                            capacity_ready: false,
                        },
                    );
                }
                inserted = true;
                disabled_route = Some(DisabledRouteGuard {
                    router: self,
                    name: name.clone(),
                    armed: true,
                });
                #[cfg(test)]
                tokio::task::yield_now().await;
            }
            if capacity.publish_route_if_current(plan.revision, || {
                let mut engines = self.engines_write();
                let entry = engines
                    .get_mut(&name)
                    .expect("capacity publication retains its disabled engine entry");
                entry.capacity_ready = true;
            }) {
                disabled_route
                    .take()
                    .expect("disabled route guard exists")
                    .publish();
                return Ok(());
            }
        }
    }

    async fn apply_cache_plan(
        &self,
        plan: &crate::capacity::CacheAllocationPlan,
        provisional: Option<(&str, &Arc<Engine>)>,
    ) -> Result<(), String> {
        let engines = {
            let loaded = self.engines_read();
            plan.allocations
                .iter()
                .filter_map(|(name, retained, prefix)| {
                    loaded
                        .get(name)
                        .map(|entry| (name.clone(), Arc::clone(&entry.engine), *retained, *prefix))
                })
                .collect::<Vec<_>>()
        };
        for (name, engine, retained, prefix) in engines {
            engine
                .apply_capacity_cache_limits(plan.revision, retained, prefix, plan.pressure)
                .await
                .map_err(|error| {
                    format!("failed to apply cache allocation for '{name}': {error}")
                })?;
        }
        if let Some((name, engine)) = provisional
            && let Some((_, retained, prefix)) =
                plan.allocations.iter().find(|(model, _, _)| model == name)
        {
            engine
                .apply_capacity_cache_limits(plan.revision, *retained, *prefix, plan.pressure)
                .await
                .map_err(|error| {
                    format!("failed to apply cache allocation for '{name}': {error}")
                })?;
        }
        Ok(())
    }

    // -- Private helpers ---------------------------------------------------

    fn engines_read(&self) -> RwLockReadGuard<'_, HashMap<String, LocalEngineEntry>> {
        self.local_engines
            .read()
            .unwrap_or_else(PoisonError::into_inner)
    }

    fn engines_write(&self) -> RwLockWriteGuard<'_, HashMap<String, LocalEngineEntry>> {
        self.local_engines
            .write()
            .unwrap_or_else(PoisonError::into_inner)
    }

    async fn try_auto_route(
        &self,
        messages: Option<&[serde_json::Value]>,
    ) -> Option<ResolvedRoute> {
        let auto_engine = self.auto_router_engine.as_ref()?;
        let msg_slice = messages?;
        if self.auto_candidates.is_empty() || msg_slice.is_empty() {
            return None;
        }

        let engine_clone = Arc::clone(auto_engine);
        let candidates = self.auto_candidates.clone();
        let messages_owned = msg_slice.to_vec();

        let timeout = std::time::Duration::from_millis(self.auto_router_timeout_ms);
        let timeout_result = tokio::time::timeout(
            timeout,
            tokio::task::spawn_blocking(move || {
                crate::auto_router::classify_local(&engine_clone, &candidates, &messages_owned)
            }),
        )
        .await;
        let name = if let Ok(join_result) = timeout_result {
            join_result.ok()??
        } else {
            warn!("auto-router classification timed out after {timeout:?}");
            return None;
        };

        let entry = self.auto_routes.iter().find(|r| r.name == name)?;
        self.resolve_target(&entry.target, "auto", RoutingMethod::Auto)
            .ok()
    }

    fn resolve_target(
        &self,
        target: &RouteTarget,
        model: &str,
        method: RoutingMethod,
    ) -> Result<ResolvedRoute, String> {
        match target {
            RouteTarget::Higgs { model_rewrite } => {
                let lookup_name = model_rewrite.as_deref().unwrap_or(model);
                let engines = self.engines_read();
                let (entry, resolved_name) = if let Some(entry) = engines.get(lookup_name) {
                    (entry.clone(), lookup_name.to_owned())
                } else if model == "auto" {
                    // "auto" is a virtual model name; pick any loaded engine
                    let (name, entry) = engines
                        .iter()
                        .next()
                        .ok_or_else(|| "no local models loaded for default route".to_owned())?;
                    (entry.clone(), name.clone())
                } else {
                    return Err(format!(
                        "model '{lookup_name}' not found among loaded local models"
                    ));
                };
                Ok(ResolvedRoute::Higgs {
                    engine: entry.engine,
                    model_name: resolved_name,
                    generation_defaults: entry.generation_defaults,
                    routing_method: method,
                })
            }
            RouteTarget::Remote {
                provider_name,
                provider_url,
                provider_format,
                model_rewrite,
                strip_auth,
                api_key,
                stub_count_tokens,
            } => {
                if model == "auto" && model_rewrite.is_none() {
                    return Err(
                        "cannot forward virtual model name \"auto\" to remote provider; \
                         configure a model rewrite on the default route or add auto-router routes"
                            .to_owned(),
                    );
                }
                Ok(ResolvedRoute::Remote {
                    provider_name: provider_name.clone(),
                    provider_url: provider_url.clone(),
                    provider_format: *provider_format,
                    model_rewrite: model_rewrite.clone(),
                    strip_auth: *strip_auth,
                    api_key: api_key.clone(),
                    stub_count_tokens: *stub_count_tokens,
                    routing_method: method,
                })
            }
        }
    }
}

fn build_route_target(
    provider_name: &str,
    model_rewrite: Option<String>,
    config: &HiggsConfig,
) -> Result<RouteTarget, String> {
    if provider_name == "higgs" {
        return Ok(RouteTarget::Higgs { model_rewrite });
    }
    let provider = config
        .providers
        .get(provider_name)
        .ok_or_else(|| format!("route provider '{provider_name}' not found in providers"))?;
    Ok(RouteTarget::Remote {
        provider_name: provider_name.to_owned(),
        provider_url: provider.url.clone(),
        provider_format: provider.format,
        model_rewrite,
        strip_auth: provider.strip_auth,
        api_key: provider.api_key.clone(),
        stub_count_tokens: provider.stub_count_tokens,
    })
}

#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::shadow_unrelated
)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_config_file;

    fn config_from_toml(toml: &str) -> HiggsConfig {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, toml).unwrap();
        load_config_file(&path, None).unwrap()
    }

    fn router_from_toml(toml: &str) -> Router {
        let config = config_from_toml(toml);
        Router::from_config(&config, HashMap::new()).unwrap()
    }

    fn production_toml() -> &'static str {
        r#"
        [provider.anthropic]
        url = "https://api.anthropic.com"
        format = "anthropic"

        [provider.ollama]
        url = "http://localhost:11434"
        strip_auth = true
        api_key = "ollama"
        stub_count_tokens = true

        [[routes]]
        pattern = "opus"
        provider = "anthropic"

        [[routes]]
        pattern = "sonnet|haiku"
        provider = "ollama"
        model = "qwen3-coder:30b"

        [default]
        provider = "anthropic"
        "#
    }

    #[tokio::test]
    async fn remote_pattern_resolves_to_anthropic() {
        let router = router_from_toml(production_toml());
        let route = router.resolve("claude-opus-4-6", None).await.unwrap();
        match route {
            ResolvedRoute::Remote {
                provider_name,
                provider_url,
                provider_format,
                model_rewrite,
                strip_auth,
                api_key,
                stub_count_tokens,
                routing_method,
            } => {
                assert_eq!(provider_name, "anthropic");
                assert_eq!(provider_url, "https://api.anthropic.com");
                assert_eq!(provider_format, ApiFormat::Anthropic);
                assert_eq!(model_rewrite, None);
                assert!(!strip_auth);
                assert_eq!(api_key, None);
                assert!(!stub_count_tokens);
                assert_eq!(routing_method, RoutingMethod::Pattern);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[tokio::test]
    async fn remote_pattern_with_model_rewrite() {
        let router = router_from_toml(production_toml());
        let route = router
            .resolve("claude-sonnet-4-5-20250929", None)
            .await
            .unwrap();
        match route {
            ResolvedRoute::Remote {
                provider_url,
                model_rewrite,
                strip_auth,
                api_key,
                stub_count_tokens,
                ..
            } => {
                assert_eq!(provider_url, "http://localhost:11434");
                assert_eq!(model_rewrite.as_deref(), Some("qwen3-coder:30b"));
                assert!(strip_auth);
                assert_eq!(api_key.as_deref(), Some("ollama"));
                assert!(stub_count_tokens);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[tokio::test]
    async fn unmatched_model_falls_to_default() {
        let router = router_from_toml(production_toml());
        let route = router.resolve("some-unknown-model", None).await.unwrap();
        match route {
            ResolvedRoute::Remote {
                provider_name,
                provider_url,
                routing_method,
                ..
            } => {
                assert_eq!(provider_name, "anthropic");
                assert_eq!(provider_url, "https://api.anthropic.com");
                assert_eq!(routing_method, RoutingMethod::Default);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[tokio::test]
    async fn empty_model_falls_to_default() {
        let router = router_from_toml(production_toml());
        let route = router.resolve("", None).await.unwrap();
        match route {
            ResolvedRoute::Remote {
                provider_url,
                routing_method,
                ..
            } => {
                assert_eq!(provider_url, "https://api.anthropic.com");
                assert_eq!(routing_method, RoutingMethod::Default);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[tokio::test]
    async fn first_matching_route_wins() {
        let router = router_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [provider.b]
            url = "http://b"
            [[routes]]
            pattern = "opus"
            provider = "a"
            [[routes]]
            pattern = "opus"
            provider = "b"
            [default]
            provider = "a"
            "#,
        );
        let route = router.resolve("opus", None).await.unwrap();
        match route {
            ResolvedRoute::Remote { provider_url, .. } => {
                assert_eq!(provider_url, "http://a");
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[test]
    fn invalid_regex_returns_error() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            pattern = "[invalid"
            provider = "a"
            [default]
            provider = "a"
            "#,
        );
        let err = Router::from_config(&config, HashMap::new())
            .err()
            .expect("should fail");
        assert!(err.contains("invalid regex"), "got: {err}");
    }

    #[test]
    fn missing_route_provider_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            pattern = "test"
            provider = "nonexistent"
            [default]
            provider = "a"
            "#,
        )
        .unwrap();
        // Config validation catches this before Router::from_config
        let result = load_config_file(&path, None);
        assert!(result.is_err());
    }

    #[test]
    fn missing_default_provider_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            pattern = "x"
            provider = "a"
            [default]
            provider = "nonexistent"
            "#,
        )
        .unwrap();
        let result = load_config_file(&path, None);
        assert!(result.is_err());
    }

    #[test]
    fn description_without_name_errors() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            description = "some task"
            provider = "a"
            [default]
            provider = "a"
            "#,
        );
        let err = Router::from_config(&config, HashMap::new())
            .err()
            .expect("should fail");
        assert!(err.contains("description but no name"), "got: {err}");
    }

    #[test]
    fn route_without_pattern_or_description_errors() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            provider = "a"
            [default]
            provider = "a"
            "#,
        );
        let err = Router::from_config(&config, HashMap::new())
            .err()
            .expect("should fail");
        assert!(
            err.contains("neither pattern nor description"),
            "got: {err}"
        );
    }

    #[test]
    fn duplicate_route_names_error() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            name = "coding"
            description = "code tasks"
            pattern = "opus"
            provider = "a"
            [[routes]]
            name = "coding"
            description = "other code tasks"
            pattern = "sonnet"
            provider = "a"
            [default]
            provider = "a"
            "#,
        );
        let err = Router::from_config(&config, HashMap::new())
            .err()
            .expect("should fail");
        assert!(err.contains("duplicate route name"), "got: {err}");
    }

    #[test]
    fn auto_candidates_built_from_descriptions() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [provider.b]
            url = "http://b"
            [[routes]]
            name = "coding"
            description = "code tasks"
            pattern = "opus"
            provider = "a"
            [[routes]]
            pattern = "sonnet"
            provider = "b"
            [default]
            provider = "a"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        assert_eq!(router.auto_candidates.len(), 1);
        assert_eq!(router.auto_candidates[0].name, "coding");
        assert_eq!(router.auto_routes.len(), 1);
        assert_eq!(router.compiled_routes.len(), 2);
    }

    #[test]
    fn description_only_route_not_in_pattern_routes() {
        let config = config_from_toml(
            r#"
            [provider.a]
            url = "http://a"
            [[routes]]
            name = "coding"
            description = "code tasks"
            provider = "a"
            [default]
            provider = "a"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        assert_eq!(router.compiled_routes.len(), 0);
        assert_eq!(router.auto_candidates.len(), 1);
    }

    #[tokio::test]
    async fn higgs_route_errors_when_model_not_found() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            [[routes]]
            pattern = "Llama.*"
            provider = "higgs"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        let result = router.resolve("Llama-3.2-1B", None).await;
        match result {
            Err(e) => assert!(
                e.contains("not found among loaded local models"),
                "got: {e}"
            ),
            Ok(_) => panic!("expected error for missing local model"),
        }
    }

    #[tokio::test]
    async fn exact_local_model_beats_regex_route() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "mlx-community/Llama-3.2-1B-Instruct-4bit"
            name = "llama"

            [provider.remote]
            url = "http://127.0.0.1:9"

            [[routes]]
            pattern = "llama"
            provider = "remote"

            [default]
            provider = "remote"
            "#,
        );
        let mut engines = HashMap::new();
        engines.insert(
            "llama".to_owned(),
            Arc::new(crate::state::Engine::test_stub("llama")),
        );

        let router = Router::from_config(&config, engines).unwrap();
        let result = router.resolve("llama", None).await.unwrap();

        match result {
            ResolvedRoute::Higgs {
                model_name,
                routing_method,
                ..
            } => {
                assert_eq!(model_name, "llama");
                assert_eq!(routing_method, RoutingMethod::Direct);
            }
            ResolvedRoute::Remote { .. } => panic!("expected exact local model to win"),
        }
    }

    #[tokio::test]
    async fn local_model_resolution_carries_generation_defaults() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "prism-ml/Ternary-Bonsai-27B-mlx-2bit"
            name = "bonsai-27b-q2"

            [models.generation_defaults]
            max_tokens = 4096
            temperature = 0.0
            top_p = 1.0
            speculation = "auto"
            enable_thinking = false
            "#,
        );
        let mut engines = HashMap::new();
        engines.insert(
            "bonsai-27b-q2".to_owned(),
            Arc::new(crate::state::Engine::test_stub("bonsai-27b-q2")),
        );

        let router = Router::from_config(&config, engines).unwrap();
        let result = router.resolve("bonsai-27b-q2", None).await.unwrap();

        match result {
            ResolvedRoute::Higgs {
                generation_defaults,
                ..
            } => {
                assert_eq!(generation_defaults.max_tokens, Some(4096));
                assert_eq!(generation_defaults.temperature, Some(0.0));
                assert_eq!(generation_defaults.top_p, Some(1.0));
                assert_eq!(generation_defaults.speculation.as_deref(), Some("auto"));
                assert_eq!(generation_defaults.enable_thinking, Some(false));
            }
            ResolvedRoute::Remote { .. } => panic!("expected Higgs route"),
        }
    }

    #[tokio::test]
    async fn higgs_default_errors_when_model_not_found() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            "#,
        );
        // No routes, default is "higgs", no engines loaded
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        let result = router.resolve("nonexistent-model", None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn resolved_route_includes_provider_name() {
        let router = router_from_toml(production_toml());

        let route = router.resolve("claude-opus-4-6", None).await.unwrap();
        match route {
            ResolvedRoute::Remote { provider_name, .. } => {
                assert_eq!(provider_name, "anthropic");
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote"),
        }

        let route = router
            .resolve("claude-sonnet-4-5-20250929", None)
            .await
            .unwrap();
        match route {
            ResolvedRoute::Remote { provider_name, .. } => {
                assert_eq!(provider_name, "ollama");
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote"),
        }
    }

    #[test]
    fn no_routes_no_providers_uses_higgs_default() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        // default_target should be Higgs
        match &router.default_target {
            RouteTarget::Higgs { model_rewrite } => {
                assert!(model_rewrite.is_none());
            }
            RouteTarget::Remote { .. } => panic!("expected Higgs default"),
        }
    }

    #[test]
    fn local_model_names_lists_engines() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        assert!(router.local_model_names().is_empty());

        let mut engines = HashMap::new();
        engines.insert(
            "b".to_owned(),
            Arc::new(crate::state::Engine::test_stub("b")),
        );
        engines.insert(
            "a".to_owned(),
            Arc::new(crate::state::Engine::test_stub("a")),
        );
        let router = Router::from_config(&config, engines).unwrap();
        assert_eq!(router.local_model_names(), vec!["a", "b"]);
    }

    #[test]
    fn insert_remove_and_contains_engine() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();

        assert!(!router.contains_engine("x"));
        router
            .insert_engine(
                "x".to_owned(),
                Arc::new(crate::state::Engine::test_stub("x")),
            )
            .unwrap();
        assert!(router.contains_engine("x"));

        // Duplicate insert is rejected with the conflicting name.
        let dup = router.insert_engine(
            "x".to_owned(),
            Arc::new(crate::state::Engine::test_stub("x")),
        );
        assert_eq!(dup, Err("x".to_owned()));

        let removed = router.remove_engine("x");
        assert!(removed.is_some());
        assert!(!router.contains_engine("x"));
        assert!(router.remove_engine("x").is_none());
    }

    #[tokio::test]
    async fn inserted_engine_and_defaults_publish_as_one_entry() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "some/model"
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        let defaults = GenerationDefaults {
            max_tokens: Some(73),
            ..GenerationDefaults::default()
        };
        router
            .insert_engine_with_defaults(
                "atomic".to_owned(),
                Arc::new(crate::state::Engine::test_stub("atomic")),
                defaults,
            )
            .map_err(|(name, _engine)| name)
            .unwrap();

        match router.resolve("atomic", None).await.unwrap() {
            ResolvedRoute::Higgs {
                generation_defaults,
                ..
            } => assert_eq!(generation_defaults.max_tokens, Some(73)),
            ResolvedRoute::Remote { .. } => panic!("expected local route"),
        }
    }

    #[tokio::test]
    async fn auto_model_with_higgs_default_picks_first_engine() {
        // When model="auto", auto-router returns None, default is higgs
        // with no model_rewrite, it should pick the first loaded engine
        // instead of trying to look up "auto" as a model name.
        let config = config_from_toml(
            r#"
            [[models]]
            path = "test/model-a"
            "#,
        );
        let engine = crate::state::Engine::test_stub("test/model-a");
        let mut engines = HashMap::new();
        engines.insert("test/model-a".to_owned(), Arc::new(engine));

        let router = Router::from_config(&config, engines).unwrap();
        // No auto-router configured, so "auto" falls through to default
        let result = router.resolve("auto", None).await;
        match result {
            Ok(ResolvedRoute::Higgs {
                model_name,
                routing_method,
                ..
            }) => {
                assert_eq!(model_name, "test/model-a");
                assert_eq!(routing_method, RoutingMethod::Default);
            }
            Ok(ResolvedRoute::Remote { .. }) => panic!("expected Higgs route"),
            Err(e) => panic!("should resolve to first engine, got error: {e}"),
        }
    }

    #[test]
    fn auto_router_model_resolved_by_name() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "/Users/someone/models/Arch-Router-1.5B-4bit"
            name = "router"

            [auto_router]
            enabled = true
            model = "router"
            "#,
        );
        let engine = crate::state::Engine::test_stub("router");
        let mut engines = HashMap::new();
        engines.insert("router".to_owned(), Arc::new(engine));

        let router = Router::from_config(&config, engines);
        assert!(router.is_ok(), "should resolve auto_router model by name");
    }

    #[test]
    fn auto_router_model_resolved_by_path() {
        let config = config_from_toml(
            r#"
            [[models]]
            path = "/Users/someone/models/Arch-Router-1.5B-4bit"
            name = "router"

            [auto_router]
            enabled = true
            model = "/Users/someone/models/Arch-Router-1.5B-4bit"
            "#,
        );
        let engine = crate::state::Engine::test_stub("router");
        let mut engines = HashMap::new();
        engines.insert("router".to_owned(), Arc::new(engine));

        let router = Router::from_config(&config, engines);
        assert!(router.is_ok(), "should resolve auto_router model by path");
    }

    #[test]
    fn auto_router_model_resolved_by_path_without_name() {
        // ensure_auto_router_model injects the model entry with name=None,
        // so the engine key is derived from engine.model_name() (the basename).
        let config = config_from_toml(
            r#"
            [[models]]
            path = "/Users/someone/models/Arch-Router-1.5B-4bit"

            [auto_router]
            enabled = true
            model = "/Users/someone/models/Arch-Router-1.5B-4bit"
            "#,
        );
        let engine = crate::state::Engine::test_stub("Arch-Router-1.5B-4bit");
        let mut engines = HashMap::new();
        engines.insert("Arch-Router-1.5B-4bit".to_owned(), Arc::new(engine));

        let router = Router::from_config(&config, engines);
        assert!(
            router.is_ok(),
            "should resolve auto_router model by path when model has no name"
        );
    }

    #[tokio::test]
    async fn auto_model_with_remote_default_rejects_without_rewrite() {
        // model="auto" falling through to a remote default with no model_rewrite
        // should error -- "auto" is not a real model name for upstream providers.
        let router = router_from_toml(production_toml());
        let result = router.resolve("auto", None).await;
        match result {
            Err(e) => assert!(e.contains("cannot forward virtual model name"), "got: {e}"),
            Ok(_) => panic!("expected error for auto with remote default"),
        }
    }

    #[tokio::test]
    async fn force_mode_falls_through_without_engine() {
        // force=true, no auto-router engine loaded, named model should
        // fall through to normal resolution (pattern/direct/default).
        let config = config_from_toml(
            r#"
            [provider.anthropic]
            url = "https://api.anthropic.com"
            format = "anthropic"

            [[routes]]
            pattern = "opus"
            provider = "anthropic"

            [default]
            provider = "anthropic"

            [auto_router]
            enabled = false
            force = true
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        let result = router.resolve("claude-opus-4-6", None).await.unwrap();
        match result {
            ResolvedRoute::Remote {
                provider_name,
                routing_method,
                ..
            } => {
                assert_eq!(provider_name, "anthropic");
                assert_eq!(routing_method, RoutingMethod::Pattern);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }

    #[tokio::test]
    async fn force_mode_unmatched_falls_to_default() {
        // force=true, no auto-router engine, unmatched model falls to default
        let config = config_from_toml(
            r#"
            [provider.anthropic]
            url = "https://api.anthropic.com"
            format = "anthropic"

            [default]
            provider = "anthropic"

            [auto_router]
            enabled = false
            force = true
            "#,
        );
        let router = Router::from_config(&config, HashMap::new()).unwrap();
        let result = router.resolve("some-unknown-model", None).await.unwrap();
        match result {
            ResolvedRoute::Remote {
                provider_name,
                routing_method,
                ..
            } => {
                assert_eq!(provider_name, "anthropic");
                assert_eq!(routing_method, RoutingMethod::Default);
            }
            ResolvedRoute::Higgs { .. } => panic!("expected Remote route"),
        }
    }
}
