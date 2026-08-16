use std::path::{Path, PathBuf};

/// Resolve a model specifier to a concrete directory path.
///
/// 1. If `path` is an existing directory, returns it directly.
/// 2. If it looks like `org/name`, resolves from the `HuggingFace` cache
///    (`~/.cache/huggingface/hub/models--org--name/snapshots/<hash>`).
pub fn resolve(path: &str) -> Result<PathBuf, String> {
    let as_path = Path::new(path);

    let expanded = path.strip_prefix("~/").map_or_else(
        || as_path.to_path_buf(),
        |rest| {
            directories::BaseDirs::new()
                .map_or_else(|| as_path.to_path_buf(), |d| d.home_dir().join(rest))
        },
    );

    if expanded.is_dir() {
        return Ok(expanded);
    }

    // Tilde paths are explicit filesystem references, not HF model IDs.
    if path.starts_with("~/") {
        return Err(format!("model directory not found: {}", expanded.display()));
    }

    resolve_with_cache(path, default_hf_cache().as_deref())
}

/// Returns `true` if `s` looks like a `org/name` `HuggingFace` model ID.
pub fn is_hf_model_id(s: &str) -> bool {
    if s.starts_with("~/") || s.starts_with('/') {
        return false;
    }
    matches!(s.split_once('/'), Some((org, name)) if !org.is_empty() && !name.is_empty() && !name.contains('/'))
}

/// Policy gate for caller-supplied runtime-load paths.
///
/// Accepts Hugging Face model ids (which resolve through the local HF cache,
/// not the caller's filesystem) unless the same string names an existing local
/// directory. Local paths are accepted only when they resolve (after `~`
/// expansion and symlink resolution) inside one of `roots`. Returns a reason
/// string on rejection.
#[cfg(test)]
fn runtime_load_path_allowed(path: &str, roots: &[String]) -> Result<(), String> {
    // A syntactically valid `org/name` can also be a relative local
    // directory. `resolve` prefers existing directories, so do not let that
    // local path bypass the filesystem allowlist.
    if is_hf_model_id(path) && !Path::new(path).is_dir() {
        return Ok(());
    }
    canonical_runtime_local_path(path, roots).map(|_| ())
}

/// Resolve and authorize one runtime-load request, returning the exact path
/// that the loader must use.
///
/// HF-shaped inputs are resolved cache-only after the initial local-directory
/// check; they are never passed back through `resolve`, which prefers
/// caller-relative directories.
pub fn resolve_runtime_model(path: &str, roots: &[String]) -> Result<PathBuf, String> {
    resolve_runtime_model_with_cache(path, roots, default_hf_cache().as_deref())
}

fn resolve_runtime_model_with_cache(
    path: &str,
    roots: &[String],
    cache_root: Option<&Path>,
) -> Result<PathBuf, String> {
    if is_hf_model_id(path) && !Path::new(path).is_dir() {
        let (org, name) = path
            .split_once('/')
            .ok_or_else(|| format!("invalid Hugging Face model id '{path}'"))?;
        let snapshot = cache_root
            .ok_or_else(|| format!("Hugging Face cache is not configured for '{path}'"))
            .and_then(|cache| resolve_hf_snapshot(cache, org, name))?;
        return snapshot
            .canonicalize()
            .map_err(|e| format!("cached model '{path}' cannot be resolved: {e}"));
    }

    canonical_runtime_local_path(path, roots)
}

fn canonical_runtime_local_path(path: &str, roots: &[String]) -> Result<PathBuf, String> {
    if roots.is_empty() {
        return Err(format!(
            "path '{path}' is not a Hugging Face model id and local.runtime_model_roots is empty; \
             use an HF model id or configure local.runtime_model_roots"
        ));
    }
    let expanded = expand_user_path(path);
    let canonical = expanded
        .canonicalize()
        .map_err(|e| format!("path '{path}' cannot be resolved: {e}"))?;
    for root in roots {
        let root_expanded = expand_user_path(root);
        let root_canonical = root_expanded.canonicalize().map_err(|e| {
            format!(
                "configured local.runtime_model_roots entry '{}' cannot be resolved: {e}",
                root_expanded.display()
            )
        })?;
        if canonical.starts_with(&root_canonical) {
            return Ok(canonical);
        }
    }
    Err(format!(
        "path '{path}' is outside all configured local.runtime_model_roots"
    ))
}

fn expand_user_path(path: &str) -> PathBuf {
    path.strip_prefix("~/").map_or_else(
        || PathBuf::from(path),
        |rest| {
            directories::BaseDirs::new()
                .map_or_else(|| PathBuf::from(path), |d| d.home_dir().join(rest))
        },
    )
}

/// Testable resolver with explicit cache root.
fn resolve_with_cache(path: &str, cache_root: Option<&Path>) -> Result<PathBuf, String> {
    let as_path = Path::new(path);
    if as_path.is_dir() {
        return Ok(as_path.to_path_buf());
    }

    if is_hf_model_id(path) {
        if let (Some((org, name)), Some(cache)) = (path.split_once('/'), cache_root) {
            return resolve_hf_snapshot(cache, org, name);
        }
    }

    Err(format!(
        "model '{path}' is not an existing directory and was not found in the HuggingFace cache"
    ))
}

/// Read `refs/main` and resolve to the snapshot directory.
/// Only the default revision (`main`) is supported; models downloaded at a
/// specific revision or branch will not be found.
fn resolve_hf_snapshot(cache_root: &Path, org: &str, name: &str) -> Result<PathBuf, String> {
    let model_dir = cache_root.join(format!("models--{org}--{name}"));
    let ref_path = model_dir.join("refs").join("main");
    let hash = std::fs::read_to_string(&ref_path)
        .map_err(|e| format!("could not read HF cache ref for '{org}/{name}': {e}"))?
        .trim()
        .to_owned();

    if hash.is_empty() {
        return Err(format!("empty ref in HF cache for '{org}/{name}'"));
    }

    let snapshot_dir = model_dir.join("snapshots").join(&hash);
    if snapshot_dir.is_dir() {
        Ok(snapshot_dir)
    } else {
        Err(format!(
            "snapshot directory missing for '{org}/{name}' (hash: {hash})"
        ))
    }
}

fn default_hf_cache() -> Option<PathBuf> {
    let env = |key| std::env::var(key).ok();
    hf_cache_from_env(
        env("HF_HUB_CACHE").as_deref(),
        env("HUGGINGFACE_HUB_CACHE").as_deref(),
        env("HF_HOME").as_deref(),
    )
    .or_else(|| {
        directories::BaseDirs::new()
            .map(|d| d.home_dir().join(".cache").join("huggingface").join("hub"))
    })
}

/// Testable env var resolution without reading actual environment.
///
/// Resolution order matches the `HuggingFace` Python SDK:
/// 1. `HF_HUB_CACHE`          (direct cache path)
/// 2. `HUGGINGFACE_HUB_CACHE`  (legacy alias)
/// 3. `HF_HOME` + `/hub`       (home override)
///
/// Empty or whitespace-only values are treated as unset.
fn hf_cache_from_env(
    hub_cache: Option<&str>,
    legacy_cache: Option<&str>,
    hf_home: Option<&str>,
) -> Option<PathBuf> {
    fn non_empty(v: Option<&str>) -> Option<&str> {
        v.filter(|s| !s.trim().is_empty())
    }

    if let Some(cache) = non_empty(hub_cache) {
        return Some(PathBuf::from(cache));
    }
    if let Some(cache) = non_empty(legacy_cache) {
        return Some(PathBuf::from(cache));
    }
    if let Some(home) = non_empty(hf_home) {
        return Some(PathBuf::from(home).join("hub"));
    }
    None
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn create_hf_cache(root: &Path, org: &str, name: &str, hash: &str) -> PathBuf {
        let model_dir = root.join(format!("models--{org}--{name}"));
        let refs_dir = model_dir.join("refs");
        let snapshot_dir = model_dir.join("snapshots").join(hash);
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::create_dir_all(&snapshot_dir).unwrap();
        std::fs::write(refs_dir.join("main"), hash).unwrap();
        snapshot_dir
    }

    #[test]
    fn test_resolve_existing_directory_returns_as_is() {
        let dir = tempfile::tempdir().unwrap();
        let result = resolve_with_cache(dir.path().to_str().unwrap(), None);
        assert_eq!(result.unwrap(), dir.path());
    }

    #[test]
    fn test_resolve_hf_model_id_from_cache() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = create_hf_cache(cache.path(), "mlx-community", "Qwen3-4bit", "abc123");
        let result = resolve_with_cache("mlx-community/Qwen3-4bit", Some(cache.path()));
        assert_eq!(result.unwrap(), snapshot);
    }

    #[test]
    fn test_resolve_hf_model_id_not_in_cache_is_err() {
        let cache = tempfile::tempdir().unwrap();
        let result = resolve_with_cache("no-org/NoModel", Some(cache.path()));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_hf_empty_refs_main_is_err() {
        let cache = tempfile::tempdir().unwrap();
        let model_dir = cache.path().join("models--org--name");
        let refs_dir = model_dir.join("refs");
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::write(refs_dir.join("main"), "  \n").unwrap();
        let result = resolve_with_cache("org/name", Some(cache.path()));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_hf_snapshot_dir_missing_is_err() {
        let cache = tempfile::tempdir().unwrap();
        let model_dir = cache.path().join("models--org--name");
        let refs_dir = model_dir.join("refs");
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::write(refs_dir.join("main"), "deadbeef").unwrap();
        let result = resolve_with_cache("org/name", Some(cache.path()));
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_nonexistent_plain_path_is_err() {
        let result = resolve_with_cache("/nonexistent/path/to/model", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_no_cache_for_hf_id_is_err() {
        let result = resolve_with_cache("org/model", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_tilde_expansion_nonexistent_is_err() {
        let result = resolve("~/nonexistent_model_dir_12345");
        assert!(result.is_err());
    }

    // --- hf_cache_from_env tests ---

    #[test]
    fn test_hf_cache_from_env_hf_hub_cache_takes_priority() {
        let dir = tempfile::tempdir().unwrap();
        let result = hf_cache_from_env(
            Some(dir.path().to_str().unwrap()),
            Some("/legacy"),
            Some("/home"),
        );
        assert_eq!(result, Some(dir.path().to_path_buf()));
    }

    #[test]
    fn test_hf_cache_from_env_legacy_over_hf_home() {
        let dir = tempfile::tempdir().unwrap();
        let result = hf_cache_from_env(None, Some(dir.path().to_str().unwrap()), Some("/home"));
        assert_eq!(result, Some(dir.path().to_path_buf()));
    }

    #[test]
    fn test_hf_cache_from_env_hf_home_appends_hub() {
        let dir = tempfile::tempdir().unwrap();
        let result = hf_cache_from_env(None, None, Some(dir.path().to_str().unwrap()));
        assert_eq!(result, Some(dir.path().join("hub")));
    }

    #[test]
    fn test_hf_cache_from_env_none_when_all_unset() {
        let result = hf_cache_from_env(None, None, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_hf_cache_from_env_empty_string_ignored() {
        let result = hf_cache_from_env(Some(""), Some(""), Some(""));
        assert!(result.is_none());
    }

    #[test]
    fn test_hf_cache_from_env_whitespace_only_ignored() {
        let result = hf_cache_from_env(Some("  "), None, None);
        assert!(result.is_none());
    }

    // --- is_hf_model_id tests ---

    #[test]
    fn test_is_hf_model_id_valid() {
        assert!(is_hf_model_id("org/model"));
        assert!(is_hf_model_id("mlx-community/Qwen3-4bit"));
    }

    #[test]
    fn test_is_hf_model_id_tilde_path_is_false() {
        assert!(!is_hf_model_id("~/models/foo"));
    }

    #[test]
    fn test_is_hf_model_id_absolute_path_is_false() {
        assert!(!is_hf_model_id("/some/absolute/path"));
    }

    #[test]
    fn test_is_hf_model_id_nested_slash_is_false() {
        assert!(!is_hf_model_id("org/name/extra"));
    }

    #[test]
    fn test_is_hf_model_id_no_slash_is_false() {
        assert!(!is_hf_model_id("justname"));
    }

    #[test]
    fn test_is_hf_model_id_empty_org_is_false() {
        assert!(!is_hf_model_id("/model"));
    }

    #[test]
    fn test_is_hf_model_id_empty_name_is_false() {
        assert!(!is_hf_model_id("org/"));
    }

    #[test]
    fn test_runtime_policy_does_not_treat_existing_relative_directory_as_hf_id() {
        assert!(Path::new("src/.").is_dir());
        assert!(runtime_load_path_allowed("src/.", &[]).is_err());
    }

    #[test]
    fn test_runtime_policy_allows_hf_id_without_local_roots() {
        assert!(runtime_load_path_allowed("org/model", &[]).is_ok());
    }

    #[test]
    fn test_runtime_policy_allows_existing_directory_inside_root() {
        assert!(runtime_load_path_allowed("src/.", &["src".to_owned()]).is_ok());
    }

    #[test]
    fn test_runtime_policy_reports_unresolvable_configured_root() {
        let model_root = tempfile::tempdir().unwrap();
        let missing_root = model_root.path().join("missing-root");
        let model = model_root.path().join("model");
        std::fs::create_dir(&model).unwrap();
        let roots = vec![missing_root.to_string_lossy().into_owned()];

        let error = runtime_load_path_allowed(model.to_str().unwrap(), &roots).unwrap_err();
        assert!(error.contains("configured local.runtime_model_roots entry"));
    }

    #[test]
    fn test_runtime_resolver_returns_canonical_hf_snapshot() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = create_hf_cache(cache.path(), "org", "model", "abc123");
        let resolved =
            resolve_runtime_model_with_cache("org/model", &[], Some(cache.path())).unwrap();

        assert_eq!(resolved, snapshot.canonicalize().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn test_runtime_policy_rejects_symlink_escape_from_root() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_model = outside.path().join("model");
        std::fs::create_dir(&outside_model).unwrap();
        std::os::unix::fs::symlink(&outside_model, root.path().join("linked-model")).unwrap();

        let roots = vec![root.path().to_string_lossy().into_owned()];
        assert!(
            runtime_load_path_allowed(root.path().join("linked-model").to_str().unwrap(), &roots)
                .is_err()
        );
    }

    // --- tilde expansion error message test ---

    #[test]
    fn test_resolve_tilde_path_error_mentions_expanded_path() {
        let result = resolve("~/nonexistent_model_dir_12345");
        let err = result.unwrap_err();
        assert!(
            !err.contains("HuggingFace cache"),
            "tilde path error should not mention HF cache, got: {err}"
        );
    }

    #[test]
    fn test_resolve_hf_hash_with_trailing_newline() {
        let cache = tempfile::tempdir().unwrap();
        let snapshot = create_hf_cache(cache.path(), "org", "model", "abc123");
        // Overwrite with trailing newline (common from `echo`)
        let ref_path = cache
            .path()
            .join("models--org--model")
            .join("refs")
            .join("main");
        std::fs::write(&ref_path, "abc123\n").unwrap();
        let result = resolve_with_cache("org/model", Some(cache.path()));
        assert_eq!(result.unwrap(), snapshot);
    }
}
