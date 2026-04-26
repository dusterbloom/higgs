use std::collections::HashMap;
use std::io::IsTerminal;
use std::net::SocketAddr;
use std::sync::Arc;

use clap::Parser;

use higgs::{
    build_router,
    config::{self, Cli, Commands, ConfigAction, HiggsConfig, MetricsLogConfig, ServeArgs},
    model_download, model_resolver,
    router::Router,
    state::{AppState, Engine},
};

#[tokio::main]
#[allow(clippy::print_stderr)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if let Some(ref name) = cli.profile {
        config::validate_profile_name(name)?;
    }
    let profile = cli.profile.as_deref();

    match cli.command {
        Commands::Serve(ref args) => cmd_serve(&cli, args).await,
        Commands::Start(ref _args) => {
            let config_path = resolve_config_path(&cli)?;
            higgs::daemon::detach(&config_path, cli.verbose, profile);
            Ok(())
        }
        Commands::Stop => {
            higgs::daemon::cmd_stop(profile);
            Ok(())
        }
        Commands::Attach => {
            let config = load_config_for_command(&cli)?;
            higgs::daemon::run_attached(&config, profile);
            Ok(())
        }
        Commands::Init => {
            higgs::daemon::cmd_init(profile);
            Ok(())
        }
        Commands::Shellenv => {
            let config = load_config_for_command(&cli).unwrap_or_default();
            higgs::daemon::cmd_shellenv(&config);
            Ok(())
        }
        Commands::Exec { ref command } => {
            let config = load_config_for_command(&cli).unwrap_or_default();
            higgs::daemon::cmd_exec(&config, command);
        }
        Commands::Config { ref action } => {
            cmd_config(&cli, action);
            Ok(())
        }
        Commands::Doctor(ref args) => {
            init_tracing(cli.verbose);
            let config = if let Some(ref path) = cli.config {
                config::load_config_file(path, Some(args))?
            } else if cli.profile.is_some() {
                let path = resolve_config_path(&cli)?;
                config::load_config_file(&path, Some(args))?
            } else {
                let default = config::default_config_path();
                if default.exists() {
                    config::load_config_file(&default, Some(args))?
                } else if !args.models.is_empty() {
                    config::build_simple_config(args)?
                } else {
                    return Err("no config to validate; use --config or 'higgs init'".into());
                }
            };
            let result = higgs::doctor::run_doctor(&config).await;
            if result.failures > 0 {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

/// Resolve a config file path from CLI args, profile, or the default location.
#[allow(clippy::print_stderr)]
fn resolve_config_path(cli: &Cli) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    if let Some(ref path) = cli.config {
        return Ok(path.clone());
    }
    if let Some(ref name) = cli.profile {
        let path = config::profile_config_path(name);
        if path.exists() {
            return Ok(path);
        }
        return Err(format!(
            "profile config not found at {}\nhint: use 'higgs init --profile {name}' to create one",
            path.display()
        )
        .into());
    }
    let default = config::default_config_path();
    if default.exists() {
        Ok(default)
    } else {
        Err(format!(
            "no config file specified or found at {}\nhint: use 'higgs init' to create one",
            default.display()
        )
        .into())
    }
}

/// Load config from CLI path or default location.
fn load_config_for_command(cli: &Cli) -> Result<HiggsConfig, Box<dyn std::error::Error>> {
    let path = resolve_config_path(cli)?;
    config::load_config_file(&path, None).map_err(Into::into)
}

async fn cmd_serve(cli: &Cli, args: &ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing(cli.verbose);
    install_crash_diagnostics();

    let profile = cli.profile.as_deref();
    let simple_mode = config::is_simple_mode(cli, args);

    // Load config: simple mode (--model) or config file mode (--config)
    let mut higgs_config = if simple_mode {
        config::build_simple_config(args)?
    } else if let Some(ref path) = cli.config {
        config::load_config_file(path, Some(args))?
    } else if cli.profile.is_some() {
        let path = resolve_config_path(cli)?;
        config::load_config_file(&path, Some(args))?
    } else if args.models.is_empty() {
        let default_path = config::default_config_path();
        if default_path.exists() {
            tracing::info!(path = %default_path.display(), "Auto-discovered config file");
            config::load_config_file(&default_path, Some(args))?
        } else {
            return Err("no --model or --config provided, and no config file found at ~/.config/higgs/config.toml\n\
                hint: use 'higgs serve --model <model>' or 'higgs init' to create a config".into());
        }
    } else {
        config::build_simple_config(args)?
    };

    // Rewrite metrics path for profile isolation if still at default
    if let Some(name) = profile {
        let default_path = config::default_metrics_log_path_for_profile(name);
        let generic_default = MetricsLogConfig::default().path;
        if higgs_config.logging.metrics.path == generic_default {
            higgs_config.logging.metrics.path = default_path;
        }
    }

    // Load all local models and build router
    let engines = load_engines(&higgs_config)?;
    let router = Router::from_config(&higgs_config, engines)?;

    // Validate timeout
    let timeout_secs = higgs_config.server.timeout;
    if !timeout_secs.is_finite() || timeout_secs <= 0.0 {
        return Err("timeout must be a positive, finite number".into());
    }

    let api_key = higgs_config.server.api_key.clone();
    let rate_limit = higgs_config.server.rate_limit;
    let bind_addr = format!("{}:{}", higgs_config.server.host, higgs_config.server.port);

    // Create metrics (config mode only)
    let metrics = if simple_mode {
        None
    } else {
        let m = higgs::daemon::create_metrics(&higgs_config);
        higgs::daemon::spawn_eviction_task(&m);
        Some(m)
    };

    // Create shared state
    let http_client = reqwest::Client::new();
    let shared_state = Arc::new(AppState {
        router,
        config: higgs_config,
        http_client,
        metrics,
    });

    // Build router with middleware
    let app = build_router(shared_state, timeout_secs, api_key, rate_limit);

    // Start server
    tracing::info!(addr = %bind_addr, "Starting server");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    // Write PID file after bind succeeds so it's never stale on bind errors
    higgs::daemon::write_pid_file(profile);

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(higgs::daemon::await_shutdown_signal())
    .await?;

    higgs::daemon::remove_pid_file(profile);
    Ok(())
}

fn load_engines(
    config: &HiggsConfig,
) -> Result<HashMap<String, Arc<Engine>>, Box<dyn std::error::Error>> {
    let mut engines = HashMap::new();

    for model_cfg in &config.models {
        let model_path = &model_cfg.path;
        tracing::info!(model = %model_path, "Resolving model path");
        let resolved = match model_resolver::resolve(model_path) {
            Ok(path) => path,
            Err(resolve_err) if model_resolver::is_hf_model_id(model_path) => {
                tracing::debug!(error = %resolve_err, "model not in cache; attempting download");
                let is_interactive = std::io::stdin().is_terminal();
                model_download::offer_download(
                    model_path,
                    is_interactive,
                    &mut std::io::stderr().lock(),
                    std::io::stdin().lock(),
                    || {
                        let status = std::process::Command::new("huggingface-cli")
                            .args(["download", model_path])
                            .status()
                            .map_err(|e| {
                                format!(
                                    "failed to run huggingface-cli: {e}\nInstall with: brew install huggingface-cli"
                                )
                            })?;
                        if status.success() {
                            Ok(())
                        } else {
                            Err(format!(
                                "huggingface-cli download failed for '{model_path}'"
                            ))
                        }
                    },
                )?;
                model_resolver::resolve(model_path)?
            }
            Err(err) => return Err(err.into()),
        };

        tracing::info!(model = %model_path, resolved = %resolved.display(), "Loading model");
        let kv_cache_config = model_cfg.kv_cache_config();
        let dflash_path =
            model_cfg
                .dflash
                .as_ref()
                .and_then(|p| match model_resolver::resolve(p) {
                    Ok(resolved) => Some(resolved),
                    Err(e) => {
                        tracing::warn!(dflash = %p, "DFlash drafter not found, skipping: {e}");
                        None
                    }
                });
        // AR-spec drafter is wired through an env var (the engine reads
        // HIGGS_AR_SPEC_DRAFT_PATH at SimpleEngine load time). Surface the
        // resolved path here so the higgs.toml entry stays self-contained
        // without spreading another path through the engine API.
        if let Some(ref p) = model_cfg.ar_spec {
            match model_resolver::resolve(p) {
                Ok(resolved) => set_ar_spec_env(&resolved),
                Err(e) => {
                    tracing::warn!(ar_spec = %p, "AR-spec drafter not found, skipping: {e}");
                }
            }
        }
        let engine = if model_cfg.batch {
            Engine::load_batch(&resolved, kv_cache_config)?
        } else if let Some(ref draft_path) = model_cfg.draft_model {
            let draft_resolved = model_resolver::resolve(draft_path)?;
            Engine::load_simple_with_draft(
                &resolved,
                &draft_resolved,
                model_cfg.num_draft,
                kv_cache_config,
            )?
        } else if model_cfg.pld {
            Engine::load_simple_with_pld(
                &resolved,
                model_cfg.num_draft,
                model_cfg.pld_max_ngram,
                model_cfg.pld_min_ngram,
                kv_cache_config,
            )?
        } else {
            Engine::load_simple_with_dflash(&resolved, kv_cache_config, dflash_path.as_deref())?
        };
        let name = model_cfg
            .name
            .clone()
            .unwrap_or_else(|| engine.model_name().to_owned());
        tracing::info!(model_name = %name, "Model loaded");

        if engines.insert(name.clone(), Arc::new(engine)).is_some() {
            return Err(format!(
                "model name collision: two model paths resolve to the same name '{name}'"
            )
            .into());
        }
    }

    Ok(engines)
}

fn cmd_config(cli: &Cli, action: &ConfigAction) {
    let config_path = cli.config.clone().unwrap_or_else(|| {
        cli.profile
            .as_ref()
            .map_or_else(config::default_config_path, |name| {
                config::profile_config_path(name)
            })
    });
    match action {
        ConfigAction::Get { key } => {
            higgs::cli_config::config_get(&config_path, key);
        }
        ConfigAction::Set { key, value } => {
            higgs::cli_config::config_set(&config_path, key, value);
        }
        ConfigAction::Path => {
            #[allow(clippy::print_stdout)]
            {
                println!("{}", config_path.display());
            }
        }
    }
}

fn init_tracing(verbose: bool) {
    let default_filter = if verbose { "higgs=debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                default_filter
                    .parse()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
            }),
        )
        .init();
}

/// Crash diagnostic: install panic hook, SIGABRT/SEGV/BUS handlers, and atexit.
/// Writes to `/tmp/higgs_crash_<pid>.log`. Enabled via `HIGGS_CRASH_DIAG=1`.
/// Set `HIGGS_AR_SPEC_DRAFT_PATH` so `SimpleEngine::load` can pick it up.
/// Called from the single-threaded init phase before any worker threads
/// are spawned, so the unsafe `set_var` cannot race with concurrent readers.
#[allow(unsafe_code)]
fn set_ar_spec_env(path: &std::path::Path) {
    unsafe {
        std::env::set_var("HIGGS_AR_SPEC_DRAFT_PATH", path);
    }
}

#[allow(unsafe_code, clippy::print_stderr)]
fn install_crash_diagnostics() {
    use std::io::Write;
    use std::os::unix::io::{AsRawFd, FromRawFd};
    use std::sync::OnceLock;

    static DIAG_FD: OnceLock<i32> = OnceLock::new();
    static DIAG_PATH: OnceLock<String> = OnceLock::new();

    if std::env::var("HIGGS_CRASH_DIAG").ok().as_deref() != Some("1") {
        return;
    }

    let pid = std::process::id();
    let path = format!("/tmp/higgs_crash_{pid}.log");
    let file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("crash_diag: failed to open {path}: {e}");
            return;
        }
    };
    let fd = file.as_raw_fd();
    // Leak the file so fd stays valid for signal handlers
    std::mem::forget(file);
    let _ = DIAG_FD.set(fd);
    let _ = DIAG_PATH.set(path.clone());

    eprintln!("crash_diag: writing to {path}");

    // Write a startup banner
    let mut f = unsafe { std::fs::File::from_raw_fd(fd) };
    let _ = writeln!(
        f,
        "=== STARTUP pid={pid} time={} ===",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let _ = f.flush();
    std::mem::forget(f); // don't close fd

    // Panic hook -> file + stderr
    std::panic::set_hook(Box::new(move |info| {
        let msg = format!(
            "=== PANIC pid={pid} at {} ===\n{info}\n{}\n",
            info.location()
                .map_or("?".to_owned(), |l| format!("{}:{}", l.file(), l.line())),
            std::backtrace::Backtrace::force_capture()
        );
        eprintln!("{msg}");
        if let Some(&fd) = DIAG_FD.get() {
            let mut f = unsafe { std::fs::File::from_raw_fd(fd) };
            let _ = f.write_all(msg.as_bytes());
            let _ = f.flush();
            std::mem::forget(f);
        }
    }));

    // Signal handlers for SIGABRT/SEGV/BUS/TERM/HUP
    extern "C" fn sig_handler(sig: libc::c_int) {
        // Async-signal-safe: only use write(2) to our fd
        if let Some(&fd) = DIAG_FD.get() {
            let name = match sig {
                libc::SIGABRT => "SIGABRT",
                libc::SIGSEGV => "SIGSEGV",
                libc::SIGBUS => "SIGBUS",
                libc::SIGTERM => "SIGTERM",
                libc::SIGHUP => "SIGHUP",
                libc::SIGPIPE => "SIGPIPE",
                libc::SIGILL => "SIGILL",
                _ => "UNKNOWN",
            };
            let msg = format!("=== SIGNAL {name} ({sig}) ===\n");
            unsafe {
                libc::write(fd, msg.as_ptr().cast(), msg.len());
                libc::fsync(fd);
            }
        }
        // Re-raise with default handler to get core/termination
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = libc::SIG_DFL;
            libc::sigaction(sig, &sa, std::ptr::null_mut());
            libc::raise(sig);
        }
    }

    for &sig in &[
        libc::SIGABRT,
        libc::SIGSEGV,
        libc::SIGBUS,
        libc::SIGILL,
        libc::SIGTERM,
        libc::SIGHUP,
    ] {
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = sig_handler as usize;
            sa.sa_flags = libc::SA_RESETHAND;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(sig, &sa, std::ptr::null_mut());
        }
    }

    // atexit for normal exits
    extern "C" fn at_exit() {
        if let Some(&fd) = DIAG_FD.get() {
            let msg = b"=== CLEAN EXIT ===\n";
            unsafe {
                libc::write(fd, msg.as_ptr().cast(), msg.len());
                libc::fsync(fd);
            }
        }
    }
    unsafe {
        libc::atexit(at_exit);
    }
}
