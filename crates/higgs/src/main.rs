use std::collections::HashMap;
use std::fs;
use std::future::{Future, IntoFuture};
use std::io::IsTerminal;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::time::SystemTime;

use clap::Parser;

use higgs::{
    build_router,
    capacity::{
        ActiveRegistration, CapacityPressureCoordinator, CapacityRegistry,
        start_capacity_pressure_observer,
    },
    config::{
        self, Cli, Commands, ConfigAction, HiggsConfig, MetricsLogConfig, ServeArgs, StartArgs,
        StopArgs,
    },
    model_download, model_resolver,
    router::Router,
    state::{
        AppState, Engine, build_engine_with_capacity, refresh_after_engine_drop,
        release_failed_engine,
    },
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
        Commands::Start(ref args) => {
            reject_legacy_start_flags(args)?;
            let config_path = resolve_config_path(&cli)?;
            higgs::daemon::detach(&config_path, cli.verbose, profile);
            Ok(())
        }
        Commands::Stop(StopArgs { force }) => {
            let exit_code = higgs::daemon::cmd_stop(profile, force);
            if exit_code != 0 {
                std::process::exit(exit_code);
            }
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
            let config = load_config_for_command(&cli)?;
            higgs::daemon::cmd_shellenv(&config)?;
            Ok(())
        }
        Commands::Exec { ref command } => {
            let config = load_config_for_command(&cli)?;
            higgs::daemon::cmd_exec(&config, command);
        }
        Commands::Config { ref action } => {
            cmd_config(&cli, action);
            Ok(())
        }
        Commands::Doctor(ref args) => {
            init_tracing(cli.verbose);
            let (config, config_path) = if let Some(ref path) = cli.config {
                (
                    config::load_config_file(path, Some(args))?,
                    Some(path.clone()),
                )
            } else if cli.profile.is_some() {
                let path = resolve_config_path(&cli)?;
                let config = config::load_config_file(&path, Some(args))?;
                (config, Some(path))
            } else {
                let default = config::default_config_path();
                if default.exists() {
                    let config = config::load_config_file(&default, Some(args))?;
                    (config, Some(default))
                } else if !args.models.is_empty() {
                    (config::build_simple_config(args)?, None)
                } else {
                    return Err("no config to validate; use --config or 'higgs init'".into());
                }
            };
            let result = higgs::doctor::run_doctor(&config, config_path.as_deref()).await;
            if result.failures > 0 {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

fn reject_legacy_start_flags(args: &StartArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.uses_serve_flags() {
        return Err(
            "higgs start is config/profile-only\nhint: use 'higgs serve' for ad hoc --model/--port/--batch flags"
                .into(),
        );
    }
    Ok(())
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

    ensure_local_runtime_ready(&higgs_config)?;
    if higgs_config.local.raise_wired_limit && higgs_config.models.len() > 1 {
        tracing::warn!(
            model_count = higgs_config.models.len(),
            "MLX wired-limit escalation is enabled with multiple resident local models; unified-memory pressure may spike"
        );
    }

    // One registry owns shared residency and pressure across every local model.
    let capacity =
        CapacityRegistry::new_with_profile_dir(std::iter::empty(), capacity_profile_dir(cli));
    // Boot-time memory authority: engines load (and admit against capacity)
    // BEFORE any AppState exists, so the registry needs one measured snapshot
    // here. Without it the first startup load fails closed with "no safe
    // process memory authority" because the fresh registry's memory state is
    // empty.
    match higgs_engine::MlxMemorySnapshot::measure() {
        Ok(memory) => {
            tracing::info!(
                limit = ?memory.memory_limit_bytes,
                working_set = ?memory.metal_recommended_working_set_bytes,
                active = memory.active_bytes,
                "boot_capacity_memory_authority"
            );
            capacity.refresh_memory(memory);
        }
        Err(error) => {
            tracing::warn!(%error, "failed to measure boot capacity memory authority");
        }
    }
    let pressure_coordinator = CapacityPressureCoordinator::new(Arc::clone(&capacity));
    let pressure_observer =
        start_capacity_pressure_observer(Arc::clone(&pressure_coordinator)).await?;

    let mut pid_written = false;
    let serve_result: Result<(), Box<dyn std::error::Error>> = async {

    // Loaded models remain provisional until the router accepts the complete set.
    #[cfg(target_os = "macos")]
    let (engines, registrations) = startup_load_once(
        || load_engines(&higgs_config, &capacity),
        offer_startup_memory_recovery,
        || {
            higgs_engine::simple::maybe_clear_mlx_cache(true, "startup memory recovery");
            let memory = higgs_engine::MlxMemorySnapshot::measure()?;
            capacity.refresh_memory(memory);
            Ok(())
        },
    ).await?;
    #[cfg(not(target_os = "macos"))]
    let (engines, registrations) = load_engines(&higgs_config, &capacity).await?;
    let cleanup_engines = engines.clone();
    let router = match Router::from_config(&higgs_config, engines) {
        Ok(router) => router,
        Err(error) => {
            drop(registrations);
            for engine in cleanup_engines.into_values() {
                if let Ok(engine) = Arc::try_unwrap(engine)
                    && let Err(shutdown_error) = engine.shutdown()
                {
                    tracing::warn!(%shutdown_error, "failed to join engine after router construction failure");
                }
            }
            let _ = refresh_after_engine_drop(&capacity, "failed router construction");
            return Err(error.into());
        }
    };
    drop(cleanup_engines);
    for registration in registrations {
        registration.publish();
    }

    // Validate timeout
    let timeout_secs = higgs_config.server.timeout;
    if !timeout_secs.is_finite() || timeout_secs <= 0.0 {
        return Err("timeout must be a positive, finite number".into());
    }

    let api_key = higgs_config.server.api_key.clone();
    let rate_limit = higgs_config.server.rate_limit;
    let max_body_size = higgs_config.server.max_body_size;
    let cors_origins = higgs_config.server.cors_origins.clone();
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
    let shared_state = Arc::new(AppState::with_capacity_registry(
        router,
        higgs_config,
        http_client,
        metrics,
        Arc::clone(&capacity),
    ));
    pressure_coordinator.attach(&shared_state).await?;

    // Build router with middleware
    let app = build_router(
        Arc::clone(&shared_state),
        timeout_secs,
        api_key,
        rate_limit,
        max_body_size,
        cors_origins,
    );

    // Start server
    tracing::info!(addr = %bind_addr, "Starting server");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    // Write PID file after bind succeeds so it's never stale on bind errors
    higgs::daemon::write_pid_file(profile);
    pid_written = true;

    let server = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(higgs::daemon::await_shutdown_signal());
    server.await?;
    Ok(())
    }
    .await;
    let (observer_result, persist_result) = stop_observer_then_cleanup(
        || pressure_observer.stop(),
        || {
            if !pid_written {
                return Ok(());
            }
            let persist_result = capacity.persist_profiles();
            higgs::daemon::remove_pid_file(profile);
            persist_result
        },
    )
    .await;
    serve_result?;
    observer_result?;
    persist_result?;
    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct StartupApp {
    pid: i32,
    uid: u32,
    started: (u64, u64),
    path: PathBuf,
    rss: u64,
}

#[cfg(target_os = "macos")]
fn startup_app_candidates(mut apps: Vec<StartupApp>, uid: u32, own_pid: u32) -> Vec<StartupApp> {
    apps.retain(|app| {
        app.pid > 1
            && u32::try_from(app.pid).ok() != Some(own_pid)
            && app.uid != 0
            && app.uid == uid
            && app.path.is_absolute()
            && !app.path.starts_with("/System")
            && !app.path.starts_with("/usr")
            && !app
                .path
                .to_string_lossy()
                .to_ascii_lowercase()
                .contains("higgs")
            && app
                .path
                .components()
                .any(|part| part.as_os_str().to_string_lossy().ends_with(".app"))
    });
    // Nested helper bundles belong to the outer user application. Their memory
    // contributes to ranking, but only a direct main executable is signalable.
    let mut bundles: HashMap<PathBuf, (u64, Option<StartupApp>)> = HashMap::new();
    for app in apps {
        let Some(bundle) = app
            .path
            .ancestors()
            .filter(|path| path.extension().is_some_and(|ext| ext == "app"))
            .last()
        else {
            continue;
        };
        let direct_main = app.path.parent() == Some(bundle.join("Contents/MacOS").as_path());
        let (rss, main) = bundles.entry(bundle.to_path_buf()).or_default();
        *rss = rss.saturating_add(app.rss);
        if direct_main && main.as_ref().is_none_or(|current| app.pid < current.pid) {
            *main = Some(app);
        }
    }
    let mut apps: Vec<_> = bundles
        .into_values()
        .filter_map(|(rss, main)| {
            main.map(|mut app| {
                app.rss = rss;
                app
            })
        })
        .collect();
    apps.sort_by(|a, b| b.rss.cmp(&a.rss).then(a.pid.cmp(&b.pid)));
    apps.truncate(3);
    apps
}

#[cfg(target_os = "macos")]
trait StartupRecovery {
    fn confirm(&mut self, message: &str) -> std::io::Result<bool>;
    fn same_process(&mut self, app: &StartupApp) -> bool;
    fn signal(&mut self, app: &StartupApp, signal: nix::sys::signal::Signal)
    -> std::io::Result<()>;
    fn wait(&mut self, apps: &[StartupApp]);
}

#[cfg(target_os = "macos")]
fn recover_startup_apps(
    apps: &[StartupApp],
    interactive: bool,
    uid: u32,
    io: &mut impl StartupRecovery,
) -> std::io::Result<bool> {
    use nix::sys::signal::Signal;
    if !interactive || uid == 0 || apps.is_empty() {
        return Ok(false);
    }
    if !io.confirm("Quit the listed apps with SIGTERM? Unsaved work may be lost. [y/N] ")? {
        return Ok(false);
    }
    let mut signaled = false;
    for app in apps {
        // Check start timestamp and executable immediately before each signal.
        // PID alone is never authority to act on a recycled process identifier.
        if io.same_process(app) {
            io.signal(app, Signal::SIGTERM)?;
            signaled = true;
        }
    }
    io.wait(apps);
    let survivors: Vec<_> = apps
        .iter()
        .filter(|app| io.same_process(app))
        .cloned()
        .collect();
    if !survivors.is_empty() {
        if !io.confirm("Some listed apps remain. Force quit those survivors with SIGKILL? Unsaved work will be lost. [y/N] ")? { return Ok(false); }
        for app in &survivors {
            if io.same_process(app) {
                io.signal(app, Signal::SIGKILL)?;
                signaled = true;
            }
        }
        // A successful kill syscall does not mean process teardown has finished.
        // Give the authorized survivors time to release residency before measuring.
        io.wait(&survivors);
    }
    Ok(signaled)
}

#[cfg(target_os = "macos")]
async fn startup_load_once<T, F, Fut>(
    mut load: F,
    recover: impl FnOnce() -> Result<bool, Box<dyn std::error::Error>>,
    refresh: impl FnOnce() -> Result<(), Box<dyn std::error::Error>>,
) -> Result<T, Box<dyn std::error::Error>>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, Box<dyn std::error::Error>>>,
{
    match load().await {
        Err(error) if error.to_string().contains("rejected by capacity policy") => {
            if !recover()? {
                return Err(error);
            }
            refresh()?;
            // Deliberately outside the match: a second rejection is terminal.
            load().await
        }
        result => result,
    }
}

#[cfg(target_os = "macos")]
// Native libproc has no safe wrapper here; all output pointers have exact-sized storage.
#[allow(unsafe_code)]
fn startup_process(pid: i32) -> Option<StartupApp> {
    // libproc fills fixed-size C records. Treat partial records as unavailable.
    unsafe {
        let mut bsd: libc::proc_bsdinfo = std::mem::zeroed();
        let bsd_size = i32::try_from(std::mem::size_of_val(&bsd)).ok()?;
        if libc::proc_pidinfo(
            pid,
            libc::PROC_PIDTBSDINFO,
            0,
            std::ptr::from_mut(&mut bsd).cast(),
            bsd_size,
        ) != bsd_size
        {
            return None;
        }
        let mut task: libc::proc_taskinfo = std::mem::zeroed();
        let task_size = i32::try_from(std::mem::size_of_val(&task)).ok()?;
        if libc::proc_pidinfo(
            pid,
            libc::PROC_PIDTASKINFO,
            0,
            std::ptr::from_mut(&mut task).cast(),
            task_size,
        ) != task_size
        {
            return None;
        }
        let mut path = [0u8; 4096];
        if libc::proc_pidpath(
            pid,
            path.as_mut_ptr().cast(),
            u32::try_from(path.len()).ok()?,
        ) <= 0
        {
            return None;
        }
        let end = path.iter().position(|byte| *byte == 0)?;
        use std::os::unix::ffi::OsStrExt;
        Some(StartupApp {
            pid,
            uid: bsd.pbi_uid,
            started: (bsd.pbi_start_tvsec, bsd.pbi_start_tvusec),
            path: PathBuf::from(std::ffi::OsStr::from_bytes(&path[..end])),
            rss: task.pti_resident_size,
        })
    }
}

#[cfg(target_os = "macos")]
fn startup_signal_result(result: Result<(), nix::errno::Errno>) -> std::io::Result<()> {
    // The process may exit after identity validation but before kill reaches it.
    // Only ESRCH means the intended outcome already happened; permission errors remain fatal.
    match result {
        Ok(()) | Err(nix::errno::Errno::ESRCH) => Ok(()),
        Err(error) => Err(std::io::Error::from(error)),
    }
}

#[cfg(target_os = "macos")]
struct StartupTerminal;

#[cfg(target_os = "macos")]
impl StartupRecovery for StartupTerminal {
    fn confirm(&mut self, message: &str) -> std::io::Result<bool> {
        use std::io::Write;
        let mut stderr = std::io::stderr().lock();
        stderr.write_all(message.as_bytes())?;
        stderr.flush()?;
        let mut reply = String::new();
        std::io::stdin().read_line(&mut reply)?;
        Ok(matches!(
            reply.trim().to_ascii_lowercase().as_str(),
            "y" | "yes"
        ))
    }
    fn same_process(&mut self, app: &StartupApp) -> bool {
        startup_process(app.pid).is_some_and(|now| {
            now.uid == app.uid && now.started == app.started && now.path == app.path
        })
    }
    fn signal(
        &mut self,
        app: &StartupApp,
        signal: nix::sys::signal::Signal,
    ) -> std::io::Result<()> {
        startup_signal_result(nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(app.pid),
            signal,
        ))
    }
    fn wait(&mut self, apps: &[StartupApp]) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while apps.iter().any(|app| self.same_process(app)) {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            std::thread::sleep(remaining.min(std::time::Duration::from_millis(100)));
        }
    }
}

#[cfg(target_os = "macos")]
// Enumeration writes only within the allocated PID buffer; geteuid takes no pointers.
#[allow(unsafe_code)]
fn offer_startup_memory_recovery() -> Result<bool, Box<dyn std::error::Error>> {
    use std::io::Write;
    // libproc returns PID counts, while its buffer capacity is specified in bytes.
    let apps = unsafe {
        let count = libc::proc_listallpids(std::ptr::null_mut(), 0);
        if count <= 0 {
            return Ok(false);
        }
        let mut pids = vec![0i32; usize::try_from(count)?.saturating_add(128)];
        let used = libc::proc_listallpids(
            pids.as_mut_ptr().cast(),
            i32::try_from(std::mem::size_of_val(pids.as_slice()))?,
        );
        if used <= 0 {
            return Ok(false);
        }
        pids.truncate(usize::try_from(used)?);
        pids.into_iter()
            .filter(|pid| *pid > 1)
            .filter_map(startup_process)
            .collect()
    };
    let uid = unsafe { libc::geteuid() };
    let apps = startup_app_candidates(apps, uid, std::process::id());
    {
        let mut stderr = std::io::stderr().lock();
        writeln!(
            stderr,
            "Startup was rejected by capacity policy. Close memory-heavy apps and retry higgs serve. Root and noninteractive runs never signal apps."
        )?;
        for app in &apps {
            writeln!(
                stderr,
                "  {} — PID {}: {} MiB — {}",
                app.path
                    .components()
                    .find(|part| part.as_os_str().to_string_lossy().ends_with(".app"))
                    .map(|part| part.as_os_str().to_string_lossy())
                    .unwrap_or_default(),
                app.pid,
                app.rss / (1024 * 1024),
                app.path.display()
            )?;
        }
    }
    recover_startup_apps(
        &apps,
        std::io::stdin().is_terminal() && std::io::stderr().is_terminal(),
        uid,
        &mut StartupTerminal,
    )
    .map_err(Into::into)
}

fn capacity_profile_dir(cli: &Cli) -> PathBuf {
    let config_path = cli.config.clone().unwrap_or_else(|| {
        cli.profile
            .as_deref()
            .map_or_else(config::default_config_path, config::profile_config_path)
    });
    config_path
        .parent()
        .map_or_else(config::config_dir, Path::to_path_buf)
        .join("capacity")
}

async fn load_engines(
    config: &HiggsConfig,
    capacity: &Arc<CapacityRegistry>,
) -> Result<(HashMap<String, Arc<Engine>>, Vec<ActiveRegistration>), Box<dyn std::error::Error>> {
    let mut engines: HashMap<String, Arc<Engine>> = HashMap::new();
    let mut registrations = Vec::new();

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
                    || download_via_hf_cli(model_path),
                )?;
                model_resolver::resolve(model_path)?
            }
            Err(err) => return Err(err.into()),
        };

        tracing::info!(model = %model_path, resolved = %resolved.display(), "Loading model");
        let (name, engine, facts) =
            build_engine_with_capacity(&resolved, model_cfg, config, capacity)?;
        if engines.contains_key(&name) {
            release_failed_engine(engine, capacity);
            return Err(format!(
                "model name collision: two model paths resolve to the same name '{name}'"
            )
            .into());
        }
        let ticket = match capacity.begin_registration(name.clone()) {
            Ok(ticket) => ticket,
            Err(error) => {
                release_failed_engine(engine, capacity);
                return Err(error.into());
            }
        };
        let registration = match capacity.commit_active(ticket, facts) {
            Ok(registration) => registration,
            Err(error) => {
                release_failed_engine(engine, capacity);
                return Err(error.into());
            }
        };
        loop {
            let plan = capacity.cache_allocation_plan();
            let Some((_, retained, prefix)) =
                plan.allocations.iter().find(|(model, _, _)| model == &name)
            else {
                drop(registration);
                release_failed_engine(engine, capacity);
                return Err(format!("cache allocation missing for '{name}'").into());
            };
            if let Err(error) = engine
                .apply_capacity_cache_limits(plan.revision, *retained, *prefix, plan.pressure)
                .await
            {
                drop(registration);
                release_failed_engine(engine, capacity);
                return Err(
                    format!("failed to apply cache allocation for '{name}': {error}").into(),
                );
            }
            for (loaded_name, loaded_engine) in &engines {
                let Some((_, retained, prefix)) = plan
                    .allocations
                    .iter()
                    .find(|(model, _, _)| model == loaded_name)
                else {
                    continue;
                };
                if let Err(error) = loaded_engine
                    .apply_capacity_cache_limits(plan.revision, *retained, *prefix, plan.pressure)
                    .await
                {
                    drop(registration);
                    release_failed_engine(engine, capacity);
                    return Err(format!(
                        "failed to apply cache allocation for '{loaded_name}': {error}"
                    )
                    .into());
                }
            }
            if capacity.publish_cache_allocation_revision(plan.revision) {
                break;
            }
        }
        tracing::info!(model_name = %name, "Model loaded");

        engines.insert(name.clone(), Arc::new(engine));
        registrations.push(registration);
    }

    Ok((engines, registrations))
}

#[cfg_attr(not(test), allow(dead_code))]
async fn await_server_then_stop<ServerFuture, Stop, StopFuture, ServerError>(
    server: ServerFuture,
    stop: Stop,
) -> (Result<(), ServerError>, Result<(), String>)
where
    ServerFuture: IntoFuture<Output = Result<(), ServerError>>,
    Stop: FnOnce() -> StopFuture,
    StopFuture: Future<Output = Result<(), String>>,
{
    let server_result = server.into_future().await;
    let observer_result = stop().await;
    (server_result, observer_result)
}

async fn stop_observer_then_cleanup<Stop, StopFuture, Cleanup, CleanupResult>(
    stop: Stop,
    cleanup: Cleanup,
) -> (Result<(), String>, CleanupResult)
where
    Stop: FnOnce() -> StopFuture,
    StopFuture: Future<Output = Result<(), String>>,
    Cleanup: FnOnce() -> CleanupResult,
{
    let observer_result = stop().await;
    let cleanup_result = cleanup();
    (observer_result, cleanup_result)
}

fn ensure_local_runtime_ready(config: &HiggsConfig) -> Result<(), Box<dyn std::error::Error>> {
    if config.models.is_empty() {
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    {
        let exe = std::env::current_exe()?;
        let metallib = exe.with_file_name("mlx.metallib");
        if !metallib.exists() {
            try_restore_metallib(&exe, &metallib)?;
        }
        if !metallib.exists() {
            return Err(format!(
                "mlx.metallib not found next to executable at {}\nhint: rebuild Higgs, reinstall the Apple Silicon Homebrew package, or use a release artifact that bundles mlx.metallib",
                metallib.display()
            )
            .into());
        }
    }
    Ok(())
}

fn download_via_hf_cli(model_path: &str) -> Result<(), String> {
    const CMD: &str = "hf";

    let status = std::process::Command::new(CMD)
        .args(["download", model_path])
        .status()
        .map_err(|e| format!("failed to run {CMD}: {e}\nInstall with: brew install {CMD}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("{CMD} download failed for '{model_path}'"))
    }
}

#[cfg(target_os = "macos")]
fn try_restore_metallib(exe: &Path, destination: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let Some(profile_dir) = derive_profile_dir(exe) else {
        return Ok(());
    };
    let Some(source) = newest_metallib_candidate(&profile_dir) else {
        return Ok(());
    };
    fs::copy(&source, destination)?;
    tracing::info!(
        source = %source.display(),
        destination = %destination.display(),
        "restored mlx.metallib next to executable from Cargo build output"
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn derive_profile_dir(exe: &Path) -> Option<PathBuf> {
    let parent = exe.parent()?;
    if parent.file_name().is_some_and(|name| name == "deps") {
        parent.parent().map(Path::to_path_buf)
    } else {
        Some(parent.to_path_buf())
    }
}

#[cfg(target_os = "macos")]
fn newest_metallib_candidate(profile_dir: &Path) -> Option<PathBuf> {
    let build_dir = profile_dir.join("build");
    let entries = fs::read_dir(build_dir).ok()?;
    let candidates = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name();
            let is_mlx_sys = name
                .to_str()
                .is_some_and(|value| value.starts_with("mlx-sys-"));
            if !is_mlx_sys {
                return None;
            }
            let candidate = entry.path().join("out/build/lib/mlx.metallib");
            candidate.exists().then(|| {
                (
                    fs::metadata(&candidate)
                        .and_then(|meta| meta.modified())
                        .ok(),
                    candidate,
                )
            })
        })
        .collect();
    select_latest_metallib_candidate(candidates)
}

#[cfg(target_os = "macos")]
fn select_latest_metallib_candidate(
    candidates: Vec<(Option<SystemTime>, PathBuf)>,
) -> Option<PathBuf> {
    candidates
        .into_iter()
        .max_by(|(left_time, left_path), (right_time, right_path)| {
            left_time
                .cmp(right_time)
                .then_with(|| left_path.cmp(right_path))
        })
        .map(|(_, path)| path)
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

#[cfg(all(test, target_os = "macos"))]
#[allow(clippy::panic, clippy::unwrap_used, clippy::tests_outside_test_module)]
mod tests {
    use super::*;

    fn app(pid: i32, rss: u64) -> StartupApp {
        StartupApp {
            pid,
            uid: 501,
            started: (10, 20),
            path: PathBuf::from(format!("/Applications/App{pid}.app/Contents/MacOS/App")),
            rss,
        }
    }
    struct RecoveryFixture {
        answers: std::collections::VecDeque<bool>,
        alive: bool,
        reused: bool,
        graceful: bool,
        signals: Vec<nix::sys::signal::Signal>,
        waits: usize,
        waited_after_kill: bool,
    }
    impl StartupRecovery for RecoveryFixture {
        fn confirm(&mut self, _: &str) -> std::io::Result<bool> {
            Ok(self.answers.pop_front().expect("unexpected prompt"))
        }
        fn same_process(&mut self, _: &StartupApp) -> bool {
            self.alive && !self.reused
        }
        fn signal(
            &mut self,
            _: &StartupApp,
            signal: nix::sys::signal::Signal,
        ) -> std::io::Result<()> {
            self.signals.push(signal);
            Ok(())
        }
        fn wait(&mut self, _: &[StartupApp]) {
            self.waits += 1;
            self.waited_after_kill =
                self.signals.last() == Some(&nix::sys::signal::Signal::SIGKILL);
            if self.graceful {
                self.alive = false;
            }
        }
    }
    #[test]
    fn startup_recovery_filters_sorts_and_caps() {
        let mut apps: Vec<_> = (2..10)
            .map(|id| app(id, u64::try_from(id).unwrap()))
            .collect();
        apps[0].uid = 0;
        apps[1].uid = 502;
        apps[2].path = PathBuf::from("/usr/bin/worker");
        apps[3].path = PathBuf::from("/System/Foo.app/worker");
        apps[4].path = PathBuf::from("/Applications/Higgs.app/higgs");
        let result = startup_app_candidates(apps, 501, 9);
        assert_eq!(result.iter().map(|a| a.pid).collect::<Vec<_>>(), [8, 7]);
        assert_eq!(
            startup_app_candidates(
                (2..9)
                    .map(|id| app(id, u64::try_from(id).unwrap()))
                    .collect(),
                501,
                99
            )
            .len(),
            3
        );
    }
    #[test]
    fn startup_recovery_groups_helpers_under_outer_main() {
        let main = app(10, 100);
        let mut helper = app(11, 900);
        helper.path = PathBuf::from(
            "/Applications/App10.app/Contents/Frameworks/Helper.app/Contents/MacOS/Helper",
        );
        let mut second_helper = helper.clone();
        second_helper.pid = 12;
        second_helper.rss = 500;
        let mut orphan = helper.clone();
        orphan.pid = 13;
        orphan.path = PathBuf::from(
            "/Applications/Orphan.app/Contents/Frameworks/Helper.app/Contents/MacOS/Helper",
        );
        let result = startup_app_candidates(
            vec![helper, app(20, 1200), orphan, main.clone(), second_helper],
            501,
            99,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].pid, main.pid);
        assert_eq!(result[0].path, main.path);
        assert_eq!(result[0].started, main.started);
        assert_eq!(result[0].rss, 1500);
        assert_eq!(result[1].pid, 20);
    }

    #[test]
    fn startup_recovery_authorization_and_identity() {
        assert!(startup_signal_result(Ok(())).is_ok());
        assert!(startup_signal_result(Err(nix::errno::Errno::ESRCH)).is_ok());
        assert_eq!(
            startup_signal_result(Err(nix::errno::Errno::EPERM))
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::PermissionDenied
        );
        use nix::sys::signal::Signal::{SIGKILL, SIGTERM};
        for (interactive, uid, answers, graceful, reused, expected) in [
            (false, 501, vec![], false, false, vec![]),
            (true, 0, vec![], false, false, vec![]),
            (true, 501, vec![false], false, false, vec![]),
            (true, 501, vec![true], true, false, vec![SIGTERM]),
            (true, 501, vec![true, false], false, false, vec![SIGTERM]),
            (
                true,
                501,
                vec![true, true],
                false,
                false,
                vec![SIGTERM, SIGKILL],
            ),
            (true, 501, vec![true], false, true, vec![]),
        ] {
            let mut io = RecoveryFixture {
                answers: answers.into(),
                alive: true,
                reused,
                graceful,
                signals: vec![],
                waits: 0,
                waited_after_kill: false,
            };
            let recovered =
                recover_startup_apps(&[app(44, 100)], interactive, uid, &mut io).unwrap();
            assert_eq!(
                recovered,
                !expected.is_empty() && (graceful || expected.contains(&SIGKILL))
            );
            assert_eq!(io.signals, expected);
            if expected.contains(&SIGKILL) {
                assert_eq!(io.waits, 2);
                assert!(io.waited_after_kill);
            } else {
                assert!(io.waits <= 1);
                assert!(!io.waited_after_kill);
            }
        }
    }
    #[tokio::test]
    async fn startup_recovery_matches_only_capacity_and_retries_once() {
        for (message, recover, expected) in [
            ("unrelated failure", true, 1),
            ("load rejected by capacity policy: RAM", false, 1),
            ("load rejected by capacity policy: RAM", true, 2),
        ] {
            let calls = std::cell::Cell::new(0);
            let refreshed = std::cell::Cell::new(false);
            let result = startup_load_once(
                || {
                    calls.set(calls.get() + 1);
                    if calls.get() == 2 {
                        assert!(refreshed.get());
                    }
                    async { Err::<(), Box<dyn std::error::Error>>(message.into()) }
                },
                || Ok(recover),
                || {
                    refreshed.set(true);
                    Ok(())
                },
            )
            .await;
            assert!(result.is_err());
            assert_eq!(calls.get(), expected);
            assert_eq!(refreshed.get(), expected == 2);
        }
    }

    #[test]
    fn select_latest_metallib_candidate_prefers_newest_timestamp() {
        let older = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(10);
        let newer = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(20);
        let first = PathBuf::from("/tmp/mlx-sys-a/out/build/lib/mlx.metallib");
        let second = PathBuf::from("/tmp/mlx-sys-b/out/build/lib/mlx.metallib");

        let selected = select_latest_metallib_candidate(vec![
            (Some(older), first),
            (Some(newer), second.clone()),
        ]);

        assert_eq!(selected, Some(second));
    }

    #[test]
    fn select_latest_metallib_candidate_breaks_none_ties_by_path() {
        let first = PathBuf::from("/tmp/mlx-sys-a/out/build/lib/mlx.metallib");
        let second = PathBuf::from("/tmp/mlx-sys-b/out/build/lib/mlx.metallib");

        let selected =
            select_latest_metallib_candidate(vec![(None, first), (None, second.clone())]);

        assert_eq!(selected, Some(second));
    }

    #[tokio::test]
    async fn observer_stop_is_awaited_when_server_returns_error() {
        let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_flag = Arc::clone(&stopped);
        let (server, observer) = await_server_then_stop(
            async { Err::<(), _>("server failed") },
            move || async move {
                stop_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok::<(), String>(())
            },
        )
        .await;

        assert_eq!(server, Err("server failed"));
        assert_eq!(observer, Ok(()));
        assert!(stopped.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn observer_is_stopped_before_pid_cleanup() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        events.lock().unwrap().push("server_return");
        let stop_events = Arc::clone(&events);
        let cleanup_events = Arc::clone(&events);

        let (observer, cleanup) = stop_observer_then_cleanup(
            move || async move {
                stop_events.lock().unwrap().push("observer_stopped");
                Ok::<(), String>(())
            },
            move || {
                cleanup_events.lock().unwrap().push("pid_cleanup");
                Ok::<(), String>(())
            },
        )
        .await;

        assert_eq!(observer, Ok(()));
        assert_eq!(cleanup, Ok(()));
        assert_eq!(
            *events.lock().unwrap(),
            ["server_return", "observer_stopped", "pid_cleanup"]
        );
    }
}
