use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::Duration,
};

use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager, RunEvent, State};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

// A one-shot sender registered before killing the sidecar; the event-drain
// task fires it once the child's Terminated event arrives, i.e. the process
// really exited and its exe is no longer locked.
type ExitListener = Arc<Mutex<Option<mpsc::Sender<()>>>>;

struct SidecarState {
    // The flag flips true when the child's Terminated event is emitted — used
    // to confirm death when kill() fails and the event may already be gone.
    child: Mutex<Option<(CommandChild, Arc<AtomicBool>)>>,
    exit_listener: ExitListener,
    // Set while a killed sidecar's Terminated event is still pending — blocks
    // further update attempts so nothing proceeds on an unconfirmed exit.
    stopping: Arc<AtomicBool>,
    // Terminated flag of a child whose kill() could not be confirmed — the
    // handle is consumed by kill(), so the flag is kept here instead. An
    // empty `child` slot must not be read as "backend stopped" while this is
    // set: the update refuses to proceed until the orphan's exit is observed.
    lost_child_termination: Mutex<Option<Arc<AtomicBool>>>,
}

#[tauri::command]
async fn api_base(port: State<'_, u16>) -> Result<String, String> {
    let port = *port;
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while std::time::Instant::now() < deadline {
        let healthy = tauri::async_runtime::spawn_blocking(move || health_check(port))
            .await
            .unwrap_or(false);
        if healthy {
            break;
        }
        let _ = tauri::async_runtime::spawn_blocking(|| {
            thread::sleep(Duration::from_millis(250))
        })
        .await;
    }
    Ok(format!("http://127.0.0.1:{port}"))
}

fn free_port() -> std::io::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

fn health_check(port: u16) -> bool {
    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(250)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    if stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut response = String::new();
    if stream.read_to_string(&mut response).is_err() {
        return false;
    }
    response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200")
}

fn sanitize_installer_name(file_name: &str) -> String {
    let sanitized: String = file_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let sanitized = sanitized.trim_matches('.').to_string();
    if sanitized.is_empty() {
        "openchat-setup.exe".to_string()
    } else if sanitized.to_lowercase().ends_with(".exe") {
        sanitized
    } else {
        format!("{sanitized}.exe")
    }
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot read downloaded installer: {e}"))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .map_err(|e| format!("Cannot hash downloaded installer: {e}"))?;
    let actual = format!("{:x}", hasher.finalize());
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        let _ = std::fs::remove_file(path);
        Err("Downloaded installer failed integrity verification.".to_string())
    }
}

// Release downloads start on github.com under this repo's release path and
// redirect to GitHub's CDN hosts.
const RELEASE_DOWNLOAD_PREFIX: &str = "/mrn1522/openchat/releases/download/";
const REDIRECT_HOST_SUFFIXES: &[&str] = &["github.com", "githubusercontent.com"];

fn download_host_allowed(url: &reqwest::Url) -> bool {
    if url.scheme() != "https" {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    REDIRECT_HOST_SUFFIXES
        .iter()
        .any(|suffix| host == *suffix || host.ends_with(&format!(".{suffix}")))
}

fn download_installer(
    download_url: &str,
    file_name: &str,
    expected_sha256: &str,
) -> Result<PathBuf, String> {
    let url = reqwest::Url::parse(download_url)
        .map_err(|e| format!("Invalid installer URL: {e}"))?;
    let initial_allowed = url.scheme() == "https"
        && url.host_str() == Some("github.com")
        && url
            .path()
            .to_lowercase()
            .starts_with(RELEASE_DOWNLOAD_PREFIX);
    if !initial_allowed {
        return Err("Installer URL must be an https OpenChat release asset on github.com.".to_string());
    }

    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("openchat-desktop/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(600))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() < 5 && download_host_allowed(attempt.url()) {
                attempt.follow()
            } else {
                attempt.stop()
            }
        }))
        .build()
        .map_err(|e| format!("HTTP client init failed: {e}"))?;
    let mut response = client
        .get(url)
        .send()
        .and_then(|response| response.error_for_status())
        .map_err(|e| format!("Installer download failed: {e}"))?;

    let dest = std::env::temp_dir().join(sanitize_installer_name(file_name));
    {
        let mut file = std::fs::File::create(&dest)
            .map_err(|e| format!("Cannot write installer to {}: {e}", dest.display()))?;
        if let Err(e) = std::io::copy(&mut response, &mut file) {
            drop(file);
            let _ = std::fs::remove_file(&dest);
            return Err(format!("Installer download failed: {e}"));
        }
    }

    verify_sha256(&dest, expected_sha256)?;
    Ok(dest)
}

#[tauri::command]
async fn install_update(
    app: AppHandle,
    download_url: String,
    sha256: String,
    file_name: String,
) -> Result<(), String> {
    if !cfg!(windows) {
        return Err("In-app updating is only supported on Windows.".to_string());
    }
    let expected = sha256.trim().to_lowercase();
    if expected.len() != 64 || !expected.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("Installer integrity digest is missing or invalid.".to_string());
    }

    // Claim the update gate before the download: overlapping calls write the
    // same temp installer path, so a second call must be rejected before it
    // can truncate the file the first call already verified. `stopping` stays
    // set through the installer launch so no second installer can spawn; it
    // is released on every early-return failure path below.
    let stopping = app.state::<SidecarState>().stopping.clone();
    if stopping
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("An update is already in progress.".to_string());
    }

    let installer = match tauri::async_runtime::spawn_blocking(move || {
        download_installer(&download_url, &file_name, &expected)
    })
    .await
    {
        Ok(Ok(installer)) => installer,
        Ok(Err(e)) => {
            stopping.store(false, Ordering::SeqCst);
            return Err(e);
        }
        Err(e) => {
            stopping.store(false, Ordering::SeqCst);
            return Err(format!("Installer download failed: {e}"));
        }
    };

    let port = *app.state::<u16>();

    // Stop the sidecar and wait for its Terminated event so the installer
    // never writes over a still-locked openchat-server.exe.
    let terminated = match stop_sidecar(&app.state::<SidecarState>()) {
        Ok(terminated) => terminated,
        Err(e) => {
            stopping.store(false, Ordering::SeqCst);
            return Err(e);
        }
    };
    if let Some(terminated) = terminated {
        match tauri::async_runtime::spawn_blocking(move || {
            match terminated.recv_timeout(Duration::from_secs(5)) {
                Ok(()) => Ok(()),
                Err(_) => Err(terminated),
            }
        })
        .await
        {
            Ok(Ok(())) => {}
            Ok(Err(rx)) => {
                // Keep update attempts blocked until the old process is
                // confirmed dead, then restore the backend so the app stays
                // usable. No upper bound here — kill() already succeeded, so
                // the process will exit eventually; a second timeout would
                // leave updates blocked forever even after it does.
                let app = app.clone();
                let stopping = stopping.clone();
                tauri::async_runtime::spawn(async move {
                    let dead = tauri::async_runtime::spawn_blocking(move || rx.recv().is_ok())
                        .await
                        .unwrap_or(false);
                    if dead {
                        // Hold the gate through the respawn so a retry can't
                        // launch an installer while the backend is coming back
                        // up, then clear it regardless — the killed process is
                        // confirmed gone either way.
                        let restored = restore_sidecar(&app, port);
                        stopping.store(false, Ordering::SeqCst);
                        if let Err(e) = restored {
                            eprintln!("Failed to restart openchat-server: {e}");
                        }
                    }
                });
                return Err(
                    "The backend is still stopping — the update was cancelled. Try again in a moment."
                        .to_string(),
                );
            }
            Err(_) => {
                stopping.store(false, Ordering::SeqCst);
                return Err(
                    "Could not confirm the backend stopped — restart OpenChat before retrying."
                        .to_string(),
                );
            }
        }
    }

    // Same flags the Tauri updater plugin passes to its NSIS installer:
    // `/S` runs fully silently (no setup UI), `/UPDATE` installs over the
    // existing install so no uninstall/reinstall prompt appears (shortcuts,
    // registry entries, and app data are preserved), and `/R` relaunches the
    // app when the install finishes. `/ARGS` clears any stray argument the
    // installer would otherwise forward to the relaunched app.
    if let Err(e) = std::process::Command::new(&installer)
        .args(["/S", "/UPDATE", "/R", "/ARGS", ""])
        .spawn()
    {
        // The sidecar is already stopped — bring the backend back so the app
        // stays usable. The gate stays held through the respawn so a retry
        // can't race the backend coming back up, then clears regardless —
        // termination is confirmed either way.
        let error = match restore_sidecar(&app, port) {
            Ok(()) => format!("Failed to launch the installer: {e}"),
            Err(restore_error) => {
                eprintln!("Failed to restart openchat-server: {restore_error}");
                format!(
                    "Failed to launch the installer: {e}; failed to restore the backend: {restore_error}"
                )
            }
        };
        stopping.store(false, Ordering::SeqCst);
        return Err(error);
    }

    // Give the IPC response a moment to reach the webview before quitting; the
    // NSIS installer takes over from there and relaunches the app.
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(400));
        app.exit(0);
    });
    Ok(())
}

fn spawn_sidecar(
    app: &AppHandle,
    port: u16,
    exit_listener: ExitListener,
) -> Result<(CommandChild, Arc<AtomicBool>), String> {
    let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;
    let port_arg = port.to_string();
    let data_dir_arg = data_dir.to_string_lossy().into_owned();
    let parent_pid_arg = std::process::id().to_string();

    let sidecar = app
        .shell()
        .sidecar("openchat-server")
        .map_err(|e| e.to_string())?
        .args([
            "--port",
            &port_arg,
            "--data-dir",
            &data_dir_arg,
            "--parent-pid",
            &parent_pid_arg,
        ])
        .spawn()
        .map_err(|e| e.to_string())?;
    let (mut events, child) = sidecar;
    let terminated_flag = Arc::new(AtomicBool::new(false));
    let drain_flag = terminated_flag.clone();

    tauri::async_runtime::spawn(async move {
        while let Some(event) = events.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => eprint!("{}", String::from_utf8_lossy(&bytes)),
                CommandEvent::Stderr(bytes) => eprint!("{}", String::from_utf8_lossy(&bytes)),
                CommandEvent::Error(error) => eprintln!("openchat-server error: {error}"),
                CommandEvent::Terminated(payload) => {
                    eprintln!("openchat-server exited: {payload:?}");
                    drain_flag.store(true, Ordering::SeqCst);
                    if let Ok(mut listener) = exit_listener.lock() {
                        if let Some(tx) = listener.take() {
                            let _ = tx.send(());
                        }
                    }
                }
                _ => {}
            }
        }
    });

    Ok((child, terminated_flag))
}

fn restore_sidecar(app: &AppHandle, port: u16) -> Result<(), String> {
    let listener = app.state::<SidecarState>().exit_listener.clone();
    let spawned = spawn_sidecar(app, port, listener)?;
    let state = app.state::<SidecarState>();
    let mut slot = state
        .child
        .lock()
        .map_err(|_| "Sidecar state is unavailable.".to_string())?;
    *slot = Some(spawned);
    Ok(())
}

// Kills the sidecar and returns a receiver that fires once the process has
// actually terminated (its Terminated command event). Returns None when the
// exit is already confirmed or no sidecar is running. The caller must already
// hold the `stopping` gate — this function does not claim or release it.
fn stop_sidecar(state: &SidecarState) -> Result<Option<mpsc::Receiver<()>>, String> {
    let mut slot = state
        .child
        .lock()
        .map_err(|_| "Sidecar state is unavailable.".to_string())?;
    let Some((child, terminated)) = slot.take() else {
        // An earlier attempt can lose the handle of a child whose kill was
        // never confirmed; an empty slot alone does not prove the backend
        // exited. Proceed only once that orphan's Terminated flag was seen.
        if let Ok(mut lost) = state.lost_child_termination.lock() {
            match lost.as_ref() {
                Some(flag) if flag.load(Ordering::SeqCst) => {
                    lost.take();
                }
                Some(_) => {
                    return Err(
                        "The backend may still be running — restart OpenChat before updating."
                            .to_string(),
                    );
                }
                None => {}
            }
        }
        return Ok(None);
    };
    let (tx, rx) = mpsc::channel::<()>();
    if let Ok(mut listener) = state.exit_listener.lock() {
        *listener = Some(tx);
    } else {
        // The slot was already emptied — put the running sidecar back so the
        // next attempt still finds (and kills) it instead of seeing no child.
        *slot = Some((child, terminated));
        return Err("Sidecar state is unavailable.".to_string());
    }
    let kill_failed = child.kill().is_err();
    // The drain task flips `terminated` on every Terminated — including an
    // exit that already happened, where kill() can still succeed but no event
    // is left to fire the receiver. When the flag is set there is nothing to
    // wait for. A dead process whose event is still in flight gets a brief
    // window to deliver it first.
    let confirmed = terminated.load(Ordering::SeqCst)
        || (kill_failed
            && (rx.recv_timeout(Duration::from_millis(500)).is_ok()
                || terminated.load(Ordering::SeqCst)));
    if confirmed {
        if let Ok(mut listener) = state.exit_listener.lock() {
            listener.take();
        }
        return Ok(None);
    }
    if kill_failed {
        // A live process that wouldn't take the kill emits no Terminated
        // event, so nothing would ever fire the receiver. kill() consumed
        // the handle, so it can't be handed back — keep its termination
        // flag so a retry can confirm the exit instead of assuming the
        // backend is gone. The caller releases the update gate.
        if let Ok(mut lost) = state.lost_child_termination.lock() {
            *lost = Some(terminated);
        }
        if let Ok(mut listener) = state.exit_listener.lock() {
            listener.take();
        }
        return Err(
            "Could not stop the backend — restart OpenChat before retrying.".to_string(),
        );
    }
    Ok(Some(rx))
}

fn kill_sidecar(state: &SidecarState) {
    if let Ok(mut child) = state.child.lock() {
        if let Some((child, _)) = child.take() {
            let _ = child.kill();
        }
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![api_base, install_update])
        .setup(|app| {
            let port = free_port()?;
            let exit_listener: ExitListener = Arc::new(Mutex::new(None));
            app.manage(SidecarState {
                child: Mutex::new(None),
                exit_listener: exit_listener.clone(),
                stopping: Arc::new(AtomicBool::new(false)),
                lost_child_termination: Mutex::new(None),
            });

            let spawned = spawn_sidecar(&app.handle(), port, exit_listener)?;
            if let Ok(mut slot) = app.state::<SidecarState>().child.lock() {
                *slot = Some(spawned);
            }

            let window = app.get_webview_window("main");
            thread::spawn(move || {
                let deadline = std::time::Instant::now() + Duration::from_secs(20);
                while std::time::Instant::now() < deadline {
                    if health_check(port) {
                        if let Some(window) = window {
                            let _ = window.show();
                        }
                        return;
                    }
                    thread::sleep(Duration::from_millis(250));
                }
                eprintln!("openchat-server did not become healthy before timeout");
                if let Some(window) = window {
                    let _ = window.show();
                }
            });

            app.manage(port);
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while running OpenChat")
        .run(|app, event| {
            if matches!(event, RunEvent::Exit | RunEvent::ExitRequested { .. }) {
                kill_sidecar(&app.state::<SidecarState>());
            }
        });
}
