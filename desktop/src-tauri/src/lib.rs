use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::Mutex,
    thread,
    time::Duration,
};

use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager, RunEvent, State};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

struct SidecarState(Mutex<Option<CommandChild>>);

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
    let installer = tauri::async_runtime::spawn_blocking(move || {
        download_installer(&download_url, &file_name, &expected)
    })
    .await
    .map_err(|e| format!("Installer download failed: {e}"))??;

    std::process::Command::new(&installer)
        .spawn()
        .map_err(|e| format!("Failed to launch the installer: {e}"))?;

    // Give the IPC response a moment to reach the webview before quitting; the
    // NSIS installer takes over from there and relaunches the app.
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(400));
        app.exit(0);
    });
    Ok(())
}

fn kill_sidecar(state: &SidecarState) {
    if let Ok(mut child) = state.0.lock() {
        if let Some(child) = child.take() {
            let _ = child.kill();
        }
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![api_base, install_update])
        .setup(|app| {
            let data_dir: PathBuf = app.path().app_data_dir()?;
            std::fs::create_dir_all(&data_dir)?;
            let port = free_port()?;
            let port_arg = port.to_string();
            let data_dir_arg = data_dir.to_string_lossy().into_owned();
            let parent_pid_arg = std::process::id().to_string();

            let sidecar = app
                .shell()
                .sidecar("openchat-server")?
                .args([
                    "--port",
                    &port_arg,
                    "--data-dir",
                    &data_dir_arg,
                    "--parent-pid",
                    &parent_pid_arg,
                ])
                .spawn()?;
            let (mut events, child) = sidecar;
            if let Ok(mut state) = app.state::<SidecarState>().0.lock() {
                *state = Some(child);
            }

            tauri::async_runtime::spawn(async move {
                while let Some(event) = events.recv().await {
                    match event {
                        CommandEvent::Stdout(bytes) => eprint!("{}", String::from_utf8_lossy(&bytes)),
                        CommandEvent::Stderr(bytes) => eprint!("{}", String::from_utf8_lossy(&bytes)),
                        CommandEvent::Error(error) => eprintln!("openchat-server error: {error}"),
                        CommandEvent::Terminated(payload) => eprintln!("openchat-server exited: {payload:?}"),
                        _ => {}
                    }
                }
            });

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
