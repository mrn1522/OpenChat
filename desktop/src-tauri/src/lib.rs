use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    sync::Mutex,
    thread,
    time::Duration,
};

use tauri::{Manager, RunEvent, State};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

struct SidecarState(Mutex<Option<CommandChild>>);

#[tauri::command]
fn api_base(port: State<'_, u16>) -> String {
    format!("http://127.0.0.1:{}", *port)
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
        .invoke_handler(tauri::generate_handler![api_base])
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
