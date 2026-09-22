---
name: openchat-settings-testing
description: Exercise first-run connection settings and isolated persistence through the web client on Linux.
---

# OpenChat settings testing

## Devin Secrets Needed
None for configuration persistence tests: use a clearly fake API key. Real model generation requires an authorized OPENAI_API_KEY.

## Local runtime
- Reuse server/.venv-desktop and client/node_modules if available; otherwise create a Python venv from server/requirements.txt and run npm ci in client/. The venv may exist but be empty — if `python -m app` fails with ModuleNotFoundError, run `.venv-desktop/bin/pip install -r requirements-desktop.txt`.
- Create a fresh writable temporary data directory. From server/, run the venv Python with `-m app --port 8765 --data-dir <absolute temporary directory>`.
- Unset inherited OPENAI_API_KEY, OPENAI_BASE_URL, OPENCHAT_DATA_DIR, and OPENCHAT_HISTORY_DB_PATH, and set CORS_ALLOW_ORIGINS=http://localhost:5173 for the backend process to isolate first-run behavior.
- From client/, run `VITE_OPENCHAT_API_BASE=http://127.0.0.1:8765 npm run dev -- --host 0.0.0.0 --port 5173 --strictPort`.
- Navigate to http://localhost:5173, not http://127.0.0.1:5173: the default CORS configuration allows the localhost origin.

## Flow
- Fresh data directory triggers Connect OpenRouter. Empty Save must show validation; no Cancel/close is offered.
- Save a fake key, verify only its masked hint appears in GET /api/settings, and inspect .env inside the temporary data directory.
- Composer sliders button (Open parameters) → API key reopens settings. Leave API key blank to preserve it when updating Base URL.
- Reload the browser and restart the backend with the same directory and clean inherited environment to prove disk persistence.
- Confirm openchat_history.db exists in the data directory and contains chats/workflows tables; no database or .env should be newly created in server/.
- Model listings may load even with a fake key; listing availability does not prove authenticated inference works.
- Linux web testing covers shared UI/backend behavior, not the Tauri command bridge, Windows installer, or sidecar lifecycle.

## Linux desktop app (Tauri webview)
- The `cargo build` debug binary (`desktop/src-tauri/target/debug/openchat-desktop`) runs in dev mode: the webview loads `devUrl` (http://localhost:5173) and shows "Could not connect to localhost" without a Vite dev server. Start `npm run dev` in client/ first (or use `npm run dev --prefix desktop` = `tauri dev`). Only release builds embed `frontendDist`.
- Window starts hidden; it appears when the sidecar `/health` check passes, or after the ~20s timeout (logged as `openchat-server did not become healthy before timeout`). Verify with `wmctrl -l` (title "OpenChat") and `pgrep -af openchat-server` — the sidecar runs as `openchat-server --port <random> --data-dir ~/.local/share/com.openchat.desktop --parent-pid <app-pid>`; PyInstaller onefile shows two PIDs (bootloader + server), both must die with the app.
- Get the live port from the pgrep args (random each launch), then `curl http://127.0.0.1:<port>/health` and `/api/settings` to verify the sidecar API without devtools (the actual Tauri command bridge is covered by the `invoke` check below).
- Sidecar lifecycle: closing the window must kill both sidecar PIDs within ~3s (Tauri kill on Exit + 1s parent-pid watchdog via os.kill(pid,0)). `wmctrl -c OpenChat` closes cleanly; verify with pgrep afterwards.
- First-run is driven by `GET /api/settings` (`api_key_configured`), not by dir existence — deleting `~/.local/share/com.openchat.desktop` fully resets; an existing dir without a saved key still shows first-run.
- `.env` lands in the data dir with 0600 (plaintext on Linux — DPAPI/`secrets_store` is Windows-only); masked hint `first6…last4` appears as the API-key input placeholder and in `api_key_hint`. Nothing may be written under server/.
- Webview devtools: Ctrl+Shift+I opens the WebKitGTK inspector in debug builds; `window.__TAURI_INTERNALS__.invoke('install_update', {downloadUrl:'https://github.com/mrn1522/OpenChat/releases/download/v0.0.0/x.exe', sha256:'<64 hex>', fileName:'x.exe'})` must reject "In-app updating is only supported on Windows." — useful when api.github.com is rate-limited (403) and the UI install button can't render.
- A clearly fake `sk-or-...` key still loads the model catalog (server fetches `/models` unauthenticated) and streams through the whole pipeline; upstream 401 errors are expected and differ from a missing-header failure — cross-check by curling OpenRouter with the same fake key.

## Clicking small UI targets
- The desktop is typically 1600x1200 while the computer tool uses 1024x768 (1.5625 scale); screenshots show real pixels, so eyeballing icon positions can miss small targets like the 34px rail buttons by a few px.
- For precise clicks, query `el.getBoundingClientRect()` via the browser console (returns viewport-relative CSS pixels), then convert: `tool_x = (window_x + real_cx * dpr) / scale`, `tool_y = (window_y + (window_height - innerHeight * dpr) + real_cy * dpr) / scale`, where `dpr` is `window.devicePixelRatio` (browser zoom/HiDPI changes it) and `scale` is the detected desktop→tool scale from `xrandr` (typically 1.5625, not hardcoded). Get window origin/size via `wmctrl -lG` or `xdotool getactivewindow getwindowgeometry`.
