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

## Clicking small UI targets
- The desktop is typically 1600x1200 while the computer tool uses 1024x768 (1.5625 scale); screenshots show real pixels, so eyeballing icon positions can miss small targets like the 34px rail buttons by a few px.
- For precise clicks, query `el.getBoundingClientRect()` via the browser console (returns viewport-relative coords), then convert: `tool_x = (window_x + real_cx) / scale`, `tool_y = (window_y + browser_chrome_height + real_cy) / scale`. Browser chrome ≈ (window height − innerHeight); get window origin and size via `wmctrl -lG` or `xdotool getactivewindow getwindowgeometry`, and the detected scale with `xrandr` instead of hardcoding 1.5625.
