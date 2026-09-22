# OpenChat (MoA: Mixture of Agents)

OpenChat is a sleek native chat interface for multi-LLM orchestration:

1. Send one prompt to multiple selected source models in parallel.
2. Run a judge model that produces structured pre-synthesis analysis:
   - Agreement
   - Key Differences
   - Partial Coverage
   - Unique Insights
   - Blind Spots
3. Run a synthesizer model that produces one final Markdown answer covering all analysis areas.

## Project Structure

- `client/`: React + Vite + TypeScript frontend.
- `server/`: FastAPI proxy that fans out requests and streams events.

## Desktop app (Windows and Linux)

Download the latest Windows installer from the [GitHub Releases](../../releases) page. Release tags must use semantic versions such as `v1.2.3`. On first launch, OpenChat opens a **Connect OpenRouter** panel for your API key. The key is stored in the desktop app data directory, not returned to the client or included in logs; on Windows it is written DPAPI-encrypted so the `.env` file is only readable by that Windows user (on Linux the `.env` stays plaintext with `0600` permissions). You can reopen the panel from the **API key** row in the composer settings popover.

OpenChat stores desktop data in the app-data directory produced by Tauri for the `com.openchat.desktop` identifier — `%APPDATA%\com.openchat.desktop` on Windows, `~/.local/share/com.openchat.desktop` on Linux. This includes the local settings `.env` file and chat history database.

Release installers are produced per-OS by CI: the Windows NSIS installer is built by `.github/workflows/desktop-build.yml` on `windows-latest`. Linux builds are supported (the shell compiles and the app runs standalone) but no Linux installer bundle is produced yet — `tauri.conf.json` only declares `nsis` bundle targets, so on Linux `npm run build --prefix desktop` yields the `target/release/openchat-desktop` binary you can run directly. In-app updating is Windows-only; on Linux the updater reports that it is unsupported.

### Building the desktop app locally

On Windows with Node.js 20, Python 3.12, Rust, and the Tauri prerequisites installed:

```powershell
cd OpenChat
python -m venv server\.venv
server\.venv\Scripts\pip install -r server\requirements-desktop.txt
server\.venv\Scripts\pyinstaller server\openchat_server.spec
Copy-Item server\dist\openchat-server.exe desktop\src-tauri\binaries\openchat-server-x86_64-pc-windows-msvc.exe
npm ci --prefix client
npm ci --prefix desktop
npm run build --prefix desktop
```

On Linux the same steps apply with POSIX equivalents (sidecar name: `openchat-server-x86_64-unknown-linux-gnu`):

```bash
cd OpenChat
python -m venv server/.venv-desktop
server/.venv-desktop/bin/pip install -r server/requirements-desktop.txt
server/.venv-desktop/bin/pyinstaller server/openchat_server.spec
cp server/dist/openchat-server desktop/src-tauri/binaries/openchat-server-x86_64-unknown-linux-gnu
npm ci --prefix client
npm ci --prefix desktop
npm run build --prefix desktop   # produces desktop/src-tauri/target/release/openchat-desktop
```

For development, run `npm run dev --prefix desktop`; the Tauri shell starts the client dev server and launches a local FastAPI sidecar. Note that `cargo build` debug binaries load `devUrl` and therefore also need the client dev server running; only release builds embed the frontend.

## Quick Start

### 1) Backend

```bash
cd OpenChat/server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy ..\.env.example .env
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

### 2) Frontend

```bash
cd OpenChat/client
npm install
npm run dev
```

Frontend runs at `http://localhost:5173` and backend at `http://localhost:8000`.

## Notes

- `OPENAI_BASE_URL` defaults to OpenRouter-compatible API format.
- `CORS_ALLOW_ORIGINS` defaults to `http://localhost:5173,http://tauri.localhost,tauri://localhost`.
- Source model failures are isolated and surfaced per model.
- Streamed server events update the UI in sequence: source outputs -> judge analysis -> final synthesis.
- Auto-Optimize now performs a pre-send prompt rewrite pass through `openai/gpt-oss-20b` with reasoning disabled, then pauses for user confirmation.
- After optimization, ✓ accepts and sends the rewritten prompt immediately; ✕ restores the original prompt, disables Auto-Optimize highlight, and waits for a new Send click.
- Optional Personas can be enabled from the settings popover. When enabled, OpenChat first calls `google/gemma-4-31b-it` (low reasoning, no tools) to assign a unique persona per selected source model based on the current question.
- Before sending, generated personas are shown in editable text boxes (one per source model). The same ✓ / ✕ controls used by Auto-Optimize now gate both prompt/persona review.
- ✓ sends the reviewed prompt and reviewed personas; ✕ immediately sends the original prompt with personas disabled for that run.
- Persona assignments are shown in the UI per source model, streamed as part of the run, and persisted in chat history records.
- Saved workflows now store whether Personas are enabled, so workflow replays preserve the same setting.
- Chat history is persisted in a local SQLite database (`server/openchat_history.db` by default via `OPENCHAT_HISTORY_DB_PATH`) running in WAL mode; the schema is versioned via `PRAGMA user_version` and migrates older history files in place. Conversations are normalized into a transcript table plus per-run source/debate result rows, so direct-chat threads reopen with their full message history.
- History growth is bounded and destructive: conversations beyond the `OPENCHAT_HISTORY_LIMIT` (default 500) most recently active are permanently deleted on the next save.
- The sidebar History button opens a dedicated history page where each saved chat can be reopened (prompt, source outputs, debate critique, and fusion answer) or deleted.
- If history is not appearing, ensure the backend process has write access to the `server/` directory and verify `OPENCHAT_HISTORY_DB_PATH` points to a valid location.
