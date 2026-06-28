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
- Source model failures are isolated and surfaced per model.
- Streamed server events update the UI in sequence: source outputs -> judge analysis -> final synthesis.
- Auto-Optimize now performs a pre-send prompt rewrite pass through `openai/gpt-oss-20b` with reasoning disabled, then pauses for user confirmation.
- After optimization, ✓ accepts and sends the rewritten prompt immediately; ✕ restores the original prompt, disables Auto-Optimize highlight, and waits for a new Send click.
- Optional Personas can be enabled from the settings popover. When enabled, OpenChat first calls `google/gemma-4-31b-it` (low reasoning, no tools) to assign a unique persona per selected source model based on the current question.
- Before sending, generated personas are shown in editable text boxes (one per source model). The same ✓ / ✕ controls used by Auto-Optimize now gate both prompt/persona review.
- ✓ sends the reviewed prompt and reviewed personas; ✕ immediately sends the original prompt with personas disabled for that run.
- Persona assignments are shown in the UI per source model, streamed as part of the run, and persisted in chat history records.
- Saved workflows now store whether Personas are enabled, so workflow replays preserve the same setting.
- Chat history is now persisted in a local SQLite database (`server/openchat_history.db` by default via `OPENCHAT_HISTORY_DB_PATH`).
- The sidebar History button opens a dedicated history page where each saved chat can be reopened (prompt, source outputs, debate critique, and fusion answer) or deleted.
- If history is not appearing, ensure the backend process has write access to the `server/` directory and verify `OPENCHAT_HISTORY_DB_PATH` points to a valid location.
