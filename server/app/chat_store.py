"""SQLite persistence for chat history and saved workflows.

Schema v2 splits the old single ``chats`` blob table into a normalized layout:

    conversations          — one row per saved chat (orchestrated run or direct thread)
    conversation_messages  — ordered user/assistant transcript per conversation
    run_source_results     — per-agent source outputs for orchestrated runs
    run_debate_results     — per-reviewer debate entries for orchestrated runs
    workflows              — saved workflow presets

``PRAGMA user_version`` tracks the schema and ``init_chat_store`` migrates
v1 databases (legacy ``chats`` table) forward without data loss.
"""

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
DEFAULT_HISTORY_LIMIT = 500
_TITLE_MAX_CHARS = 500

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        conversation_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL CHECK(kind IN ('orchestrated', 'direct')),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        status TEXT NOT NULL,
        elapsed_ms INTEGER,
        run_id TEXT NOT NULL,
        title TEXT NOT NULL,
        request_json TEXT NOT NULL,
        critique_output TEXT NOT NULL,
        fusion_output TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
    ON conversations(updated_at DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL
            REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
        model TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE(conversation_id, seq)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation
    ON conversation_messages(conversation_id, seq)
    """,
    """
    CREATE TABLE IF NOT EXISTS run_source_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL
            REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        model TEXT NOT NULL,
        agent_id TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL,
        content TEXT NOT NULL,
        persona_json TEXT,
        error TEXT,
        latency_ms INTEGER,
        UNIQUE(conversation_id, seq)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_run_source_results_conversation
    ON run_source_results(conversation_id, seq)
    """,
    """
    CREATE TABLE IF NOT EXISTS run_debate_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL
            REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        seq INTEGER NOT NULL,
        target_model TEXT NOT NULL,
        reviewer_model TEXT NOT NULL,
        target_agent_id TEXT NOT NULL DEFAULT '',
        reviewer_agent_id TEXT NOT NULL DEFAULT '',
        content TEXT NOT NULL,
        UNIQUE(conversation_id, seq)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_run_debate_results_conversation
    ON run_debate_results(conversation_id, seq)
    """,
    """
    CREATE TABLE IF NOT EXISTS workflows (
        workflow_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        name TEXT NOT NULL UNIQUE COLLATE NOCASE,
        config_json TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_workflows_created_at
    ON workflows(created_at DESC)
    """,
)


class WorkflowNameExistsError(Exception):
    """Raised when trying to create a workflow with a duplicate name."""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    # WAL + NORMAL sync is the standard desktop-app profile: readers never block
    # the writer and a crash leaves the journal recoverable. busy_timeout covers
    # the brief overlap between the streaming write and a history-list read.
    try:
        connection.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError as exc:
        # A transient lock can busy-timeout the pragma; later writes still
        # carry busy_timeout protection in whatever journal mode remains.
        if not exc.sqlite_errorname.startswith("SQLITE_BUSY"):
            raise
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _apply_migrations(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    # Statements are all CREATE IF NOT EXISTS — always run them so a partially
    # versioned DB still ends up with the full schema.
    for statement in _SCHEMA_STATEMENTS:
        connection.execute(statement)
    if version < 2:
        _migrate_legacy_chats(connection)
    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def _migrate_legacy_chats(connection: sqlite3.Connection) -> None:
    """Copy rows from the v1 ``chats`` blob table into the normalized schema."""
    if not _table_exists(connection, "chats"):
        return

    rows = connection.execute(
        "SELECT * FROM chats ORDER BY datetime(created_at) ASC"
    ).fetchall()
    for row in rows:
        request_payload = json.loads(row["request_json"])
        source_results = json.loads(row["source_results_json"])
        debate_results = json.loads(row["debate_results_json"])
        kind = "direct" if str(row["status"]).startswith("direct") else "orchestrated"
        prompt = str(request_payload.get("prompt", ""))
        fusion_output = str(row["fusion_output"])

        messages: list[dict[str, str]] = []
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if fusion_output.strip():
            model = str(request_payload.get("fusion_model", ""))
            messages.append({"role": "assistant", "content": fusion_output, "model": model})

        connection.execute(
            """
            INSERT OR REPLACE INTO conversations (
                conversation_id, kind, created_at, updated_at, status,
                elapsed_ms, run_id, title, request_json, critique_output,
                fusion_output
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["chat_id"],
                kind,
                row["created_at"],
                row["created_at"],
                row["status"],
                row["elapsed_ms"],
                row["run_id"],
                _truncate_title(prompt),
                row["request_json"],
                row["critique_output"],
                row["fusion_output"],
            ),
        )
        connection.execute(
            "DELETE FROM conversation_messages WHERE conversation_id = ?",
            (row["chat_id"],),
        )
        _insert_messages(connection, row["chat_id"], messages, row["created_at"])
        connection.execute(
            "DELETE FROM run_source_results WHERE conversation_id = ?",
            (row["chat_id"],),
        )
        _insert_source_results(connection, row["chat_id"], source_results)
        connection.execute(
            "DELETE FROM run_debate_results WHERE conversation_id = ?",
            (row["chat_id"],),
        )
        _insert_debate_results(connection, row["chat_id"], debate_results)

    connection.execute("DROP TABLE chats")
    logger.info("Migrated %d legacy chat rows into schema v2", len(rows))


def init_chat_store(db_path: str) -> None:
    with _connect(db_path) as connection:
        _apply_migrations(connection)
        connection.commit()


def _history_limit() -> int:
    raw = os.environ.get("OPENCHAT_HISTORY_LIMIT", "").strip()
    if not raw:
        # Settings picks up the desktop data-dir .env; read lazily so this
        # module stays importable without the app config loaded.
        try:
            from app.config import settings

            raw = str(settings.openchat_history_limit).strip()
        except Exception:
            raw = ""
    try:
        return max(int(raw), 1)
    except ValueError:
        return DEFAULT_HISTORY_LIMIT


def _prune_conversations(connection: sqlite3.Connection) -> None:
    """Bound history growth by evicting the least-recently-active conversations."""
    cursor = connection.execute(
        """
        DELETE FROM conversations
        WHERE conversation_id IN (
            SELECT conversation_id FROM conversations
            ORDER BY datetime(updated_at) DESC, updated_at DESC, rowid DESC
            LIMIT -1 OFFSET ?
        )
        """,
        (_history_limit(),),
    )
    if cursor.rowcount:
        logger.info("Pruned %d conversations beyond history limit", cursor.rowcount)


def _truncate_title(text: str) -> str:
    return text[:_TITLE_MAX_CHARS]


def _insert_messages(
    connection: sqlite3.Connection,
    conversation_id: str,
    messages: list[dict[str, Any]],
    created_at: str,
) -> None:
    connection.executemany(
        """
        INSERT INTO conversation_messages (
            conversation_id, seq, role, model, content, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                conversation_id,
                seq,
                str(message.get("role", "user")),
                str(message.get("model") or ""),
                str(message.get("content", "")),
                created_at,
            )
            for seq, message in enumerate(messages)
            if str(message.get("content", "")).strip()
        ],
    )


def _insert_source_results(
    connection: sqlite3.Connection,
    conversation_id: str,
    source_results: list[dict[str, Any]],
) -> None:
    connection.executemany(
        """
        INSERT INTO run_source_results (
            conversation_id, seq, model, agent_id, status, content,
            persona_json, error, latency_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                conversation_id,
                seq,
                str(result.get("model", "")),
                str(result.get("agent_id") or ""),
                str(result.get("status", "ok")),
                str(result.get("content", "")),
                json.dumps(result["persona"]) if result.get("persona") else None,
                result.get("error"),
                result.get("latency_ms"),
            )
            for seq, result in enumerate(source_results)
        ],
    )


def _insert_debate_results(
    connection: sqlite3.Connection,
    conversation_id: str,
    debate_results: list[dict[str, Any]],
) -> None:
    connection.executemany(
        """
        INSERT INTO run_debate_results (
            conversation_id, seq, target_model, reviewer_model,
            target_agent_id, reviewer_agent_id, content
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                conversation_id,
                seq,
                str(result.get("target_model", "")),
                str(result.get("reviewer_model", "")),
                str(result.get("target_agent_id") or ""),
                str(result.get("reviewer_agent_id") or ""),
                str(result.get("content", "")),
            )
            for seq, result in enumerate(debate_results)
        ],
    )


def _default_run_messages(
    request_payload: dict[str, Any], fusion_output: str
) -> list[dict[str, str]]:
    prompt = str(request_payload.get("prompt", ""))
    messages: list[dict[str, str]] = []
    if prompt.strip():
        messages.append({"role": "user", "content": prompt})
    if fusion_output.strip():
        messages.append(
            {
                "role": "assistant",
                "content": fusion_output,
                "model": str(request_payload.get("fusion_model", "")),
            }
        )
    return messages


def save_chat_record(
    db_path: str,
    *,
    chat_id: str,
    run_id: str,
    status: str,
    elapsed_ms: int | None,
    request_payload: dict[str, Any],
    source_results: list[dict[str, Any]],
    debate_results: list[dict[str, Any]],
    critique_output: str,
    fusion_output: str,
    kind: str = "orchestrated",
    messages: list[dict[str, Any]] | None = None,
) -> None:
    """Persist a chat turn, replacing any earlier partial state for ``chat_id``.

    ``messages`` is the full ordered transcript to store; when omitted a
    prompt + fusion-output pair is derived from the run payload. Message and
    result rows are deleted and reinserted so repeated saves stay consistent.
    """
    now = _utc_now_iso()
    if messages is None:
        messages = _default_run_messages(request_payload, fusion_output)
    first_user = next(
        (str(m.get("content", "")) for m in messages if m.get("role") == "user"),
        str(request_payload.get("prompt", "")),
    )

    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO conversations (
                conversation_id, kind, created_at, updated_at, status,
                elapsed_ms, run_id, title, request_json, critique_output,
                fusion_output
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                status = excluded.status,
                elapsed_ms = excluded.elapsed_ms,
                run_id = excluded.run_id,
                title = excluded.title,
                request_json = excluded.request_json,
                critique_output = excluded.critique_output,
                fusion_output = excluded.fusion_output
            """,
            (
                chat_id,
                kind,
                now,
                now,
                status,
                elapsed_ms,
                run_id,
                _truncate_title(first_user),
                json.dumps(request_payload),
                critique_output,
                fusion_output,
            ),
        )
        connection.execute(
            "DELETE FROM conversation_messages WHERE conversation_id = ?",
            (chat_id,),
        )
        _insert_messages(connection, chat_id, messages, now)
        connection.execute(
            "DELETE FROM run_source_results WHERE conversation_id = ?",
            (chat_id,),
        )
        _insert_source_results(connection, chat_id, source_results)
        connection.execute(
            "DELETE FROM run_debate_results WHERE conversation_id = ?",
            (chat_id,),
        )
        _insert_debate_results(connection, chat_id, debate_results)
        _prune_conversations(connection)
        connection.commit()


def list_chat_records(db_path: str, *, limit: int = 100) -> list[dict[str, Any]]:
    with _connect(db_path) as connection:
        # fusion_output can be tens of KB per row; the list view only needs to
        # know whether it is non-empty, so compute that in SQL instead of
        # hauling every blob into memory. TRIM's second argument lists every
        # codepoint Python's str.strip() removes (its full str.isspace() set:
        # ASCII whitespace incl. \x1c-\x1f, NEL, NBSP, and the Unicode
        # space separators).
        rows = connection.execute(
            """
            SELECT conversation_id, created_at, updated_at, status, title,
                   request_json,
                   (LENGTH(TRIM(fusion_output, char(9,10,11,12,13,28,29,30,31,32,
                           133,160,5760,8192,8193,8194,8195,8196,8197,8198,8199,
                           8200,8201,8202,8232,8233,8239,8287,12288))) > 0)
                       AS has_fusion_output
            FROM conversations
            ORDER BY datetime(updated_at) DESC, updated_at DESC, rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records: list[dict[str, Any]] = []
    for row in rows:
        request_payload = json.loads(row["request_json"])
        preview = row["title"]
        if len(preview) > 160:
            preview = f"{preview[:160]}…"

        records.append(
            {
                "chat_id": row["conversation_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "status": row["status"],
                "prompt_preview": preview,
                "source_models": request_payload.get("source_models", []),
                "fusion_model": request_payload.get("fusion_model", ""),
                "has_fusion_output": bool(row["has_fusion_output"]),
            }
        )

    return records


def _fetch_messages(
    connection: sqlite3.Connection, conversation_id: str
) -> list[dict[str, str]]:
    rows = connection.execute(
        """
        SELECT role, content
        FROM conversation_messages
        WHERE conversation_id = ?
        ORDER BY seq ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def _fetch_source_results(
    connection: sqlite3.Connection, conversation_id: str
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT model, agent_id, status, content, persona_json, error, latency_ms
        FROM run_source_results
        WHERE conversation_id = ?
        ORDER BY seq ASC
        """,
        (conversation_id,),
    ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        results.append(
            {
                "model": row["model"],
                "agent_id": row["agent_id"],
                "status": row["status"],
                "content": row["content"],
                "persona": json.loads(row["persona_json"]) if row["persona_json"] else None,
                "error": row["error"],
                "latency_ms": row["latency_ms"],
            }
        )
    return results


def _fetch_debate_results(
    connection: sqlite3.Connection, conversation_id: str
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT target_model, reviewer_model, target_agent_id, reviewer_agent_id,
               content
        FROM run_debate_results
        WHERE conversation_id = ?
        ORDER BY seq ASC
        """,
        (conversation_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_chat_record(db_path: str, chat_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT conversation_id, kind, created_at, run_id, status, elapsed_ms,
                   request_json, critique_output, fusion_output
            FROM conversations
            WHERE conversation_id = ?
            """,
            (chat_id,),
        ).fetchone()
        if row is None:
            return None
        messages = (
            _fetch_messages(connection, chat_id) if row["kind"] == "direct" else []
        )
        source_results = _fetch_source_results(connection, chat_id)
        debate_results = _fetch_debate_results(connection, chat_id)

    return {
        "chat_id": row["conversation_id"],
        "created_at": row["created_at"],
        "run_id": row["run_id"],
        "status": row["status"],
        "elapsed_ms": row["elapsed_ms"],
        "request": json.loads(row["request_json"]),
        "source_results": source_results,
        "debate_results": debate_results,
        "critique_output": row["critique_output"],
        "fusion_output": row["fusion_output"],
        "messages": messages,
    }


def get_conversation_kind(db_path: str, conversation_id: str) -> str | None:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT kind FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    return str(row["kind"]) if row is not None else None


def delete_chat_record(db_path: str, chat_id: str) -> bool:
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM conversations WHERE conversation_id = ?",
            (chat_id,),
        )
        connection.commit()
        return cursor.rowcount > 0


def save_workflow_record(
    db_path: str,
    *,
    workflow_id: str,
    name: str,
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    created_at = _utc_now_iso()

    with _connect(db_path) as connection:
        try:
            connection.execute(
                """
                INSERT INTO workflows (
                    workflow_id,
                    created_at,
                    name,
                    config_json
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    created_at,
                    name,
                    json.dumps(config_payload),
                ),
            )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            if "workflows.name" in str(exc):
                raise WorkflowNameExistsError(name) from exc
            raise

    return {
        "workflow_id": workflow_id,
        "created_at": created_at,
        "name": name,
        "config": config_payload,
    }


def list_workflow_records(db_path: str, *, limit: int = 200) -> list[dict[str, Any]]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT workflow_id, created_at, name, config_json
            FROM workflows
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "workflow_id": row["workflow_id"],
                "created_at": row["created_at"],
                "name": row["name"],
                "config": json.loads(row["config_json"]),
            }
        )

    return records


def delete_workflow_record(db_path: str, workflow_id: str) -> bool:
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        )
        connection.commit()
        return cursor.rowcount > 0
