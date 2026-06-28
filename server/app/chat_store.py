import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class WorkflowNameExistsError(Exception):
    """Raised when trying to create a workflow with a duplicate name."""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def init_chat_store(db_path: str) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                run_id TEXT NOT NULL,
                status TEXT NOT NULL,
                elapsed_ms INTEGER,
                request_json TEXT NOT NULL,
                source_results_json TEXT NOT NULL,
                debate_results_json TEXT NOT NULL,
                critique_output TEXT NOT NULL,
                fusion_output TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chats_created_at
            ON chats(created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                config_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_workflows_created_at
            ON workflows(created_at DESC)
            """
        )
        connection.commit()


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
) -> None:
    created_at = _utc_now_iso()

    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO chats (
                chat_id,
                created_at,
                run_id,
                status,
                elapsed_ms,
                request_json,
                source_results_json,
                debate_results_json,
                critique_output,
                fusion_output
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                run_id = excluded.run_id,
                status = excluded.status,
                elapsed_ms = excluded.elapsed_ms,
                request_json = excluded.request_json,
                source_results_json = excluded.source_results_json,
                debate_results_json = excluded.debate_results_json,
                critique_output = excluded.critique_output,
                fusion_output = excluded.fusion_output
            """,
            (
                chat_id,
                created_at,
                run_id,
                status,
                elapsed_ms,
                json.dumps(request_payload),
                json.dumps(source_results),
                json.dumps(debate_results),
                critique_output,
                fusion_output,
            ),
        )
        connection.commit()


def list_chat_records(db_path: str, *, limit: int = 100) -> list[dict[str, Any]]:
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT chat_id, created_at, status, request_json, fusion_output
            FROM chats
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records: list[dict[str, Any]] = []
    for row in rows:
        request_payload = json.loads(row["request_json"])
        prompt = str(request_payload.get("prompt", ""))
        preview = prompt[:160]
        if len(prompt) > 160:
            preview = f"{preview}…"

        records.append(
            {
                "chat_id": row["chat_id"],
                "created_at": row["created_at"],
                "status": row["status"],
                "prompt_preview": preview,
                "source_models": request_payload.get("source_models", []),
                "fusion_model": request_payload.get("fusion_model", ""),
                "has_fusion_output": bool(str(row["fusion_output"]).strip()),
            }
        )

    return records


def get_chat_record(db_path: str, chat_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                chat_id,
                created_at,
                run_id,
                status,
                elapsed_ms,
                request_json,
                source_results_json,
                debate_results_json,
                critique_output,
                fusion_output
            FROM chats
            WHERE chat_id = ?
            """,
            (chat_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "chat_id": row["chat_id"],
        "created_at": row["created_at"],
        "run_id": row["run_id"],
        "status": row["status"],
        "elapsed_ms": row["elapsed_ms"],
        "request": json.loads(row["request_json"]),
        "source_results": json.loads(row["source_results_json"]),
        "debate_results": json.loads(row["debate_results_json"]),
        "critique_output": row["critique_output"],
        "fusion_output": row["fusion_output"],
    }


def delete_chat_record(db_path: str, chat_id: str) -> bool:
    with _connect(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM chats WHERE chat_id = ?",
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

