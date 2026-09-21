import json
import sqlite3

import pytest

from app.chat_store import (
    SCHEMA_VERSION,
    WorkflowNameExistsError,
    delete_chat_record,
    delete_workflow_record,
    get_chat_record,
    get_conversation_kind,
    init_chat_store,
    list_chat_records,
    list_workflow_records,
    save_chat_record,
    save_workflow_record,
)


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "history.db")
    init_chat_store(path)
    return path


def _save_chat(db_path: str, chat_id: str, prompt: str = "p", **overrides) -> None:
    kwargs = {
        "run_id": "run-1",
        "status": "ok",
        "elapsed_ms": 123,
        "request_payload": {
            "prompt": prompt,
            "source_models": ["openai/a"],
            "fusion_model": "openai/f",
        },
        "source_results": [{"model": "openai/a", "content": "x", "status": "ok"}],
        "debate_results": [],
        "critique_output": "",
        "fusion_output": "final",
    }
    kwargs.update(overrides)
    save_chat_record(db_path, chat_id=chat_id, **kwargs)


def _save_direct_turn(db_path: str, chat_id: str, messages: list[dict]) -> None:
    request_messages = [
        {"role": m["role"], "content": m["content"]} for m in messages[:-1]
    ]
    save_chat_record(
        db_path,
        chat_id=chat_id,
        run_id="run-1",
        status="direct_completed",
        elapsed_ms=10,
        request_payload={
            "prompt": request_messages[-1]["content"],
            "source_models": ["openai/d"],
            "fusion_model": "openai/d",
        },
        source_results=[],
        debate_results=[],
        critique_output="",
        fusion_output=messages[-1]["content"],
        kind="direct",
        messages=messages,
    )


class TestInit:
    def test_idempotent(self, tmp_path):
        path = str(tmp_path / "db.sqlite")
        init_chat_store(path)
        init_chat_store(path)  # second call must not fail

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "nested" / "deeper" / "db.sqlite")
        init_chat_store(path)


class TestChatRecords:
    def test_roundtrip(self, db_path):
        _save_chat(db_path, "chat-1", prompt="hello world")
        record = get_chat_record(db_path, "chat-1")
        assert record is not None
        assert record["chat_id"] == "chat-1"
        assert record["run_id"] == "run-1"
        assert record["elapsed_ms"] == 123
        assert record["request"]["prompt"] == "hello world"
        assert record["source_results"][0]["model"] == "openai/a"
        assert record["fusion_output"] == "final"

    def test_upsert_updates_existing(self, db_path):
        _save_chat(db_path, "chat-1", status="running", fusion_output="")
        _save_chat(db_path, "chat-1", status="ok", fusion_output="done")
        record = get_chat_record(db_path, "chat-1")
        assert record["status"] == "ok"
        assert record["fusion_output"] == "done"

    def test_get_missing_returns_none(self, db_path):
        assert get_chat_record(db_path, "nope") is None

    def test_list_prompt_preview_truncates(self, db_path):
        _save_chat(db_path, "chat-long", prompt="x" * 200)
        (record,) = list_chat_records(db_path)
        assert record["prompt_preview"].endswith("…")
        assert len(record["prompt_preview"]) == 161
        assert record["source_models"] == ["openai/a"]
        assert record["has_fusion_output"] is True

    def test_list_reports_missing_fusion(self, db_path):
        _save_chat(db_path, "chat-1", fusion_output="  ")
        (record,) = list_chat_records(db_path)
        assert record["has_fusion_output"] is False

    def test_delete(self, db_path):
        _save_chat(db_path, "chat-1")
        assert delete_chat_record(db_path, "chat-1") is True
        assert delete_chat_record(db_path, "chat-1") is False


class TestWorkflowRecords:
    def _config(self) -> dict:
        return {"source_models": ["a"], "fusion_model": "f", "temperature": 0.2}

    def test_save_and_list(self, db_path):
        record = save_workflow_record(
            db_path, workflow_id="wf-1", name="My Flow", config_payload=self._config()
        )
        assert record["workflow_id"] == "wf-1"
        assert record["config"]["fusion_model"] == "f"

        (listed,) = list_workflow_records(db_path)
        assert listed["name"] == "My Flow"
        assert listed["config"]["source_models"] == ["a"]

    def test_duplicate_name_raises(self, db_path):
        save_workflow_record(
            db_path, workflow_id="wf-1", name="My Flow", config_payload=self._config()
        )
        with pytest.raises(WorkflowNameExistsError):
            save_workflow_record(
                db_path, workflow_id="wf-2", name="my flow", config_payload=self._config()
            )

    def test_delete(self, db_path):
        save_workflow_record(
            db_path, workflow_id="wf-1", name="My Flow", config_payload=self._config()
        )
        assert delete_workflow_record(db_path, "wf-1") is True
        assert delete_workflow_record(db_path, "wf-1") is False


class TestSchemaAndPragmas:
    def test_wal_mode_and_user_version(self, db_path):
        with sqlite3.connect(db_path) as connection:
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            user_version = connection.execute("PRAGMA user_version").fetchone()[0]
        assert journal_mode == "wal"
        assert user_version == SCHEMA_VERSION

    def test_delete_cascades_child_rows(self, db_path):
        _save_direct_turn(
            db_path,
            "conv-1",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        assert delete_chat_record(db_path, "conv-1") is True
        with sqlite3.connect(db_path) as connection:
            orphan_messages = connection.execute(
                "SELECT COUNT(*) FROM conversation_messages WHERE conversation_id = 'conv-1'"
            ).fetchone()[0]
        assert orphan_messages == 0

    def test_foreign_keys_enforced_on_writes(self, db_path):
        with sqlite3.connect(db_path) as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO conversation_messages (conversation_id, seq, role, content, created_at) "
                    "VALUES ('missing', 0, 'user', 'x', 'now')"
                )


class TestDirectTranscripts:
    def test_full_transcript_roundtrip(self, db_path):
        _save_direct_turn(
            db_path,
            "conv-1",
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "a1", "model": "openai/d"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "a2", "model": "openai/d"},
            ],
        )
        record = get_chat_record(db_path, "conv-1")
        assert [m["content"] for m in record["messages"]] == [
            "first",
            "a1",
            "second",
            "a2",
        ]
        assert [m["role"] for m in record["messages"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    def test_upsert_replaces_transcript(self, db_path):
        _save_direct_turn(
            db_path,
            "conv-1",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a1"}],
        )
        _save_direct_turn(
            db_path,
            "conv-1",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "more"},
                {"role": "assistant", "content": "a2"},
            ],
        )
        record = get_chat_record(db_path, "conv-1")
        assert len(record["messages"]) == 4
        assert record["fusion_output"] == "a2"

    def test_messages_empty_for_orchestrated_detail(self, db_path):
        _save_chat(db_path, "chat-1")
        assert get_chat_record(db_path, "chat-1")["messages"] == []

    def test_title_is_first_user_message(self, db_path):
        _save_direct_turn(
            db_path,
            "conv-1",
            [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "second question"},
                {"role": "assistant", "content": "a2"},
            ],
        )
        (record,) = list_chat_records(db_path)
        assert record["prompt_preview"] == "first question"


class TestLegacyMigration:
    def _create_v1_db(self, path: str, user_version: int = 0) -> None:
        with sqlite3.connect(path) as connection:
            connection.execute(f"PRAGMA user_version={user_version}")
            connection.execute(
                """
                CREATE TABLE chats (
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
                INSERT INTO chats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "legacy-1",
                    "2026-01-01T00:00:00+00:00",
                    "run-9",
                    "completed",
                    42,
                    json.dumps(
                        {
                            "prompt": "legacy prompt",
                            "source_models": ["m/a"],
                            "fusion_model": "m/f",
                        }
                    ),
                    json.dumps([{"model": "m/a", "content": "src", "status": "ok"}]),
                    json.dumps(
                        [
                            {
                                "target_model": "m/a",
                                "reviewer_model": "m/b",
                                "content": "debate",
                            }
                        ]
                    ),
                    "critique text",
                    "fused answer",
                ),
            )
            connection.commit()

    def test_v1_rows_migrate(self, tmp_path):
        path = str(tmp_path / "legacy.db")
        self._create_v1_db(path)
        init_chat_store(path)

        record = get_chat_record(path, "legacy-1")
        assert record is not None
        assert record["request"]["prompt"] == "legacy prompt"
        assert record["fusion_output"] == "fused answer"
        assert record["critique_output"] == "critique text"
        assert record["source_results"][0]["model"] == "m/a"
        assert record["debate_results"][0]["reviewer_model"] == "m/b"
        assert record["messages"] == []  # orchestrated kind exposes no transcript

        (summary,) = list_chat_records(path)
        assert summary["prompt_preview"] == "legacy prompt"
        assert summary["has_fusion_output"] is True

        with sqlite3.connect(path) as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='chats'"
                ).fetchone()[0]
                == 0
            )

    def test_user_version_1_db_gets_schema_then_migrates(self, tmp_path):
        path = str(tmp_path / "v1.db")
        self._create_v1_db(path, user_version=1)
        init_chat_store(path)

        record = get_chat_record(path, "legacy-1")
        assert record is not None
        assert record["fusion_output"] == "fused answer"


class TestRetention:
    def test_evicts_oldest_beyond_limit(self, db_path, monkeypatch):
        monkeypatch.setenv("OPENCHAT_HISTORY_LIMIT", "3")
        for index in range(5):
            _save_chat(db_path, f"chat-{index}", prompt=f"p{index}")

        remaining = {row["chat_id"] for row in list_chat_records(db_path, limit=50)}
        assert len(remaining) == 3
        assert "chat-0" not in remaining
        assert "chat-1" not in remaining
        assert "chat-4" in remaining

    def test_invalid_limit_falls_back(self, db_path, monkeypatch):
        monkeypatch.setenv("OPENCHAT_HISTORY_LIMIT", "bogus")
        _save_chat(db_path, "chat-1")
        assert len(list_chat_records(db_path)) == 1

    def test_limit_from_settings_file(self, db_path, monkeypatch):
        from app.config import settings

        monkeypatch.delenv("OPENCHAT_HISTORY_LIMIT", raising=False)
        monkeypatch.setattr(settings, "openchat_history_limit", "2")
        for index in range(3):
            _save_chat(db_path, f"chat-{index}", prompt=f"p{index}")

        remaining = {row["chat_id"] for row in list_chat_records(db_path, limit=50)}
        assert remaining == {"chat-1", "chat-2"}


class TestConversationKinds:
    def test_kind_lookup(self, db_path):
        _save_chat(db_path, "chat-1")
        _save_direct_turn(
            db_path,
            "conv-1",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"}],
        )
        assert get_conversation_kind(db_path, "chat-1") == "orchestrated"
        assert get_conversation_kind(db_path, "conv-1") == "direct"
        assert get_conversation_kind(db_path, "missing") is None

    def test_resumed_thread_sorts_to_top(self, db_path, monkeypatch):
        import time as _time

        _save_direct_turn(
            db_path,
            "conv-1",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"}],
        )
        _time.sleep(0.01)
        for index in range(3):
            _save_chat(db_path, f"chat-{index}", prompt=f"p{index}")
            _time.sleep(0.01)
        _save_direct_turn(
            db_path,
            "conv-1",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "more"},
                {"role": "assistant", "content": "a2"},
            ],
        )

        ids = [row["chat_id"] for row in list_chat_records(db_path, limit=50)]
        assert ids[0] == "conv-1"
