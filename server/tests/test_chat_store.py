import pytest

from app.chat_store import (
    WorkflowNameExistsError,
    delete_chat_record,
    delete_workflow_record,
    get_chat_record,
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
