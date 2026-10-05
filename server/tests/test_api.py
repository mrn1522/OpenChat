import pytest


class TestHealthAndSettings:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert "env" in body

    def test_settings_shape(self, client):
        response = client.get("/api/settings")
        assert response.status_code == 200
        body = response.json()
        assert body["api_key_configured"] is False
        assert body["api_key_hint"] is None
        assert body["base_url"].startswith("http")


class TestChatsApi:
    def test_empty_list(self, client):
        response = client.get("/api/chats")
        assert response.status_code == 200
        assert response.json() == {"data": []}

    def test_missing_chat_404(self, client):
        assert client.get("/api/chats/nope").status_code == 404
        assert client.delete("/api/chats/nope").status_code == 404


class TestWorkflowsApi:
    def _config(self) -> dict:
        return {"source_models": ["a", "b"], "fusion_model": "f"}

    def test_create_list_delete(self, client):
        created = client.post(
            "/api/workflows",
            json={"name": "Review Loop", "config": self._config()},
        )
        assert created.status_code == 201
        workflow_id = created.json()["workflow_id"]

        listed = client.get("/api/workflows")
        assert listed.status_code == 200
        assert any(w["workflow_id"] == workflow_id for w in listed.json()["data"])

        assert client.delete(f"/api/workflows/{workflow_id}").status_code == 200
        assert client.delete(f"/api/workflows/{workflow_id}").status_code == 404

    def test_duplicate_name_conflict_case_insensitive(self, client):
        first = client.post(
            "/api/workflows", json={"name": "Dup Flow", "config": self._config()}
        )
        assert first.status_code == 201
        second = client.post(
            "/api/workflows", json={"name": "dup flow", "config": self._config()}
        )
        assert second.status_code == 409
        client.delete(f"/api/workflows/{first.json()['workflow_id']}")

    def test_blank_name_rejected(self, client):
        response = client.post(
            "/api/workflows", json={"name": "   ", "config": self._config()}
        )
        assert response.status_code == 422

    def test_invalid_config_rejected(self, client):
        response = client.post(
            "/api/workflows", json={"name": "Bad", "config": {"source_models": []}}
        )
        assert response.status_code == 422


class TestMissingApiKey:
    def test_run_stream_500_without_key(self, client, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "openai_api_key", "")
        response = client.post(
            "/api/run/stream",
            json={
                "prompt": "hi",
                "source_models": ["a"],
                "fusion_model": "f",
            },
        )
        assert response.status_code == 500
        assert "OPENAI_API_KEY" in response.json()["detail"]


class TestDirectChatHistoryReopen:
    def test_binary_attachment_chat_reopens(self, client):
        import base64

        from app import main
        from app.chat_store import save_chat_record
        from app.config import settings
        from app.models import AttachmentInput

        attachments = [
            AttachmentInput(
                name="shot.png", size=4, content_type="image/png", content="QUJD"
            ),
            AttachmentInput(
                name="report.pdf",
                size=12,
                content_type="application/pdf",
                content=base64.b64encode(b"%PDF-1.7 doc").decode(),
            ),
        ]
        save_chat_record(
            settings.openchat_history_db_path,
            chat_id="direct-binary",
            run_id="run-1",
            status="direct_completed",
            elapsed_ms=5,
            kind="direct",
            request_payload={
                "prompt": "Summarize this",
                "source_models": ["m"],
                "fusion_model": "m",
                "debate_mode": "off",
                "attachments": [main._history_attachment_payload(a) for a in attachments],
            },
            source_results=[],
            debate_results=[],
            critique_output="",
            fusion_output="done",
            messages=[
                {"role": "user", "content": "Summarize this"},
                {"role": "assistant", "content": "done", "model": "m"},
            ],
        )

        response = client.get("/api/chats/direct-binary")
        assert response.status_code == 200
        names = [a["name"] for a in response.json()["request"]["attachments"]]
        assert names == ["shot.png", "report.pdf"]
