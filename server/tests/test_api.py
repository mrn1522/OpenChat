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
