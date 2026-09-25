"""API-key persistence across app-data resets (Windows Credential Manager).

The mirrored credential is what restores the key after a reinstall wipes the
app-data directory. These tests exercise the wiring; the wincred calls
themselves are no-ops off Windows.
"""

import app.main as main_module
from app.secrets_store import (
    delete_credential,
    read_credential,
    write_credential,
)


class TestSettingsCredentialMirror:
    def test_save_mirrors_key_to_credential_store(self, client, monkeypatch):
        writes = []
        monkeypatch.setattr(main_module, "write_credential", writes.append)
        monkeypatch.setattr(
            main_module, "delete_credential", lambda: writes.append(None)
        )
        try:
            response = client.put(
                "/api/settings", json={"api_key": "sk-or-test-key"}
            )
            assert response.status_code == 200
            assert response.json()["api_key_configured"] is True
            assert writes == ["sk-or-test-key"]
        finally:
            client.put("/api/settings", json={"api_key": ""})

    def test_clear_deletes_mirrored_credential(self, client, monkeypatch):
        deletes = []
        monkeypatch.setattr(main_module, "write_credential", lambda v: None)
        monkeypatch.setattr(
            main_module, "delete_credential", lambda: deletes.append(True)
        )
        client.put("/api/settings", json={"api_key": "sk-or-test-key"})
        response = client.put("/api/settings", json={"api_key": ""})
        assert response.status_code == 200
        assert response.json()["api_key_configured"] is False
        assert deletes == [True]

    def test_base_url_only_update_skips_credential_store(
        self, client, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(
            main_module, "write_credential", lambda v: calls.append("write")
        )
        monkeypatch.setattr(
            main_module, "delete_credential", lambda: calls.append("delete")
        )
        try:
            response = client.put(
                "/api/settings", json={"base_url": "https://example.com/v1"}
            )
            assert response.status_code == 200
            assert calls == []
        finally:
            client.put("/api/settings", json={"base_url": ""})


class TestCredentialFallback:
    def test_reload_restores_key_from_credential_store(
        self, tmp_path, monkeypatch
    ):
        import app.config as config

        monkeypatch.setattr(
            config, "read_credential", lambda: "sk-or-restored"
        )
        # Fresh data dir with no .env and no inherited key env var.
        monkeypatch.setenv("OPENCHAT_DATA_DIR", str(tmp_path / "fresh"))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config.reload_settings()
        assert config.settings.openai_api_key == "sk-or-restored"
        monkeypatch.undo()
        config.reload_settings()

    def test_env_key_wins_over_credential_store(self, tmp_path, monkeypatch):
        import app.config as config

        monkeypatch.setattr(config, "read_credential", lambda: "sk-or-stale")
        monkeypatch.setenv("OPENCHAT_DATA_DIR", str(tmp_path / "fresh"))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-or-live")
        config.reload_settings()
        assert config.settings.openai_api_key == "sk-or-live"
        monkeypatch.undo()
        config.reload_settings()


class TestCredentialStoreNoOpOffWindows:
    def test_public_helpers_are_safe_on_non_windows(self):
        assert read_credential() == ""
        write_credential("sk-or-test")
        delete_credential()
