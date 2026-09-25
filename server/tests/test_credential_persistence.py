"""API-key persistence across app-data resets (Windows Credential Manager).

The mirrored credential is what restores the key after a reinstall wipes the
app-data directory. These tests exercise the wiring; the wincred calls
themselves are no-ops off Windows.
"""

import os

import pytest

import app.main as main_module
from app.secrets_store import (
    delete_credential,
    read_credential,
    write_credential,
)


class TestSettingsCredentialMirror:
    def test_save_mirrors_key_to_credential_store(self, client, monkeypatch):
        writes = []
        monkeypatch.setattr(
            main_module,
            "write_credential",
            lambda value: writes.append(value) or True,
        )
        monkeypatch.setattr(
            main_module, "delete_credential", lambda: writes.append(None) or True
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
        monkeypatch.setattr(main_module, "write_credential", lambda v: True)
        monkeypatch.setattr(
            main_module,
            "delete_credential",
            lambda: deletes.append(True) or True,
        )
        client.put("/api/settings", json={"api_key": "sk-or-test-key"})
        response = client.put("/api/settings", json={"api_key": ""})
        assert response.status_code == 200
        assert response.json()["api_key_configured"] is False
        assert deletes == [True]

    def test_clear_surfaces_failed_credential_delete(self, client, monkeypatch):
        """A failed delete must not half-apply: a stale credential would
        resurrect the cleared key on the next settings reload."""
        try:
            response = client.put(
                "/api/settings", json={"api_key": "sk-or-test-key"}
            )
            assert response.status_code == 200

            monkeypatch.setattr(main_module, "delete_credential", lambda: False)
            failed = client.put("/api/settings", json={"api_key": ""})
            assert failed.status_code == 500
            # Nothing was cleared — the stored key is still reported configured.
            assert client.get("/api/settings").json()["api_key_configured"] is True
        finally:
            monkeypatch.undo()
            client.put("/api/settings", json={"api_key": ""})

    def test_save_surfaces_failed_mirror_cleanup(self, client, monkeypatch):
        """When the mirror write fails and the stale credential cannot be
        dropped either, the save must fail before touching .env — otherwise
        the old key would come back after an app-data reset."""
        try:
            response = client.put(
                "/api/settings", json={"api_key": "sk-or-first"}
            )
            assert response.status_code == 200

            monkeypatch.setattr(main_module, "write_credential", lambda v: False)
            monkeypatch.setattr(main_module, "delete_credential", lambda: False)
            failed = client.put(
                "/api/settings", json={"api_key": "sk-or-second"}
            )
            assert failed.status_code == 500
            # The save was rejected atomically: the previous key is still the
            # one in effect.
            assert os.environ["OPENAI_API_KEY"] == "sk-or-first"
            assert client.get("/api/settings").json()["api_key_configured"] is True
        finally:
            monkeypatch.undo()
            client.put("/api/settings", json={"api_key": ""})

    def test_failed_mirror_write_drops_stale_credential(
        self, client, monkeypatch
    ):
        deletes = []
        monkeypatch.setattr(main_module, "write_credential", lambda v: False)
        monkeypatch.setattr(
            main_module,
            "delete_credential",
            lambda: deletes.append(True) or True,
        )
        try:
            response = client.put(
                "/api/settings", json={"api_key": "sk-or-test-key"}
            )
            assert response.status_code == 200
            assert response.json()["api_key_configured"] is True
            assert deletes == [True]
        finally:
            client.put("/api/settings", json={"api_key": ""})

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
        try:
            config.reload_settings()
            assert config.settings.openai_api_key == "sk-or-restored"
        finally:
            monkeypatch.undo()
            config.reload_settings()

    def test_env_key_wins_over_credential_store(self, tmp_path, monkeypatch):
        import app.config as config

        monkeypatch.setattr(config, "read_credential", lambda: "sk-or-stale")
        monkeypatch.setenv("OPENCHAT_DATA_DIR", str(tmp_path / "fresh"))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-or-live")
        try:
            config.reload_settings()
            assert config.settings.openai_api_key == "sk-or-live"
        finally:
            monkeypatch.undo()
            config.reload_settings()


class TestCredentialStoreNoOpOffWindows:
    @pytest.mark.skipif(
        os.name == "nt",
        reason="would read and overwrite the real Windows Credential Manager",
    )
    def test_public_helpers_are_safe_on_non_windows(self):
        assert read_credential() == ""
        assert write_credential("sk-or-test") is True
        assert delete_credential() is True
