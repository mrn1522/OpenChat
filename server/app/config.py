import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.secrets_store import read_credential, unprotect_secret


class Settings(BaseSettings):
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openchat_timeout_seconds: float = Field(default=60.0)
    openchat_max_parallel_sources: int = Field(default=6)
    app_env: str = Field(default="development")
    cors_allow_origins: str = Field(
        default="http://localhost:5173,http://tauri.localhost,tauri://localhost"
    )
    openchat_history_db_path: str = Field(default="openchat_history.db")
    openchat_data_dir: str = Field(default="")
    openchat_history_limit: str = Field(default="")

    model_config = SettingsConfigDict(
        env_file=None,
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


def _data_dir_from_environment() -> Path | None:
    value = os.environ.get("OPENCHAT_DATA_DIR", "").strip()
    return Path(value).expanduser() if value else None


def _load_settings() -> Settings:
    data_dir = _data_dir_from_environment()
    env_file = (
        (data_dir / ".env")
        if data_dir
        else Path(__file__).resolve().parents[1] / ".env"
    )
    if data_dir:
        data_dir.mkdir(parents=True, exist_ok=True)

    loaded = Settings(_env_file=str(env_file))
    if data_dir:
        loaded.openchat_data_dir = str(data_dir)
        configured_history_path = os.environ.get("OPENCHAT_HISTORY_DB_PATH")
        if not configured_history_path:
            configured_history_path = str(dotenv_values(env_file).get("OPENCHAT_HISTORY_DB_PATH") or "")
        if not configured_history_path:
            loaded.openchat_history_db_path = str(data_dir / "openchat_history.db")
    loaded.openai_api_key = unprotect_secret(loaded.openai_api_key)
    if not loaded.openai_api_key.strip():
        # The OS credential store outlives the settings file, so it restores
        # the key after a reinstall wiped the app data directory.
        loaded.openai_api_key = read_credential()
    return loaded


settings = _load_settings()


def reload_settings() -> Settings:
    """Reload settings from the active environment and env file in place."""
    refreshed = _load_settings()
    settings.__dict__.clear()
    settings.__dict__.update(refreshed.__dict__)
    return settings


def settings_env_file() -> Path:
    data_dir = settings.openchat_data_dir.strip()
    if data_dir:
        path = Path(data_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path / ".env"

    return Path(__file__).resolve().parents[1] / ".env"
