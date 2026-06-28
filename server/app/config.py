from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openchat_timeout_seconds: float = Field(default=60.0)
    openchat_max_parallel_sources: int = Field(default=6)
    app_env: str = Field(default="development")
    cors_allow_origins: str = Field(default="http://localhost:5173")
    openchat_history_db_path: str = Field(default="openchat_history.db")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]


settings = Settings()
