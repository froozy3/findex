from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration from the environment / `.env`:  INDEX_PATH=index.pkl uvicorn ..."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    index_path: Path = Path("index.pkl")
    embeddings_path: Path | None = None  # set to enable semantic and hybrid modes
    default_page_size: int = 10
    max_page_size: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
