"""
환경변수 설정 — Pydantic BaseSettings로 중앙 관리
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://220.118.0.205:11434"
    ollama_model_name: str = "gpt-oss:20b"
    ollama_embedding_model: str = "bge-m3"
    redis_url: str = "redis://220.118.0.205:6379/0"
    qdrant_host: str = "220.118.0.205"
    qdrant_port: int = 6333
    cors_origins: list[str] = ["*"]

    model_config = {"env_prefix": "", "case_sensitive": False, "env_file": ".env"}


settings = Settings()
