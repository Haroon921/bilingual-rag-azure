"""Configuration from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    aoai_endpoint: str
    aoai_api_version: str
    aoai_api_key: str  # empty -> use Microsoft Entra ID (recommended)
    chat_deployment: str
    embedding_deployment: str
    embedding_dims: int
    search_endpoint: str
    search_api_key: str  # empty -> use Microsoft Entra ID (recommended)
    search_index: str
    top_k: int


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        aoai_endpoint=env("AZURE_OPENAI_ENDPOINT", ""),
        aoai_api_version=env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        aoai_api_key=env("AZURE_OPENAI_API_KEY", ""),
        chat_deployment=env("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
        embedding_deployment=env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
        embedding_dims=int(env("EMBEDDING_DIMENSIONS", "3072")),
        search_endpoint=env("AZURE_SEARCH_ENDPOINT", ""),
        search_api_key=env("AZURE_SEARCH_API_KEY", ""),
        search_index=env("AZURE_SEARCH_INDEX", "bilingual-policies"),
        top_k=int(env("TOP_K", "4")),
    )
