"""Azure client factories. Keyless (Microsoft Entra ID) by default; API keys are an optional fallback."""
from __future__ import annotations

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from openai import AzureOpenAI

from .config import Settings


def openai_client(s: Settings) -> AzureOpenAI:
    if s.aoai_api_key:
        return AzureOpenAI(
            azure_endpoint=s.aoai_endpoint, api_version=s.aoai_api_version, api_key=s.aoai_api_key
        )
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureOpenAI(
        azure_endpoint=s.aoai_endpoint,
        api_version=s.aoai_api_version,
        azure_ad_token_provider=token_provider,
    )


def _search_credential(s: Settings):
    return AzureKeyCredential(s.search_api_key) if s.search_api_key else DefaultAzureCredential()


def search_client(s: Settings) -> SearchClient:
    return SearchClient(s.search_endpoint, s.search_index, _search_credential(s))


def index_client(s: Settings) -> SearchIndexClient:
    return SearchIndexClient(s.search_endpoint, _search_credential(s))
