"""Retrieval-augmented answering: hybrid search -> grounded, cited answer, or an explicit refusal."""
from __future__ import annotations

import time
from dataclasses import dataclass

from azure.search.documents.models import VectorizedQuery

from .clients import openai_client, search_client
from .config import Settings, get_settings
from .prompts import REFUSAL, build_messages, is_refusal
from .text_utils import detect_language, extract_citation_ids


@dataclass
class Source:
    n: int
    doc_id: str
    title: str
    language: str
    chunk_index: int
    content: str
    source: str


@dataclass
class Answer:
    question: str
    language: str
    text: str
    refused: bool
    retrieved: list[Source]
    cited: list[Source]
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RagPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.oai = openai_client(self.s)
        self.search = search_client(self.s)

    def embed(self, text: str) -> list[float]:
        resp = self.oai.embeddings.create(model=self.s.embedding_deployment, input=[text])
        return resp.data[0].embedding

    def retrieve(self, question: str, k: int | None = None) -> list[Source]:
        """Hybrid retrieval: BM25 over the language-specific fields + vector similarity, fused by RRF."""
        k = k or self.s.top_k
        results = self.search.search(
            search_text=question,
            vector_queries=[
                VectorizedQuery(vector=self.embed(question), k_nearest_neighbors=k, fields="content_vector")
            ],
            search_fields=["title", "content_en", "content_ar"],
            select=["doc_id", "title", "language", "chunk_index", "content", "source"],
            top=k,
        )
        return [
            Source(
                n=i,
                doc_id=r["doc_id"],
                title=r["title"],
                language=r["language"],
                chunk_index=r["chunk_index"],
                content=r["content"],
                source=r["source"],
            )
            for i, r in enumerate(results, start=1)
        ]

    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        language = detect_language(question)
        sources = self.retrieve(question)

        if not sources:  # nothing retrieved: refuse without spending an LLM call
            return Answer(question, language, REFUSAL[language], True, [], [],
                          (time.perf_counter() - started) * 1000)

        # Note: temperature=0 for repeatable answers. Remove it if you deploy a reasoning model.
        resp = self.oai.chat.completions.create(
            model=self.s.chat_deployment,
            messages=build_messages(question, language, sources),
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        refused = is_refusal(text)
        cited_ids = set() if refused else set(extract_citation_ids(text))
        usage = resp.usage
        return Answer(
            question=question,
            language=language,
            text=text,
            refused=refused,
            retrieved=sources,
            cited=[src for src in sources if src.n in cited_ids],
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
