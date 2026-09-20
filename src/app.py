"""Minimal HTTP API.   uvicorn src.app:app --reload

Demo only: no authentication. See README, "Production considerations".
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Bilingual enterprise RAG (Arabic + English)")
_pipeline = None


def pipeline():
    global _pipeline
    if _pipeline is None:  # lazy, so /health works without Azure configured
        from .rag import RagPipeline

        _pipeline = RagPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    a = pipeline().ask(req.question)
    return {
        "answer": a.text,
        "language": a.language,
        "refused": a.refused,
        "citations": [
            {"n": s.n, "title": s.title, "source": s.source, "excerpt": s.content[:200]} for s in a.cited
        ],
        "latency_ms": round(a.latency_ms),
        "tokens": {"prompt": a.prompt_tokens, "completion": a.completion_tokens},
    }
