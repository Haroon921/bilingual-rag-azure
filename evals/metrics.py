"""Pure evaluation metrics (no Azure dependencies, unit-tested)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from src.text_utils import normalize


def load_questions(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieval_hit(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> bool:
    """True if at least one expected document is among the retrieved chunks."""
    return any(doc_id in set(retrieved_doc_ids) for doc_id in expected_doc_ids)


def facts_present(answer: str, facts: list[str]) -> bool:
    """True if every expected fact appears in the answer, after normalisation."""
    normalized = normalize(answer)
    return all(normalize(fact) in normalized for fact in facts)


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _mean(values: list) -> float | None:
    vals = [int(v) if isinstance(v, bool) else v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if not r["should_refuse"]]
    must_refuse = [r for r in rows if r["should_refuse"]]
    languages = sorted({r["lang"] for r in answerable})
    return {
        "n_answerable": len(answerable),
        "n_should_refuse": len(must_refuse),
        "retrieval_hit_rate": _mean([r["hit"] for r in answerable]),
        "fact_accuracy": _mean([r["facts_ok"] for r in answerable]),
        "fact_accuracy_by_language": {
            lang: _mean([r["facts_ok"] for r in answerable if r["lang"] == lang]) for lang in languages
        },
        "false_refusal_rate": _mean([r["refused"] for r in answerable]),
        "correct_refusal_rate": _mean([r["refused"] for r in must_refuse]),
        "groundedness": _mean([r.get("grounded") for r in answerable]),
        "latency_ms_p50": percentile([r["latency_ms"] for r in rows], 50),
        "latency_ms_p95": percentile([r["latency_ms"] for r in rows], 95),
        "avg_prompt_tokens": _mean([r["prompt_tokens"] for r in rows]),
        "avg_completion_tokens": _mean([r["completion_tokens"] for r in rows]),
    }
