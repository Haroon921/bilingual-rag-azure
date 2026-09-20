"""Run the evaluation set against the live pipeline.

    python -m evals.run_eval            # deterministic checks
    python -m evals.run_eval --judge    # + LLM-as-judge groundedness

Metrics:
  retrieval_hit_rate    an expected document was retrieved
  fact_accuracy         the answer contains the expected key facts (digit/format-normalised)
  false_refusal_rate    answerable questions that were wrongly refused
  correct_refusal_rate  out-of-scope questions that were correctly refused
  groundedness          (--judge) every claim is supported by the retrieved sources
"""
from __future__ import annotations

import argparse
import json

from src.rag import Answer, RagPipeline

from .metrics import facts_present, load_questions, retrieval_hit, summarize

JUDGE_PROMPT = """You are a strict grader of retrieval-augmented answers.
Given SOURCES and an ANSWER, decide whether EVERY factual claim in the ANSWER is supported by the SOURCES.
Judge the facts only, not the language the answer is written in.
Reply with JSON only: {"grounded": true|false, "reason": "<one short sentence>"}"""


def judge_groundedness(pipe: RagPipeline, answer: Answer) -> bool:
    sources = "\n\n".join(f"[{s.n}] {s.content}" for s in answer.retrieved)
    resp = pipe.oai.chat.completions.create(
        model=pipe.s.chat_deployment,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"SOURCES:\n{sources}\n\nANSWER:\n{answer.text}"},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        return bool(json.loads(resp.choices[0].message.content)["grounded"])
    except (ValueError, KeyError, TypeError):
        return False


def fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if isinstance(value, float) and value <= 1 else f"{value:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default="evals/questions.jsonl")
    parser.add_argument("--out", default="evals/results.json")
    parser.add_argument("--judge", action="store_true", help="add LLM-as-judge groundedness")
    args = parser.parse_args()

    pipe = RagPipeline()
    rows = []
    for q in load_questions(args.questions):
        ans = pipe.ask(q["question"])
        row = {
            "id": q["id"],
            "lang": q["lang"],
            "question": q["question"],
            "should_refuse": q["should_refuse"],
            "refused": ans.refused,
            "hit": retrieval_hit([s.doc_id for s in ans.retrieved], q["expected_docs"])
            if q["expected_docs"] else None,
            "facts_ok": facts_present(ans.text, q["expected_facts"]) if q["expected_facts"] else None,
            "answer": ans.text,
            "cited_docs": sorted({s.doc_id for s in ans.cited}),
            "latency_ms": ans.latency_ms,
            "prompt_tokens": ans.prompt_tokens,
            "completion_tokens": ans.completion_tokens,
        }
        if args.judge and not q["should_refuse"] and not ans.refused:
            row["grounded"] = judge_groundedness(pipe, ans)
        rows.append(row)
        status = "refused" if ans.refused else ("ok" if row["facts_ok"] else "CHECK")
        print(f"  {q['id']:<22} {status}")

    summary = summarize(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    for key, value in summary.items():
        if isinstance(value, dict):
            value = ", ".join(f"{k}={fmt(v)}" for k, v in value.items())
            print(f"{key:<28} {value}")
        else:
            print(f"{key:<28} {fmt(value)}")
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
