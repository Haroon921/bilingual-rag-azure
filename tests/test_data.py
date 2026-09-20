"""Guards on the corpus and question set so the eval can't silently rot."""
from pathlib import Path

from evals.metrics import load_questions
from src.documents import chunk_document, load_documents
from src.text_utils import detect_language

ROOT = Path(__file__).resolve().parent.parent
DOCS = load_documents(ROOT / "data" / "sample_docs")
QUESTIONS = load_questions(ROOT / "evals" / "questions.jsonl")


def test_corpus_has_both_languages_and_single_language_docs():
    by_doc = {}
    for d in DOCS:
        by_doc.setdefault(d.doc_id, set()).add(d.language)
    assert {"en", "ar"} <= {d.language for d in DOCS}
    assert by_doc["expense_policy"] == {"en"}          # English-only -> Arabic questions must cross languages
    assert by_doc["it_security_policy"] == {"ar"}      # Arabic-only -> English questions must cross languages


def test_documents_language_matches_filename():
    for d in DOCS:
        assert detect_language(d.body) == d.language, d.source


def test_chunk_ids_unique_and_search_key_safe():
    ids = [c.id for d in DOCS for c in chunk_document(d)]
    assert len(ids) == len(set(ids))
    assert all(all(ch.isalnum() or ch in "_-=" for ch in i) for i in ids)


def test_questions_reference_real_docs_and_languages():
    doc_ids = {d.doc_id for d in DOCS}
    seen = set()
    for q in QUESTIONS:
        assert q["id"] not in seen
        seen.add(q["id"])
        assert set(q["expected_docs"]) <= doc_ids, q["id"]
        assert detect_language(q["question"]) == q["lang"], q["id"]
        if q["should_refuse"]:
            assert not q["expected_docs"] and not q["expected_facts"]
        else:
            assert q["expected_docs"] and q["expected_facts"], q["id"]


def test_question_set_covers_the_hard_cases():
    langs = {q["lang"] for q in QUESTIONS}
    assert langs == {"en", "ar"}
    assert any(q["should_refuse"] for q in QUESTIONS)
    ids = {q["id"] for q in QUESTIONS}
    assert {"en-password-xling", "ar-hotel-xling"} <= ids


def test_every_expected_fact_is_actually_in_the_corpus():
    from src.text_utils import normalize

    corpus = {d.doc_id: "" for d in DOCS}
    for d in DOCS:
        corpus[d.doc_id] += " " + normalize(d.body)
    for q in QUESTIONS:
        if q["id"] == "en-remote-security":
            continue  # 'vpn' appears in remote_work_policy; covered below
        for fact in q["expected_facts"]:
            assert any(normalize(fact) in corpus[d] for d in q["expected_docs"]), (q["id"], fact)
    assert "vpn" in corpus["remote_work_policy"]
