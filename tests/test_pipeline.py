"""Exercise RagPipeline.ask() end-to-end with fake Azure clients (no network)."""
from types import SimpleNamespace

from src.config import Settings
from src.prompts import REFUSAL
from src.rag import RagPipeline

SETTINGS = Settings("https://x", "2024-10-21", "", "chat", "embed", 3072, "https://y", "", "idx", 4)

HITS = [
    {"doc_id": "leave_policy", "title": "Leave Policy", "language": "en", "chunk_index": 0,
     "content": "Employees receive 25 working days of paid annual leave.", "source": "leave_policy.en.md"},
    {"doc_id": "expense_policy", "title": "Expenses", "language": "en", "chunk_index": 1,
     "content": "Receipts are required above SAR 100.", "source": "expense_policy.en.md"},
]


class FakeOpenAI:
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.embeddings = SimpleNamespace(create=lambda **kw: SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.0] * 3)]))
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _chat(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=12),
        )


class FakeSearch:
    def __init__(self, hits):
        self.hits, self.kwargs = hits, None

    def search(self, **kw):
        self.kwargs = kw
        return iter(self.hits)


def make(reply, hits):
    p = object.__new__(RagPipeline)
    p.s, p.oai, p.search = SETTINGS, FakeOpenAI(reply), FakeSearch(hits)
    return p


def test_answer_with_citation():
    p = make("You get 25 working days [1].", HITS)
    a = p.ask("How many days of annual leave do I get?")
    assert not a.refused and a.language == "en"
    assert [s.doc_id for s in a.cited] == ["leave_policy"]   # only the cited source is returned
    assert len(a.retrieved) == 2
    assert (a.prompt_tokens, a.completion_tokens) == (120, 12)


def test_hybrid_query_uses_text_and_vector_and_language_fields():
    p = make("x [1]", HITS)
    p.ask("How many days of annual leave do I get?")
    kw = p.search.kwargs
    assert kw["search_text"] and kw["vector_queries"]
    assert {"content_en", "content_ar"} <= set(kw["search_fields"])


def test_model_refusal_is_detected_and_has_no_citations():
    p = make(REFUSAL["ar"], HITS)
    a = p.ask("ما هي سياسة الزيادة السنوية للرواتب؟")
    assert a.refused and a.language == "ar" and a.cited == []


def test_no_retrieval_results_refuses_without_calling_the_llm():
    p = make("should never be used", [])
    a = p.ask("What is the stock option vesting schedule?")
    assert a.refused and a.text == REFUSAL["en"]
    assert p.oai.calls == []


def test_prompt_asks_for_question_language_and_marks_sources_untrusted():
    p = make("ok [1]", HITS)
    p.ask("ما مدة الإجازة السنوية؟")
    system = p.oai.calls[0]["messages"][0]["content"]
    assert "Arabic" in system and "untrusted" in system
    assert REFUSAL["ar"] in system
