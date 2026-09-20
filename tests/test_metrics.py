from evals.metrics import facts_present, percentile, retrieval_hit, summarize


def test_retrieval_hit():
    assert retrieval_hit(["a", "b"], ["b"])
    assert retrieval_hit(["a"], ["x", "a"])
    assert not retrieval_hit(["a"], ["b"])
    assert not retrieval_hit([], ["a"])


def test_facts_present_is_format_robust():
    assert facts_present("The limit is SAR 1,200 per night [1].", ["1200"])
    assert facts_present("الحد الأقصى ١٬٢٠٠ ريال", ["1200"])
    assert facts_present("Use the company VPN [2]", ["vpn"])
    assert not facts_present("The limit is SAR 800", ["1200"])
    assert not facts_present("You get 25 days", ["25", "30"])


def test_citation_markers_do_not_create_false_matches():
    assert not facts_present("Answer is unknown [2]", ["2"])


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == 5


def _row(**kw):
    base = dict(id="x", lang="en", should_refuse=False, refused=False, hit=True, facts_ok=True,
                latency_ms=100.0, prompt_tokens=10, completion_tokens=5)
    base.update(kw)
    return base


def test_summarize():
    rows = [
        _row(),
        _row(lang="ar", facts_ok=False, hit=False),
        _row(should_refuse=True, refused=True, hit=None, facts_ok=None),
        _row(should_refuse=True, refused=False, hit=None, facts_ok=None),
        _row(refused=True, facts_ok=False),
    ]
    s = summarize(rows)
    assert s["n_answerable"] == 3 and s["n_should_refuse"] == 2
    assert s["correct_refusal_rate"] == 0.5
    assert abs(s["false_refusal_rate"] - 1 / 3) < 1e-9
    assert abs(s["fact_accuracy"] - 1 / 3) < 1e-9
    assert s["fact_accuracy_by_language"]["ar"] == 0.0
    assert s["groundedness"] is None  # no judge rows


def test_summarize_handles_empty_groups():
    s = summarize([_row(should_refuse=True, refused=True, hit=None, facts_ok=None)])
    assert s["fact_accuracy"] is None and s["correct_refusal_rate"] == 1.0
