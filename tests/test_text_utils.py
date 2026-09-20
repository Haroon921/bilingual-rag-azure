from src.prompts import REFUSAL, is_refusal
from src.text_utils import chunk_text, detect_language, extract_citation_ids, normalize


def test_detect_language():
    assert detect_language("How many days of leave do I get?") == "en"
    assert detect_language("كم يوماً من الإجازة السنوية أستحق؟") == "ar"
    assert detect_language("ما هو الـ VPN المطلوب؟") == "ar"  # mixed, Arabic dominates
    assert detect_language("12345 ???") == "en"  # no letters -> default


def test_normalize_digits_and_separators():
    assert "1200" in normalize("Up to SAR 1,200 per night")
    assert "1200" in normalize("حتى ١٬٢٠٠ ريال")
    assert "25" in normalize("٢٥ يوماً")
    assert normalize("Use the VPN [1][2]").split() == ["use", "the", "vpn"]


def test_normalize_strips_arabic_diacritics():
    assert normalize("إجازةٌ") == normalize("إجازة")


def test_extract_citation_ids():
    assert extract_citation_ids("Yes [2]. Also see [1][2] and [10].") == [1, 2, 10]
    assert extract_citation_ids("no citations") == []


def test_is_refusal_both_languages():
    assert is_refusal(REFUSAL["en"])
    assert is_refusal(REFUSAL["ar"])
    assert is_refusal("  " + REFUSAL["en"].upper() + "  ")
    assert not is_refusal("You get 25 days [1].")


def test_chunks_respect_max_size_english_and_arabic():
    en = "\n\n".join(f"Paragraph {i}. " + "Sentence about policy. " * 6 for i in range(12))
    ar = "\n\n".join(f"فقرة {i}. " + "جملة عن السياسة. " * 8 for i in range(12))
    for text in (en, ar):
        chunks = chunk_text(text, max_chars=300, overlap=80)
        assert len(chunks) > 1
        assert all(0 < len(c) <= 300 for c in chunks)


def test_chunks_cover_all_content_and_overlap():
    paras = [f"Section {i}: " + "x" * 60 for i in range(10)]
    chunks = chunk_text("\n\n".join(paras), max_chars=200, overlap=90)
    joined = "\n".join(chunks)
    assert all(p in joined for p in paras)
    # consecutive chunks share a paragraph (overlap)
    assert any(set(a.split("\n\n")) & set(b.split("\n\n")) for a, b in zip(chunks, chunks[1:]))


def test_long_paragraph_is_split_on_sentences():
    text = " ".join(f"This is sentence number {i}." for i in range(60))
    chunks = chunk_text(text, max_chars=200, overlap=0)
    assert all(len(c) <= 200 for c in chunks)
    assert "sentence number 59." in chunks[-1]


def test_single_giant_token_is_hard_split():
    chunks = chunk_text("a" * 1000, max_chars=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks) and sum(len(c) for c in chunks) == 1000


def test_invalid_overlap_rejected():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("text", max_chars=100, overlap=100)
