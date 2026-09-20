"""Pure text helpers: language detection, chunking, normalisation.

No Azure dependencies here, so everything in this module is unit-testable offline.
"""
from __future__ import annotations

import re
import unicodedata

_ARABIC_CHAR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟۔])\s+")

_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS_AND_TATWEEL = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d)[,،٬](?=\d{3})")
_CITATION = re.compile(r"\[\d+\]")


def detect_language(text: str) -> str:
    """Return 'ar' if Arabic-script letters dominate the text, otherwise 'en'."""
    letters = _LETTER.findall(text)
    if not letters:
        return "en"
    arabic = sum(1 for ch in letters if _ARABIC_CHAR.match(ch))
    return "ar" if arabic / len(letters) > 0.3 else "en"


def normalize(text: str) -> str:
    """Normalise text for robust matching in evaluation.

    Maps Arabic-Indic digits to ASCII, drops diacritics/tatweel and [n] citation markers,
    removes thousands separators (1,200 -> 1200) and lower-cases.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ARABIC_INDIC_DIGITS)
    text = _DIACRITICS_AND_TATWEEL.sub("", text)
    text = _CITATION.sub(" ", text)
    text = _THOUSANDS_SEPARATOR.sub("", text)
    return text.lower()


def extract_citation_ids(text: str) -> list[int]:
    """Return the sorted unique source numbers cited as [n] in an answer."""
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries (Latin and Arabic punctuation)."""
    pieces: list[str] = []
    buf = ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:  # a single sentence longer than the limit
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not sentence:
            continue
        candidate = f"{buf} {sentence}" if buf else sentence
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            pieces.append(buf)
            buf = sentence
    if buf:
        pieces.append(buf)
    return pieces


def chunk_text(text: str, max_chars: int = 450, overlap: int = 100) -> list[str]:
    """Chunk text on paragraph/sentence boundaries with paragraph-level overlap.

    Character-based and language-agnostic, so it works for Arabic and English alike.
    Each chunk is at most ``max_chars`` long. Consecutive chunks share trailing
    paragraphs of up to ``overlap`` characters so facts near a boundary keep context.
    """
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("require 0 <= overlap < max_chars")

    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            units.extend(_split_long_paragraph(paragraph, max_chars))

    def size(parts: list[str]) -> int:
        return sum(len(p) for p in parts) + 2 * max(len(parts) - 1, 0)

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and size(current + [unit]) > max_chars:
            chunks.append("\n\n".join(current))
            carry: list[str] = []
            for previous in reversed(current):
                if size([previous] + carry) > overlap:
                    break
                carry.insert(0, previous)
            current = carry if size(carry + [unit]) <= max_chars else []
        current.append(unit)
    if current:
        chunks.append("\n\n".join(current))
    return chunks
