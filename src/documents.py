"""Load the markdown corpus and turn it into chunks. No Azure dependencies.

File naming convention: ``<doc_id>.<lang>.md`` (for example ``leave_policy.ar.md``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .text_utils import chunk_text

SUPPORTED_LANGUAGES = {"en", "ar"}


@dataclass(frozen=True)
class Document:
    doc_id: str
    language: str
    title: str
    body: str
    source: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    language: str
    title: str
    source: str
    chunk_index: int
    text: str


def parse_document(path: Path) -> Document:
    parts = path.stem.rsplit(".", 1)
    if len(parts) != 2 or parts[1] not in SUPPORTED_LANGUAGES:
        raise ValueError(f"{path.name}: expected '<doc_id>.<en|ar>.md'")
    doc_id, language = parts
    body = path.read_text(encoding="utf-8")
    title = next((ln[2:].strip() for ln in body.splitlines() if ln.startswith("# ")), doc_id)
    return Document(doc_id=doc_id, language=language, title=title, body=body, source=path.name)


def load_documents(folder: Path) -> list[Document]:
    return [parse_document(p) for p in sorted(Path(folder).glob("*.md"))]


def chunk_document(doc: Document, max_chars: int = 450, overlap: int = 100) -> list[Chunk]:
    return [
        Chunk(
            id=f"{doc.doc_id}-{doc.language}-{i}",  # valid Azure AI Search key characters
            doc_id=doc.doc_id,
            language=doc.language,
            title=doc.title,
            source=doc.source,
            chunk_index=i,
            text=text,
        )
        for i, text in enumerate(chunk_text(doc.body, max_chars, overlap))
    ]
