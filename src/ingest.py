"""Create the Azure AI Search index and load the corpus.

    python -m src.ingest --recreate

Index design:
  * content_en / content_ar   lexical (BM25) search with the language-specific Microsoft analyzers
  * content_vector            multilingual embeddings, which is what makes cross-language questions work
  * hybrid queries fuse both rankings with Reciprocal Rank Fusion (RRF)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from .clients import index_client, openai_client, search_client
from .config import Settings, get_settings
from .documents import chunk_document, load_documents


def build_index(s: Settings) -> SearchIndex:
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SimpleField(name="language", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SimpleField(name="source", type=SearchFieldDataType.String),
        SimpleField(name="content", type=SearchFieldDataType.String),  # returned to the app
        SearchableField(name="content_en", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="content_ar", type=SearchFieldDataType.String, analyzer_name="ar.microsoft"),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=s.embedding_dims,
            vector_search_profile_name="hnsw-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")],
    )
    return SearchIndex(name=s.search_index, fields=fields, vector_search=vector_search)


def embed_texts(oai, s: Settings, texts: list[str], batch_size: int = 16) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = oai.embeddings.create(model=s.embedding_deployment, input=texts[i : i + batch_size])
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", default="data/sample_docs", help="folder of <doc_id>.<en|ar>.md files")
    parser.add_argument("--recreate", action="store_true", help="delete and rebuild the index first")
    args = parser.parse_args()

    s = get_settings()
    ic, oai = index_client(s), openai_client(s)

    if args.recreate:
        try:
            ic.delete_index(s.search_index)
            print(f"Deleted index '{s.search_index}'")
        except ResourceNotFoundError:
            pass
    ic.create_or_update_index(build_index(s))

    chunks = [c for doc in load_documents(Path(args.docs)) for c in chunk_document(doc)]
    # Prepend the title so every chunk carries its document context into the embedding.
    vectors = embed_texts(oai, s, [f"{c.title}\n\n{c.text}" for c in chunks])

    records = [
        {
            "id": c.id,
            "doc_id": c.doc_id,
            "title": c.title,
            "language": c.language,
            "chunk_index": c.chunk_index,
            "source": c.source,
            "content": c.text,
            "content_en": c.text if c.language == "en" else None,
            "content_ar": c.text if c.language == "ar" else None,
            "content_vector": vec,
        }
        for c, vec in zip(chunks, vectors)
    ]
    results = search_client(s).upload_documents(documents=records)
    failed = [r.key for r in results if not r.succeeded]
    print(f"Indexed {len(records) - len(failed)}/{len(records)} chunks from {args.docs}")
    if failed:
        raise SystemExit(f"Failed to index: {failed}")


if __name__ == "__main__":
    main()
