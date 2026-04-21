"""Thin service layer reusable by CLI, API, and UI."""

import uuid
from pathlib import Path

from loguru import logger

from src.config import settings
from src.indexing import build_chunks, ingest as _run_ingest
from src.learning import (
    generate_flashcards as _generate_flashcards,
    generate_quiz as _generate_quiz,
    summarize as _summarize,
)
from src.rag import answer as _answer, fetch_all_chunks
from src.schemas import FlashcardSet, QuizSet, RagAnswer, Summary
from src.store import ensure_collection, get_vector_store


def ask(
    question: str,
    k: int | None = None,
    filters: dict[str, str | int] | None = None,
) -> RagAnswer:
    """Grounded Q&A over the indexed corpus."""
    return _answer(question, k=k, filters=filters)


def summarize(
    document: str | None = None,
    query: str | None = None,
    filters: dict[str, str | int] | None = None,
    k: int | None = None,
) -> Summary:
    """Produce a grounded summary for a document, topic, or filter."""
    return _summarize(document=document, query=query, filters=filters, k=k)


def quiz(
    document: str | None = None,
    query: str | None = None,
    filters: dict[str, str | int] | None = None,
    count: int | None = None,
    k: int | None = None,
) -> QuizSet:
    """Generate a grounded multiple-choice quiz."""
    return _generate_quiz(
        document=document, query=query, filters=filters, count=count, k=k
    )


def flashcards(
    document: str | None = None,
    query: str | None = None,
    filters: dict[str, str | int] | None = None,
    count: int | None = None,
    k: int | None = None,
) -> FlashcardSet:
    """Generate a grounded flashcard set for spaced repetition."""
    return _generate_flashcards(
        document=document, query=query, filters=filters, count=count, k=k
    )


def list_documents() -> list[dict]:
    """Return indexed documents with filename, document_id, pages, chunk counts."""
    chunks = fetch_all_chunks(filters=None)
    by_doc: dict[str, dict] = {}
    for c in chunks:
        meta = c.metadata
        info = by_doc.setdefault(
            meta.filename,
            {
                "filename": meta.filename,
                "document_id": meta.document_id,
                "pages": set(),
                "chunk_count": 0,
            },
        )
        info["pages"].add(meta.page)
        info["chunk_count"] += 1

    docs = []
    for info in by_doc.values():
        pages = sorted(info["pages"])
        docs.append(
            {
                "filename": info["filename"],
                "document_id": info["document_id"],
                "pages": pages,
                "page_count": len(pages),
                "chunk_count": info["chunk_count"],
            }
        )
    return sorted(docs, key=lambda d: d["filename"])


def save_and_ingest_pdf(file_bytes: bytes, filename: str) -> dict:
    """Persist an uploaded PDF under data_dir and index it into Qdrant.

    Returns basic metadata: the saved filename and number of chunks indexed.
    Raises ValueError if the filename is not a PDF or the payload is empty.
    """
    if not filename:
        raise ValueError("Filename is required.")
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are accepted.")
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    safe_name = Path(filename).name
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.data_dir / safe_name
    dest.write_bytes(file_bytes)
    logger.info("Saved uploaded PDF: {}", dest)

    ensure_collection(recreate=False)
    chunks = build_chunks([dest])
    if not chunks:
        logger.warning("No chunks produced for uploaded file {}", safe_name)
        return {"filename": safe_name, "chunks_indexed": 0}

    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks
    ]
    store = get_vector_store()
    store.add_documents(chunks, ids=ids)
    logger.info("Indexed {} chunks from {}", len(chunks), safe_name)
    return {"filename": safe_name, "chunks_indexed": len(chunks)}


def ingest_data_dir(recreate: bool = False) -> int:
    """Re-run the full ingestion over every PDF in data_dir."""
    return _run_ingest(recreate=recreate)
