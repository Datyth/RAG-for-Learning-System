"""Load PDFs, split into chunks with metadata, and index into Qdrant."""

import hashlib
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import settings
from src.schemas import ChunkMetadata
from src.store import ensure_collection, get_vector_store


class Chunker(Protocol):
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split page-level documents into chunk-level documents."""


def _splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )


def _document_id(path: Path) -> str:
    raw = f"{path.name}:{path.stat().st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _chunk_id(doc_id: str, page: int, index: int) -> str:
    return f"{doc_id}:{page}:{index}"


def _load_pdf(path: Path) -> list[Document]:
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    doc_id = _document_id(path)
    for doc in pages:
        page_number = int(doc.metadata.get("page", 0)) + 1
        doc.metadata = {
            "document_id": doc_id,
            "filename": path.name,
            "source": str(path.resolve()),
            "page": page_number,
            "section": doc.metadata.get("section"),
        }
    return pages


def discover_pdfs(data_dir: Path | None = None) -> list[Path]:
    directory = data_dir or settings.data_dir
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")


def build_chunks(
    pdf_paths: list[Path],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    chunker: Chunker | None = None,
) -> list[Document]:
    page_docs: list[Document] = []
    for path in pdf_paths:
        logger.info("Loading PDF: {}", path.name)
        page_docs.extend(_load_pdf(path))

    if chunker is None:
        chunks = _splitter(chunk_size, chunk_overlap).split_documents(page_docs)
    else:
        chunks = chunker.split_documents(page_docs)

    per_doc_counter: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        idx = per_doc_counter[doc_id]
        per_doc_counter[doc_id] += 1
        meta = ChunkMetadata(
            document_id=doc_id,
            filename=chunk.metadata["filename"],
            source=chunk.metadata["source"],
            page=chunk.metadata["page"],
            chunk_id=_chunk_id(doc_id, chunk.metadata["page"], idx),
            section=chunk.metadata.get("section"),
        )
        chunk.metadata = meta.model_dump()
    return chunks


def index_chunks(chunks: list[Document], collection_name: str | None = None) -> int:
    """Compute deterministic UUIDs and add chunks to the vector store.

    Re-ingesting the same content upserts instead of creating duplicates.
    """
    if not chunks:
        return 0
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    get_vector_store(collection_name=collection_name).add_documents(chunks, ids=ids)
    return len(chunks)


def ingest(
    recreate: bool = False,
    collection_name: str | None = None,
    chunker: Chunker | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    pdfs = discover_pdfs()
    if not pdfs:
        logger.warning("No PDF files found in {}", settings.data_dir)
        return 0

    ensure_collection(recreate=recreate, collection_name=collection_name)
    chunks = build_chunks(
        pdfs, chunker=chunker,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        logger.warning("No chunks produced from {} PDF(s)", len(pdfs))
        return 0

    count = index_chunks(chunks, collection_name=collection_name)
    logger.info("Ingested {} chunks from {} PDF(s)", count, len(pdfs))
    return count
