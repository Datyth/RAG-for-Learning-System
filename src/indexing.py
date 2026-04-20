"""Load PDFs, split into chunks with metadata, and index into Qdrant."""

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import settings
from src.schemas import ChunkMetadata
from src.store import ensure_collection, get_vector_store


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
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
    for p in pages:
        page_number = int(p.metadata.get("page", 0)) + 1
        p.metadata = {
            "document_id": doc_id,
            "filename": path.name,
            "source": str(path.resolve()),
            "page": page_number,
            "section": p.metadata.get("section"),
        }
    return pages


def discover_pdfs(data_dir: Path | None = None) -> list[Path]:
    directory = data_dir or settings.data_dir
    return sorted(p for p in directory.glob("*.pdf") if p.is_file())


def build_chunks(pdf_paths: list[Path]) -> list[Document]:
    page_docs: list[Document] = []
    for path in pdf_paths:
        logger.info("Loading PDF: {}", path.name)
        page_docs.extend(_load_pdf(path))

    chunks = _splitter().split_documents(page_docs)

    per_doc_counter: dict[str, int] = {}
    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        idx = per_doc_counter.get(doc_id, 0)
        per_doc_counter[doc_id] = idx + 1
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


def ingest(recreate: bool = False) -> int:
    pdfs = discover_pdfs()
    if not pdfs:
        logger.warning("No PDF files found in {}", settings.data_dir)
        return 0

    ensure_collection(recreate=recreate)
    chunks = build_chunks(pdfs)

    if not chunks:
        logger.warning("No chunks produced from {} PDF(s)", len(pdfs))
        return 0

    store = get_vector_store()
    store.add_documents(chunks)
    logger.info("Ingested {} chunks from {} PDF(s)", len(chunks), len(pdfs))
    return len(chunks)
