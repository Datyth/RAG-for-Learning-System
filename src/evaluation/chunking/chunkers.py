"""Chunker implementations used by indexing + evaluation experiments.

Each chunker exposes `split_documents(pages) -> chunks` using LangChain Documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class ChunkingStrategy:
    """A named chunking setup usable for indexing + evaluation."""

    strategy_id: str
    chunker: object
    params: dict[str, object]


@dataclass(frozen=True)
class RecursiveChunker:
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: list[str] | None = None

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators or DEFAULT_SEPARATORS,
            is_separator_regex=False,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        return self._splitter().split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        return self._splitter().split_text(text)


class SemanticChunker:
    """Placeholder for semantic chunking.

    This project currently evaluates chunking strategies by building isolated
    Qdrant collections per strategy. Semantic chunking requires an embedding-
    aware splitter, which is not wired up yet in this repo.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
        embedding: object | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self.embedding = embedding

    def split_documents(self, documents: list[Document]) -> list[Document]:
        raise NotImplementedError(
            "SemanticChunker is not implemented yet. Use RecursiveChunker-based strategies."
        )


def default_strategies() -> list[ChunkingStrategy]:
    """Return a small baseline set of chunking strategies for experiments."""

    candidates = [
        ("rc_500_50", 500, 50),
        ("rc_800_100", 800, 100),
        ("rc_1000_150", 1000, 150),
        ("rc_1200_200", 1200, 200),
        ("rc_1500_200", 1500, 200),
    ]
    strategies: list[ChunkingStrategy] = []
    for strategy_id, size, overlap in candidates:
        strategies.append(
            ChunkingStrategy(
                strategy_id=strategy_id,
                chunker=RecursiveChunker(chunk_size=size, chunk_overlap=overlap),
                params={"chunk_size": size, "chunk_overlap": overlap, "type": "recursive"},
            )
        )
    return strategies
