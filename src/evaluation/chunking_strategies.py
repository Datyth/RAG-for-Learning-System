"""Chunker wrappers and strategy registry for evaluation experiments."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_RECURSIVE_CONFIGS = [
    ("rc_500_50", 500, 50),
    ("rc_800_100", 800, 100),
    ("rc_1000_150", 1000, 150),
]

_SEMANTIC_CONFIGS = [
    ("semantic_percentile", "percentile"),
    ("semantic_std_dev", "standard_deviation"),
    ("semantic_interquartile", "interquartile"),
]


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


@dataclass(frozen=True)
class SemanticChunkerWrapper:
    """Wrapper for LangChain SemanticChunker."""

    embeddings: Embeddings
    breakpoint_type: str = "percentile"

    def _splitter(self) -> SemanticChunker:
        return SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.breakpoint_type,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        return self._splitter().split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        return self._splitter().split_text(text)


def default_strategies(
    embeddings_model: Embeddings | None = None,
    option: str = "recursive",
) -> list[ChunkingStrategy]:
    """Return chunking strategies for experiments.

    option: "recursive" | "semantic" | "both".
    Semantic strategies require embeddings_model.
    """

    if option == "semantic" and not embeddings_model:
        raise ValueError("embeddings_model required for option='semantic'")

    strategies: list[ChunkingStrategy] = []

    if option in ("recursive", "both"):
        strategies.extend(
            ChunkingStrategy(
                strategy_id=sid,
                chunker=RecursiveChunker(chunk_size=size, chunk_overlap=overlap),
                params={"chunk_size": size, "chunk_overlap": overlap, "type": "recursive"},
            )
            for sid, size, overlap in _RECURSIVE_CONFIGS
        )

    if option in ("semantic", "both") and embeddings_model:
        strategies.extend(
            ChunkingStrategy(
                strategy_id=sid,
                chunker=SemanticChunkerWrapper(embeddings=embeddings_model, breakpoint_type=b_type),
                params={"breakpoint_type": b_type, "type": "semantic"},
            )
            for sid, b_type in _SEMANTIC_CONFIGS
        )

    return strategies
