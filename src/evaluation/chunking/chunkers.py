"""Chunker implementations used by indexing + evaluation experiments.

Each chunker exposes `split_documents(pages) -> chunks` using LangChain Documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_experimental.text_splitter import SemanticChunker


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


@dataclass(frozen=True)
class SemanticChunkerWrapper:
    """Wrapper cho Langchain Semantic Chunker."""
    embeddings: Embeddings
    breakpoint_type: str = "percentile"

    def _splitter(self) -> SemanticChunker:
        return SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.breakpoint_type
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        return self._splitter().split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        return self._splitter().split_text(text)


def default_strategies(embeddings_model: Optional[Embeddings] = None, option: str = "recursive") -> list[ChunkingStrategy]:
    """Return a combined set of chunking strategies (Recursive + Semantic) for experiments.
    If `option` is set to "recursive", only return recursive strategies.
    If `option` is set to "semantic", only return semantic strategies (requires `embeddings_model`).
    """

    if option == "recursive":
        # Only return recursive strategies
        return [
            ChunkingStrategy(
                strategy_id=sid,
                chunker=RecursiveChunker(chunk_size=size, chunk_overlap=overlap),
                params={"chunk_size": size, "chunk_overlap": overlap, "type": "recursive"},
            )
            for sid, size, overlap in [
                ("rc_500_50", 500, 50),
                ("rc_800_100", 800, 100),
                ("rc_1000_150", 1000, 150),
            ]
        ]
        
    elif option == "semantic": 
        if not embeddings_model:
            raise ValueError("Cần phải cung cấp 'embeddings_model' khi chọn option='semantic'")
            
        # Only return semantic strategies
        return [
            ChunkingStrategy(
                strategy_id=sid,
                chunker=SemanticChunkerWrapper(
                    embeddings=embeddings_model, 
                    breakpoint_type=b_type
                ),
                params={"breakpoint_type": b_type, "type": "semantic"},
            )
            for sid, b_type in [
                ("semantic_percentile", "percentile"),
                ("semantic_std_dev", "standard_deviation"),
                ("semantic_interquartile", "interquartile"),
            ]
        ]
        
    else:  #both
        # 1. Base Recursive Strategies
        recursive_candidates = [
            ("rc_500_50", 500, 50),
            ("rc_800_100", 800, 100),
            ("rc_1000_150", 1000, 150),
        ]
        
        strategies = [
            ChunkingStrategy(
                strategy_id=sid,
                chunker=RecursiveChunker(chunk_size=size, chunk_overlap=overlap),
                params={"chunk_size": size, "chunk_overlap": overlap, "type": "recursive"},
            )
            for sid, size, overlap in recursive_candidates
        ]

        # 2. Semantic Strategies 
        if embeddings_model:
            semantic_candidates = [
                ("semantic_percentile", "percentile"),
                ("semantic_std_dev", "standard_deviation"),
                ("semantic_interquartile", "interquartile"),
            ]
            
            semantic_strategies = [
                ChunkingStrategy(
                    strategy_id=sid,
                    chunker=SemanticChunkerWrapper(
                        embeddings=embeddings_model, 
                        breakpoint_type=b_type
                    ),
                    params={"breakpoint_type": b_type, "type": "semantic"},
                )
                for sid, b_type in semantic_candidates
            ]
            strategies.extend(semantic_strategies)
            
        return strategies