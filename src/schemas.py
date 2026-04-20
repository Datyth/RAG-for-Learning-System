"""Pydantic schemas for chunk metadata, retrieved hits, and final answers."""

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Stable metadata attached to every chunk stored in Qdrant."""

    document_id: str
    filename: str
    source: str
    page: int
    chunk_id: str
    section: str | None = None


class RetrievedChunk(BaseModel):
    """A retrieved chunk with its score and metadata."""

    text: str
    score: float
    metadata: ChunkMetadata


class Citation(BaseModel):
    """Citation extracted from a retrieved chunk's metadata."""

    source_marker: str  # e.g. "S1", "S2" — matches inline markers in the answer
    filename: str
    page: int
    section: str | None = None
    chunk_id: str


class RagAnswer(BaseModel):
    """Final grounded answer returned to the caller."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
