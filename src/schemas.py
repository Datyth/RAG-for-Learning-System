"""Pydantic schemas for chunks, answers, and learning outputs."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


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

    source_index: int
    source_marker: str
    filename: str
    page: int
    section: str | None = None
    chunk_id: str | None = None


class RagAnswer(BaseModel):
    """Final grounded answer returned to the caller."""

    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class Summary(BaseModel):
    """Grounded study-oriented summary of a document or subset."""

    scope: Literal["query", "document", "filter", "corpus"]
    target: str | None = None
    summary: str
    key_points: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class QuizItem(BaseModel):
    """A single multiple-choice quiz item grounded in the source material."""

    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int
    explanation: str
    source_markers: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    topic: str | None = None

    @model_validator(mode="after")
    def _validate_correct_index(self) -> "QuizItem":
        if not 0 <= self.correct_index < len(self.options):
            raise ValueError(
                f"correct_index {self.correct_index} out of range for {len(self.options)} options"
            )
        return self


class QuizSet(BaseModel):
    """A reusable set of grounded quiz items with resolved citations."""

    scope: Literal["query", "document", "filter", "corpus"]
    target: str | None = None
    items: list[QuizItem] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class Flashcard(BaseModel):
    """A single study flashcard grounded in the source material."""

    front: str
    back: str
    hint: str | None = None
    topic: str | None = None
    source_markers: list[str] = Field(default_factory=list)


class FlashcardSet(BaseModel):
    """A reusable set of grounded flashcards with resolved citations."""

    scope: Literal["query", "document", "filter", "corpus"]
    target: str | None = None
    cards: list[Flashcard] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
