"""FastAPI application exposing RAG, summary, quiz, and flashcards endpoints."""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import services
from src.learning import GenerationError
from src.schemas import FlashcardSet, QuizSet, RagAnswer, Summary


class MetadataFilter(BaseModel):
    """Retrieval filter applied against indexed chunk metadata."""

    filename: str | None = None
    page: int | None = None
    section: str | None = None
    document_id: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="User question in any language.")
    k: int | None = Field(default=None, ge=1, le=64)
    filters: MetadataFilter | None = None


class SummarizeRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    k: int | None = Field(default=None, ge=1, le=64)


class QuizRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    count: int | None = Field(default=None, ge=1, le=50)
    k: int | None = Field(default=None, ge=1, le=64)


class FlashcardsRequest(QuizRequest):
    """Same shape as QuizRequest."""


class DocumentInfo(BaseModel):
    filename: str
    document_id: str
    page_count: int
    pages: list[int]
    chunk_count: int


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int


def _filters_to_dict(f: MetadataFilter | None) -> dict[str, str | int] | None:
    if f is None:
        return None
    data = f.model_dump(exclude_none=True)
    return data or None


app = FastAPI(
    title="RAG Learning API",
    description="Grounded Q&A, summaries, quizzes, and flashcards over indexed PDFs.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[DocumentInfo]:
    """List every document currently indexed in the vector store."""
    return [DocumentInfo(**d) for d in services.list_documents()]


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Accept a PDF upload, persist it to data_dir, and index its chunks."""
    content = await file.read()
    try:
        return UploadResponse(**services.save_and_ingest_pdf(content, file.filename or ""))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask", response_model=RagAnswer)
def ask(req: AskRequest) -> RagAnswer:
    """Grounded Q&A with inline source citations."""
    return services.ask(
        req.question,
        k=req.k,
        filters=_filters_to_dict(req.filters),
    )


@app.post("/summarize", response_model=Summary)
def summarize(req: SummarizeRequest) -> Summary:
    """Grounded summary scoped by document, query, or filter."""
    try:
        return services.summarize(
            document=req.document,
            query=req.query,
            filters=_filters_to_dict(req.filters),
            k=req.k,
        )
    except GenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/quiz", response_model=QuizSet)
def quiz(req: QuizRequest) -> QuizSet:
    """Generate a grounded multiple-choice quiz."""
    try:
        return services.quiz(
            document=req.document,
            query=req.query,
            filters=_filters_to_dict(req.filters),
            count=req.count,
            k=req.k,
        )
    except GenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/flashcards", response_model=FlashcardSet)
def flashcards(req: FlashcardsRequest) -> FlashcardSet:
    """Generate a grounded flashcard set."""
    try:
        return services.flashcards(
            document=req.document,
            query=req.query,
            filters=_filters_to_dict(req.filters),
            count=req.count,
            k=req.k,
        )
    except GenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def main() -> None:
    """Entrypoint for `rag-api` launching uvicorn on 0.0.0.0:8000."""
    import uvicorn

    uvicorn.run("src.interfaces.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
