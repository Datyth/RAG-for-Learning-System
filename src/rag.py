"""Retrieval, prompts, LLM, citations, and grounded answers."""

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client.http import models as qmodels

from src.config import settings
from src.schemas import ChunkMetadata, Citation, RagAnswer, RetrievedChunk
from src.store import get_vector_store

PROMPTS_DIR = Path(__file__).parent / "prompts"
ANSWER_TEMPLATE = "answer.jinja2"


def _metadata_filter(filters: dict[str, str | int] | None) -> qmodels.Filter | None:
    if not filters:
        return None
    conditions = [
        qmodels.FieldCondition(
            key=f"metadata.{field}",
            match=qmodels.MatchValue(value=value),
        )
        for field, value in filters.items()
        if value is not None
    ]
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


def retrieve(
    query: str,
    k: int | None = None,
    filters: dict[str, str | int] | None = None,
) -> list[RetrievedChunk]:
    store = get_vector_store()
    hits = store.similarity_search_with_score(
        query=query,
        k=k or settings.top_k,
        filter=_metadata_filter(filters),
    )
    return [
        RetrievedChunk(
            text=doc.page_content,
            score=float(score),
            metadata=ChunkMetadata(**doc.metadata),
        )
        for doc, score in hits
    ]


@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=select_autoescape(disabled_extensions=("jinja2",), default=False),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_answer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return _jinja_env().get_template(ANSWER_TEMPLATE).render(question=question, chunks=chunks)


def format_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    seen: set[tuple[str, int]] = set()
    citations: list[Citation] = []
    for c in chunks:
        key = (c.metadata.filename, c.metadata.page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                filename=c.metadata.filename,
                page=c.metadata.page,
                section=c.metadata.section,
                chunk_id=c.metadata.chunk_id,
            )
        )
    return citations


@lru_cache(maxsize=1)
def _llm() -> ChatGoogleGenerativeAI:
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env before calling the LLM."
        )
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )


def answer(
    question: str,
    k: int | None = None,
    filters: dict[str, str | int] | None = None,
) -> RagAnswer:
    chunks = retrieve(question, k=k, filters=filters)
    if not chunks:
        return RagAnswer(
            question=question,
            answer="I don't have enough information in the provided context to answer.",
            citations=[],
            chunks=[],
        )

    prompt = render_answer_prompt(question, chunks)
    response = _llm().invoke([HumanMessage(content=prompt)])
    text = response.content if isinstance(response.content, str) else str(response.content)

    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(chunks),
        chunks=chunks,
    )
