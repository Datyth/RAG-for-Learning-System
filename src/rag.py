"""Retrieval, prompts, LLM, citations, and grounded answers."""

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from qdrant_client.http import models as qmodels

from src.config import settings
from src.schemas import ChunkMetadata, Citation, RagAnswer, RetrievedChunk
from src.store import get_client, get_vector_store

PROMPTS_DIR = Path(__file__).parent / "prompts"
ANSWER_TEMPLATE = "answer.jinja2"
SCROLL_PAGE_SIZE = 256


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


def fetch_all_chunks(
    filters: dict[str, str | int] | None = None,
) -> list[RetrievedChunk]:
    """Scroll every chunk in the collection matching the filter, ordered by page."""
    client = get_client()
    q_filter = _metadata_filter(filters)
    results: list[RetrievedChunk] = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=q_filter,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            meta = payload.get("metadata") or {}
            text = payload.get("page_content") or ""
            if not meta or not text:
                continue
            results.append(
                RetrievedChunk(
                    text=text,
                    score=0.0,
                    metadata=ChunkMetadata(**meta),
                )
            )
        if next_offset is None:
            break
        offset = next_offset

    results.sort(
        key=lambda r: (r.metadata.filename, r.metadata.page, r.metadata.chunk_id)
    )
    return results


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


def render_prompt(template_name: str, **context: object) -> str:
    """Render an arbitrary Jinja template from the prompts directory."""
    return _jinja_env().get_template(template_name).render(**context)


def format_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            source_index=i,
            source_marker=f"S{i}",
            filename=c.metadata.filename,
            page=c.metadata.page,
            section=c.metadata.section,
            chunk_id=c.metadata.chunk_id,
        )
        for i, c in enumerate(chunks, start=1)
    ]

def _build_hf_local() -> BaseChatModel:
    """Build a chat model backed by a local Transformers pipeline."""
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    do_sample = settings.llm_temperature > 0
    pipeline_kwargs: dict = {
        "max_new_tokens": settings.hf_max_new_tokens,
        "do_sample": do_sample, "max_length": None,
        "return_full_text": False,
    }
    if do_sample:
        pipeline_kwargs["temperature"] = settings.llm_temperature

    try:
        tokenizer = AutoTokenizer.from_pretrained(settings.hf_model)
        model = AutoModelForCausalLM.from_pretrained(settings.hf_model, dtype=torch.bfloat16)

        text_gen_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device=settings.hf_device,
            **pipeline_kwargs,
        )
        llm = HuggingFacePipeline(pipeline=text_gen_pipeline, pipeline_kwargs=pipeline_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Hugging Face model '{settings.hf_model}': {e}"
        ) from e

    return ChatHuggingFace(llm=llm)


def _build_gemini() -> BaseChatModel:
    """Build a Gemini chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to .env before using provider 'gemini'."
        )
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
    )


@lru_cache(maxsize=1)
def _llm() -> BaseChatModel:
    provider = settings.llm_provider
    if provider == "hf_local":
        return _build_hf_local()
    if provider == "gemini":
        return _build_gemini()
    raise ValueError(
        f"Unknown llm_provider '{provider}'. Expected 'hf_local' or 'gemini'."
    )


def invoke_llm(prompt: str) -> str:
    """Invoke the configured chat model with a single user prompt and return text."""
    response = _llm().invoke([HumanMessage(content=prompt)])
    return response.content if isinstance(response.content, str) else str(response.content)


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
    text = invoke_llm(prompt)

    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(chunks),
        chunks=chunks,
    )
