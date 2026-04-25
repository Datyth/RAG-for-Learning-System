"""Grounded learning features: summarization, quiz, and flashcard generation."""

import json
import re

from loguru import logger
from pydantic import ValidationError

from src.config import settings
from src.rag import (
    fetch_all_chunks,
    format_citations,
    invoke_llm,
    render_prompt,
    retrieve,
)
from src.schemas import (
    Flashcard,
    FlashcardSet,
    QuizItem,
    QuizSet,
    RetrievedChunk,
    Summary,
)

SUMMARY_SINGLE_TEMPLATE = "summary_single.jinja2"
SUMMARY_MAP_TEMPLATE = "summary_map.jinja2"
SUMMARY_REDUCE_TEMPLATE = "summary_reduce.jinja2"
QUIZ_TEMPLATE = "quiz.jinja2"
FLASHCARDS_TEMPLATE = "flashcards.jinja2"


class GenerationError(RuntimeError):
    """Raised when LLM output cannot be parsed or validated."""


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        if t.endswith("```"):
            t = t[: -len("```")]
    return t.strip()


def _parse_json(text: str) -> dict | list:
    """Parse the first JSON object/array found in model output."""
    cleaned = _strip_code_fences(text)
    decoder = json.JSONDecoder()

    # Fast path: whole output is valid JSON.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, (dict, list)):
            return obj
    except json.JSONDecodeError:
        pass

    # Common path: model wraps JSON with prose; decode from first plausible start.
    starts = sorted(i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1)
    if not starts:
        raise GenerationError("No JSON object found in model output.")

    for start in starts:
        try:
            obj, _end = decoder.raw_decode(cleaned, idx=start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, (dict, list)):
            return obj

    raise GenerationError("Invalid JSON from model output.")


def _resolve_target(
    document: str | None,
    query: str | None,
    filters: dict[str, str | int] | None,
    k: int | None,
    retrieval_k: int,
) -> tuple[list[RetrievedChunk], str, str | None]:
    """Resolve input options into (chunks, scope, target_label)."""
    effective_filters: dict[str, str | int] = dict(filters or {})
    if document:
        effective_filters["filename"] = document

    if query:
        chunks = retrieve(query, k=k or retrieval_k, filters=effective_filters)
        target: str | None = query
        scope = "query"
    elif effective_filters:
        chunks = fetch_all_chunks(filters=effective_filters)
        target = ", ".join(f"{fk}={fv}" for fk, fv in effective_filters.items())
        scope = "document" if document else "filter"
    else:
        chunks = fetch_all_chunks(filters=None)
        target = None
        scope = "corpus"

    return chunks, scope, target


def _batch(items: list[RetrievedChunk], size: int) -> list[list[RetrievedChunk]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _validate_items(
    payload: object,
    key: str,
    model_class: type,
    dedup_field: str,
    label: str,
    valid_markers: set[str],
) -> list:
    if not isinstance(payload, dict):
        raise GenerationError(f"Expected JSON object for {label}.")
    raw_items = payload.get(key)
    if not isinstance(raw_items, list):
        raise GenerationError(f"Expected '{key}' to be a list for {label}.")

    items: list = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        try:
            item = model_class.model_validate(raw)
        except ValidationError as e:
            logger.warning("Dropping invalid {}: {}", label, e)
            continue
        norm = str(getattr(item, dedup_field, "")).strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        markers = [m for m in item.source_markers if m in valid_markers]
        items.append(item.model_copy(update={"source_markers": markers}))

    if not items:
        raise GenerationError(f"No valid {label} produced.")
    return items


def _validate_summary_payload(payload: object) -> tuple[str, list[str]]:
    if not isinstance(payload, dict):
        raise GenerationError("Expected a JSON object for summary.")
    summary = payload.get("summary")
    key_points = payload.get("key_points", [])
    if not isinstance(summary, str):
        raise GenerationError("Summary payload missing 'summary' string.")
    if not isinstance(key_points, list) or not all(isinstance(x, str) for x in key_points):
        raise GenerationError("Summary payload 'key_points' must be a list of strings.")
    return summary.strip(), [kp.strip() for kp in key_points if kp.strip()]


def summarize(
    document: str | None = None,
    query: str | None = None,
    filters: dict[str, str | int] | None = None,
    k: int | None = None,
) -> Summary:
    """Grounded summary; uses map-reduce when chunk count exceeds batch size."""
    chunks, scope, target = _resolve_target(
        document=document,
        query=query,
        filters=filters,
        k=k,
        retrieval_k=settings.summarize_retrieval_k,
    )

    if not chunks:
        raise GenerationError("No chunks available for summarization.")

    batch_size = settings.summarize_batch_size
    if len(chunks) <= batch_size:
        prompt = render_prompt(SUMMARY_SINGLE_TEMPLATE, chunks=chunks)
        payload = _parse_json(invoke_llm(prompt))
        summary_text, key_points = _validate_summary_payload(payload)
    else:
        n_batches = (len(chunks) + batch_size - 1) // batch_size
        partials: list[dict] = []
        for i, batch in enumerate(_batch(chunks, batch_size), start=1):
            logger.info("Summarizing batch {}/{}", i, n_batches)
            prompt = render_prompt(SUMMARY_MAP_TEMPLATE, chunks=batch)
            payload = _parse_json(invoke_llm(prompt))
            s, kp = _validate_summary_payload(payload)
            partials.append({"summary": s, "key_points": kp})

        reduce_prompt = render_prompt(SUMMARY_REDUCE_TEMPLATE, partials=partials)
        payload = _parse_json(invoke_llm(reduce_prompt))
        summary_text, key_points = _validate_summary_payload(payload)

    return Summary(
        scope=scope,
        target=target,
        summary=summary_text,
        key_points=key_points,
        citations=format_citations(chunks),
    )



def generate_quiz(
    document: str | None = None,
    query: str | None = None,
    filters: dict[str, str | int] | None = None,
    count: int | None = None,
    k: int | None = None,
) -> QuizSet:
    """Grounded multiple-choice quiz; raises GenerationError if output is unparseable."""
    chunks, scope, target = _resolve_target(
        document=document, query=query,
        filters=filters, k=k,
        retrieval_k=settings.generation_retrieval_k,
    )
    if not chunks:
        raise GenerationError("No chunks available for quiz generation.")

    n = count or settings.quiz_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}

    prompt = render_prompt(QUIZ_TEMPLATE, chunks=chunks, count=n)
    payload = _parse_json(invoke_llm(prompt))
    items = _validate_items(payload, "items", QuizItem, "question", "quiz items", valid_markers)

    return QuizSet(
        scope=scope,
        target=target,
        items=items,
        citations=format_citations(chunks),
    )


def generate_flashcards(
    document: str | None = None,
    query: str | None = None,
    filters: dict[str, str | int] | None = None,
    count: int | None = None,
    k: int | None = None,
) -> FlashcardSet:
    """Grounded flashcard set for spaced repetition; raises GenerationError if output is unparseable."""
    chunks, scope, target = _resolve_target(
        document=document,
        query=query,
        filters=filters,
        k=k,
        retrieval_k=settings.generation_retrieval_k,
    )
    if not chunks:
        raise GenerationError("No chunks available for flashcard generation.")

    n = count or settings.flashcards_default_count
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}

    prompt = render_prompt(FLASHCARDS_TEMPLATE, chunks=chunks, count=n)
    payload = _parse_json(invoke_llm(prompt))
    cards = _validate_items(payload, "cards", Flashcard, "front", "flashcards", valid_markers)

    return FlashcardSet(
        scope=scope,
        target=target,
        cards=cards,
        citations=format_citations(chunks),
    )
