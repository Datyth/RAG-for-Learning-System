"""Command-line interface: ingest, ask, debug-retrieval, summarize, quiz, flashcards."""

import json
from pathlib import Path

import typer
from loguru import logger
from pydantic import BaseModel

from src.export import export
from src.indexing import ingest as ingest_data_dir
from src.learning import (
    generate_flashcards,
    generate_quiz,
    summarize as summarize_learning,
)
from src.rag import answer, retrieve
from src.schemas import RetrievedChunk
from src.store import close_client

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Minimal RAG CLI over local PDFs with learning features.",
)

LEARNING_FORMATS = {"text", "json", "md"}


def _print_section(title: str) -> None:
    typer.echo(title)
    typer.echo("─" * len(title))


def _print_answer(text: str) -> None:
    _print_section("Answer")
    typer.echo(text.strip())


def _print_sources(chunks: list[RetrievedChunk]) -> None:
    if not chunks:
        return
    typer.echo()
    _print_section("Sources")
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        line = f"[S{i}] {meta.filename} p.{meta.page}"
        if meta.section:
            line += f" | {meta.section}"
        typer.echo(line)


def _parse_filters(pairs: list[str] | None) -> dict[str, str | int] | None:
    """Parse CLI 'key=value' filters into a flat dict (page is cast to int)."""
    if not pairs:
        return None
    parsed: dict[str, str | int] = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"Filter '{pair}' must be in key=value form.")
        key, value = pair.split("=", 1)
        key, value = key.strip(), value.strip()
        parsed[key] = int(value) if key == "page" else value
    return parsed


def _validate_format(fmt: str) -> str:
    fmt = fmt.lower()
    if fmt not in LEARNING_FORMATS:
        raise typer.BadParameter(
            f"--format must be one of {sorted(LEARNING_FORMATS)}, got '{fmt}'."
        )
    return fmt


def _emit(model: BaseModel, output: Path | None, fmt: str) -> None:
    if output:
        written = export(model, fmt=fmt, output=output)
        typer.echo(f"Wrote {written}")
        return
    typer.echo(export(model, fmt=fmt))


@app.command()
def ingest(
    recreate: bool = typer.Option(False, "--recreate", help="Drop and recreate the collection."),
) -> None:
    """Ingest every PDF under ./data into Qdrant."""
    count = ingest_data_dir(recreate=recreate)
    typer.echo(f"Done. {count} chunks indexed.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="The user question."),
    k: int | None = typer.Option(None, "--k", help="Number of chunks to retrieve."),
    filters: list[str] | None = typer.Option(
        None,
        "--filter",
        "-f",
        help="Metadata filter, e.g. -f filename=foo.pdf -f page=3",
    ),
) -> None:
    """Answer a question using retrieved context only."""
    result = answer(question, k=k, filters=_parse_filters(filters))
    _print_answer(result.answer)
    _print_sources(result.chunks)


@app.command("debug-retrieval")
def debug_retrieval(
    question: str = typer.Argument(..., help="Query to retrieve chunks for."),
    k: int | None = typer.Option(None, "--k", help="Number of chunks to retrieve."),
    filters: list[str] | None = typer.Option(
        None,
        "--filter",
        "-f",
        help="Metadata filter, e.g. -f filename=foo.pdf",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Print top retrieved chunks with scores and metadata, without calling the LLM."""
    chunks = retrieve(question, k=k, filters=_parse_filters(filters))

    if as_json:
        payload = [c.model_dump() for c in chunks]
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if not chunks:
        typer.echo("No chunks retrieved.")
        return

    for i, c in enumerate(chunks, start=1):
        typer.echo(
            f"[{i}] score={c.score:.4f} | {c.metadata.filename} p.{c.metadata.page} "
            f"| chunk_id={c.metadata.chunk_id}"
            + (f" | section={c.metadata.section}" if c.metadata.section else "")
        )
        preview = c.text.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:120] + "…"
        typer.echo(f"    {preview}\n")


@app.command("summarize")
def summarize(
    document: str | None = typer.Option(
        None, "--document", "-d", help="Target filename (e.g. paper.pdf)."
    ),
    query: str | None = typer.Option(
        None, "--query", "-q", help="Topic or question for retrieval-guided summary."
    ),
    filters: list[str] | None = typer.Option(
        None, "--filter", "-f", help="Extra metadata filter, e.g. -f page=3."
    ),
    k: int | None = typer.Option(None, "--k", help="Retrieval top-k (query mode)."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write output to this file path."
    ),
    fmt: str = typer.Option("text", "--format", help="Output format: text | json | md."),
) -> None:
    """Generate a grounded study summary of a document, filter, or topic."""
    fmt = _validate_format(fmt)
    try:
        result = summarize_learning(
            document=document,
            query=query,
            filters=_parse_filters(filters),
            k=k,
        )
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    _emit(result, output, fmt)


@app.command("quiz")
def quiz(
    document: str | None = typer.Option(None, "--document", "-d", help="Target filename."),
    query: str | None = typer.Option(
        None, "--query", "-q", help="Topic or question for retrieval-guided quiz."
    ),
    filters: list[str] | None = typer.Option(None, "--filter", "-f", help="Metadata filter."),
    count: int | None = typer.Option(None, "--count", "-n", help="Number of quiz items."),
    k: int | None = typer.Option(None, "--k", help="Retrieval top-k (query mode)."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write output to file."),
    fmt: str = typer.Option("text", "--format", help="Output format: text | json | md."),
) -> None:
    """Generate a grounded multiple-choice quiz set."""
    fmt = _validate_format(fmt)
    try:
        result = generate_quiz(
            document=document,
            query=query,
            filters=_parse_filters(filters),
            count=count,
            k=k,
        )
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    _emit(result, output, fmt)


@app.command("flashcards")
def flashcards(
    document: str | None = typer.Option(None, "--document", "-d", help="Target filename."),
    query: str | None = typer.Option(
        None, "--query", "-q", help="Topic or question for retrieval-guided flashcards."
    ),
    filters: list[str] | None = typer.Option(None, "--filter", "-f", help="Metadata filter."),
    count: int | None = typer.Option(None, "--count", "-n", help="Number of flashcards."),
    k: int | None = typer.Option(None, "--k", help="Retrieval top-k (query mode)."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write output to file."),
    fmt: str = typer.Option("text", "--format", help="Output format: text | json | md."),
) -> None:
    """Generate a grounded flashcard set for study and review."""
    fmt = _validate_format(fmt)
    try:
        result = generate_flashcards(
            document=document,
            query=query,
            filters=_parse_filters(filters),
            count=count,
            k=k,
        )
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    _emit(result, output, fmt)


def main() -> None:
    logger.remove()
    logger.add(
        lambda m: typer.echo(m, err=True),
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <6}</level> | {message}",
    )
    try:
        app()
    finally:
        close_client()


if __name__ == "__main__":
    main()
