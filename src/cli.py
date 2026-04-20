"""Command-line interface: ingest, ask, debug-retrieval."""

import json

import typer
from loguru import logger

from src.indexing import ingest as run_ingest
from src.rag import answer as run_answer, retrieve
from src.store import close_client

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Minimal RAG CLI over local PDFs.",
)

def _print_section(title: str) -> None:
    typer.echo(title)
    typer.echo("─" * len(title))

def _print_answer(text: str) -> None:
    _print_section("Answer")
    typer.echo(text.strip())
    typer.echo()
    
def _print_sources(chunks: list) -> None:
    if not chunks:
        return
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


@app.command()
def ingest(
    recreate: bool = typer.Option(False, "--recreate", help="Drop and recreate the collection."),
) -> None:
    """Ingest every PDF under ./data into Qdrant."""
    run_ingest(recreate=recreate)


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
    result = run_answer(question, k=k, filters=_parse_filters(filters))
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
        typer.echo(f"    {preview}\n")


def main() -> None:
    logger.remove()
    logger.add(lambda m: typer.echo(m, err=True), level="INFO")
    try:
        app()
    finally:
        close_client()


if __name__ == "__main__":
    main()
