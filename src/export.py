"""Export learning outputs to JSON and Markdown."""

from pathlib import Path

from pydantic import BaseModel

from src.schemas import Citation, FlashcardSet, QuizSet, Summary


def _citation_line(c: Citation) -> str:
    parts = [f"[{c.source_marker}] {c.filename} p.{c.page}"]
    if c.section:
        parts.append(f"section: {c.section}")
    if c.chunk_id:
        parts.append(f"chunk: {c.chunk_id}")
    return " | ".join(parts)


def _citations_block(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["## Sources", ""]
    lines.extend(f"- {_citation_line(c)}" for c in citations)
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def to_json(model: BaseModel) -> str:
    """Serialize any learning output model to stable JSON."""
    return model.model_dump_json(indent=2)


def write_json(model: BaseModel, path: Path) -> Path:
    """Write a learning output model as JSON to `path`."""
    return _write_text(path, to_json(model) + "\n")


def summary_to_markdown(summary: Summary) -> str:
    title = f"# Summary"
    if summary.target:
        title += f": {summary.target}"

    lines = [title, ""]
    lines.append(f"_Scope: {summary.scope}_")
    lines.append("")

    if summary.summary:
        lines.append(summary.summary.strip())
        lines.append("")

    if summary.key_points:
        lines.append("## Key Points")
        lines.append("")
        lines.extend(f"- {kp}" for kp in summary.key_points)
        lines.append("")

    citations_md = _citations_block(summary.citations)
    if citations_md:
        lines.append(citations_md)

    return "\n".join(lines).rstrip() + "\n"


def quiz_to_markdown(quiz: QuizSet) -> str:
    title = "# Quiz"
    if quiz.target:
        title += f": {quiz.target}"

    lines = [title, "", f"_Scope: {quiz.scope} | Items: {len(quiz.items)}_", ""]

    for idx, item in enumerate(quiz.items, start=1):
        meta_parts: list[str] = []
        if item.topic:
            meta_parts.append(f"topic: {item.topic}")
        if item.difficulty:
            meta_parts.append(f"difficulty: {item.difficulty}")
        meta_suffix = f" _({' | '.join(meta_parts)})_" if meta_parts else ""

        lines.append(f"## Q{idx}.{meta_suffix}")
        lines.append("")
        lines.append(item.question.strip())
        lines.append("")
        for opt_idx, option in enumerate(item.options):
            marker = "x" if opt_idx == item.correct_index else " "
            letter = chr(ord("A") + opt_idx)
            lines.append(f"- [{marker}] {letter}. {option}")
        lines.append("")
        lines.append(f"**Answer:** {chr(ord('A') + item.correct_index)}")
        lines.append("")
        if item.explanation:
            lines.append(f"**Explanation:** {item.explanation.strip()}")
            lines.append("")
        if item.source_markers:
            lines.append(f"**Sources:** {', '.join(item.source_markers)}")
            lines.append("")

    citations_md = _citations_block(quiz.citations)
    if citations_md:
        lines.append(citations_md)

    return "\n".join(lines).rstrip() + "\n"


def flashcards_to_markdown(flashcards: FlashcardSet) -> str:
    title = "# Flashcards"
    if flashcards.target:
        title += f": {flashcards.target}"

    lines = [title, "", f"_Scope: {flashcards.scope} | Cards: {len(flashcards.cards)}_", ""]

    for idx, card in enumerate(flashcards.cards, start=1):
        header = f"## Card {idx}"
        if card.topic:
            header += f" — {card.topic}"
        lines.append(header)
        lines.append("")
        lines.append(f"**Front:** {card.front.strip()}")
        lines.append("")
        lines.append(f"**Back:** {card.back.strip()}")
        lines.append("")
        if card.hint:
            lines.append(f"**Hint:** {card.hint.strip()}")
            lines.append("")
        if card.source_markers:
            lines.append(f"**Sources:** {', '.join(card.source_markers)}")
            lines.append("")

    citations_md = _citations_block(flashcards.citations)
    if citations_md:
        lines.append(citations_md)

    return "\n".join(lines).rstrip() + "\n"


def write_markdown(model: BaseModel, path: Path) -> Path:
    """Write a learning output model as Markdown to `path`."""
    if isinstance(model, Summary):
        text = summary_to_markdown(model)
    elif isinstance(model, QuizSet):
        text = quiz_to_markdown(model)
    elif isinstance(model, FlashcardSet):
        text = flashcards_to_markdown(model)
    else:
        raise TypeError(f"Unsupported model for markdown export: {type(model).__name__}")
    return _write_text(path, text)
