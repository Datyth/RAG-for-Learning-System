"""Export learning outputs to JSON or Markdown."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from src.schemas import Citation, FlashcardSet, QuizSet, Summary

ExportFormat = Literal["text", "md", "json"]


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


def _to_markdown(model: BaseModel) -> str:
    if isinstance(model, Summary):
        title = "# Summary" + (f": {model.target}" if model.target else "")
        lines: list[str] = [title, "", f"_Scope: {model.scope}_", ""]
        if model.summary:
            lines.extend([model.summary.strip(), ""])
        if model.key_points:
            lines.extend(["## Key Points", "", *[f"- {kp}" for kp in model.key_points], ""])
        c = _citations_block(model.citations)
        if c:
            lines.append(c)
        return "\n".join(lines).rstrip() + "\n"

    if isinstance(model, QuizSet):
        title = "# Quiz" + (f": {model.target}" if model.target else "")
        lines = [title, "", f"_Scope: {model.scope} | Items: {len(model.items)}_", ""]
        for idx, item in enumerate(model.items, start=1):
            meta_parts: list[str] = []
            if item.topic:
                meta_parts.append(f"topic: {item.topic}")
            if item.difficulty:
                meta_parts.append(f"difficulty: {item.difficulty}")
            meta_suffix = f" _({' | '.join(meta_parts)})_" if meta_parts else ""

            lines.extend([f"## Q{idx}.{meta_suffix}", "", item.question.strip(), ""])
            for opt_idx, option in enumerate(item.options):
                lines.append(f"- {chr(ord('A') + opt_idx)}) {option}")
            lines.append("")
            lines.append(f"**Answer:** {chr(ord('A') + item.correct_index)}")
            if item.explanation:
                lines.append(f"**Explanation:** {item.explanation.strip()}")
            if item.source_markers:
                lines.append(f"**Sources:** {', '.join(item.source_markers)}")
            lines.append("")

        c = _citations_block(model.citations)
        if c:
            lines.append(c)
        return "\n".join(lines).rstrip() + "\n"

    if isinstance(model, FlashcardSet):
        title = "# Flashcards" + (f": {model.target}" if model.target else "")
        lines = [title, "", f"_Scope: {model.scope} | Cards: {len(model.cards)}_", ""]
        for idx, card in enumerate(model.cards, start=1):
            topic = f" — {card.topic}" if card.topic else ""
            lines.extend([f"## Card {idx}{topic}", ""])
            lines.append(f"**Front:** {card.front.strip()}")
            lines.append(f"**Back:** {card.back.strip()}")
            if card.hint:
                lines.append(f"**Hint:** {card.hint.strip()}")
            if card.source_markers:
                lines.append(f"**Sources:** {', '.join(card.source_markers)}")
            lines.append("")

        c = _citations_block(model.citations)
        if c:
            lines.append(c)
        return "\n".join(lines).rstrip() + "\n"

    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def export(
    model: BaseModel, *, fmt: ExportFormat = "text", output: Path | None = None
) -> str | Path:
    """Render model to a string, optionally writing it to disk.

    Args: model, fmt, output (optional).
    Returns: rendered string if output is None; otherwise the written path.
    Raises: TypeError for unsupported model type; ValueError for unknown fmt.
    """
    if fmt == "json":
        text = model.model_dump_json(indent=2) + "\n"
    elif fmt in {"text", "md"}:
        text = _to_markdown(model)
    else:
        raise ValueError(f"Unknown fmt '{fmt}'. Expected 'text' | 'md' | 'json'.")

    if output is None:
        return text

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output
