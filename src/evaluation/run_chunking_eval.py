"""Run Ragas evaluation across multiple chunking strategies.

This script loops strategies and uses a separate Qdrant collection per strategy:
recreate collection -> ingest PDFs -> run Ragas -> write artifact files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import settings
from src.evaluation.chunking.chunkers import ChunkingStrategy, default_strategies
from src.evaluation.evaluator import run_evaluation
from src.indexing import ingest
from src.rag import answer


def _collection_for(strategy_id: str) -> str:
    return f"{settings.qdrant_collection}__{strategy_id}"


def _safe_serialize_result(result: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"repr": repr(result)}

    for attr in ("to_pandas", "to_dict", "model_dump"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                value = fn()
                if attr == "to_pandas":
                    payload["pandas"] = value.to_dict()  # type: ignore[no-any-return]
                else:
                    payload[attr] = value
            except Exception as exc:  # pragma: no cover
                payload[f"{attr}_error"] = str(exc)

    return payload


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_test_cases() -> list[dict[str, str]]:
    return [
        {
            "question": "Fine tuning là gì?",
            "ground_truth": (
                "Kĩ thuật điều chỉnh một mô hình đã được huấn luyện trước đó "
                "trên một tập dữ liệu nhỏ hơn và cụ thể hơn để cải thiện hiệu "
                "suất của nó trên một nhiệm vụ cụ thể."
            ),
        },
        {
            "question": "Pretraining là gì?",
            "ground_truth": (
                "Quá trình huấn luyện ban đầu của một mô hình học máy trên một "
                "tập dữ liệu lớn và đa dạng trước khi tiến hành huấn luyện cụ "
                "thể hơn cho một nhiệm vụ nhất định."
            ),
        },
    ]


def _evaluate_strategy(strategy: ChunkingStrategy, output_dir: Path) -> dict[str, Any]:
    collection_name = _collection_for(strategy.strategy_id)
    logger.info("Strategy={} collection={}", strategy.strategy_id, collection_name)

    chunk_count = ingest(
        recreate=True,
        collection_name=collection_name,
        chunker=strategy.chunker,  # must provide split_documents()
    )

    test_cases = _default_test_cases()
    strategy_out = {
        "strategy_id": strategy.strategy_id,
        "collection_name": collection_name,
        "chunk_count": chunk_count,
        "params": strategy.params,
        "ragas_result": None,
        "error": None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        def answer_fn(q: str):
            return answer(q, collection_name=collection_name)

        result = run_evaluation(test_cases, answer_fn=answer_fn)
        strategy_out["ragas_result"] = _safe_serialize_result(result)
    except Exception as exc:  # pragma: no cover
        strategy_out["error"] = str(exc)

    _write_json(output_dir / f"{strategy.strategy_id}.json", strategy_out)
    return strategy_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple chunking strategies.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval/chunking"),
        help="Directory to write per-strategy results and summary.",
    )
    args = parser.parse_args()

    strategies = default_strategies()
    logger.info("Running {} strategies -> {}", len(strategies), args.output_dir)

    summary: dict[str, Any] = {
        "base_collection": settings.qdrant_collection,
        "output_dir": str(args.output_dir),
        "strategies": [asdict(s) for s in strategies],
        "results": [],
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }

    for strategy in strategies:
        summary["results"].append(_evaluate_strategy(strategy, args.output_dir))

    _write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()

