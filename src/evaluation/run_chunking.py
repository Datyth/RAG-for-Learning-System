"""Evaluate chunking strategies with Ragas.

Loops over strategies using a separate Qdrant collection per strategy:
recreate collection → ingest PDFs → run Ragas → write artifact files.

Usage:
  uv run python src/evaluation/run_chunking.py --mode recursive
  uv run python src/evaluation/run_chunking.py --mode semantic
  uv run python src/evaluation/run_chunking.py --mode both
"""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger
from tabulate import tabulate

from src.config import settings
from src.evaluation.chunking_strategies import ChunkingStrategy, default_strategies
from src.evaluation.ragas_evaluator import (
    load_test_cases,
    run_evaluation,
    summary_metrics,
    write_json,
)
from src.indexing import ingest
from src.rag import answer
from src.schemas import RagAnswer
from src.store import get_embeddings


def _evaluate_strategy(
    strategy: ChunkingStrategy, output_dir: Path, test_cases: list[dict]
) -> dict[str, object]:
    collection_name = f"{settings.qdrant_collection}__{strategy.strategy_id}"
    logger.info("Strategy={} collection={}", strategy.strategy_id, collection_name)

    chunk_count = ingest(recreate=True, collection_name=collection_name, chunker=strategy.chunker)

    result_out: dict[str, object] = {
        "strategy_id": strategy.strategy_id,
        "collection_name": collection_name,
        "chunk_count": chunk_count,
        "params": strategy.params,
        "summary_metrics": {},
        "ragas_result": None,
        "error": None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:

        def answer_fn(q: str) -> RagAnswer:
            return answer(q, collection_name=collection_name)

        result = run_evaluation(test_cases, answer_fn=answer_fn, llm_provider="vllm")
        df = result.to_pandas()
        result_out["summary_metrics"] = summary_metrics(df)

        try:
            repr_data = dict(result)
        except Exception:
            repr_data = str(result)

        result_out["ragas_result"] = {"repr": repr_data, "per_case": df.to_dict(orient="list")}

    except Exception as exc:
        logger.error("Error evaluating {}: {}", strategy.strategy_id, exc)
        logger.error(traceback.format_exc())
        result_out["error"] = str(exc)

    write_json(output_dir / f"{strategy.strategy_id}.json", result_out)
    return result_out


def main() -> None:
    logger.remove()
    logger.add(
        lambda m: print(m, end=""),
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    parser = argparse.ArgumentParser(description="Evaluate chunking strategies with Ragas.")
    parser.add_argument(
        "--mode",
        choices=["recursive", "semantic", "both"],
        default="recursive",
        help="Chunking strategy mode (default: recursive).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval/chunking"),
        help="Directory to write per-strategy results and summary.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="src/evaluation/benchmark_rag.csv",
        help="Path to the benchmark CSV file.",
    )
    args = parser.parse_args()

    test_cases = load_test_cases(args.test_path)

    embeddings = None
    if args.mode in ("semantic", "both"):
        logger.info("Loading embedding model: {}", settings.embedding_model)
        embeddings = get_embeddings()

    strategies = default_strategies(embeddings_model=embeddings, option=args.mode)
    logger.info("Running {} strategies (mode={})", len(strategies), args.mode)

    results = []
    for strategy in strategies:
        results.append(_evaluate_strategy(strategy, args.output_dir, test_cases))

    comparison_data = [
        {
            "Strategy": r["strategy_id"],
            "Chunks": r["chunk_count"],
            **{k: round(float(v), 4) for k, v in r["summary_metrics"].items()},
        }
        for r in results
        if not r.get("error") and r.get("summary_metrics")
    ]

    if comparison_data:
        df_cmp = pd.DataFrame(comparison_data)
        if "faithfulness" in df_cmp.columns:
            df_cmp = df_cmp.sort_values("faithfulness", ascending=False)
        print("\n" + "=" * 80)
        print(f"Chunking Strategy Comparison — mode={args.mode} (Average)")
        print("=" * 80)
        print(tabulate(df_cmp, headers="keys", tablefmt="psql", showindex=False))
        print("=" * 80 + "\n")
        df_cmp.to_csv(args.output_dir / f"{args.mode}_comparison_table.csv", index=False)

    write_json(
        args.output_dir / f"{args.mode}_summary.json",
        {
            "mode": args.mode,
            "base_collection": settings.qdrant_collection,
            "results": results,
            "ran_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
