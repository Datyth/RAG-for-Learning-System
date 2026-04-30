"""Run Ragas evaluation across multiple chunking strategies.

This script loops strategies and uses a separate Qdrant collection per strategy:
recreate collection -> ingest PDFs -> run Ragas -> write artifact files.
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from datasets import Dataset
from src.config import settings
from src.evaluation.chunking.chunkers import ChunkingStrategy, default_strategies
from src.evaluation.evaluator import run_evaluation
from src.indexing import ingest
from src.rag import answer
from src.schemas import RagAnswer
from src.store import get_embeddings

from tabulate import tabulate


def _summary_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return {}
    means = numeric.mean().to_dict()
    return {str(k): float(v) for k, v in means.items() if pd.notna(v)}
    

def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_test_cases(path: str) -> list[dict[str, str]]:
    dataset = Dataset.from_csv(path)
    if "ground truth" in dataset.column_names:
        dataset = dataset.rename_column("ground truth", "ground_truth")
    return [{"question": r["question"], "ground_truth": r["ground_truth"]} for r in dataset]


def _evaluate_strategy(
    strategy: ChunkingStrategy, output_dir: Path, test_cases: list[dict]
) -> dict[str, object]:
    collection_name = f"{settings.qdrant_collection}__{strategy.strategy_id}"
    logger.info("Strategy={} collection={}", strategy.strategy_id, collection_name)

    chunk_count = ingest(
        recreate=True,
        collection_name=collection_name,
        chunker=strategy.chunker,
    )

    strategy_out = {
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
        strategy_out["summary_metrics"] = _summary_metrics(df)
        
        try:
            repr_data = dict(result)
        except Exception:
            repr_data = str(result)

        strategy_out["ragas_result"] = {
            "repr": repr_data,
            "per_case": df.to_dict(orient="list"),
        }

    except Exception as exc:
        logger.error("Error evaluating {}: {}", strategy.strategy_id, exc)
        logger.error(traceback.format_exc())
        strategy_out["error"] = str(exc)

    _write_json(output_dir / f"{strategy.strategy_id}.json", strategy_out)
    return strategy_out


def main() -> None:
    logger.remove()
    logger.add(
        lambda m: print(m, end=""),
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    parser = argparse.ArgumentParser(description="Evaluate semantic chunking strategies.")
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

    test_cases = _load_test_cases(args.test_path)

    logger.info("Loading embedding model from settings: {}", settings.embedding_model)
    embeddings = get_embeddings()

    logger.info("Initializing Semantic Chunking Strategies...")
    strategies = default_strategies(embeddings_model=embeddings, option="semantic") #option: semantic, recursive, both
    
    logger.info("Running {} semantic strategies", len(strategies))

    results = []
    for strategy in strategies:
        res = _evaluate_strategy(strategy, args.output_dir, test_cases)
        results.append(res)

    comparison_data = []
    for res in results:
        if res.get("error") or not res.get("summary_metrics"):
            continue

        row = {"Strategy": res["strategy_id"], "Chunks": res["chunk_count"]}
        for metric_name, score in res["summary_metrics"].items():
            row[metric_name] = round(float(score), 4)

        comparison_data.append(row)

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)

        if "faithfulness" in df_comparison.columns:
            df_comparison = df_comparison.sort_values(by="faithfulness", ascending=False)

        print("\n" + "=" * 80)
        print("Comparison Table of Semantic Chunking Strategies (Average)")
        print("=" * 80)
        print(tabulate(df_comparison, headers="keys", tablefmt="psql", showindex=False))
        print("=" * 80 + "\n")

        df_comparison.to_csv(args.output_dir / "semantic_comparison_table.csv", index=False)

    summary = {
        "base_collection": settings.qdrant_collection,
        "results": results,
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(args.output_dir / "semantic_summary.json", summary)


if __name__ == "__main__":
    main()