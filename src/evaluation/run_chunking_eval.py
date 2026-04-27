"""Run Ragas evaluation across multiple chunking strategies.

This script loops strategies and uses a separate Qdrant collection per strategy:
recreate collection -> ingest PDFs -> run Ragas -> write artifact files.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd  
from loguru import logger

from src.config import settings
from src.evaluation.chunking.chunkers import ChunkingStrategy, default_strategies
from src.evaluation.evaluator import run_evaluation
from src.indexing import ingest
from src.rag import answer

from datasets import Dataset
from tabulate import tabulate


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
                    payload["pandas"] = value.to_dict() 
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


def _evaluate_strategy(strategy: ChunkingStrategy, output_dir: Path, test_cases: list[dict]) -> dict[str, Any]:
    collection_name = _collection_for(strategy.strategy_id)
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

        def answer_fn(q: str):
            return answer(q, collection_name=collection_name)

        result = run_evaluation(test_cases, answer_fn=answer_fn, llm_provider="vllm")
        strategy_out["ragas_result"] = _safe_serialize_result(result)
        
    except Exception as exc:
        logger.error("Error evaluating {}: {}", strategy.strategy_id, exc)
        logger.error(traceback.format_exc())
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
    parser.add_argument(
        "--test-path",
        type=str,
        default="/home/datpt/AIO/STA/Module11/RAG-for-Learning-System/src/evaluation/benchmark_rag.csv", 
        help="Path to the benchmark CSV file."
    )
    args = parser.parse_args()

    dataset = Dataset.from_csv(args.test_path)
    if "ground truth" in dataset.column_names:
        dataset = dataset.rename_column("ground truth", "ground_truth")
    test_cases = list(dataset)

    strategies = default_strategies()
    logger.info("Running {} strategies", len(strategies))

    results = []
    for strategy in strategies:
        res = _evaluate_strategy(strategy, args.output_dir, test_cases)
        results.append(res)

    comparison_data = []
    for res in results:
        if res.get("error") or not res.get("summary_metrics"):
            continue
            
        row = {
            "Strategy": res["strategy_id"],
            "Chunks": res["chunk_count"]
        }
        for metric_name, score in res["summary_metrics"].items():
            if pd.isna(score):
                row[metric_name] = 0.0
            else:
                row[metric_name] = round(float(score), 4)
            
        comparison_data.append(row)

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        if "faithfulness" in df_comparison.columns:
            df_comparison = df_comparison.sort_values(by="faithfulness", ascending=False)

        print("\n" + "="*80)
        print("BẢNG SO SÁNH HIỆU SUẤT CÁC CHIẾN LƯỢC CHUNKING (TRUNG BÌNH)")
        print("="*80)
        print(tabulate(df_comparison, headers='keys', tablefmt='psql', showindex=False))
        print("="*80 + "\n")

        df_comparison.to_csv(args.output_dir / "comparison_table.csv", index=False)

    summary = {
        "base_collection": settings.qdrant_collection,
        "results": results,
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(args.output_dir / "summary.json", summary)

if __name__ == "__main__":
    main()
