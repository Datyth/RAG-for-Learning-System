"""Evaluate RAG pipeline with a reranking step using Ragas and benchmark_rag.csv.

Creates a collection with RecursiveChunker (1000, 150), ingests PDFs,
runs Ragas over the reranking pipeline, and writes artifacts.
"""

from __future__ import annotations

import os

# Must be set before sentence_transformers/torch initializes CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger
from sentence_transformers import CrossEncoder
from tabulate import tabulate

from src.config import settings
from src.evaluation.chunking_strategies import RecursiveChunker
from src.evaluation.ragas_evaluator import (
    load_test_cases,
    run_evaluation,
    summary_metrics,
    write_json,
)
from src.indexing import ingest
from src.llm import invoke_llm
from src.rag import ANSWER_TEMPLATE, format_citations, render_prompt, retrieve
from src.schemas import RagAnswer

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
CHUNKING_STRATEGY_ID = "rc_1000_150"


def answer_with_reranker(
    question: str,
    collection_name: str,
    reranker: CrossEncoder,
    initial_k: int = 15,
    rerank_k: int = 5,
    filters: dict[str, object] | None = None,
) -> RagAnswer:
    chunks = retrieve(question, k=initial_k, filters=filters, collection_name=collection_name)
    if not chunks:
        return RagAnswer(
            question=question,
            answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời.",
        )

    scores = reranker.predict([[question, chunk.text] for chunk in chunks])
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)

    reranked = sorted(chunks, key=lambda c: c.score, reverse=True)[:rerank_k]
    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=reranked)
    text = invoke_llm(prompt)

    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(reranked),
        chunks=reranked,
    )


def main() -> None:
    logger.remove()
    logger.add(
        lambda m: print(m, end=""),
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    parser = argparse.ArgumentParser(description="Evaluate reranking strategy with Ragas.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval/reranking"),
        help="Directory to write reranking results.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="src/evaluation/benchmark_rag.csv",
        help="Path to the benchmark CSV file.",
    )
    args = parser.parse_args()

    try:
        test_cases = load_test_cases(args.test_path)
        logger.info("Loaded {} test cases from {}", len(test_cases), args.test_path)
    except Exception as exc:
        logger.error("Failed to load dataset: {}", exc)
        sys.exit(1)

    if not test_cases:
        logger.error("Dataset is empty.")
        sys.exit(1)

    try:
        logger.info("Loading reranker: {} on GPU:1", RERANKER_MODEL)
        reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device="cuda")
    except Exception as exc:
        logger.error("Failed to initialize reranker: {}", exc)
        sys.exit(1)

    collection_name = f"{settings.qdrant_collection}_rerank_{CHUNKING_STRATEGY_ID}"
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=150)

    logger.info("Ingesting into collection: {}", collection_name)
    chunk_count = ingest(recreate=True, collection_name=collection_name, chunker=chunker)
    logger.info("Ingested {} chunks.", chunk_count)

    eval_out: dict[str, object] = {
        "pipeline": "retrieve_rerank_generate",
        "reranker_model": RERANKER_MODEL,
        "chunking_strategy": CHUNKING_STRATEGY_ID,
        "collection_name": collection_name,
        "chunk_count": chunk_count,
        "summary_metrics": {},
        "error": None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        logger.info("Starting Ragas evaluation (vLLM)...")

        def eval_fn(q: str) -> RagAnswer:
            return answer_with_reranker(q, collection_name=collection_name, reranker=reranker)

        result = run_evaluation(test_cases, answer_fn=eval_fn, llm_provider="vllm")
        df = result.to_pandas()
        eval_out["summary_metrics"] = summary_metrics(df)
        eval_out["ragas_result"] = {"per_case": df.to_dict(orient="list")}

    except Exception as exc:
        logger.error("Ragas evaluation failed: {}", exc)
        logger.error(traceback.format_exc())
        eval_out["error"] = str(exc)

    write_json(args.output_dir / "reranking_report.json", eval_out)

    if eval_out.get("summary_metrics"):
        df_cmp = pd.DataFrame([eval_out["summary_metrics"]])
        print("\n" + "=" * 80)
        print(f"Reranking Evaluation (GPU:1) — {RERANKER_MODEL}")
        print("=" * 80)
        print(tabulate(df_cmp, headers="keys", tablefmt="psql", showindex=False))
        print("=" * 80 + "\n")
        df_cmp.to_csv(args.output_dir / "reranking_metrics.csv", index=False)
        logger.info("Results saved to {}", args.output_dir)


if __name__ == "__main__":
    main()
