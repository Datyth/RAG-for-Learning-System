"""Evaluate RAG pipeline with a Reranking step using Ragas and benchmark_rag.csv.

This script creates a collection with Recursive Chunking (1500, 150),
ingests PDFs, runs Ragas over the Reranking pipeline, and writes artifacts.
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import json
import traceback
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger
from datasets import Dataset
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tabulate import tabulate

from src.config import settings
from src.llm import invoke_llm
from src.rag import retrieve, render_prompt, format_citations, ANSWER_TEMPLATE
from src.schemas import RagAnswer
from src.evaluation.evaluator import run_evaluation
from src.indexing import ingest

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

try:
    logger.info(f"Downloading Reranker model: {RERANKER_MODEL_NAME} on GPU:1...")
    reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=512, device='cuda')
except Exception as e:
    logger.error(f"Critical error occurred while initializing Reranker: {e}")
    sys.exit(1)


def _summary_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Summarize average metrics from Ragas results."""
    if df.empty:
        return {}
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return {}
    means = numeric.mean().to_dict()
    return {str(k): float(v) for k, v in means.items() if pd.notna(v)}
    

def _write_json(path: Path, obj: object) -> None:
    """Write results to a JSON file with readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_test_cases(path: str) -> list[dict[str, str]]:
    """Load evaluation data from a CSV file using datasets[cite: 1]."""
    dataset = Dataset.from_csv(path)
    if "ground truth" in dataset.column_names:
        dataset = dataset.rename_column("ground truth", "ground_truth")
    return [{"question": r["question"], "ground_truth": r["ground_truth"]} for r in dataset]


def answer_with_reranker(
    question: str, 
    collection_name: str,
    initial_k: int = 15, 
    rerank_k: int = 5,
    filters: dict[str, object] | None = None,
) -> RagAnswer:
    """Function for RAG pipeline with integrated Reranker."""
    # 1. Retrieval
    chunks = retrieve(question, k=initial_k, filters=filters, collection_name=collection_name)
    
    if not chunks:
        return RagAnswer(
            question=question,
            answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời."
        )

    # 2. Reranking
    pairs = [[question, chunk.text] for chunk in chunks]
    scores = reranker.predict(pairs)
    
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)
        
    reranked_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)[:rerank_k]

    # 3. Generation
    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=reranked_chunks)
    text = invoke_llm(prompt)

    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(reranked_chunks),
        chunks=reranked_chunks,
    )


def main() -> None:
    logger.remove()
    logger.add(
        lambda m: print(m, end=""),
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    
    parser = argparse.ArgumentParser(description="Evaluate Reranking Strategy.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval/reranking"),
        help="Directory to write reranking results and summary.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="src/evaluation/benchmark_rag.csv",
        help="Path to the benchmark CSV file.",
    )
    args = parser.parse_args()

    # 1. Load test cases
    try:
        test_cases = _load_test_cases(args.test_path)
        logger.info(f"Loaded {len(test_cases)} test cases from {args.test_path}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    if not test_cases:
        logger.error("Dataset is empty!")
        sys.exit(1)

    # 2. Cấu hình Chunking và tạo Collection
    chunker_id = "recursive_1500_150"
    collection_name = f"{settings.qdrant_collection}_rerank_{chunker_id}"
    
    logger.info("Initializing Recursive Chunking (1500, 150)...")
    recursive_chunker = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    logger.info(f"Ingesting into collection: {collection_name}")
    chunk_count = ingest(
        recreate=True,
        collection_name=collection_name,
        chunker=recursive_chunker,
    )
    logger.info(f"Ingested {chunk_count} chunks.")

    # 3. Đánh giá bằng Ragas
    eval_out = {
        "pipeline": "retrieve_rerank_generate",
        "reranker_model": RERANKER_MODEL_NAME,
        "chunking_strategy": chunker_id,
        "collection_name": collection_name,
        "summary_metrics": {},
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        logger.info("Starting Ragas Evaluation (LLM Provider: vLLM)...")
        
        def eval_answer_fn(q: str) -> RagAnswer:
            return answer_with_reranker(q, collection_name=collection_name)

        result = run_evaluation(test_cases, answer_fn=eval_answer_fn, llm_provider="vllm")
        df = result.to_pandas()
        
        eval_out["summary_metrics"] = _summary_metrics(df)
        eval_out["ragas_result"] = {
            "per_case": df.to_dict(orient="list"),
        }

    except Exception as exc:
        logger.error(f"Error during Ragas evaluation: {exc}")
        logger.error(traceback.format_exc())
        eval_out["error"] = str(exc)

    # 4. Xuất kết quả
    _write_json(args.output_dir / "reranking_report.json", eval_out)

    if eval_out.get("summary_metrics"):
        df_comparison = pd.DataFrame([eval_out["summary_metrics"]])
        print("\n" + "=" * 80)
        print(f"Reranking Evaluation (GPU 1) - {RERANKER_MODEL_NAME}")
        print("=" * 80)
        print(tabulate(df_comparison, headers="keys", tablefmt="psql", showindex=False))
        print("=" * 80 + "\n")
        
        df_comparison.to_csv(args.output_dir / "reranking_metrics.csv", index=False)
        logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()