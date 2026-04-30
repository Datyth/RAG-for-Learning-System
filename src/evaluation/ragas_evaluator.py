"""Ragas evaluator utilities for offline benchmarks."""

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import ragas
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from src.llm import get_llm
from src.rag import answer as default_answer_fn
from src.schemas import RagAnswer
from src.store import get_embeddings


def get_ragas_metrics(llm, embeddings) -> list[ragas.metrics.Metric]:
    """Return Ragas metrics wired to the project's LLM and embeddings."""

    faithfulness.llm = llm
    answer_relevancy.llm = llm
    context_precision.llm = llm
    context_precision.embeddings = embeddings
    context_recall.llm = llm
    context_recall.embeddings = embeddings

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def load_test_cases(path: str) -> list[dict[str, str]]:
    """Load benchmark CSV rows as {"question","ground_truth"} pairs."""

    dataset = Dataset.from_csv(path)
    if "ground truth" in dataset.column_names:
        dataset = dataset.rename_column("ground truth", "ground_truth")
    return [{"question": r["question"], "ground_truth": r["ground_truth"]} for r in dataset]


def summary_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Compute mean scores for all numeric metric columns."""

    if df.empty:
        return {}
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return {}
    means = numeric.mean().to_dict()
    return {str(k): float(v) for k, v in means.items() if pd.notna(v)}


def write_json(path: Path, obj: object) -> None:
    """Write JSON to disk (UTF-8, pretty-printed)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_evaluation(
    test_cases: list[dict[str, str]],
    *,
    answer_fn: Callable[[str], RagAnswer] = default_answer_fn,
    llm_provider: str | None = None,
    timeout_s: int = 180,
    max_retries: int = 3,
    max_workers: int = 4,
):
    """Run Ragas evaluation over live RAG answers.

    Args:
        test_cases: List of {"question","ground_truth"} rows.
        answer_fn: Function mapping question -> RagAnswer.
        llm_provider: Optional override for evaluation LLM provider.
    Returns:
        ragas EvaluationResult.
    """

    data: dict[str, list] = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }
    for case in test_cases:
        rag_response = answer_fn(case["question"])
        data["user_input"].append(case["question"])
        data["response"].append(rag_response.answer)
        data["retrieved_contexts"].append([chunk.text for chunk in rag_response.chunks])
        data["reference"].append(case["ground_truth"])

    eval_dataset = Dataset.from_dict(data)

    llm = LangchainLLMWrapper(get_llm(provider=llm_provider))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = get_ragas_metrics(llm, embeddings)
    config = RunConfig(
        timeout=timeout_s,
        max_retries=max_retries,
        max_workers=max_workers,
    )

    return evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=config,
    )
