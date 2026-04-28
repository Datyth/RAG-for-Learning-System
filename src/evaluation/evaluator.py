"""Ragas evaluation harness: build dataset from live RAG calls and score."""

from collections.abc import Callable

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

from src.evaluation.metrics import get_ragas_metrics
from src.llm import get_llm
from src.rag import answer as default_answer_fn
from src.schemas import RagAnswer
from src.store import get_embeddings


def run_evaluation(
    test_cases: list[dict[str, str]],
    *,
    answer_fn: Callable[[str], RagAnswer] = default_answer_fn,
    llm_provider: str | None = None,
    timeout_s: int = 180,
    max_retries: int = 3,
    max_workers: int = 4,
):
    """Run Ragas evaluation.

    Args: test_cases ({"question","ground_truth"}), answer_fn, llm_provider.
    Returns: ragas EvaluationResult.
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
