"""Ragas evaluation harness: prepare dataset from live RAG calls and score."""

from collections.abc import Callable

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

from src.evaluation.metrics import get_ragas_metrics
from src.rag import answer as get_rag_answer, get_llm
from src.schemas import RagAnswer
from src.store import get_embeddings


def prepare_evaluation_dataset(
    test_cases: list[dict[str, str]],
    answer_fn: Callable[[str], RagAnswer] = get_rag_answer,
) -> Dataset:
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

    return Dataset.from_dict(data)


def run_evaluation(
    test_cases: list[dict[str, str]],
    answer_fn: Callable[[str], RagAnswer] = get_rag_answer,
    llm_provider: str | None = None,
):
    """Run Ragas evaluation on the given test cases and return the result."""
    eval_dataset = prepare_evaluation_dataset(test_cases, answer_fn=answer_fn)

    llm = LangchainLLMWrapper(get_llm(provider=llm_provider))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = get_ragas_metrics(llm, embeddings)

    config = RunConfig(timeout=180, max_retries=3, max_workers=4)

    return evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=config,
    )


if __name__ == "__main__":
    test_path = "/home/datpt/AIO/STA/Module11/RAG-for-Learning-System/src/evaluation/benchmark_rag.csv"
    
    dataset = Dataset.from_csv(test_path)
    
    if "ground truth" in dataset.column_names:
        dataset = dataset.rename_column("ground truth", "ground_truth")
    
    test_cases = list(dataset)

    evaluation_results = run_evaluation(test_cases, llm_provider="vllm")
    print(evaluation_results)
    
    # Optional: Xuất kết quả ra file csv nếu muốn
    # df = evaluation_results.to_pandas()
    # df.to_csv("evaluation_report.csv", index=False)
