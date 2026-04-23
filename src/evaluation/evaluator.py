"""Ragas evaluation harness: prepare dataset from live RAG calls and score."""

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig

from src.evaluation.metrics import get_ragas_metrics
from src.rag import answer as get_rag_answer, get_llm
from src.store import get_embeddings


def prepare_evaluation_dataset(test_cases: list[dict]) -> Dataset:
    """Build a HuggingFace Dataset by running RAG on each test case."""
    data: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for case in test_cases:
        rag_response = get_rag_answer(case["question"])
        data["question"].append(case["question"])
        data["answer"].append(rag_response.answer)
        data["contexts"].append([chunk.text for chunk in rag_response.chunks])
        data["ground_truth"].append(case["ground_truth"])

    return Dataset.from_dict(data)


def run_evaluation(test_cases: list[dict]):
    """Run Ragas evaluation on the given test cases and return the result."""
    eval_dataset = prepare_evaluation_dataset(test_cases)

    llm = LangchainLLMWrapper(get_llm())
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = get_ragas_metrics(llm, embeddings)

    config = RunConfig(timeout=180, max_retries=3, max_workers=4)

    return evaluate(
        dataset=eval_dataset, metrics=metrics,
        llm=llm, embeddings=embeddings,
        run_config=config,
    )


if __name__ == "__main__":
    test_cases = [
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

    evaluation_results = run_evaluation(test_cases)
    print(evaluation_results)
