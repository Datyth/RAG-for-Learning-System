import pandas as pd
from datasets import Dataset
from ragas import evaluate

from src.evaluation.metrics import get_ragas_metrics
from src.rag import answer as get_rag_answer

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.rag import _llm
from src.store import get_embeddings

from ragas.run_config import RunConfig

def prepare_evaluation_dataset(test_cases: list[dict]) -> Dataset:
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for case in test_cases:
        rag_response = get_rag_answer(case["question"])

        data["question"].append(case["question"])
        data["answer"].append(rag_response.answer)
        data["contexts"].append([chunk.text for chunk in rag_response.chunks])
        data["ground_truth"].append(case["ground_truth"])
        
    return Dataset.from_dict(data)

def run_evaluation(test_cases: list[dict]):
    eval_dataset = prepare_evaluation_dataset(test_cases)
    metrics = get_ragas_metrics()
    
    local_llm = LangchainLLMWrapper(_llm())
    local_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    # config = RunConfig(
    #     timeout=180,      # Tăng thời gian chờ lên 3 phút cho mỗi yêu cầu
    #     max_retries=10,    # Thử lại nhiều lần nếu gặp lỗi mạng
    #     max_workers=1      # QUAN TRỌNG: Chỉ chạy 1 luồng duy nhất để tránh bị Gemini chặn do quá nhanh
    # )

    config = RunConfig(
        timeout=180,
        max_retries=3,
        max_workers=4 # vLLM xử lý song song rất tốt
    )
    
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=local_llm,              
        embeddings=local_embeddings, 
        run_config=config
    )
    
    # df = result.to_pandas()
    return result

if __name__ == "__main__":


    # Example test cases
    test_cases = [
        {
            "question": "Fine tuning là gì?",
            "ground_truth": "Kĩ thuật điều chỉnh một mô hình đã được huấn luyện trước đó trên một tập dữ liệu nhỏ hơn và cụ thể hơn để cải thiện hiệu suất của nó trên một nhiệm vụ cụ thể."
        },
        {
            "question": "Pretraining là gì?",
            "ground_truth": "Quá trình huấn luyện ban đầu của một mô hình học máy trên một tập dữ liệu lớn và đa dạng trước khi tiến hành huấn luyện cụ thể hơn cho một nhiệm vụ nhất định."
        }
    ]
    
    evaluation_results = run_evaluation(test_cases)
    print(evaluation_results)