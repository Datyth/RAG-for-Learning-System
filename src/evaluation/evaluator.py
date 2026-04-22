import pandas as pd
from datasets import Dataset
from ragas import evaluate

from src.evaluation.metrics import get_ragas_metrics
from src.rag import answer as get_rag_answer

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
    
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
    )
    
    df = result.to_pandas()
    return df