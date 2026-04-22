from ragas.metrics import (faithfulness, answer_relevancy, context_precision, context_recall)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.config import settings
from src.rag import _llm  
from src.store import get_embeddings


def get_ragas_metrics() -> list[ragas.metrics.Metric]:
    #1. Initialize LLM that can be used for all metrics
    langchain_llm = _llm()
    llm = LangchainLLMWrapper(langchain_llm)

    #2. Initialize embeddings for context relevance metrics
    langchain_embeddings = get_embeddings()
    embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    #3. Create metrics
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    context_precision.llm = llm
    context_precision.embeddings = embeddings
    context_recall.llm = llm
    context_recall.embeddings = embeddings

    return [faithfulness, answer_relevancy, context_precision, context_recall]


