"""Configure Ragas metric objects with the project's LLM and embeddings."""

import ragas
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness


def get_ragas_metrics(llm, embeddings) -> list[ragas.metrics.Metric]:
    """Return Ragas metrics wired to the project's LLM and embeddings."""
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    context_precision.llm = llm
    context_precision.embeddings = embeddings
    context_recall.llm = llm
    context_recall.embeddings = embeddings

    return [faithfulness, answer_relevancy, context_precision, context_recall]