"""Application configuration. All runtime options live here; `.env` is for secrets only."""

from dataclasses import dataclass
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _secret(name: str) -> str | None:
    v = getenv(name)
    return v.strip() if v and v.strip() else None


@dataclass(frozen=True)
class Settings:
    """Edit runtime options below. Only secrets should ever come from `.env`."""

    data_dir: Path
    storage_dir: Path
    qdrant_collection: str

    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    top_k: int

    llm_provider: str  # "hf_local" | "gemini" | "vllm"
    llm_temperature: float

    hf_model: str
    hf_device: int  # -1 = CPU, 0+ = CUDA device index
    hf_max_new_tokens: int

    gemini_model: str
    google_api_key: str | None

    vllm_api_base: str
    vllm_api_key: str

    summarize_batch_size: int
    summarize_retrieval_k: int
    quiz_default_count: int
    flashcards_default_count: int
    generation_retrieval_k: int


settings = Settings(
    data_dir=Path("./data"),
    storage_dir=Path("./storage/qdrant"),
    qdrant_collection="rag_chunks",
    chunk_size=1000,
    chunk_overlap=150,
    embedding_model="GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1",
    top_k=5,
    llm_provider="vllm",
    llm_temperature=0.1,
    hf_model="/mnt/pretrained_fm/Qwen_Qwen3-4B-Instruct-2507",
    hf_device=1,
    hf_max_new_tokens=2048,
    gemini_model="gemini-2.5-flash",
    google_api_key=_secret("GOOGLE_API_KEY"),
    vllm_api_base="http://localhost:8001/v1",
    vllm_api_key="EMPTY",
    summarize_batch_size=10,
    summarize_retrieval_k=12,
    quiz_default_count=8,
    flashcards_default_count=15,
    generation_retrieval_k=16,
)
