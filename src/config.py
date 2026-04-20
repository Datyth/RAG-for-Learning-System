"""Application configuration"""

from dataclasses import dataclass
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _google_api_key() -> str | None:
    k = getenv("GOOGLE_API_KEY")
    return k.strip() if k else None


@dataclass(frozen=True)
class Settings:
    """Options are defined below. Put secrets in `.env`."""
    data_dir: Path
    storage_dir: Path
    qdrant_collection: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    llm_model: str
    llm_temperature: float
    top_k: int
    google_api_key: str | None


settings = Settings(
    data_dir=Path("./data"),
    storage_dir=Path("./storage/qdrant"),
    qdrant_collection="rag_chunks",
    chunk_size=1000,
    chunk_overlap=150,
    embedding_model="GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1",
    llm_model="gemini-2.5-flash",
    llm_temperature=0.1,
    top_k=5,
    google_api_key=_google_api_key(),
)
