"""Embeddings, Qdrant client, collection setup, and vector store."""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import settings

INDEXED_PAYLOAD_FIELDS: dict[str, qmodels.PayloadSchemaType] = {
    "metadata.document_id": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.filename": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.source": qmodels.PayloadSchemaType.KEYWORD,
    "metadata.page": qmodels.PayloadSchemaType.INTEGER,
    "metadata.section": qmodels.PayloadSchemaType.KEYWORD,
}


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model, 
        encode_kwargs={"normalize_embeddings": True},
    )

def close_client() -> None:
    client = get_client()
    client.close()
    get_client.cache_clear()

def embedding_dim() -> int:
    return len(get_embeddings().embed_query("dimension probe"))


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Return a cached local Qdrant client backed by on-disk storage."""
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.storage_dir))


def ensure_collection(recreate: bool = False) -> None:
    """Create the collection and payload indexes if they do not exist."""
    client = get_client()
    name = settings.qdrant_collection
    dim = embedding_dim()

    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
        )

    collection_info = client.get_collection(name)
    existing_payload_schema = collection_info.payload_schema or {}

    for field_name, field_schema in INDEXED_PAYLOAD_FIELDS.items():
        existing_index = existing_payload_schema.get(field_name)
        if existing_index is None:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=field_schema,
            )
            continue

        existing_schema = getattr(existing_index, "data_type", None)
        if existing_schema != field_schema:
            raise ValueError(
                f"Payload index for '{field_name}' already exists with schema "
                f"{existing_schema!r}, expected {field_schema!r}."
            )
def get_vector_store() -> QdrantVectorStore:
    """Return a QdrantVectorStore bound to the current client and collection."""
    return QdrantVectorStore(
        client=get_client(),
        collection_name=settings.qdrant_collection,
        embedding=get_embeddings(),
    )
