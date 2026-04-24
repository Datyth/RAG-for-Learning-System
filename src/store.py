"""Embeddings, Qdrant client, collection setup, and vector store."""

from collections.abc import Iterator
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import settings

_SCROLL_PAGE_SIZE = 256

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
        model_kwargs={"device": settings.hf_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def close_client() -> None:
    if get_client.cache_info().currsize == 0:
        return
    client = get_client()
    client.close()
    get_client.cache_clear()


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Return a cached local Qdrant client backed by on-disk storage."""
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.storage_dir))


def ensure_collection(recreate: bool = False, collection_name: str | None = None) -> None:
    """Create the collection and payload indexes if they do not exist."""
    client = get_client()
    name = collection_name or settings.qdrant_collection

    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False

    if not exists:
        dim = len(get_embeddings().embed_query("dimension probe"))
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
        )

    payload_schema = client.get_collection(name).payload_schema or {}

    for field_name, field_schema in INDEXED_PAYLOAD_FIELDS.items():
        existing = payload_schema.get(field_name)
        if existing is None:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=field_schema,
            )
            continue

        existing_schema = getattr(existing, "data_type", None)
        if existing_schema != field_schema:
            raise ValueError(
                f"Payload index for '{field_name}' has schema "
                f"{existing_schema!r}, expected {field_schema!r}."
            )


def scroll_all(
    collection_name: str,
    scroll_filter: qmodels.Filter | None = None,
    with_payload: bool | list[str] = True,
) -> Iterator[list]:
    """Yield pages of Qdrant points (no vectors) until the collection is exhausted."""
    client = get_client()
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )
        yield points
        if next_offset is None:
            break
        offset = next_offset


def get_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_client(),
        collection_name=collection_name or settings.qdrant_collection,
        embedding=get_embeddings(),
    )
