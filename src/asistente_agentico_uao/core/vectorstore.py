"""ChromaDB persistente: colección, upsert idempotente y utilidades (Fase 2).

Colección única ``uao_normativa`` con espacio ``cosine`` (los embeddings E5 se
entregan normalizados). Los IDs son deterministas (SHA-256 de doc_name +
chunk_index): correr la ingesta dos veces no duplica chunks, solo los
reemplaza.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from ..ingestion.chunk import Chunk
from .config import settings

COLLECTION_NAME = "uao_normativa"
UPSERT_BATCH = 256


def get_client():
    """PersistentClient sobre ``Data/chroma`` (crea el directorio si falta)."""
    import chromadb

    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def get_collection(client=None, rebuild: bool = False):
    """Colección de trabajo; ``rebuild=True`` borra y recrea (índice limpio)."""
    client = client or get_client()
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001, S110 - la colección aún no existía
            pass
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_id(doc_name: str, chunk_index: int) -> str:
    """ID determinista del chunk: sha256(doc_name + chunk_index)."""
    return hashlib.sha256(f"{doc_name}{chunk_index}".encode()).hexdigest()


def upsert_chunks(
    collection,
    chunks: list[Chunk],
    vectors,
    batch_size: int = UPSERT_BATCH,
) -> int:
    """Upsert por lotes; devuelve el número de chunks procesados."""
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vecs = vectors[start : start + batch_size]
        collection.upsert(
            ids=[chunk_id(c.doc_name, c.chunk_index) for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "doc_name": c.doc_name,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "is_table": c.is_table,
                    "n_tokens": c.n_tokens,
                }
                for c in batch
            ],
            embeddings=[[float(x) for x in vec] for vec in vecs],
        )
    return len(chunks)


def prune_missing_docs(collection, valid_doc_names: Iterable[str]) -> int:
    """Borra chunks de documentos que ya no están en ``Data/Documentos_MD``."""
    valid = set(valid_doc_names)
    got = collection.get(include=["metadatas"])
    stale = [
        meta_id
        for meta_id, meta in zip(got["ids"], got["metadatas"])
        if meta["doc_name"] not in valid
    ]
    if stale:
        collection.delete(ids=stale)
    return len(stale)
