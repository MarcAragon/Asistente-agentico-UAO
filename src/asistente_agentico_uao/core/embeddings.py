"""Modelo de embeddings E5 + utilidades de tokenización (Fase 2).

Usa ``intfloat/multilingual-e5-base`` (ventana de 512 tokens): los chunks de
~400 tokens caben completos, a diferencia de mpnet-base (128 tokens, que
truncaría silenciosamente). E5 exige prefijos asimétricos:

- ``passage:`` para los chunks al indexar (``embed_passages``).
- ``query:``   para las preguntas al recuperar (``embed_query``, Fase 3).

Todos los vectores se normalizan (``normalize_embeddings=True``) → la
similitud coseno es un simple producto punto y Chroma usa ``cosine``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import resolve_embedding_device, settings

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_model = None  # singleton perezoso (carga única por proceso)


def get_model():
    """Carga (una vez) el SentenceTransformer con el device resuelto."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(
            settings.embedding_model, device=resolve_embedding_device()
        )
    return _model


def token_counter() -> Callable[[str], int]:
    """Contador de tokens con el tokenizador real del modelo (sin especiales).

    Se usa para presupuestar el chunking; los 2 tokens de especiales (<s></s>)
    quedan de margen entre ``chunk_max_tokens`` y los 512 del modelo.
    """
    tokenizer = get_model().tokenizer

    def count(text: str) -> int:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(ids[0] if ids and isinstance(ids[0], list) else ids)

    return count


def max_seq_length() -> int:
    """Ventana máxima del modelo (p.ej. 512 en E5)."""
    return int(get_model().max_seq_length)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embeddings de chunks/indexación con prefijo ``passage:``."""
    model = get_model()
    return model.encode(
        [PASSAGE_PREFIX + t for t in texts],
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 32,
    )


def embed_query(text: str) -> np.ndarray:
    """Embedding de una pregunta con prefijo ``query:`` (Fase 3)."""
    model = get_model()
    return model.encode(
        [QUERY_PREFIX + text],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
