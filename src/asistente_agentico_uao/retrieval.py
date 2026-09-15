"""Motor de recuperación sobre ChromaDB (Fase 3).

``Retriever.retrieve(question)``: embed de la pregunta con prefijo ``query:``
(E5, ver ``embeddings.embed_query``), búsqueda bruta de ``top_k * 2``
candidatos en la colección, re-ordenamiento por similitud coseno
(``score = 1 - distance``, espacio ``cosine``), descarte de lo que no supere
``min_similarity`` y corte a ``top_k``.

Si ningún fragmento supera el umbral la lista queda vacía: la cadena RAG
(Fase 4) responderá "no tengo información suficiente" sin llamar al LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, settings
from .embeddings import embed_query

# Factor de sobre-muestreo en la query bruta: pide top_k * OVERSAMPLE para
# que, tras el filtro por umbral, queden suficientes candidatos para top_k.
OVERSAMPLE = 2


@dataclass
class RetrievedChunk:
    """Fragmento recuperado, listo para citar (doc + sección) en la respuesta."""

    doc_name: str
    section: str
    text: str
    score: float  # similitud coseno pregunta-chunk (1.0 = idénticos)
    chunk_index: int
    is_table: bool = False


def _as_str(value: object) -> str:
    """Chroma puede devolver None en metadata: se normaliza a str."""
    return value if isinstance(value, str) else ""


def _as_int(value: object, default: int = -1) -> int:
    """Metadata de Chroma tipada como ``str | int | float | bool | None``.

    Solo ``int`` (y ``bool``, que es subclase) es convertible con seguridad;
    ``str``/``float`` se aceptan si el valor es realmente numérico y el resto
    cae al ``default``. Evita pasar un ``SparseVector`` a ``int()``.
    """
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


class Retriever:
    """Recuperador top-k con umbral de similitud sobre la colección Chroma."""

    def __init__(self, collection=None, config: Settings | None = None):
        self.settings = config or settings
        self._collection = collection  # inyectable (tests); perezosa si no

    @property
    def collection(self):
        if self._collection is None:
            from .vectorstore import get_collection

            self._collection = get_collection()
        return self._collection

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Devuelve hasta ``top_k`` chunks con ``score >= min_similarity``."""
        top_k = self.settings.top_k
        if self.collection.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=[embed_query(question).tolist()],
            n_results=max(1, top_k * OVERSAMPLE),
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        distances = result.get("distances") or []
        if not documents or not metadatas or not distances:
            return []

        candidates: list[RetrievedChunk] = []
        for doc, meta, dist in zip(
            documents[0], metadatas[0], distances[0], strict=True
        ):
            score = 1.0 - float(dist)  # espacio cosine: 0 distancia = idénticos
            if score < self.settings.min_similarity:
                continue
            candidates.append(
                RetrievedChunk(
                    doc_name=_as_str(meta.get("doc_name")),
                    section=_as_str(meta.get("section")),
                    text=doc if isinstance(doc, str) else "",
                    score=score,
                    chunk_index=_as_int(meta.get("chunk_index")),
                    is_table=bool(meta.get("is_table", False)),
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Renderiza los chunks como bloques numerados para citación del LLM.

    Formato (plan 3.2)::

        [1] (doc_name — sección): texto

    El número de bloque es la referencia que el prompt de Fase 4 exigirá
    citar al responder.
    """
    blocks: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] ({chunk.doc_name}"
        if chunk.section:
            header += f" — {chunk.section}"
        blocks.append(f"{header}): {chunk.text.strip()}")
    return "\n\n".join(blocks)