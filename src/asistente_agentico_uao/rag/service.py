"""Capa de servicio compartida por las dos APIs (Fase 5).

``AppState`` concentra el estado vivo del sistema (Settings + Retriever +
LLM), cargado UNA vez por proceso y consumido por:

- la API REST (``api/main.py``): consulta pública ``/ask`` — data plane;
- el servicio gRPC (``grpc_impl/servicer.py``): ingesta y administración
  del índice — control plane.

Al compartir el mismo ``AppState`` ambos protocolos usan la misma colección
Chroma, el mismo singleton ``CerebrasLLM`` (una sola rotación de claves por
proceso) y el mismo modelo de embeddings. El dispositivo de inferencia se
resuelve perezosamente (importar ``torch`` solo si se necesita).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.config import Settings, resolve_embedding_device
from ..core.llm import CerebrasLLM
from .cache import SemanticCache
from .chain import RagAnswer, answer_question
from .retrieval import Retriever


@dataclass
class DocumentSummary:
    """Documento indexado (agregación de sus chunks)."""

    doc_name: str
    chunks: int


@dataclass
class AppState:
    """Estado vivo compartido por REST y gRPC (uno por proceso)."""

    config: Settings
    retriever: Retriever
    llm: CerebrasLLM
    _device: str | None = field(default=None, repr=False)
    cache: SemanticCache = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache = SemanticCache(config=self.config)

    def ask(self, question: str) -> RagAnswer:
        """Ejecuta la cadena RAG completa (data plane, solo lectura), con cache."""
        cacheada = self.cache.buscar(question)
        if cacheada is not None:
            return cacheada
        respuesta = answer_question(
            question, retriever=self.retriever, llm=self.llm, config=self.config
        )
        self.cache.guardar(question, respuesta)
        return respuesta

    def index_chunks(self) -> int:
        """Número de chunks en la colección (dispara la carga perezosa)."""
        return self.retriever.collection.count()

    def documents(self) -> list[DocumentSummary]:
        """Documentos indexados con su conteo de chunks (trazabilidad)."""
        got = self.retriever.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in got.get("metadatas") or []:
            if not isinstance(meta, dict):
                continue
            # El valor se extrae a una variable para estrecharlo con
            # isinstance: Chroma tipa la metadata como unión y el subscript
            # directo no se estrecha (el `key` de `counts` debe ser str).
            doc_name = meta.get("doc_name")
            if isinstance(doc_name, str):
                counts[doc_name] = counts.get(doc_name, 0) + 1
        return [
            DocumentSummary(doc_name=name, chunks=n)
            for name, n in sorted(counts.items())
        ]

    def on_index_rebuilt(self) -> None:
        """Tras un rebuild la coleccion anterior se borro: invalidar retriever y cache."""
        self.retriever.reset()
        self.cache.invalidar_todo()

    @property
    def device(self) -> str:
        """Dispositivo de inferencia de embeddings (resuelto una sola vez)."""
        if self._device is None:
            self._device = self.config.embedding_device or resolve_embedding_device()
        return self._device
