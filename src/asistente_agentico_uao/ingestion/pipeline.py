"""Pipeline de ingesta reutilizable (Fase 5): markdown → chunks → Chroma.

Extrae la lógica del CLI ``scripts/ingest.py`` (Fase 2) a funciones
invocables para que el CLI, el servicio gRPC de administración
(``grpc_impl.servicer``, plano de control) y los tests ejecuten el mismo
código.

- ``markdown_paths(file_match)``: markdowns a procesar (ordenados).
- ``pdf_stems()`` / ``resolve_doc_name()``: ``doc_name`` para citación.
- ``validate_chunk_budget(seq_len)``: presupuesto de tokens vs ventana E5.
- ``ingest_documents(...)``: chunking + embeddings + upsert por documento;
  el callback ``on_summary`` habilita progreso en vivo (streaming gRPC).
- ``prune_index()``: borra chunks de documentos ya ausentes.

Las dependencias pesadas (modelo de embeddings, tokenizador, colección
Chroma) son inyectables: si no se pasan, se cargan perezosamente. Esto
permite tests con dobles de prueba sin descargar el modelo real.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..core.config import settings
from ..core.embeddings import (
    embed_passages,
    get_model,
    max_seq_length,
    token_counter,
)
from ..core.vectorstore import get_collection, prune_missing_docs, upsert_chunks
from .chunk import chunk_markdown


@dataclass
class IngestSummary:
    """Resultado de ingestar un documento (fila del reporte CLI/progreso gRPC)."""

    doc_name: str
    n_chunks: int
    seconds: float
    token_p50: int = 0
    token_p90: int = 0
    token_max: int = 0


def pdf_stems() -> dict[str, str]:
    """Mapa stem -> nombre real del PDF (para doc_name de citación)."""
    return {p.stem: p.name for p in settings.docs_dir.glob("*.pdf")}


def resolve_doc_name(md_path: Path, stems: dict[str, str]) -> str:
    """Nombre del documento tal como se citará en las fuentes."""
    return stems.get(md_path.stem, md_path.stem)


def markdown_paths(file_match: str | None = None) -> list[Path]:
    """Markdowns ordenados de ``markdown_dir``; ``file_match`` filtra parcial."""
    mds = sorted(settings.markdown_dir.glob("*.md"))
    if file_match:
        mds = [p for p in mds if file_match.lower() in p.name.lower()]
    return mds


def validate_chunk_budget(seq_len: int) -> None:
    """El chunk final debe caber en la ventana del modelo de embeddings."""
    if settings.chunk_max_tokens > seq_len - 4:
        raise ValueError(
            f"chunk_max_tokens={settings.chunk_max_tokens} excede "
            f"max_seq_length={seq_len}: los chunks se truncarían."
        )


def ingest_documents(
    file_match: str | None = None,
    rebuild: bool = False,
    *,
    model=None,
    seq_len: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
    collection=None,
    on_summary: Callable[[IngestSummary], None] | None = None,
) -> list[IngestSummary]:
    """Ingresa los markdowns al índice; devuelve un resumen por documento.

    Idempotente: los IDs son sha256(doc_name + chunk_index), así que
    re-ingestar reemplaza los mismos chunks en vez de duplicarlos. Con
    ``rebuild=True`` la colección se borra y recrea antes de indexar.
    ``on_summary`` se invoca tras cada documento (progreso en vivo).
    """
    mds = markdown_paths(file_match)
    if not mds:
        raise ValueError(
            f"No hay markdown en {settings.markdown_dir}"
            + (f" que coincida con '{file_match}'" if file_match else "")
        )

    if model is None:
        model = get_model()
    validate_chunk_budget(seq_len if seq_len is not None else max_seq_length())
    if count_tokens is None:
        count_tokens = token_counter()
    if collection is None:
        collection = get_collection(rebuild=rebuild)

    stems = pdf_stems()
    summaries: list[IngestSummary] = []
    for md in mds:
        started = time.perf_counter()
        doc_name = resolve_doc_name(md, stems)
        chunks = chunk_markdown(
            md.read_text(encoding="utf-8"),
            doc_name=doc_name,
            count_tokens=count_tokens,
            target_tokens=settings.chunk_size_tokens,
            max_tokens=settings.chunk_max_tokens,
            min_tokens=settings.chunk_min_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        if chunks:
            vectors = embed_passages([c.text for c in chunks])
            upsert_chunks(collection, chunks, vectors)

        toks = sorted(c.n_tokens for c in chunks)
        summary = IngestSummary(
            doc_name=doc_name,
            n_chunks=len(chunks),
            seconds=time.perf_counter() - started,
            token_p50=toks[len(toks) // 2] if toks else 0,
            token_p90=toks[min(len(toks) - 1, int(len(toks) * 0.9))] if toks else 0,
            token_max=toks[-1] if toks else 0,
        )
        summaries.append(summary)
        if on_summary is not None:
            on_summary(summary)
    return summaries


def prune_index(collection=None) -> int:
    """Borra chunks de documentos ausentes en ``markdown_dir``; los cuenta."""
    collection = collection if collection is not None else get_collection()
    stems = pdf_stems()
    valid = {resolve_doc_name(p, stems) for p in settings.markdown_dir.glob("*.md")}
    return prune_missing_docs(collection, valid)
