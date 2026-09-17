"""Servicer gRPC ``IndexAdmin`` (Fase 5): control plane de ingesta.

Implementa el contrato de ``protos/index_admin.proto`` sobre el
``AppState`` compartido con la API REST. Funciones (NUNCA duplica /ask):

- ``Ingest``: ingesta de markdowns al índice con progreso en streaming
  (un ``IngestProgress`` por documento). Ejecuta el pipeline pesado
  (chunking + embeddings) en un hilo worker y va reenviando los resúmenes
  al stream; ``rebuild=True`` invalida la colección cacheada del Retriever.
- ``PruneIndex``: borra chunks de documentos ausentes.
- ``IndexStatus``: chunks, documentos, modelo y dispositivo de inferencia.

Códigos de estado: ``INVALID_ARGUMENT`` sin markdowns que procesar,
``INTERNAL`` ante fallos del pipeline (misma semántica que 422/500 en REST).
"""

from __future__ import annotations

import asyncio
import queue
import threading

import grpc

from ..ingestion.pipeline import (
    IngestSummary,
    ingest_documents,
    markdown_paths,
    prune_index,
)
from .stubs import index_admin_pb2, index_admin_pb2_grpc

_DONE = object()  # sentinela de fin de ingesta en la cola worker→stream


class IndexAdminServicer(index_admin_pb2_grpc.IndexAdminServicer):
    """Control plane del índice sobre el ``AppState`` compartido."""

    def __init__(self, state):
        self._state = state

    async def Ingest(self, request, context):
        total = len(markdown_paths(request.file_match or None))
        if total == 0:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "No hay markdown que indexar (file_match sin coincidencias).",
            )

        progress: queue.Queue = queue.Queue()

        def worker():
            try:
                ingest_documents(
                    file_match=request.file_match or None,
                    rebuild=request.rebuild,
                    on_summary=progress.put,
                )
                progress.put(_DONE)
            except BaseException as exc:  # noqa: BLE001 - se reporta por el stream
                progress.put(exc)

        threading.Thread(target=worker, daemon=True, name="ingesta-indexadmin").start()

        loop = asyncio.get_running_loop()
        done = 0
        while True:
            item = await loop.run_in_executor(None, progress.get)
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                await context.abort(
                    grpc.StatusCode.INTERNAL, f"La ingesta falló: {item}"
                )
            done += 1
            summary: IngestSummary = item
            yield index_admin_pb2.IngestProgress(
                doc_name=summary.doc_name,
                chunks=summary.n_chunks,
                seconds=summary.seconds,
                token_p50=summary.token_p50,
                token_p90=summary.token_p90,
                token_max=summary.token_max,
                done=done,
                total_docs=total,
            )
        if request.rebuild:
            # La colección anterior fue borrada por el pipeline: el Retriever
            # (compartido con la REST en el mismo proceso) debe recargarla.
            self._state.on_index_rebuilt()

    async def PruneIndex(self, request, context):
        try:
            removed = await asyncio.to_thread(prune_index)
        except Exception as exc:  # noqa: BLE001 - error controlado INTERNAL
            await context.abort(grpc.StatusCode.INTERNAL, f"El prune falló: {exc}")
        if removed:
            # Los chunks podados pueden ser la fuente de respuestas ya
            # cacheadas: se invalida el cache para no servir citas muertas.
            self._state.cache.invalidar_todo()
        return index_admin_pb2.PruneResult(removed_chunks=removed)

    async def IndexStatus(self, request, context):
        def _collect():
            collection = self._state.retriever.collection
            count = collection.count()
            docs: set[str] = set()
            if count:
                got = collection.get(include=["metadatas"])
                for meta in got.get("metadatas") or []:
                    if not isinstance(meta, dict):
                        continue
                    # Variable local para estrechar el tipo (misma razón que
                    # en AppState.documents).
                    doc_name = meta.get("doc_name")
                    if isinstance(doc_name, str):
                        docs.add(doc_name)
            return count, docs

        try:
            chunks, docs = await asyncio.to_thread(_collect)
        except Exception as exc:  # noqa: BLE001 - error controlado INTERNAL
            await context.abort(
                grpc.StatusCode.INTERNAL, f"No se pudo leer el índice: {exc}"
            )
        return index_admin_pb2.IndexStatusResponse(
            chunks=chunks,
            documents=len(docs),
            embedding_model=self._state.config.embedding_model,
            device=self._state.device,
        )
