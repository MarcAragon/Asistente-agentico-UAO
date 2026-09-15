"""Pruebas unitarias del servicio gRPC ``IndexAdmin`` (Fase 5).

Control plane de ingesta sobre un servidor ``grpc.aio`` in-proceso
(puerto efímero) con el pipeline mockeado: sin modelo de embeddings real,
sin Chroma ni red externa. Verifica streaming de progreso, códigos de
estado (INVALID_ARGUMENT/INTERNAL) y los RPC PruneIndex/IndexStatus.
"""

import asyncio
from types import SimpleNamespace

import grpc
import pytest

from asistente_agentico_uao.grpc_impl import servicer as servicer_mod
from asistente_agentico_uao.grpc_impl.servicer import IndexAdminServicer
from asistente_agentico_uao.grpc_impl.stubs import (
    index_admin_pb2,
    index_admin_pb2_grpc,
)
from asistente_agentico_uao.ingestion.pipeline import IngestSummary


class FakeCollection:
    def __init__(self, count: int, doc_names: list[str]):
        self._count = count
        self._doc_names = doc_names

    def count(self):
        return self._count

    def get(self, include=None):
        return {
            "metadatas": [
                {"doc_name": d, "chunk_index": 0} for d in self._doc_names
            ]
        }


class FakeRetriever:
    def __init__(self, collection):
        self.collection = collection
        self.resets = 0

    def reset(self):
        self.resets += 1


class FakeState:
    """Estado mínimo (duck-typing de AppState) para el servicer."""

    def __init__(self):
        self.retriever = FakeRetriever(FakeCollection(42, ["A.pdf", "B.md"]))
        self.config = SimpleNamespace(embedding_model="fake-model")
        self.device = "cpu"
        self.rebuilds = 0

    def on_index_rebuilt(self):
        self.rebuilds += 1


def _fake_summaries() -> list[IngestSummary]:
    return [
        IngestSummary(
            doc_name="A.pdf", n_chunks=3, seconds=0.1,
            token_p50=100, token_p90=120, token_max=140,
        ),
        IngestSummary(
            doc_name="B.md", n_chunks=5, seconds=0.2,
            token_p50=90, token_p90=110, token_max=130,
        ),
    ]


async def _serve(servicer: IndexAdminServicer):
    """Servidor aio in-proceso con puerto efímero."""
    server = grpc.aio.server()
    index_admin_pb2_grpc.add_IndexAdminServicer_to_server(servicer, server)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    return server, port


def grpc_test(scenario):
    """Ejecuta un escenario async con event loop propio (sin pytest-asyncio)."""

    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(scenario(*args, **kwargs))
        finally:
            loop.close()

    return wrapper


def test_ingest_emite_progreso_en_streaming(monkeypatch):
    @grpc_test
    async def scenario():
        ingest_calls: list[dict] = []

        async def fake_ingest(file_match=None, rebuild=False, on_summary=None):
            ingest_calls.append({"file_match": file_match, "rebuild": rebuild})
            for summary in _fake_summaries():
                on_summary(summary)

        monkeypatch.setattr(
            servicer_mod, "markdown_paths", lambda fm=None: ["A", "B"]
        )
        monkeypatch.setattr(servicer_mod, "ingest_documents", fake_ingest)

        state = FakeState()
        server, port = await _serve(IndexAdminServicer(state))
        try:
            async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                stub = index_admin_pb2_grpc.IndexAdminStub(channel)
                request = index_admin_pb2.IngestRequest(rebuild=True)
                progress = [p async for p in stub.Ingest(request, timeout=10)]
        finally:
            await server.stop(None)

        assert [p.doc_name for p in progress] == ["A.pdf", "B.md"]
        assert [p.chunks for p in progress] == [3, 5]
        assert progress[-1].done == 2 and progress[-1].total_docs == 2
        assert ingest_calls == [{"file_match": None, "rebuild": True}]
        assert state.rebuilds == 1  # reset de la colección tras el rebuild


def test_ingest_sin_markdown_invalid_argument(monkeypatch):
    @grpc_test
    async def scenario():
        monkeypatch.setattr(servicer_mod, "markdown_paths", lambda fm=None: [])
        server, port = await _serve(IndexAdminServicer(FakeState()))
        try:
            async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                stub = index_admin_pb2_grpc.IndexAdminStub(channel)
                with pytest.raises(grpc.RpcError) as exc_info:
                    async for _ in stub.Ingest(
                        index_admin_pb2.IngestRequest(file_match="nada"),
                        timeout=10,
                    ):
                        pass
        finally:
            await server.stop(None)

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_ingest_fallo_interno(monkeypatch):
    @grpc_test
    async def scenario():
        def fake_ingest(file_match=None, rebuild=False, on_summary=None):
            raise RuntimeError("embeddings caídos")

        monkeypatch.setattr(servicer_mod, "markdown_paths", lambda fm=None: ["A"])
        monkeypatch.setattr(servicer_mod, "ingest_documents", fake_ingest)
        server, port = await _serve(IndexAdminServicer(FakeState()))
        try:
            async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                stub = index_admin_pb2_grpc.IndexAdminStub(channel)
                with pytest.raises(grpc.RpcError) as exc_info:
                    async for _ in stub.Ingest(
                        index_admin_pb2.IngestRequest(), timeout=10
                    ):
                        pass
        finally:
            await server.stop(None)

        assert exc_info.value.code() == grpc.StatusCode.INTERNAL


def test_prune_index(monkeypatch):
    @grpc_test
    async def scenario():
        monkeypatch.setattr(servicer_mod, "prune_index", lambda: 7)
        server, port = await _serve(IndexAdminServicer(FakeState()))
        try:
            async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                stub = index_admin_pb2_grpc.IndexAdminStub(channel)
                response = await stub.PruneIndex(index_admin_pb2.Empty(), timeout=10)
        finally:
            await server.stop(None)

        assert response.removed_chunks == 7


def test_index_status(monkeypatch):
    @grpc_test
    async def scenario():
        server, port = await _serve(IndexAdminServicer(FakeState()))
        try:
            async with grpc.aio.insecure_channel(f"localhost:{port}") as channel:
                stub = index_admin_pb2_grpc.IndexAdminStub(channel)
                response = await stub.IndexStatus(index_admin_pb2.Empty(), timeout=10)
        finally:
            await server.stop(None)

        assert response.chunks == 42
        assert response.documents == 2
        assert response.embedding_model == "fake-model"
        assert response.device == "cpu"

