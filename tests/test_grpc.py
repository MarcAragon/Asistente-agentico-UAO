"""Pruebas unitarias del servicio gRPC ``IndexAdmin`` (Fase 5).

El servicio ``IndexAdmin`` representa el control plane encargado de administrar
el índice vectorial mediante operaciones gRPC.

Estas pruebas ejecutan un servidor ``grpc.aio`` en memoria utilizando un puerto
efímero y componentes simulados para evitar dependencias externas como modelos
de embeddings, ChromaDB o conexiones de red reales.

Se validan:
- Streaming de progreso durante la ingesta.
- Manejo de errores INVALID_ARGUMENT e INTERNAL.
- Operación de limpieza del índice.
- Consulta del estado actual del índice.
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
    """Simula una colección Chroma para pruebas unitarias."""

    def __init__(
        self,
        count: int,
        doc_names: list[str],
    ):
        """Inicializa una colección falsa."""

        self._count = count
        self._doc_names = doc_names

    def count(self):
        """Retorna la cantidad simulada de chunks almacenados."""

        return self._count

    def get(self, include=None):
        """Simula la recuperación de metadatos del índice."""

        return {
            "metadatas": [
                {
                    "doc_name": doc,
                    "chunk_index": 0,
                }
                for doc in self._doc_names
            ]
        }


class FakeRetriever:
    """Simula el componente Retriever utilizado por AppState."""

    def __init__(self, collection):
        """Inicializa un retriever falso."""

        self.collection = collection
        self.resets = 0

    def reset(self):
        """Simula la limpieza del retriever."""

        self.resets += 1


class FakeCache:
    """Simula la caché utilizada por el estado de aplicación."""

    def __init__(self):
        """Inicializa el contador de invalidaciones."""

        self.invalidations = 0

    def invalidar_todo(self):
        """Registra una invalidación completa de la caché."""

        self.invalidations += 1


class FakeState:
    """Estado mínimo compatible con el servicio IndexAdmin.

    Implementa únicamente los atributos requeridos por el servicer,
    siguiendo el patrón duck-typing utilizado por AppState.
    """

    def __init__(self):
        """Inicializa el estado falso del servicio."""

        self.retriever = FakeRetriever(
            FakeCollection(
                42,
                ["A.pdf", "B.md"],
            )
        )

        self.config = SimpleNamespace(
            embedding_model="fake-model"
        )

        self.device = "cpu"
        self.rebuilds = 0
        self.cache = FakeCache()

    def on_index_rebuilt(self):
        """Registra una reconstrucción del índice."""

        self.rebuilds += 1


def _fake_summaries() -> list[IngestSummary]:
    """Genera resúmenes falsos utilizados durante la ingesta."""

    return [
        IngestSummary(
            doc_name="A.pdf",
            n_chunks=3,
            seconds=0.1,
            token_p50=100,
            token_p90=120,
            token_max=140,
        ),
        IngestSummary(
            doc_name="B.md",
            n_chunks=5,
            seconds=0.2,
            token_p50=90,
            token_p90=110,
            token_max=130,
        ),
    ]


async def _serve(servicer: IndexAdminServicer):
    """Levanta un servidor gRPC temporal para pruebas."""

    server = grpc.aio.server()

    index_admin_pb2_grpc.add_IndexAdminServicer_to_server(
        servicer,
        server,
    )

    port = server.add_insecure_port(
        "localhost:0"
    )

    await server.start()

    return server, port


def grpc_test(scenario):
    """Ejecuta escenarios async sin pytest-asyncio."""

    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()

        try:
            return loop.run_until_complete(
                scenario(
                    *args,
                    **kwargs,
                )
            )
        finally:
            loop.close()

    return wrapper


def test_ingest_emite_progreso_en_streaming(monkeypatch):
    """Verifica que Ingest envíe progreso mediante streaming gRPC."""

    @grpc_test
    async def scenario():

        ingest_calls: list[dict] = []

        def fake_ingest(
            file_match=None,
            rebuild=False,
            on_summary=None,
        ):
            ingest_calls.append(
                {
                    "file_match": file_match,
                    "rebuild": rebuild,
                }
            )

            for summary in _fake_summaries():
                on_summary(summary)

        monkeypatch.setattr(
            servicer_mod,
            "markdown_paths",
            lambda fm=None: ["A", "B"],
        )

        monkeypatch.setattr(
            servicer_mod,
            "ingest_documents",
            fake_ingest,
        )

        state = FakeState()

        server, port = await _serve(
            IndexAdminServicer(state)
        )

        try:
            async with grpc.aio.insecure_channel(
                f"localhost:{port}"
            ) as channel:

                stub = index_admin_pb2_grpc.IndexAdminStub(
                    channel
                )

                progress = [
                    item
                    async for item in stub.Ingest(
                        index_admin_pb2.IngestRequest(
                            rebuild=True
                        ),
                        timeout=10,
                    )
                ]

        finally:
            await server.stop(None)

        assert [p.doc_name for p in progress] == [
            "A.pdf",
            "B.md",
        ]

        assert [p.chunks for p in progress] == [
            3,
            5,
        ]

        assert progress[-1].done == 2
        assert progress[-1].total_docs == 2

        assert ingest_calls == [
            {
                "file_match": None,
                "rebuild": True,
            }
        ]

        assert state.rebuilds == 1

    scenario()


def test_ingest_sin_markdown_invalid_argument(monkeypatch):
    """Verifica que una ingesta sin documentos retorne INVALID_ARGUMENT."""

    @grpc_test
    async def scenario():

        monkeypatch.setattr(
            servicer_mod,
            "markdown_paths",
            lambda fm=None: [],
        )

        server, port = await _serve(
            IndexAdminServicer(
                FakeState()
            )
        )

        try:
            async with grpc.aio.insecure_channel(
                f"localhost:{port}"
            ) as channel:

                stub = index_admin_pb2_grpc.IndexAdminStub(
                    channel
                )

                with pytest.raises(grpc.RpcError) as exc_info:

                    async for _ in stub.Ingest(
                        index_admin_pb2.IngestRequest(
                            file_match="nada"
                        ),
                        timeout=10,
                    ):
                        pass

        finally:
            await server.stop(None)

        assert exc_info.value.code() == (
            grpc.StatusCode.INVALID_ARGUMENT
        )

    scenario()


def test_ingest_fallo_interno(monkeypatch):
    """Verifica que errores internos retornen estado INTERNAL."""

    @grpc_test
    async def scenario():

        def fake_ingest(
            file_match=None,
            rebuild=False,
            on_summary=None,
        ):
            raise RuntimeError(
                "embeddings caídos"
            )

        monkeypatch.setattr(
            servicer_mod,
            "markdown_paths",
            lambda fm=None: ["A"],
        )

        monkeypatch.setattr(
            servicer_mod,
            "ingest_documents",
            fake_ingest,
        )

        server, port = await _serve(
            IndexAdminServicer(
                FakeState()
            )
        )

        try:
            async with grpc.aio.insecure_channel(
                f"localhost:{port}"
            ) as channel:

                stub = index_admin_pb2_grpc.IndexAdminStub(
                    channel
                )

                with pytest.raises(grpc.RpcError) as exc_info:

                    async for _ in stub.Ingest(
                        index_admin_pb2.IngestRequest(),
                        timeout=10,
                    ):
                        pass

        finally:
            await server.stop(None)

        assert exc_info.value.code() == (
            grpc.StatusCode.INTERNAL
        )

    scenario()


def test_prune_index(monkeypatch):
    """Verifica la eliminación de chunks mediante RPC PruneIndex."""

    @grpc_test
    async def scenario():

        monkeypatch.setattr(
            servicer_mod,
            "prune_index",
            lambda: 7,
        )

        server, port = await _serve(
            IndexAdminServicer(
                FakeState()
            )
        )

        try:
            async with grpc.aio.insecure_channel(
                f"localhost:{port}"
            ) as channel:

                stub = index_admin_pb2_grpc.IndexAdminStub(
                    channel
                )

                response = await stub.PruneIndex(
                    index_admin_pb2.Empty(),
                    timeout=10,
                )

        finally:
            await server.stop(None)

        assert response.removed_chunks == 7

    scenario()


def test_index_status(monkeypatch):
    """Verifica la información entregada por el RPC IndexStatus."""

    @grpc_test
    async def scenario():

        server, port = await _serve(
            IndexAdminServicer(
                FakeState()
            )
        )

        try:
            async with grpc.aio.insecure_channel(
                f"localhost:{port}"
            ) as channel:

                stub = index_admin_pb2_grpc.IndexAdminStub(
                    channel
                )

                response = await stub.IndexStatus(
                    index_admin_pb2.Empty(),
                    timeout=10,
                )

        finally:
            await server.stop(None)

        assert response.chunks == 42
        assert response.documents == 2
        assert response.embedding_model == "fake-model"
        assert response.device == "cpu"

    scenario()