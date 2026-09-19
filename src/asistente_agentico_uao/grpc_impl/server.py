"""Servidor gRPC ``IndexAdmin`` (Fase 5): plano de control de ingesta.

Dos modos:

- **Embebido** (recomendado, un solo proceso): el lifespan de la API REST
  llama ``create_grpc_server(state)`` y lo arranca en el mismo event loop;
  modelo de embeddings y colección Chroma se comparten con la REST.
- **Standalone** (dev/debug): ``uv run python -m asistente_agentico_uao.grpc_impl.server``.

Solo escucha en texto plano (insecure): el tráfico es interno; el proxy
TLS (Fase 7) queda para la superficie pública HTTP.
"""

from __future__ import annotations

import grpc

from ..core.config import Settings
from ..core.llm import CerebrasLLM
from ..rag.retrieval import Retriever
from ..rag.service import AppState
from .servicer import IndexAdminServicer
from .stubs import index_admin_pb2_grpc


def create_grpc_server(state: AppState, config: Settings | None = None):
    """``grpc.aio.Server`` con ``IndexAdmin`` registrado (sin arrancar)."""
    cfg = config or state.config
    server = grpc.aio.server()
    index_admin_pb2_grpc.add_IndexAdminServicer_to_server(
        IndexAdminServicer(state), server
    )
    bound = server.add_insecure_port(f"[::]:{cfg.grpc_port}")
    if bound == 0:
        raise RuntimeError(f"Puerto gRPC {cfg.grpc_port} no disponible")
    return server


async def serve() -> None:
    """Modo standalone: sirve solo el control plane gRPC hasta Ctrl-C."""
    cfg = Settings()
    state = AppState(config=cfg, retriever=Retriever(config=cfg), llm=CerebrasLLM(cfg))
    server = create_grpc_server(state, cfg)
    await server.start()
    print(f"[grpc] IndexAdmin (standalone) escuchando en :{cfg.grpc_port}")
    await server.wait_for_termination()


if __name__ == "__main__":
    import asyncio

    asyncio.run(serve())
