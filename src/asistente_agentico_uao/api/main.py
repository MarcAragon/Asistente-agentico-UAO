"""API REST pública del asistente (Fase 5): consulta con citas verificables.

Plano de datos (data plane): ``POST /ask`` es la única operación de
negocio — pregunta → ``RagAnswer`` con fuentes. Solo LEE el índice: la
escritura (ingesta/prune/rebuild) vive en el plano de control gRPC
(``grpc_impl``, servicio ``IndexAdmin``) y nunca se expone por HTTP.

Endpoints:
- ``POST /ask``: 200 con respuesta+fuentes; 422 validación; 503 sin
  ``CEREBRAS_API_KEY``; 500 error interno.
- ``GET /health``: estado, chunks indexados y dispositivo.
- ``GET /documents``: documentos indexados (trazabilidad).
- CORS ``*`` para el futuro frontend.

Ejecución:
    uv run uvicorn asistente_agentico_uao.api.main:app --host 0.0.0.0 --port 8000

El servidor gRPC embebido (``IndexAdmin``, :50051) arranca en el lifespan
si ``UAO_RAG__GRPC_ENABLED=1`` (default); su fallo no impide la REST.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from ..config import Settings
from ..llm import CerebrasLLM, collect_api_keys
from ..retrieval import Retriever
from ..service import AppState
from .schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    HealthResponse,
    SourceModel,
)

NO_KEY_DETAIL = (
    "CEREBRAS_API_KEY no configurada: define al menos una clave en .env "
    "(o CEREBRAS_API_KEYS para rotación)."
)


def create_app(config: Settings | None = None, state: AppState | None = None) -> FastAPI:
    """Fábrica de la app; ``state`` inyectable para tests (sin lifespan real)."""
    cfg = config or (state.config if state is not None else Settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag = state or AppState(
            config=cfg, retriever=Retriever(config=cfg), llm=CerebrasLLM(cfg)
        )
        if cfg.grpc_enabled:
            try:
                from ..grpc_impl.server import create_grpc_server

                app.state.grpc_server = create_grpc_server(app.state.rag, cfg)
                await app.state.grpc_server.start()
                print(f"[grpc] IndexAdmin (control plane) en :{cfg.grpc_port}")
            except Exception as exc:  # noqa: BLE001 - REST no depende de gRPC
                print(f"[grpc] no se pudo iniciar (REST sigue activo): {exc}")
        yield
        grpc_server = getattr(app.state, "grpc_server", None)
        if grpc_server is not None:
            await grpc_server.stop(grace=2)

    app = FastAPI(
        title="Asistente RAG UAO",
        description="Consulta de normativa UAO con fuentes citadas (RAG).",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post(
        "/ask",
        response_model=AskResponse,
        summary="Responde una pregunta con fuentes verificables",
    )
    def ask(payload: AskRequest, request: Request) -> AskResponse:
        rag: AppState = request.app.state.rag
        if not collect_api_keys(rag.config):
            raise HTTPException(status_code=503, detail=NO_KEY_DETAIL)
        try:
            result = rag.ask(payload.question)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Error interno generando la respuesta.",
            ) from exc
        return AskResponse(
            answer=result.answer,
            sources=[SourceModel(**asdict(s)) for s in result.sources],
            model=result.model,
            used_fallback=result.used_fallback,
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Estado del servicio y del índice",
    )
    def health(request: Request) -> HealthResponse:
        rag: AppState = request.app.state.rag
        return HealthResponse(
            status="ok", index_chunks=rag.index_chunks(), device=rag.device
        )

    @app.get(
        "/documents",
        response_model=list[DocumentInfo],
        summary="Documentos indexados (trazabilidad)",
    )
    def documents(request: Request) -> list[DocumentInfo]:
        rag: AppState = request.app.state.rag
        return [
            DocumentInfo(doc_name=d.doc_name, chunks=d.chunks)
            for d in rag.documents()
        ]

    return app


app = create_app()
