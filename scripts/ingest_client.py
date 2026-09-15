"""Cliente gRPC de humo para ``IndexAdmin`` (plano de control, Fase 5).

Uso:
    uv run python scripts/ingest_client.py                  # ingesta completa con progreso
    uv run python scripts/ingest_client.py --file 666       # un documento
    uv run python scripts/ingest_client.py --rebuild        # borra la colección antes
    uv run python scripts/ingest_client.py --prune          # prune de docs ausentes
    uv run python scripts/ingest_client.py --status         # estado del índice

Contra un servidor gRPC ya levantado (embebido en la API REST o standalone:
``uv run python -m asistente_agentico_uao.grpc_impl.server``).
"""

from __future__ import annotations

import argparse
import asyncio

import grpc

from asistente_agentico_uao.grpc_impl.stubs import index_admin_pb2, index_admin_pb2_grpc


async def _ingest(stub, file_match: str | None, rebuild: bool) -> None:
    header = f"{'documento':<72}{'chunks':>7}{'seg':>7}  progreso"
    print(f"\n{header}\n{'-' * len(header)}")
    stream = stub.Ingest(
        index_admin_pb2.IngestRequest(file_match=file_match or "", rebuild=rebuild)
    )
    async for p in stream:
        print(
            f"{p.doc_name:<72}{p.chunks:>7}{p.seconds:>7.1f}  "
            f"{p.done}/{p.total_docs}"
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description="Cliente gRPC de IndexAdmin.")
    parser.add_argument("--file", help="Procesa un solo markdown (match parcial)")
    parser.add_argument("--rebuild", action="store_true", help="Borra la colección antes")
    parser.add_argument("--prune", action="store_true", help="Borra chunks de docs ausentes")
    parser.add_argument("--status", action="store_true", help="Estado del índice")
    parser.add_argument("--port", type=int, default=50051, help="Puerto gRPC")
    args = parser.parse_args()

    async with grpc.aio.insecure_channel(f"localhost:{args.port}") as channel:
        stub = index_admin_pb2_grpc.IndexAdminStub(channel)
        if args.status:
            status = await stub.IndexStatus(index_admin_pb2.Empty())
            print(
                f"chunks={status.chunks} documentos={status.documents} "
                f"modelo={status.embedding_model} device={status.device}"
            )
        if args.prune:
            result = await stub.PruneIndex(index_admin_pb2.Empty())
            print(f"Prune: {result.removed_chunks} chunk(s) eliminados")
        if not args.status and not args.prune:
            await _ingest(stub, args.file, args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
