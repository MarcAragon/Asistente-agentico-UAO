"""CLI Fase 2: markdown → chunks → embeddings E5 → ChromaDB.

Uso:
    uv run python scripts/ingest.py               # ingesta completa
    uv run python scripts/ingest.py --file 666    # un solo documento (match parcial)
    uv run python scripts/ingest.py --rebuild     # borra la colección antes
    uv run python scripts/ingest.py --prune       # borra chunks de docs ausentes

Idempotente: los IDs son sha256(doc_name + chunk_index); correr dos veces
reemplaza los mismos chunks en vez de duplicarlos. Al final corre una query
de humo (desactivable con --no-smoke).

Desde la Fase 5 la lógica vive en ``ingestion.pipeline`` (compartida con el
servicio gRPC ``IndexAdmin``); este CLI es un wrapper de consola.
"""

from __future__ import annotations

import argparse
import sys

from asistente_agentico_uao.core.config import settings
from asistente_agentico_uao.core.embeddings import (
    embed_query,
    get_model,
    max_seq_length,
    token_counter,
)
from asistente_agentico_uao.core.vectorstore import COLLECTION_NAME, get_collection
from asistente_agentico_uao.ingestion.pipeline import (
    IngestSummary,
    ingest_documents,
    markdown_paths,
    prune_index,
)


def _print_row(s: IngestSummary) -> None:
    print(f"{s.doc_name:<72}{s.n_chunks:>7}{s.seconds:>7.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingesta RAG: chunking + embeddings + indexación Chroma."
    )
    parser.add_argument("--file", help="Procesa un solo markdown (match parcial)")
    parser.add_argument(
        "--rebuild", action="store_true", help="Borra la colección antes"
    )
    parser.add_argument(
        "--prune", action="store_true", help="Borra chunks de documentos ya ausentes"
    )
    parser.add_argument(
        "--no-smoke", action="store_true", help="Omite la query de humo"
    )
    args = parser.parse_args()

    if not markdown_paths(args.file):
        print(f"No hay markdown en {settings.markdown_dir}", file=sys.stderr)
        return 1

    model = get_model()
    seq_len = max_seq_length()
    print(
        f"[modelo] {settings.embedding_model} | max_seq_length={seq_len} | "
        f"batch={settings.embedding_batch_size} | device={model.device}"
    )

    collection = get_collection(rebuild=args.rebuild)
    header = f"{'documento':<72}{'chunks':>7}{'p50':>6}{'p90':>6}{'max':>6}{'seg':>7}"
    print(f"\n{header}\n{'-' * len(header)}")
    try:
        summaries = ingest_documents(
            file_match=args.file,
            rebuild=args.rebuild,
            model=model,
            seq_len=seq_len,
            count_tokens=token_counter(),
            collection=collection,
            on_summary=_print_row,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    total_chunks = sum(s.n_chunks for s in summaries)
    print(f"\nTotal: {total_chunks} chunks en {len(summaries)} doc(s)")
    print(
        f"Colección '{COLLECTION_NAME}': {collection.count()} chunks en {settings.chroma_dir}"
    )

    if args.prune:
        removed = prune_index(collection)
        print(f"Prune: {removed} chunk(s) de documentos ausentes eliminados")

    if not args.no_smoke:
        question = "¿Cuál es la ultima fecha de cancelaciones voluntarias? Dame la fecha específica para el semestre 2026-2"
        result = collection.query(
            query_embeddings=[embed_query(question).tolist()],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        documents = result["documents"]
        metadatas = result["metadatas"]
        distances = result["distances"]
        if documents is None or metadatas is None or distances is None:
            print(
                "[humo] la query no devolvió documents/metadatas/distances",
                file=sys.stderr,
            )
            return 1
        print(f"\n[humo] '{question}'")
        for doc, meta, dist in zip(documents[0], metadatas[0], distances[0]):
            raw_doc_name = meta.get("doc_name")
            raw_section = meta.get("section")
            doc_name = raw_doc_name if isinstance(raw_doc_name, str) else ""
            section = raw_section if isinstance(raw_section, str) else ""
            print(f"  sim={1 - dist:.3f}  {doc_name} | {section[:70]}")
            excerpt = " ".join(doc.split())[:130]
            print(f"          {excerpt}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
