"""CLI Fase 2: markdown → chunks → embeddings E5 → ChromaDB.

Uso:
    uv run python scripts/ingest.py               # ingesta completa
    uv run python scripts/ingest.py --file 666    # un solo documento (match parcial)
    uv run python scripts/ingest.py --rebuild     # borra la colección antes
    uv run python scripts/ingest.py --prune       # borra chunks de docs ausentes

Idempotente: los IDs son sha256(doc_name + chunk_index); correr dos veces
reemplaza los mismos chunks en vez de duplicarlos. Al final corre una query
de humo (desactivable con --no-smoke).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from asistente_agentico_uao.config import settings
from asistente_agentico_uao.embeddings import (
    embed_passages,
    embed_query,
    get_model,
    max_seq_length,
    token_counter,
)
from asistente_agentico_uao.ingestion.chunk import chunk_markdown
from asistente_agentico_uao.vectorstore import (
    COLLECTION_NAME,
    get_collection,
    prune_missing_docs,
    upsert_chunks,
)


def _pdf_stems() -> dict[str, str]:
    """Mapa stem -> nombre real del PDF (para doc_name de citación)."""
    return {p.stem: p.name for p in settings.docs_dir.glob("*.pdf")}


def _resolve_doc_name(md_path: Path, stems: dict[str, str]) -> str:
    return stems.get(md_path.stem, md_path.stem)


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

    mds = sorted(settings.markdown_dir.glob("*.md"))
    if args.file:
        mds = [p for p in mds if args.file.lower() in p.name.lower()]
    if not mds:
        print(f"No hay markdown en {settings.markdown_dir}", file=sys.stderr)
        return 1

    model = get_model()
    seq_len = max_seq_length()
    print(
        f"[modelo] {settings.embedding_model} | max_seq_length={seq_len} | "
        f"batch={settings.embedding_batch_size} | device={model.device}"
    )
    if settings.chunk_max_tokens > seq_len - 4:
        print(
            f"ERROR: chunk_max_tokens={settings.chunk_max_tokens} excede "
            f"max_seq_length={seq_len}: los chunks se truncarían.",
            file=sys.stderr,
        )
        return 1

    count_tokens = token_counter()
    stems = _pdf_stems()
    collection = get_collection(rebuild=args.rebuild)

    total_chunks = 0
    total_started = time.perf_counter()
    header = f"{'documento':<72}{'chunks':>7}{'p50':>6}{'p90':>6}{'max':>6}{'seg':>7}"
    print(f"\n{header}\n{'-' * len(header)}")
    for md in mds:
        doc_name = _resolve_doc_name(md, stems)
        started = time.perf_counter()
        chunks = chunk_markdown(
            md.read_text(encoding="utf-8"),
            doc_name=doc_name,
            count_tokens=count_tokens,
            target_tokens=settings.chunk_size_tokens,
            max_tokens=settings.chunk_max_tokens,
            min_tokens=settings.chunk_min_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        if not chunks:
            print(f"{md.name:<72}{'0':>7}")
            continue
        vectors = embed_passages([c.text for c in chunks])
        upsert_chunks(collection, chunks, vectors)
        elapsed = time.perf_counter() - started
        toks = sorted(c.n_tokens for c in chunks)
        p50 = toks[len(toks) // 2]
        p90 = toks[min(len(toks) - 1, int(len(toks) * 0.9))]
        print(
            f"{md.name:<72}{len(chunks):>7}{p50:>6}{p90:>6}{toks[-1]:>6}{elapsed:>7.1f}"
        )
        total_chunks += len(chunks)

    print(
        f"\nTotal: {total_chunks} chunks en {len(mds)} doc(s) "
        f"({time.perf_counter() - total_started:.1f}s)"
    )
    print(
        f"Colección '{COLLECTION_NAME}': {collection.count()} chunks en {settings.chroma_dir}"
    )

    if args.prune:
        valid = {
            _resolve_doc_name(p, stems) for p in settings.markdown_dir.glob("*.md")
        }
        removed = prune_missing_docs(collection, valid)
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
