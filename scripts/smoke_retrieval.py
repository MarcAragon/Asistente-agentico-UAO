"""CLI Fase 3: humo del motor de recuperación sobre el índice real.

Uso:
    uv run python scripts/smoke_retrieval.py            # banco de preguntas fijo
    uv run python scripts/smoke_retrieval.py "pregunta" # una pregunta ad-hoc

Criterio de aceptación F3: para ~10 preguntas de humo sobre los reglamentos,
el chunk correcto debe aparecer en top-5 en >= 8 casos (medición informal,
antes de la evaluación formal de F6).
"""

from __future__ import annotations

import sys

from asistente_agentico_uao.rag.retrieval import Retriever, format_context

# Banco informal de humo: preguntas reales de estudiante (con jerga y typos)
# sobre calendarios, cancelaciones, créditos, admisiones y reglamentos.
SMOKE_QUESTIONS = [
    "¿Cuál es la ultima fecha de cancelaciones voluntarias para el semestre 2026-2?",
    "¿Hasta cuando se puede cancelar una asignatura sin que quede registrada en la hoja de vida?",
    "¿Cuántos créditos tiene el programa de Ingeniería Industrial?",
    "¿Qué es la admisión diferida y quiénes pueden aplicarla?",
    "¿Cuál es el número mínimo de créditos para ser estudiante de tiempo completo en posgrados?",
    "¿Que pasa si repruebo tres veces una misma asignatura?",
    "¿Cuáles son los requisitos para obtener el título de magíster?",
    "¿Cómo funciona el plan de mejoramiento para estudiantes en bajo desempeño académico?",
    "¿Se puede hacer transferencia interna entre programas de la Universidad?",
    "¿Cuál es el plazo máximo para culminar los estudios de doctorado?",
]


def _print_hits(question: str, chunks) -> None:
    print(f"\n{'=' * 78}\nPregunta: {question}\n{'-' * 78}")
    if not chunks:
        print("  (sin resultados sobre el umbral → respondería 'no sé')")
        return
    for i, c in enumerate(chunks, start=1):
        excerpt = " ".join(c.text.split())[:140]
        table = " [tabla]" if c.is_table else ""
        print(f"  {i}. sim={c.score:.3f}  {c.doc_name}{table}")
        print(f"     sección: {c.section[:80]}")
        print(f"     {excerpt}...")


def main() -> int:
    questions = sys.argv[1:] or SMOKE_QUESTIONS
    retriever = Retriever()
    total = retriever.collection.count()
    print(
        f"Índice: {total} chunks | top_k={retriever.settings.top_k} | "
        f"min_similarity={retriever.settings.min_similarity}"
    )
    if total == 0:
        print(
            "ERROR: la colección está vacía; corre scripts/ingest.py primero.",
            file=sys.stderr,
        )
        return 1

    all_chunks: list[list] = []
    for q in questions:
        chunks = retriever.retrieve(q)
        _print_hits(q, chunks)
        all_chunks.append(chunks)

    print(
        f"\n{'=' * 78}\nContexto renderizado (format_context) de la primera "
        "pregunta con resultados:"
    )
    for chunks in all_chunks:
        if chunks:
            print(format_context(chunks[:2])[:1200] + "\n...")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
