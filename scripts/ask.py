"""CLI Fase 4: prueba end-to-end de la cadena RAG sin levantar la API.

Uso:
    uv run python scripts/ask.py "pregunta"              # una pregunta ad-hoc
    uv run python scripts/ask.py                          # banco de humo F4

El banco incluye las 5 preguntas de humo del criterio de aceptación F4 más
2 fuera de dominio (hallazgo de F3: el umbral no las discrimina; el "no sé"
debe venir del prompt, verificado aquí de facto antes de la recalibración
de F6).
"""

from __future__ import annotations

import sys
import time

from asistente_agentico_uao.rag.chain import answer_question

# 5 in-dominio (respuestas verificables manualmente, hallazgos de F3) +
# 2 fuera de dominio que DEBEN recibir el mensaje de no-información.
SMOKE_QUESTIONS = [
    "¿Cuál es la última fecha de cancelaciones voluntarias para el semestre 2026-2?",
    "¿Que pasa si repruebo tres veces una misma asignatura?",
    "¿Cuál es el número mínimo de créditos para ser estudiante de tiempo completo en posgrados?",
    "¿Cuáles son los requisitos para obtener el título de magíster?",
    "¿Se puede hacer transferencia interna entre programas de la Universidad?",
    # --- fuera de dominio (no deben alucinar) ---
    "¿Cuál es la receta traditional de las arepas antioqueñas?",
    "¿Qué selección nacional ganó la Copa Mundial de fútbol de 2022?",
]


def _print_answer(question: str, result) -> None:
    print(f"\n{'=' * 78}\nPregunta: {question}\n{'-' * 78}")
    print(f"Respuesta ({result.model}):\n{result.answer}")
    if result.used_fallback:
        print("  [fallback: sin llamada al LLM o respuesta de no-información]")
    print(f"Fuentes ({len(result.sources)}):")
    for i, s in enumerate(result.sources, start=1):
        print(f"  {i}. sim={s.score:.3f}  {s.doc_name} — {s.section[:80]}")
        print(f"     {s.excerpt[:140]}...")


def main() -> int:
    questions = sys.argv[1:] or SMOKE_QUESTIONS
    print(f"Cadena RAG F4 | preguntas: {len(questions)}")
    for q in questions:
        start = time.perf_counter()
        try:
            result = answer_question(q)
        except Exception as exc:  # noqa: BLE001 - reporte de humo por pregunta
            print(f"\n{'=' * 78}\nPregunta: {q}\nERROR: {type(exc).__name__}: {exc}")
            return 1
        elapsed = time.perf_counter() - start
        _print_answer(q, result)
        print(f"  ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
