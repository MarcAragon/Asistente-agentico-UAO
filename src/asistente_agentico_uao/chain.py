"""Cadena RAG LCEL completa (Fase 4): retrieval → prompt → LLM → fuentes.

Flujo de ``answer_question(question) -> RagAnswer``:

1. ``Retriever.retrieve(question)``. Si no hay fragmentos sobre el umbral
   (o el índice está vacío) se responde ``NO_INFO_MESSAGE`` sin llamar al
   LLM: no se gastan tokens (tarea 4.4).
2. Generación LCEL: ``prompt | llm | StrOutputParser``. El LLM
   (``CerebrasLLM``) rota claves API ante límites de cuota (429/401/403)
   y reintenta con backoff ante errores transitorios.
   El prompt (tarea 4.2) ordena responder SOLO con el contexto numerado
   ``[1]..[k]``, citar como ``(Documento, sección)`` y, si el contexto no
   alcanza, copiar literalmente ``NO_INFO_MESSAGE``.
3. Post-proceso (tarea 4.3): se extraen las citas ``(Documento, sección)``
   de la respuesta y se mapean a los chunks recuperados (con score y
   excerpt). Las citas no verificables se descartan: nunca se inventan
   fuentes. Si la respuesta es el mensaje de no-información, ``sources=[]``.

⚠ Hallazgo de la Fase 3 (mitigación aquí): las similitudes coseno de E5 son
altas siempre (~0.81 incluso para preguntas fuera de dominio), por lo que el
umbral ``min_similarity`` casi nunca dispara el "no sé" pre-LLM. La defensa
principal es la capa 2 (juicio del LLM sobre el contexto, reglas 2-3 del
prompt); la capa 1 (umbral) queda como piso duro y la recalibración del
umbral se hace en F6 con el banco de preguntas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from .config import Settings, settings
from .llm import NO_INFO_MESSAGE, get_llm
from .retrieval import RetrievedChunk, Retriever, format_context

# Longitud del excerpt citado en Source (texto continuo, sin saltos).
EXCERPT_CHARS = 240

# Citas "(Documento, sección)" tolerantes a espacios; sin corchetes dentro
# para no chocar con las referencias [N] del propio formato de contexto.
CITATION_RE = re.compile(r"\(\s*([^()[\]]+?)\s*,\s*([^()[\]]+?)\s*\)")

SYSTEM_TEMPLATE = """\
Eres el asistente de normativa institucional de la Universidad Autónoma de \
Occidente (UAO). Respondes preguntas de estudiantes y funcionarios sobre \
reglamentos, resoluciones y políticas de la Universidad.

Responde EXCLUSIVAMENTE con la información del CONTEXTO numerado [1]..[k] \
que se entrega con la pregunta. Reglas:

1. Cita la fuente de cada afirmación con el formato (Documento, sección), \
usando el nombre del documento y la sección indicados en la referencia [N] \
del CONTEXTO que respalda la afirmación.
2. Si el CONTEXTO no contiene la información necesaria para responder la \
pregunta, responde EXACTAMENTE la siguiente frase y nada más:
{no_info_message}
3. No inventes artículos, secciones, fechas ni documentos. No uses \
conocimiento externo a la UAO ni del CONTEXTO.
4. Si el CONTEXTO respalda solo parte de la pregunta, responde únicamente \
esa parte e indica qué no se encuentra en la normativa proporcionada.
5. Responde en español, claro y directo."""

synthesis_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", "Pregunta: {question}\n\nCONTEXTO:\n{context}"),
    ]
).partial(no_info_message=NO_INFO_MESSAGE)


@dataclass
class Source:
    """Fuente citada en la respuesta (contrato API §3.3, Fase 5)."""

    doc_name: str
    section: str
    score: float  # similitud coseno pregunta-chunk
    excerpt: str  # recorte del texto del chunk


@dataclass
class RagAnswer:
    """Resultado de la cadena para la API (Fase 5) y el CLI (4.5)."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    model: str = ""
    # True si la respuesta no proviene del LLM (umbral pre-LLM) o si el LLM
    # respondió el mensaje de no-información (capa 2 del hallazgo de F3).
    used_fallback: bool = False


def _excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """Texto del chunk colapsado a una línea y recortado."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def extract_citations(answer: str) -> list[tuple[str, str]]:
    """Citas "(Documento, sección)" presentes en la respuesta del LLM."""
    seen: set[tuple[str, str]] = set()
    citations: list[tuple[str, str]] = []
    for doc, section in CITATION_RE.findall(answer or ""):
        key = (doc.strip(), section.strip())
        if key not in seen:
            seen.add(key)
            citations.append(key)
    return citations


def build_sources(answer: str, chunks: list[RetrievedChunk]) -> list[Source]:
    """Mapea las citas de la respuesta a los chunks recuperados.

    - Coincidencia exacta (doc, sección); si falla, flexible (mismo doc con
      sección contenida, p.ej. el LLM citó un artículo dentro de la sección).
    - Si el LLM respondió algo pero no se pudo verificar ninguna cita, se
      devuelven todos los chunks recuperados (trazabilidad, sin inventar).
    - Respuesta == NO_INFO_MESSAGE → sin fuentes.
    """
    if (answer or "").strip() == NO_INFO_MESSAGE or not chunks:
        return []

    by_key: dict[tuple[str, str], RetrievedChunk] = {}
    by_doc: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        by_key.setdefault((chunk.doc_name, chunk.section), chunk)
        by_doc.setdefault(chunk.doc_name, chunk)

    sources: list[Source] = []
    used: set[int] = set()  # id() de chunks ya citados (dataclass no hasheable)
    for doc, section in extract_citations(answer):
        chunk = by_key.get((doc, section))
        if chunk is None:  # sección citada con variaciones menores
            for (d, s), candidate in by_key.items():
                if d == doc and (section in s or s in section):
                    chunk = candidate
                    break
        if chunk is None:  # cita el documento correcto, sección distinta
            chunk = by_doc.get(doc)
        if chunk is None or id(chunk) in used:
            continue  # cita no verificable o duplicada: se descarta
        used.add(id(chunk))
        sources.append(
            Source(
                doc_name=chunk.doc_name,
                section=chunk.section,
                score=chunk.score,
                excerpt=_excerpt(chunk.text),
            )
        )

    if not sources:  # respuesta sin citas verificables: trazabilidad completa
        sources = [
            Source(
                doc_name=c.doc_name,
                section=c.section,
                score=c.score,
                excerpt=_excerpt(c.text),
            )
            for c in chunks
        ]
    return sources


def _generation_chain(llm):
    """LCEL 4.3: prompt → LLM (rotación de claves + reintentos) → texto."""

    def _invoke(prompt_value):
        return llm.invoke(prompt_value)

    return synthesis_prompt | RunnableLambda(_invoke) | StrOutputParser()


def answer_question(
    question: str,
    retriever: Retriever | None = None,
    llm=None,
    config: Settings | None = None,
) -> RagAnswer:
    """Ejecuta la cadena RAG completa para una pregunta (Fase 4)."""
    cfg = config or settings
    retriever = retriever or Retriever(config=cfg)
    llm = llm or get_llm()

    chunks = retriever.retrieve(question)
    if not chunks:
        # Tarea 4.4: sin contexto no se gasta el LLM. Con el hallazgo de F3
        # este camino casi no dispara; la defensa real está en el prompt.
        return RagAnswer(
            answer=NO_INFO_MESSAGE,
            sources=[],
            model=cfg.llm_model,
            used_fallback=True,
        )

    answer = _generation_chain(llm).invoke(
        {"question": question, "context": format_context(chunks)}
    ).strip()

    if not answer:
        # Defensa: el LLM no debe devolver vacío (solo ocurrió pre-disable
        # de razonamiento, cuando thinking agotaba max_tokens). Nunca se
        # responde con texto vacío: se degrada a no-información.
        answer = NO_INFO_MESSAGE

    return RagAnswer(
        answer=answer,
        sources=build_sources(answer, chunks),
        model=cfg.llm_model,
        used_fallback=(answer == NO_INFO_MESSAGE),
    )

