"""Pruebas unitarias de la Fase 4: ``chain.py`` (cadena RAG LCEL).

Cubre la post-procesización de la respuesta del LLM y la cadena completa
``answer_question`` con retriever y LLM falsos (sin llamadas reales ni
tokens consumidos):

- ``extract_citations``: extracción y deduplicación de citas.
- ``build_sources``: mapeo de citas verificables a ``Source``, descarte de
  citas no verificables (nunca se inventan fuentes), trazabilidad cuando no
  hay citas verificables y ``sources=[]`` para el mensaje de no-información.
- ``answer_question``: umbral pre-LLM (sin contexto no se gasta el LLM),
  marca de ``used_fallback``, degradación de respuesta vacía y verificación
  de que el prompt renderizado incluye pregunta, contexto y
  ``NO_INFO_MESSAGE``.
"""

from types import SimpleNamespace

from asistente_agentico_uao.core.llm import NO_INFO_MESSAGE
from asistente_agentico_uao.rag.chain import (
    EXCERPT_CHARS,
    RagAnswer,
    _excerpt,
    answer_question,
    build_sources,
    extract_citations,
)
from asistente_agentico_uao.rag.retrieval import RetrievedChunk


def make_chunk(doc: str, section: str, text: str, score: float = 0.9):
    return RetrievedChunk(
        doc_name=doc,
        section=section,
        text=text,
        score=score,
        chunk_index=0,
    )


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.questions: list[str] = []

    def retrieve(self, question):
        self.questions.append(question)
        return self.chunks


class FakeLLM:
    """LLM falso: graba el prompt renderizado y devuelve una respuesta fija."""

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def invoke(self, prompt_value):
        self.prompts.append(prompt_value.to_string())
        return self.reply


class GuardLLM:
    """LLM que falla el test si llega a invocarse (no se deben gastar tokens)."""

    def invoke(self, prompt_value):  # pragma: no cover - guarda del test
        raise AssertionError("el LLM no debe invocarse sin contexto")


# --- _excerpt ---------------------------------------------------------------


def test_excerpt_colapsa_saltos_y_recorta():
    text = "línea uno\n\nlínea\ttres " + "x" * 400

    excerpt = _excerpt(text)

    assert "\n" not in excerpt and "\t" not in excerpt
    assert len(excerpt) == EXCERPT_CHARS
    assert excerpt.endswith("…")


# --- extract_citations ------------------------------------------------------


def test_extract_citations_deduplica_y_tolera_espacios():
    answer = (
        "Según (Res-CA-6744.md, Artículo 70º-2) y con espacios "
        "( Res-CA-6744.md , Artículo 70º-2 ); también (Reso-CS-666.md, Art. 19°)."
    )

    citations = extract_citations(answer)

    assert citations == [
        ("Res-CA-6744.md", "Artículo 70º-2"),
        ("Reso-CS-666.md", "Art. 19°"),
    ]


def test_extract_citations_sin_citas():
    assert extract_citations("respuesta sin citas") == []


# --- build_sources ----------------------------------------------------------


def test_citas_se_mapean_a_fuentes():
    chunks = [
        make_chunk("Res-CA-6744.md", "Artículo 70º-2", "texto sobre repitencias."),
        make_chunk("Res-CA-6605.md", "Artículo 35º", "requisitos de admisión.", 0.8),
    ]
    answer = (
        "Puedes repetir hasta tres veces (Res-CA-6744.md, Artículo 70º-2); "
        "los requisitos en (Res-CA-6605.md, Artículo 35º)."
    )

    sources = build_sources(answer, chunks)

    assert [(s.doc_name, s.section) for s in sources] == [
        ("Res-CA-6744.md", "Artículo 70º-2"),
        ("Res-CA-6605.md", "Artículo 35º"),
    ]
    assert sources[0].score == 0.9
    assert "repitencias" in sources[0].excerpt


def test_cita_no_verificable_se_descarta():
    """Cita de documento inexistente: se descarta y queda trazabilidad total."""
    chunks = [
        make_chunk("Doc-Real.md", "Artículo 1", "contenido real."),
        make_chunk("Doc-Real-2.md", "Artículo 2", "otro contenido."),
    ]
    answer = "Invención total (Doc-Inexistente.md, Artículo 99)."

    sources = build_sources(answer, chunks)

    # Nunca se inventan fuentes: cae al fallback de todos los chunks.
    assert [(s.doc_name, s.section) for s in sources] == [
        ("Doc-Real.md", "Artículo 1"),
        ("Doc-Real-2.md", "Artículo 2"),
    ]


def test_cita_con_variacion_de_seccion_se_verifica():
    """El LLM citó una subsección: coincide por contención con la sección."""
    chunks = [make_chunk("Reglamento.md", "Capítulo III — Requisitos", "texto.")]
    answer = "Los requisitos son... (Reglamento.md, Requisitos)"

    sources = build_sources(answer, chunks)

    assert len(sources) == 1
    assert sources[0].section == "Capítulo III — Requisitos"


def test_no_info_retorna_sources_vacios():
    chunks = [make_chunk("Doc.md", "Artículo 1", "texto.")]

    assert build_sources(NO_INFO_MESSAGE, chunks) == []


def test_chunks_vacios_retorna_sources_vacios():
    assert build_sources("cualquier respuesta (Doc.md, Art. 1)", []) == []


# --- answer_question (cadena completa) --------------------------------------


def test_umbral_pre_llm_no_gasta_tokens():
    """Sin chunks sobre el umbral: respuesta inmediata sin invocar el LLM."""
    retriever = FakeRetriever(chunks=[])
    llm = GuardLLM()
    config = SimpleNamespace(llm_model="fake-model")

    result = answer_question("¿pregunta?", retriever=retriever, llm=llm, config=config)

    assert isinstance(result, RagAnswer)
    assert result.answer == NO_INFO_MESSAGE
    assert result.sources == []
    assert result.used_fallback is True
    assert result.model == "fake-model"
    assert retriever.questions == ["¿pregunta?"]


def test_llm_no_info_marca_fallback():
    """El LLM copió el mensaje de no-información: sources=[] y fallback=True."""
    chunks = [make_chunk("Doc.md", "Artículo 1", "texto sin relación.")]
    retriever = FakeRetriever(chunks=chunks)
    llm = FakeLLM(NO_INFO_MESSAGE)
    config = SimpleNamespace(llm_model="fake-model")

    result = answer_question("¿pregunta?", retriever=retriever, llm=llm, config=config)

    assert result.answer == NO_INFO_MESSAGE
    assert result.sources == []
    assert result.used_fallback is True


def test_respuesta_vacia_se_degrada_a_no_info():
    """Respuesta vacía del LLM (thinking agotó max_tokens): no-información."""
    chunks = [make_chunk("Doc.md", "Artículo 1", "texto.")]
    retriever = FakeRetriever(chunks=chunks)
    llm = FakeLLM("   ")
    config = SimpleNamespace(llm_model="fake-model")

    result = answer_question("¿pregunta?", retriever=retriever, llm=llm, config=config)

    assert result.answer == NO_INFO_MESSAGE
    assert result.used_fallback is True
    assert result.sources == []


def test_respuesta_con_citas_genera_sources():
    """Flujo feliz: respuesta del LLM con citas verificables → Source."""
    chunks = [make_chunk("Res-CA-6744.md", "Artículo 70º-2", "límite de repitencias.")]
    retriever = FakeRetriever(chunks=chunks)
    llm = FakeLLM("Máximo tres repitencias (Res-CA-6744.md, Artículo 70º-2).")
    config = SimpleNamespace(llm_model="fake-model")

    result = answer_question(
        "¿cuántas repitencias puedo tener?",
        retriever=retriever,
        llm=llm,
        config=config,
    )

    assert result.used_fallback is False
    assert result.model == "fake-model"
    assert len(result.sources) == 1
    assert result.sources[0].doc_name == "Res-CA-6744.md"
    assert "repitencias" in result.sources[0].excerpt


def test_prompt_incluye_pregunta_contexto_y_no_info():
    """El prompt renderizado lleva la pregunta, el contexto y la regla 2."""
    chunks = [make_chunk("Doc.md", "Artículo 1", "contenido del contexto.")]
    retriever = FakeRetriever(chunks=chunks)
    llm = FakeLLM("respuesta con (Doc.md, Artículo 1).")
    config = SimpleNamespace(llm_model="fake-model")

    answer_question("¿mi pregunta?", retriever=retriever, llm=llm, config=config)

    prompt = llm.prompts[0]
    assert "¿mi pregunta?" in prompt
    assert "contenido del contexto." in prompt
    assert NO_INFO_MESSAGE in prompt
