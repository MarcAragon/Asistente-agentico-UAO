"""Tests del cache semantico (Fase 7, tarea 7.8): SemanticCache con fakeredis,
sin Redis real ni modelo de embeddings descargado de mas (usa el mismo modelo
E5 que el resto del proyecto, ya que la similitud tiene que ser real).
"""

from __future__ import annotations

import fakeredis
import pytest

from asistente_agentico_uao.cache import SemanticCache
from asistente_agentico_uao.chain import RagAnswer, Source
from asistente_agentico_uao.config import Settings


@pytest.fixture
def cliente_falso():
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def cache(cliente_falso):
    config = Settings(
        cache_similarity=0.97,
        cache_ttl_seconds=86400,
    )
    return SemanticCache(
        config=config,
        client=cliente_falso,
    )


RESPUESTA_EJEMPLO = RagAnswer(
    answer="Si repruebas tres veces, entras a prueba academica por repitencia.",
    sources=[
        Source(
            doc_name="Res-CA-6744.pdf",
            section="ARTICULO 70º-2",
            score=0.88,
            excerpt="...",
        )
    ],
    model="qwen-3.8-27b",
    used_fallback=False,
)


@pytest.mark.slow
def test_hit_por_pregunta_identica(cache):
    cache.guardar(
        "¿Que pasa si repruebo tres veces una asignatura?",
        RESPUESTA_EJEMPLO,
    )

    encontrada = cache.buscar(
        "¿Que pasa si repruebo tres veces una asignatura?"
    )

    assert encontrada is not None
    assert encontrada.answer == RESPUESTA_EJEMPLO.answer
    assert encontrada.sources[0].doc_name == "Res-CA-6744.pdf"


@pytest.mark.slow
def test_miss_pregunta_sin_relacion(cache):
    cache.guardar(
        "¿Que pasa si repruebo tres veces una asignatura?",
        RESPUESTA_EJEMPLO,
    )

    encontrada = cache.buscar(
        "¿Cual es la receta de las arepas?"
    )

    assert encontrada is None


@pytest.mark.slow
def test_miss_con_cache_vacio(cache):
    assert cache.buscar(
        "cualquier pregunta"
    ) is None


def test_degradacion_sin_redis():
    """Sin conexion a Redis la aplicación continúa funcionando."""

    config = Settings(
        redis_url="redis://host-que-no-existe:6379/0"
    )

    cache = SemanticCache(
        config=config
    )

    cache.guardar(
        "una pregunta",
        RESPUESTA_EJEMPLO,
    )

    assert cache.buscar(
        "una pregunta"
    ) is None


def test_degradacion_con_cache_deshabilitado(cliente_falso):
    config = Settings(
        cache_enabled=False
    )

    cache = SemanticCache(
        config=config,
        client=cliente_falso,
    )

    cache.guardar(
        "una pregunta",
        RESPUESTA_EJEMPLO,
    )

    assert cache.buscar(
        "una pregunta"
    ) is None


@pytest.mark.slow
def test_invalidar_todo(cache):
    cache.guardar(
        "¿Que pasa si repruebo tres veces una asignatura?",
        RESPUESTA_EJEMPLO,
    )

    cache.invalidar_todo()

    assert cache.buscar(
        "¿Que pasa si repruebo tres veces una asignatura?"
    ) is None


@pytest.mark.slow
def test_estadisticas_hit_miss(cache):
    cache.guardar(
        "¿Que pasa si repruebo tres veces una asignatura?",
        RESPUESTA_EJEMPLO,
    )

    cache.buscar(
        "¿Que pasa si repruebo tres veces una asignatura?"
    )

    cache.buscar(
        "¿Cual es la receta de las arepas?"
    )

    stats = cache.estadisticas()

    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_ratio"] == 0.5


def test_guardar_y_recuperar_varias_respuestas(cache):
    """Verifica múltiples entradas."""

    segunda_respuesta = RagAnswer(
        answer="La matrícula debe realizarse dentro de las fechas establecidas.",
        sources=[],
        model="qwen-3.8-27b",
        used_fallback=False,
    )

    cache.guardar(
        "Pregunta uno",
        RESPUESTA_EJEMPLO,
    )

    cache.guardar(
        "Pregunta dos",
        segunda_respuesta,
    )

    resultado = cache.buscar(
        "Pregunta dos"
    )

    assert resultado is not None
    assert resultado.answer == segunda_respuesta.answer


def test_cache_respuesta_con_fuentes_vacias(cache):
    """Verifica respuestas sin fuentes."""

    respuesta = RagAnswer(
        answer="Respuesta sin documentos asociados.",
        sources=[],
        model="qwen-3.8-27b",
        used_fallback=False,
    )

    cache.guardar(
        "Pregunta sin fuentes",
        respuesta,
    )

    resultado = cache.buscar(
        "Pregunta sin fuentes"
    )

    assert resultado is not None
    assert resultado.sources == []


def test_invalidar_todo_elimina_varias_entradas(cache):
    """Verifica limpieza completa."""

    cache.guardar(
        "Pregunta uno",
        RESPUESTA_EJEMPLO,
    )

    cache.guardar(
        "Pregunta dos",
        RESPUESTA_EJEMPLO,
    )

    cache.invalidar_todo()

    assert cache.buscar("Pregunta uno") is None
    assert cache.buscar("Pregunta dos") is None


def test_estadisticas_iniciales_cache_vacio(cache):
    """Verifica estadísticas iniciales."""

    stats = cache.estadisticas()

    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["hit_ratio"] == 0


def test_guardar_no_falla_con_texto_vacio(cache):
    """Verifica preguntas vacías."""

    cache.guardar(
        "",
        RESPUESTA_EJEMPLO,
    )

    resultado = cache.buscar("")

    assert resultado is not None


# ==========================
# NUEVAS PRUEBAS DE COBERTURA
# ==========================


def test_cache_retorna_mismo_modelo_guardado(cache):
    """Verifica que se conserve el modelo asociado."""

    cache.guardar(
        "Pregunta modelo",
        RESPUESTA_EJEMPLO,
    )

    resultado = cache.buscar(
        "Pregunta modelo"
    )

    assert resultado.model == "qwen-3.8-27b"


def test_cache_conserva_estado_fallback(cache):
    """Verifica persistencia del indicador fallback."""

    respuesta = RagAnswer(
        answer="Respuesta fallback",
        sources=[],
        model="qwen",
        used_fallback=True,
    )

    cache.guardar(
        "Pregunta fallback",
        respuesta,
    )

    resultado = cache.buscar(
        "Pregunta fallback"
    )

    assert resultado.used_fallback is True


def test_cache_fuente_conserva_score(cache):
    """Verifica conservación del puntaje de fuente."""

    cache.guardar(
        "Pregunta score",
        RESPUESTA_EJEMPLO,
    )

    resultado = cache.buscar(
        "Pregunta score"
    )

    assert resultado.sources[0].score == 0.88


def test_cache_respuesta_no_modifica_objeto_original(cache):
    """Verifica que guardar no altere la respuesta original."""

    original = RESPUESTA_EJEMPLO.answer

    cache.guardar(
        "Pregunta copia",
        RESPUESTA_EJEMPLO,
    )

    assert RESPUESTA_EJEMPLO.answer == original