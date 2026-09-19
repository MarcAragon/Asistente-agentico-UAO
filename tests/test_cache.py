"""Tests del cache semantico (Fase 7, tarea 7.8): SemanticCache con fakeredis,
sin Redis real ni modelo de embeddings descargado de mas (usa el mismo modelo
E5 que el resto del proyecto, ya que la similitud tiene que ser real)."""

from __future__ import annotations

import fakeredis
import pytest

from asistente_agentico_uao.core.config import Settings
from asistente_agentico_uao.rag.cache import SemanticCache
from asistente_agentico_uao.rag.chain import RagAnswer, Source


@pytest.fixture
def cliente_falso():
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def cache(cliente_falso):
    config = Settings(cache_similarity=0.97, cache_ttl_seconds=86400)
    return SemanticCache(config=config, client=cliente_falso)


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
    cache.guardar("¿Que pasa si repruebo tres veces una asignatura?", RESPUESTA_EJEMPLO)

    encontrada = cache.buscar("¿Que pasa si repruebo tres veces una asignatura?")

    assert encontrada is not None
    assert encontrada.answer == RESPUESTA_EJEMPLO.answer
    assert encontrada.sources[0].doc_name == "Res-CA-6744.pdf"


@pytest.mark.slow
def test_miss_pregunta_sin_relacion(cache):
    cache.guardar("¿Que pasa si repruebo tres veces una asignatura?", RESPUESTA_EJEMPLO)

    encontrada = cache.buscar("¿Cual es la receta de las arepas?")

    assert encontrada is None


@pytest.mark.slow
def test_miss_con_cache_vacio(cache):
    assert cache.buscar("cualquier pregunta") is None


def test_degradacion_sin_redis():
    """Sin conexion a Redis, buscar/guardar no truenan: la app sigue viva."""
    config = Settings(redis_url="redis://host-que-no-existe:6379/0")
    cache = SemanticCache(config=config)

    cache.guardar("una pregunta", RESPUESTA_EJEMPLO)  # no debe lanzar excepcion
    assert cache.buscar("una pregunta") is None


def test_degradacion_con_cache_deshabilitado(cliente_falso):
    config = Settings(cache_enabled=False)
    cache = SemanticCache(config=config, client=cliente_falso)

    cache.guardar("una pregunta", RESPUESTA_EJEMPLO)
    assert cache.buscar("una pregunta") is None


@pytest.mark.slow
def test_invalidar_todo(cache):
    cache.guardar("¿Que pasa si repruebo tres veces una asignatura?", RESPUESTA_EJEMPLO)
    cache.invalidar_todo()

    assert cache.buscar("¿Que pasa si repruebo tres veces una asignatura?") is None


@pytest.mark.slow
def test_estadisticas_hit_miss(cache):
    cache.guardar("¿Que pasa si repruebo tres veces una asignatura?", RESPUESTA_EJEMPLO)
    cache.buscar("¿Que pasa si repruebo tres veces una asignatura?")  # hit
    cache.buscar("¿Cual es la receta de las arepas?")  # miss

    stats = cache.estadisticas()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_ratio"] == 0.5
