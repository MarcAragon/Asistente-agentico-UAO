"""Tests del cliente HTTP del frontend (Fase 7, tarea 7.8).

Se validan las funciones encargadas de comunicarse con la API REST desde
el frontend mediante solicitudes HTTP simuladas.

Las pruebas utilizan ``httpx`` mockeado para evitar levantar Streamlit,
la API real o servicios externos durante la ejecución.
"""

from __future__ import annotations

import httpx

from asistente_agentico_uao.frontend.client import preguntar_api


URL_PRUEBA = "http://localhost:8000/ask"


def _respuesta(
    status_code: int,
    payload: dict | None = None,
) -> httpx.Response:
    """Construye una respuesta HTTP simulada para las pruebas.

    Args:
        status_code:
            Código HTTP que será retornado por la respuesta falsa.
        payload:
            Contenido JSON opcional de la respuesta.

    Returns:
        Respuesta HTTP simulada compatible con httpx.
    """

    return httpx.Response(
        status_code,
        json=payload or {},
        request=httpx.Request(
            "POST",
            URL_PRUEBA,
        ),
    )


def test_pregunta_exitosa_devuelve_respuesta_y_fuentes(monkeypatch):
    """Verifica el flujo exitoso de consulta contra la API.

    Comprueba que una respuesta válida entregue correctamente:
    - Texto generado.
    - Fuentes asociadas.
    - Modelo utilizado.
    - Estado del fallback.
    """

    payload = {
        "answer": "Si repruebas tres veces...",
        "sources": [
            {
                "doc_name": "Res-CA-6744.pdf",
                "section": "Art 70",
                "score": 0.88,
                "excerpt": "...",
            }
        ],
        "model": "qwen-3.8-27b",
        "used_fallback": False,
    }

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _respuesta(
            200,
            payload,
        ),
    )

    mensaje = preguntar_api(
        "¿Que pasa si repruebo tres veces?"
    )

    assert mensaje["rol"] == "assistant"
    assert mensaje["texto"] == payload["answer"]
    assert mensaje["fuentes"] == payload["sources"]
    assert mensaje["fallback"] is False


def test_error_422_pregunta_invalida(monkeypatch):
    """Verifica el manejo de preguntas inválidas mediante HTTP 422."""

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _respuesta(422),
    )

    mensaje = preguntar_api("")

    assert mensaje["fallback"] is True
    assert "no es válida" in mensaje["texto"]


def test_error_503_sin_api_key(monkeypatch):
    """Verifica el comportamiento cuando la API no tiene credenciales."""

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _respuesta(503),
    )

    mensaje = preguntar_api(
        "¿Cualquier pregunta?"
    )

    assert mensaje["fallback"] is True
    assert "CEREBRAS_API_KEY" in mensaje["texto"]


def test_error_500_interno(monkeypatch):
    """Verifica el manejo de errores internos del servidor."""

    monkeypatch.setattr(
        httpx,
        "post",
        lambda *a, **k: _respuesta(500),
    )

    mensaje = preguntar_api(
        "¿Cualquier pregunta?"
    )

    assert mensaje["fallback"] is True
    assert "500" in mensaje["texto"]


def test_error_conexion(monkeypatch):
    """Verifica el comportamiento cuando la API no está disponible."""

    def _lanzar(*args, **kwargs):
        """Simula una falla de conexión HTTP."""

        raise httpx.ConnectError(
            "no se pudo conectar"
        )

    monkeypatch.setattr(
        httpx,
        "post",
        _lanzar,
    )

    mensaje = preguntar_api(
        "¿Cualquier pregunta?"
    )

    assert mensaje["fallback"] is True
    assert "No se pudo conectar" in mensaje["texto"]
