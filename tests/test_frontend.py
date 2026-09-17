"""Tests del cliente HTTP del frontend (Fase 7, tarea 7.8): funciones puras,
con httpx mockeado, sin levantar Streamlit ni la API real."""

from __future__ import annotations

import httpx

from asistente_agentico_uao.frontend.client import preguntar_api

URL_PRUEBA = "http://localhost:8000/ask"


def _respuesta(status_code: int, payload: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload or {},
        request=httpx.Request("POST", URL_PRUEBA),
    )


def test_pregunta_exitosa_devuelve_respuesta_y_fuentes(monkeypatch):
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
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _respuesta(200, payload))

    mensaje = preguntar_api("¿Que pasa si repruebo tres veces?")

    assert mensaje["rol"] == "assistant"
    assert mensaje["texto"] == payload["answer"]
    assert mensaje["fuentes"] == payload["sources"]
    assert mensaje["fallback"] is False


def test_error_422_pregunta_invalida(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _respuesta(422))

    mensaje = preguntar_api("")

    assert mensaje["fallback"] is True
    assert "no es válida" in mensaje["texto"]


def test_error_503_sin_api_key(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _respuesta(503))

    mensaje = preguntar_api("¿Cualquier pregunta?")

    assert mensaje["fallback"] is True
    assert "CEREBRAS_API_KEY" in mensaje["texto"]


def test_error_500_interno(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _respuesta(500))

    mensaje = preguntar_api("¿Cualquier pregunta?")

    assert mensaje["fallback"] is True
    assert "500" in mensaje["texto"]


def test_error_conexion(monkeypatch):
    def _lanzar(*args, **kwargs):
        raise httpx.ConnectError("no se pudo conectar")

    monkeypatch.setattr(httpx, "post", _lanzar)

    mensaje = preguntar_api("¿Cualquier pregunta?")

    assert mensaje["fallback"] is True
    assert "No se pudo conectar" in mensaje["texto"]
