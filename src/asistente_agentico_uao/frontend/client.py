"""Cliente HTTP puro del frontend (Fase 7, tarea 7.8).

Sin nada de Streamlit aqui a proposito: asi se puede probar con httpx
mockeado, sin levantar la interfaz ni la API real.
"""

from __future__ import annotations

import os

import httpx

API_BASE_URL = os.getenv("UAO_RAG__API_BASE_URL", "http://localhost:8000")


def preguntar_api(pregunta: str, base_url: str = API_BASE_URL) -> dict:
    """Llama a POST /ask y devuelve un mensaje listo para el historial del chat."""
    try:
        respuesta = httpx.post(
            f"{base_url}/ask",
            json={"question": pregunta},
            timeout=30.0,
        )
        respuesta.raise_for_status()
        datos = respuesta.json()
        return {
            "rol": "assistant",
            "texto": datos["answer"],
            "fuentes": datos["sources"],
            "fallback": datos["used_fallback"],
        }
    except httpx.HTTPStatusError as exc:
        codigo = exc.response.status_code
        if codigo == 422:
            texto = "La pregunta no es válida (vacía o muy larga)."
        elif codigo == 503:
            texto = "El asistente no tiene configurada la clave del modelo (CEREBRAS_API_KEY)."
        else:
            texto = f"Error interno de la API ({codigo})."
        return {"rol": "assistant", "texto": texto, "fuentes": [], "fallback": True}
    except httpx.RequestError:
        texto = f"No se pudo conectar con la API. ¿Está corriendo en {base_url}?"
        return {"rol": "assistant", "texto": texto, "fuentes": [], "fallback": True}