"""Esquemas Pydantic de la API REST (contrato §3.3, Fase 5).

Espejo de los dataclasses ``RagAnswer``/``Source`` de ``chain.py``: la API
serializa hacia el exterior, la cadena mantiene su contrato interno.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Pregunta del estudiante: 1..500 caracteres, no vacía ni solo blancos."""

    question: str = Field(
        min_length=1,
        max_length=500,
        description="Pregunta sobre normativa UAO (1..500 caracteres).",
    )

    @field_validator("question")
    @classmethod
    def _sin_blancos(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("la pregunta no puede estar vacía ni ser solo espacios")
        return value


class SourceModel(BaseModel):
    """Fuente citada en la respuesta (verificación de trazabilidad)."""

    doc_name: str
    section: str
    score: float
    excerpt: str


class AskResponse(BaseModel):
    """Respuesta del asistente con sus fuentes verificables."""

    answer: str
    sources: list[SourceModel]
    model: str
    used_fallback: bool


class HealthResponse(BaseModel):
    """Estado del servicio (monitoreo)."""

    status: str
    index_chunks: int
    device: str
    cache_hits: int
    cache_misses: int
    cache_hit_ratio: float


class DocumentInfo(BaseModel):
    """Documento indexado con conteo de chunks (trazabilidad)."""

    doc_name: str
    chunks: int
