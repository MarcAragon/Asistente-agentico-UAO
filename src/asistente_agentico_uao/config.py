"""Configuración central del asistente RAG UAO.

Lee variables de entorno con prefijo ``UAO_RAG__`` (doble guion bajo para
anidar) y también un archivo ``.env`` en la raíz del proyecto.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Raíz del proyecto (dos niveles arriba de este archivo: src/asistente_agentico_uao/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Ajustes del sistema, sobrescribibles por variables de entorno."""

    model_config = SettingsConfigDict(
        env_prefix="UAO_RAG__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Rutas ---
    docs_dir: Path = PROJECT_ROOT / "Data" / "Documentos"
    # Salida del parseo con LlamaCloud Parse (markdown limpio, entrada del RAG)
    markdown_dir: Path = PROJECT_ROOT / "Data" / "Documentos_MD"
    chroma_dir: Path = PROJECT_ROOT / "Data" / "chroma"

    # --- Recuperación ---
    top_k: int = 5
    min_similarity: float = 0.35

    # --- Embeddings (E5: requiere prefijos query:/passage:, ver embeddings.py) ---
    embedding_model: str = "intfloat/multilingual-e5-base"
    # Vacío -> detección automática (CUDA -> ROCm -> CPU).
    embedding_device: str = ""
    embedding_batch_size: int = 16

    # --- Chunking (Fase 2) ---
    # Objetivo de tokens del chunk final (incluye el prefijo contextual).
    chunk_size_tokens: int = 400
    # Techo duro del chunk final; debe quedar bajo max_seq_length del modelo (512).
    chunk_max_tokens: int = 450
    # Oraciones finales que se repiten en el chunk siguiente (solo intra-sección).
    chunk_overlap_tokens: int = 70
    # Chunks menores se fusionan con el anterior dentro de la misma sección.
    chunk_min_tokens: int = 80

    # --- LLM (Cerebras) ---
    llm_model: str = "qwen-3.8-27b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    # qwen-3.8-27b razona por defecto: en F4 agotó los max_tokens en tokens
    # de thinking (finish_reason=length, content vacío). Se desactiva: la
    # síntesis solo-contexto no lo necesita. Hallazgo F4 2026-09-15.
    llm_disable_reasoning: bool = True
    # Reintentos ante 429/timeout con backoff exponencial (ver llm.py).
    llm_max_retries: int = 3

    # --- gRPC: plano de control de ingesta (Fase 5) ---
    # La consulta pública (pregunta→respuesta) vive en la API REST (:8000).
    # gRPC expone SOLO la administración del índice (IndexAdmin: Ingest con
    # progreso en streaming, PruneIndex, IndexStatus) en :50051; no duplica
    # funciones de la REST. Se puede desactivar (solo REST) con 0.
    grpc_enabled: bool = True
    grpc_port: int = 50051

    # --- Cache semantico (Redis, Fase 7) ---
    # Se degrada solo: sin Redis disponible o con cache_enabled=False, buscar()
    # siempre devuelve None (miss) y la app responde igual, solo mas lento.
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_similarity: float = 0.97
    cache_ttl_seconds: int = 86400

    # --- Claves de API (sin prefijo; solo .env o entorno, nunca en el repo) ---
    cerebras_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("CEREBRAS_API_KEY", "UAO_RAG__CEREBRAS_API_KEY"),
    )
    # Claves adicionales de Cerebras separadas por coma: rotación ante
    # límites de cuota/429 por clave (ver llm.py, CerebrasLLM).
    cerebras_api_keys: str = Field(
        default="",
        validation_alias=AliasChoices("CEREBRAS_API_KEYS", "UAO_RAG__CEREBRAS_API_KEYS"),
    )
    llama_cloud_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLAMA_CLOUD_API_KEY", "UAO_RAG__LLAMA_CLOUD_API_KEY"),
    )


settings = Settings()


def resolve_embedding_device() -> str:
    """Resuelve el dispositivo de inferencia del modelo de embeddings.

    Orden de preferencia: CUDA (NVIDIA), ROCm (AMD), CPU. El mismo
    ``device="cuda"`` sirve para NVIDIA (CUDA) y AMD (ROCm) porque PyTorch
    expone ambas bajo la API de CUDA cuando la compilación lo soporta.
    """
    import torch

    if settings.embedding_device:
        return settings.embedding_device
    if torch.cuda.is_available():
        # ROCm reporta hip; CUDA reporta cuXX. Ambos usan device "cuda".
        backend = "rocm" if torch.version.hip else "cuda"
        name = torch.cuda.get_device_name(0)
        print(f"[device] Embeddings en GPU ({backend}): {name}")
        return "cuda"
    print("[device] GPU no disponible; embeddings en CPU")
    return "cpu"
