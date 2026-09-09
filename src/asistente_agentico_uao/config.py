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

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # Vacío -> detección automática (CUDA -> ROCm -> CPU).
    embedding_device: str = ""

    # --- LLM (Cerebras) ---
    llm_model: str = "qwen-3.8-27b"

    # --- Claves de API (sin prefijo; solo .env o entorno, nunca en el repo) ---
    cerebras_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("CEREBRAS_API_KEY", "UAO_RAG__CEREBRAS_API_KEY"),
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
