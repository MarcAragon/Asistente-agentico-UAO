"""Pruebas unitarias para el módulo de configuración."""

import pytest

from asistente_agentico_uao import config


def test_resolve_embedding_device_returns_configured_device(monkeypatch):
    """Verifica que se respete el dispositivo configurado explícitamente."""
    # Arrange
    monkeypatch.setattr(config.settings, "embedding_device", "cpu")

    # Act
    result = config.resolve_embedding_device()

    # Assert
    assert result == "cpu"


@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda"],
)
def test_resolve_embedding_device_returns_configured_devices(monkeypatch, device):
    """Verifica que se puedan configurar distintos dispositivos de embeddings."""
    # Arrange
    monkeypatch.setattr(config.settings, "embedding_device", device)

    # Act
    result = config.resolve_embedding_device()

    # Assert
    assert result == device


def test_settings_has_default_chunk_values():
    """Verifica que la configuración de chunking tenga valores definidos."""
    assert config.settings.chunk_size_tokens > 0
    assert config.settings.chunk_overlap_tokens >= 0
    assert config.settings.chunk_max_tokens > 0


def test_chunk_overlap_is_smaller_than_chunk_size():
    """Verifica que el solapamiento no supere el tamaño del chunk."""
    assert (
        config.settings.chunk_overlap_tokens
        < config.settings.chunk_size_tokens
    )


def test_embedding_model_is_defined():
    """Verifica que exista un modelo de embeddings configurado."""
    assert config.settings.embedding_model is not None
    assert len(config.settings.embedding_model) > 0