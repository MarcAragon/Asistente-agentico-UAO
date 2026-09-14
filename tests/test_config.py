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