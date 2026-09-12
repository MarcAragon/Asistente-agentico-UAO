"""Pruebas unitarias para el módulo llama_cloud_parsing."""

import pytest

from scripts.llama_cloud_parsing import clean_markdown, load_api_key, main


def test_clean_markdown_removes_layout_noise_and_preserves_content():
    """Verifica que se elimine ruido de layout y se conserve el contenido útil."""
    # Arrange
    source = """
    # Reglamento UAO

    logo: Universidad Autónoma de Occidente
    Página 1 de 2

    ## Artículo 1

    Contenido     del artículo.

    ![imagen](imagen.png)

    ## Artículo 2

    Contenido del segundo artículo.
    """

    # Act
    result = clean_markdown(source)

    # Assert
    assert "logo:" not in result
    assert "Página 1 de 2" not in result
    assert "![imagen]" not in result
    assert "# Reglamento UAO" in result
    assert "## Artículo 1" in result
    assert "## Artículo 2" in result
    assert "Contenido del artículo." in result
    assert "Contenido del segundo artículo." in result


@pytest.mark.parametrize(
    "noise",
    [
        "logo: Universidad Autónoma de Occidente",
        "signature: firma del documento",
        "stamp: sello institucional",
        "handwriting: texto manuscrito",
        "handwritten: anotación",
        "Página 1 de 2",
        "![imagen](imagen.png)",
    ],
)
def test_clean_markdown_removes_noise_patterns(noise):
    """Verifica que clean_markdown elimine los patrones de ruido conocidos."""
    # Arrange
    source = f"# Documento UAO\n\n{noise}\n\nContenido útil."

    # Act
    result = clean_markdown(source)

    # Assert
    assert noise not in result
    assert "# Documento UAO" in result
    assert "Contenido útil." in result


def test_clean_markdown_normalizes_spacing():
    """Verifica que se normalicen espacios y saltos de línea."""
    # Arrange
    source = (
        "\u00a0# Documento UAO\u00a0\n"
        "\n\n\n"
        "Contenido     con     espacios.\n"
        "   Segunda línea.   \n"
        "\n\n\n"
    )

    # Act
    result = clean_markdown(source)

    # Assert
    assert result == (
        "# Documento UAO\n\n"
        "Contenido con espacios.\n"
        "Segunda línea.\n"
    )


def test_load_api_key_returns_configured_key(monkeypatch):
    """Verifica que load_api_key devuelva la API key configurada."""
    # Arrange
    monkeypatch.setattr(
        "scripts.llama_cloud_parsing.settings.llama_cloud_api_key",
        "  test-api-key-123  ",
    )

    # Act
    result = load_api_key()

    # Assert
    assert result == "test-api-key-123"


def test_load_api_key_exits_when_key_is_not_configured(monkeypatch, capsys):
    """Verifica que load_api_key termine si no existe una API key."""
    # Arrange
    monkeypatch.setattr(
        "scripts.llama_cloud_parsing.settings.llama_cloud_api_key",
        "",
    )

    # Act
    with pytest.raises(SystemExit) as exc_info:
        load_api_key()

    # Assert
    assert exc_info.value.code == 1

    captured = capsys.readouterr()
    assert "LLAMA_CLOUD_API_KEY no está configurada" in captured.err


def test_main_returns_one_when_no_pdfs_are_found(monkeypatch, tmp_path):
    """Verifica que main devuelva 1 cuando no encuentra archivos PDF."""
    # Arrange
    monkeypatch.setattr(
        "scripts.llama_cloud_parsing.settings.docs_dir",
        tmp_path,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["llama_cloud_parsing.py"],
    )

    # Act
    result = main()

    # Assert
    assert result == 1