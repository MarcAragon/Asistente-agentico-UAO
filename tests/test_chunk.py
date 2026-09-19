from asistente_agentico_uao.ingestion.chunk import (
    _units,
    split_sections,
)


def test_split_sections_separates_markdown_headers():
    md = """# Introducción

Este es el texto inicial.

## Artículo 1

Este es el contenido del artículo.
"""

    result = split_sections(md)

    assert result == [
        ("Introducción", "Este es el texto inicial."),
        ("Artículo 1", "Este es el contenido del artículo."),
    ]


def test_split_sections_detects_article_as_section():
    md = """Texto inicial del documento.

**ARTÍCULO 1º:** Disposiciones generales.

Este es el contenido del artículo.
"""

    result = split_sections(md)

    assert result[0] == ("", "Texto inicial del documento.")
    assert result[1][0] == "ARTÍCULO 1º: Disposiciones generales."
    assert "Este es el contenido del artículo." in result[1][1]


def test_units_keeps_small_table_as_one_unit():
    md = """Contenido antes de la tabla.

<table>
<thead>
<tr><th>Nombre</th><th>Valor</th></tr>
</thead>
<tbody>
<tr><td>Uno</td><td>1</td></tr>
<tr><td>Dos</td><td>2</td></tr>
</tbody>
</table>
"""

    result = _units(md, lambda text: len(text.split()), max_tokens=50)

    assert len(result) == 2
    assert result[0] == ("Contenido antes de la tabla.", False)
    assert result[1][1] is True
    assert "<thead>" in result[1][0]
    assert "<tr><td>Uno</td><td>1</td></tr>" in result[1][0]


def test_units_splits_large_table_and_repeats_header():
    md = """<table>
<thead>
<tr><th>Nombre</th><th>Valor</th></tr>
</thead>
<tbody>
<tr><td>Uno</td><td>11111111111111111111</td></tr>
<tr><td>Dos</td><td>22222222222222222222</td></tr>
<tr><td>Tres</td><td>33333333333333333333</td></tr>
</tbody>
</table>
"""

    result = _units(md, lambda text: len(text.split()), max_tokens=8)

    assert len(result) > 1

    for table, is_table in result:
        assert is_table is True
        assert "<thead>" in table
        assert "<tr><th>Nombre</th><th>Valor</th></tr>" in table


def test_split_sections_returns_empty_for_empty_markdown():
    """Verifica que un markdown vacío no genere secciones."""
    result = split_sections("")

    assert result == []


def test_split_sections_keeps_plain_text_without_headers():
    """Verifica que un texto sin encabezados se conserve como una sección."""
    md = "Contenido simple del documento."

    result = split_sections(md)

    assert result == [("", "Contenido simple del documento.")]


def test_units_splits_long_text_into_multiple_units():
    """Verifica que textos largos sean divididos en unidades."""
    md = (
        "Este es un texto muy largo con muchas palabras "
        "que debe dividirse correctamente en varias unidades."
    )

    result = _units(md, lambda text: len(text.split()), max_tokens=5)

    assert len(result) > 1
    assert all(is_table is False for _, is_table in result)


def test_units_keeps_short_text_as_single_unit():
    """Verifica que textos pequeños permanezcan como una unidad."""
    md = "Texto corto."

    result = _units(md, lambda text: len(text.split()), max_tokens=50)

    assert len(result) == 1
    assert result[0][0] == md
    assert result[0][1] is False


def test_units_handles_empty_text():
    """Verifica el comportamiento con texto vacío."""
    result = _units("", lambda text: len(text.split()), max_tokens=10)

    assert result == []


def test_units_identifies_table_content():
    """Verifica que una tabla HTML sea identificada correctamente."""
    md = """
<table>
<tr><td>Dato</td></tr>
</table>
"""

    result = _units(md, lambda text: len(text.split()), max_tokens=50)

    assert len(result) == 1
    assert result[0][1] is True
    assert "<table>" in result[0][0]