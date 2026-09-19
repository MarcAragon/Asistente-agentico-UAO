import numpy as np

from asistente_agentico_uao.embeddings import (
    embed_passages,
    embed_query,
)


def test_embed_query_uses_query_prefix(monkeypatch):
    """Verifica que embed_query agregue el prefijo query:"""

    captured_texts = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            captured_texts.extend(texts)
            return np.array([[0.1, 0.2, 0.3]])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    result = embed_query("¿Cuál es la fecha de cancelaciones?")

    assert captured_texts == [
        "query: ¿Cuál es la fecha de cancelaciones?"
    ]
    assert result.shape == (3,)


def test_embed_passages_uses_passage_prefix(monkeypatch):
    """Verifica que embed_passages agregue el prefijo passage:"""

    captured_texts = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            captured_texts.extend(texts)
            return np.array(
                [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ]
            )

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    result = embed_passages(
        ["Primer documento", "Segundo documento"]
    )

    assert captured_texts == [
        "passage: Primer documento",
        "passage: Segundo documento",
    ]
    assert result.shape == (2, 3)


def test_embed_query_returns_numpy_array(monkeypatch):
    """Verifica que embed_query retorne un arreglo numpy."""

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([[0.5, 0.6, 0.7]])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    result = embed_query("Pregunta de prueba")

    assert isinstance(result, np.ndarray)


def test_embed_query_handles_empty_question(monkeypatch):
    """Verifica que embed_query procese una pregunta vacía."""

    captured_texts = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            captured_texts.extend(texts)
            return np.array([[0.1, 0.2]])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    result = embed_query("")

    assert captured_texts == ["query: "]
    assert result.shape == (2,)


def test_embed_passages_returns_numpy_array(monkeypatch):
    """Verifica que embed_passages retorne un arreglo numpy."""

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([[0.1, 0.2]])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    result = embed_passages(["Documento"])

    assert isinstance(result, np.ndarray)


def test_embed_passages_empty_list(monkeypatch):
    """Verifica el comportamiento cuando no existen documentos."""

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    result = embed_passages([])

    assert isinstance(result, np.ndarray)


def test_embed_passages_preserves_document_order(monkeypatch):
    """Verifica que los documentos mantengan el orden original."""

    captured_texts = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            captured_texts.extend(texts)
            return np.array(
                [
                    [1, 0],
                    [0, 1],
                ]
            )

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    embed_passages(["A", "B"])

    assert captured_texts == [
        "passage: A",
        "passage: B",
    ]


def test_embed_query_uses_model_encode(monkeypatch):
    """Verifica que embed_query utilice el método encode del modelo."""

    called = False

    class FakeModel:
        def encode(self, texts, **kwargs):
            nonlocal called
            called = True
            return np.array([[1, 2, 3]])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    embed_query("Consulta")

    assert called is True


# ==============================
# NUEVAS PRUEBAS PARA CALIDAD
# ==============================


def test_embed_query_preserves_embedding_values(monkeypatch):
    """Verifica que embed_query conserve los valores generados."""

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([[0.8, 0.9]])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    result = embed_query("consulta")

    assert np.array_equal(
        result,
        np.array([0.8, 0.9]),
    )


def test_embed_passages_handles_multiple_documents(monkeypatch):
    """Verifica procesamiento de varios documentos."""

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 3))

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    result = embed_passages(
        [
            "Documento uno",
            "Documento dos",
            "Documento tres",
        ]
    )

    assert result.shape == (3, 3)


def test_embed_query_calls_model_once(monkeypatch):
    """Verifica que una consulta use una sola llamada al modelo."""

    calls = 0

    class FakeModel:
        def encode(self, texts, **kwargs):
            nonlocal calls
            calls += 1
            return np.array([[1, 1]])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    embed_query("consulta")

    assert calls == 1


def test_embed_passages_calls_model_once(monkeypatch):
    """Verifica que varios documentos se procesen en una llamada."""

    calls = 0

    class FakeModel:
        def encode(self, texts, **kwargs):
            nonlocal calls
            calls += 1
            return np.array([[1, 2]])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    embed_passages(
        [
            "A",
            "B",
        ]
    )

    assert calls == 1


def test_embed_query_handles_unicode_text(monkeypatch):
    """Verifica consultas con caracteres especiales."""

    captured = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            captured.extend(texts)
            return np.array([[1]])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    embed_query("¿Información académica?")

    assert captured == [
        "query: ¿Información académica?"
    ]


def test_embed_passages_handles_unicode_documents(monkeypatch):
    """Verifica documentos con caracteres especiales."""

    captured = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            captured.extend(texts)
            return np.array([[1]])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    embed_passages(
        [
            "Artículo académico ñ",
        ]
    )

    assert captured == [
        "passage: Artículo académico ñ"
    ]


def test_embed_query_returns_flat_vector(monkeypatch):
    """Verifica que la consulta retorne un vector unidimensional."""

    class FakeModel:
        def encode(self, texts, **kwargs):
            return np.array([[1, 2, 3, 4]])

    monkeypatch.setattr(
        "asistente_agentico_uao.embeddings.get_model",
        lambda: FakeModel(),
    )

    result = embed_query("consulta")

    assert result.ndim == 1
    assert result.shape[0] == 4