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