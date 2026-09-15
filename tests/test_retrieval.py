from types import SimpleNamespace

import numpy as np

from asistente_agentico_uao.retrieval import Retriever, format_context


class FakeCollection:
    def __init__(self, documents, metadatas, distances):
        self.documents = documents
        self.metadatas = metadatas
        self.distances = distances
        self.last_n_results = None

    def count(self):
        return len(self.documents[0])

    def query(self, **kwargs):
        self.last_n_results = kwargs["n_results"]
        return {
            "documents": self.documents,
            "metadatas": self.metadatas,
            "distances": self.distances,
        }


def test_retrieve_filters_sorts_and_limits_results(monkeypatch):
    collection = FakeCollection(
        documents=[["Documento A", "Documento B", "Documento C"]],
        metadatas=[
            [
                {
                    "doc_name": "doc-a.md",
                    "section": "Sección A",
                    "chunk_index": 1,
                    "is_table": False,
                },
                {
                    "doc_name": "doc-b.md",
                    "section": "Sección B",
                    "chunk_index": 2,
                    "is_table": True,
                },
                {
                    "doc_name": "doc-c.md",
                    "section": "Sección C",
                    "chunk_index": 3,
                    "is_table": False,
                },
            ]
        ],
        distances=[[0.20, 0.05, 0.50]],
    )

    config = SimpleNamespace(
        top_k=2,
        min_similarity=0.6,
    )

    monkeypatch.setattr(
        "asistente_agentico_uao.retrieval.embed_query",
        lambda question: np.array([0.1, 0.2, 0.3]),
    )

    retriever = Retriever(collection=collection, config=config)

    results = retriever.retrieve("pregunta de prueba")

    assert collection.last_n_results == 4
    assert len(results) == 2

    assert results[0].doc_name == "doc-b.md"
    assert results[0].score == 0.95
    assert results[0].is_table is True

    assert results[1].doc_name == "doc-a.md"
    assert results[1].score == 0.80


def test_retrieve_returns_empty_when_collection_is_empty():
    collection = FakeCollection(
        documents=[[]],
        metadatas=[[]],
        distances=[[]],
    )

    config = SimpleNamespace(
        top_k=5,
        min_similarity=0.7,
    )

    retriever = Retriever(collection=collection, config=config)

    assert retriever.retrieve("pregunta") == []


def test_format_context_numbers_chunks():
    chunks = [
        SimpleNamespace(
            doc_name="reglamento.md",
            section="Artículo 10",
            text="Contenido del artículo.",
            score=0.9,
            chunk_index=1,
            is_table=False,
        ),
        SimpleNamespace(
            doc_name="calendario.md",
            section="Cancelaciones",
            text="Fecha de cancelación.",
            score=0.8,
            chunk_index=2,
            is_table=False,
        ),
    ]

    context = format_context(chunks)

    assert "[1] (reglamento.md — Artículo 10): Contenido del artículo." in context
    assert "[2] (calendario.md — Cancelaciones): Fecha de cancelación." in context