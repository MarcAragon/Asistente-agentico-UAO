"""Pruebas unitarias del pipeline de ingesta (Fase 5).

``ingestion.pipeline`` contiene la lógica compartida entre el CLI de ingesta
y el servicio gRPC ``IndexAdmin``. Estas pruebas validan el procesamiento de
documentos, generación de chunks, indexación en Chroma y limpieza del índice.

Se utilizan dobles de prueba para evitar depender de modelos reales de
embeddings o una base de datos vectorial Chroma durante la ejecución.
"""

import numpy as np
import pytest

from asistente_agentico_uao.config import settings
from asistente_agentico_uao.ingestion.pipeline import (
    IngestSummary,
    ingest_documents,
    markdown_paths,
    prune_index,
    validate_chunk_budget,
)
from asistente_agentico_uao.vectorstore import chunk_id


class FakeCollection:
    """Colección falsa que simula operaciones básicas de ChromaDB."""

    def __init__(self):
        """Inicializa almacenamiento interno para pruebas."""
        self.upserts: list[dict] = []
        self.delete_calls: list[list] = []

    def count(self):
        """Retorna la cantidad total de documentos indexados."""
        return sum(len(u["ids"]) for u in self.upserts)

    def upsert(self, **kwargs):
        """Simula la inserción o actualización de documentos."""
        self.upserts.append(kwargs)

    def get(self, include=None):
        """Simula la recuperación de documentos almacenados."""
        metadatas = [m for u in self.upserts for m in u["metadatas"]]
        ids = [i for u in self.upserts for i in u["ids"]]

        return {
            "ids": ids,
            "metadatas": metadatas,
        }

    def delete(self, ids):
        """Simula eliminación de documentos del índice."""
        self.delete_calls.append(ids)


@pytest.fixture
def index_dirs(tmp_path, monkeypatch):
    """Crea documentos temporales para validar la ingesta.

    Genera dos archivos Markdown y un PDF asociado para comprobar la relación
    entre documentos originales y archivos procesados.
    """

    md_dir = tmp_path / "md"
    docs_dir = tmp_path / "docs"

    md_dir.mkdir()
    docs_dir.mkdir()

    (docs_dir / "Doc-Prueba.pdf").write_bytes(b"%PDF-1.4 fake")

    contenido = (
        "# Reglamento de prueba\n\n"
        "**ARTÍCULO 1º:** El estudiante podrá cancelar asignaturas "
        "hasta la fecha establecida en el calendario académico.\n\n"
        "**ARTÍCULO 2º:** La prueba académica se activa por repitencia.\n"
    )

    (md_dir / "Doc-Prueba.md").write_text(
        contenido,
        encoding="utf-8",
    )

    (md_dir / "Otro-Doc.md").write_text(
        "# Otro\n\n**ARTÍCULO 1º:** Contenido de prueba.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(settings, "markdown_dir", md_dir)
    monkeypatch.setattr(settings, "docs_dir", docs_dir)

    return md_dir


def _ingest_fake(collection, **kwargs):
    """Ejecuta la ingesta utilizando modelos simulados.

    Evita descargar modelos reales de embeddings y permite probar únicamente
    la lógica del pipeline.
    """

    return ingest_documents(
        model=object(),
        seq_len=512,
        count_tokens=lambda text: len(text.split()),
        collection=collection,
        **kwargs,
    )


def test_ingest_indexa_chunks_con_doc_name_del_pdf(index_dirs, monkeypatch):
    """Verifica que la ingesta genere chunks asociados al documento correcto."""

    monkeypatch.setattr(
        "asistente_agentico_uao.ingestion.pipeline.embed_passages",
        lambda texts: np.zeros((len(texts), 4)),
    )

    collection = FakeCollection()

    summaries = _ingest_fake(collection)

    assert [s.doc_name for s in summaries] == [
        "Doc-Prueba.pdf",
        "Otro-Doc",
    ]

    assert all(s.n_chunks > 0 for s in summaries)

    assert collection.upserts, "el upsert no se invocó"

    for upsert in collection.upserts:
        for meta, chunk_id_hash in zip(
            upsert["metadatas"],
            upsert["ids"],
            strict=True,
        ):
            assert chunk_id_hash == chunk_id(
                meta["doc_name"],
                meta["chunk_index"],
            )


def test_ingest_reporta_progreso_por_documento(index_dirs, monkeypatch):
    """Verifica que la ingesta reporte progreso por cada documento procesado."""

    monkeypatch.setattr(
        "asistente_agentico_uao.ingestion.pipeline.embed_passages",
        lambda texts: np.zeros((len(texts), 4)),
    )

    recibidos: list[IngestSummary] = []

    _ingest_fake(
        FakeCollection(),
        on_summary=recibidos.append,
    )

    assert len(recibidos) == 2


def test_ingest_filtra_por_file_match(index_dirs, monkeypatch):
    """Verifica que la ingesta permita procesar archivos específicos."""

    monkeypatch.setattr(
        "asistente_agentico_uao.ingestion.pipeline.embed_passages",
        lambda texts: np.zeros((len(texts), 4)),
    )

    collection = FakeCollection()

    summaries = _ingest_fake(
        collection,
        file_match="prueba",
    )

    assert [s.doc_name for s in summaries] == [
        "Doc-Prueba.pdf"
    ]


def test_ingest_sin_markdown_raise_valueerror(index_dirs):
    """Verifica que falle correctamente cuando no existen documentos."""

    with pytest.raises(ValueError, match="No hay markdown"):
        _ingest_fake(
            FakeCollection(),
            file_match="inexistente",
        )


def test_validate_chunk_budget_excede_ventana():
    """Verifica que se detecten configuraciones inválidas de chunks."""

    with pytest.raises(ValueError, match="excede"):
        validate_chunk_budget(100)


def test_prune_borra_docs_ausentes(index_dirs):
    """Verifica que prune elimine documentos que ya no existen."""

    collection = FakeCollection()

    collection.upsert(
        ids=["1", "2"],
        documents=["a", "b"],
        metadatas=[
            {
                "doc_name": "Doc-Prueba.pdf",
                "chunk_index": 0,
            },
            {
                "doc_name": "Borrado.pdf",
                "chunk_index": 0,
            },
        ],
        embeddings=[
            [0.0],
            [0.0],
        ],
    )

    removed = prune_index(collection)

    assert removed == 1
    assert collection.delete_calls == [["2"]]


def test_markdown_paths_ordenados_y_filtrados(index_dirs):
    """Verifica que la búsqueda de Markdown filtre y ordene correctamente."""

    assert [p.name for p in markdown_paths()] == [
        "Doc-Prueba.md",
        "Otro-Doc.md",
    ]

    assert [p.name for p in markdown_paths("otro")] == [
        "Otro-Doc.md"
    ]