"""Pruebas unitarias del pipeline de ingesta (Fase 5).

``ingestion.pipeline`` es la lógica compartida por el CLI (Fase 2) y el
servicio gRPC ``IndexAdmin`` (control plane). Se prueba con dobles de
prueba: sin modelo real de embeddings ni Chroma.
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
    def __init__(self):
        self.upserts: list[dict] = []
        self.delete_calls: list[list] = []

    def count(self):
        return sum(len(u["ids"]) for u in self.upserts)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def get(self, include=None):
        metadatas = [m for u in self.upserts for m in u["metadatas"]]
        ids = [i for u in self.upserts for i in u["ids"]]
        return {"ids": ids, "metadatas": metadatas}

    def delete(self, ids):
        self.delete_calls.append(ids)


@pytest.fixture
def index_dirs(tmp_path, monkeypatch):
    """Documentos de prueba: 2 markdown + 1 PDF con stem coincidente."""
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
    (md_dir / "Doc-Prueba.md").write_text(contenido, encoding="utf-8")
    (md_dir / "Otro-Doc.md").write_text(
        "# Otro\n\n**ARTÍCULO 1º:** Contenido de prueba.\n", encoding="utf-8"
    )
    monkeypatch.setattr(settings, "markdown_dir", md_dir)
    monkeypatch.setattr(settings, "docs_dir", docs_dir)
    return md_dir


def _ingest_fake(collection, **kwargs):
    """Ingesta con modelo/tokenizador falsos (sin descargas)."""
    return ingest_documents(
        model=object(),  # no se usa: embed_passages está mockeado
        seq_len=512,
        count_tokens=lambda text: len(text.split()),
        collection=collection,
        **kwargs,
    )


def test_ingest_indexa_chunks_con_doc_name_del_pdf(index_dirs, monkeypatch):
    monkeypatch.setattr(
        "asistente_agentico_uao.ingestion.pipeline.embed_passages",
        lambda texts: np.zeros((len(texts), 4)),
    )
    collection = FakeCollection()

    summaries = _ingest_fake(collection)

    assert [s.doc_name for s in summaries] == [
        "Doc-Prueba.pdf",
        "Otro-Doc",  # sin PDF con ese stem: cae al stem del markdown
    ]
    assert all(s.n_chunks > 0 for s in summaries)
    # Los chunks se subieron con IDs deterministas (sha256 doc+index).
    assert collection.upserts, "el upsert no se invocó"
    for upsert in collection.upserts:
        for meta, chunk_id_hash in zip(
            upsert["metadatas"], upsert["ids"], strict=True
        ):
            assert chunk_id_hash == chunk_id(
                meta["doc_name"], meta["chunk_index"]
            )


def test_ingest_reporta_progreso_por_documento(index_dirs, monkeypatch):
    monkeypatch.setattr(
        "asistente_agentico_uao.ingestion.pipeline.embed_passages",
        lambda texts: np.zeros((len(texts), 4)),
    )
    recibidos: list[IngestSummary] = []

    _ingest_fake(FakeCollection(), on_summary=recibidos.append)

    assert len(recibidos) == 2  # un resumen por documento (streaming gRPC)


def test_ingest_filtra_por_file_match(index_dirs, monkeypatch):
    monkeypatch.setattr(
        "asistente_agentico_uao.ingestion.pipeline.embed_passages",
        lambda texts: np.zeros((len(texts), 4)),
    )
    collection = FakeCollection()

    summaries = _ingest_fake(collection, file_match="prueba")

    assert [s.doc_name for s in summaries] == ["Doc-Prueba.pdf"]


def test_ingest_sin_markdown_raise_valueerror(index_dirs):
    with pytest.raises(ValueError, match="No hay markdown"):
        _ingest_fake(FakeCollection(), file_match="inexistente")


def test_validate_chunk_budget_excede_ventana():
    with pytest.raises(ValueError, match="excede"):
        validate_chunk_budget(100)  # chunk_max_tokens default 450 >> 96


def test_prune_borra_docs_ausentes(index_dirs):
    collection = FakeCollection()
    collection.upsert(
        ids=["1", "2"],
        documents=["a", "b"],
        metadatas=[
            {"doc_name": "Doc-Prueba.pdf", "chunk_index": 0},
            {"doc_name": "Borrado.pdf", "chunk_index": 0},
        ],
        embeddings=[[0.0], [0.0]],
    )

    removed = prune_index(collection)

    assert removed == 1  # solo Borrado.pdf está ausente del markdown_dir
    assert collection.delete_calls == [["2"]]


def test_markdown_paths_ordenados_y_filtrados(index_dirs):
    assert [p.name for p in markdown_paths()] == [
        "Doc-Prueba.md",
        "Otro-Doc.md",
    ]
    assert [p.name for p in markdown_paths("otro")] == ["Otro-Doc.md"]
