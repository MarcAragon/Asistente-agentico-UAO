"""Pruebas unitarias de la API REST pública (Fase 5).

``POST /ask`` con cadena RAG mockeada (sin gastar tokens), validaciones
422, 503 sin clave de API, 500 con fallo interno, y los endpoints de
monitoreo ``/health`` y ``/documents``.
"""

import pytest
from fastapi.testclient import TestClient

from asistente_agentico_uao.api.main import create_app
from asistente_agentico_uao.config import Settings
from asistente_agentico_uao.llm import NO_INFO_MESSAGE
from asistente_agentico_uao.retrieval import RetrievedChunk
from asistente_agentico_uao.service import AppState


def make_chunk(doc: str, section: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        doc_name=doc, section=section, text=text, score=0.9, chunk_index=0
    )


class FakeCollection:
    def __init__(self, count: int, doc_names: list[str]):
        self._count = count
        self._doc_names = doc_names

    def count(self):
        return self._count

    def get(self, include=None):
        return {
            "metadatas": [
                {"doc_name": d, "chunk_index": i}
                for d in self._doc_names
                for i in range(2)
            ]
        }


class FakeRetriever:
    def __init__(self, chunks, collection=None):
        self.chunks = chunks
        self.collection = collection or FakeCollection(0, [])
        self.resets = 0
        self.questions: list[str] = []

    def retrieve(self, question):
        self.questions.append(question)
        return self.chunks

    def reset(self):
        self.resets += 1


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, prompt_value):
        return self.reply


class ExplodingRetriever(FakeRetriever):
    def retrieve(self, question):
        raise RuntimeError("índice no disponible")


def make_state(retriever, llm=None, **config_kwargs) -> AppState:
    config = Settings(
        **{
            "cerebras_api_key": "k1",
            "grpc_enabled": False,  # los tests no levantan el gRPC embebido
            "embedding_device": "cpu",
            **config_kwargs,
        }
    )
    return AppState(config=config, retriever=retriever, llm=llm or FakeLLM("ok"))


def make_client(state: AppState) -> TestClient:
    app = create_app(state=state)  # el lifespan inyecta el state sin construirlo
    return TestClient(app)


@pytest.fixture
def client():
    retriever = FakeRetriever(
        chunks=[make_chunk("Res-CA-6744.md", "Artículo 70º-2", "texto real.")],
        collection=FakeCollection(1284, ["Res-CA-6744.md", "Reso-CS-666.md"]),
    )
    llm = FakeLLM("Respuesta con cita (Res-CA-6744.md, Artículo 70º-2).")
    with make_client(make_state(retriever, llm)) as c:
        yield c, retriever


def test_ask_flujo_feliz_con_fuentes(client):
    c, retriever = client

    response = c.post("/ask", json={"question": "¿cuántas repitencias?"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "qwen-3.8-27b"
    assert body["used_fallback"] is False
    assert len(body["sources"]) == 1
    assert body["sources"][0]["doc_name"] == "Res-CA-6744.md"
    assert body["sources"][0]["excerpt"] == "texto real."
    assert retriever.questions == ["¿cuántas repitencias?"]


def test_ask_pregunta_vacia_o_blancos_422(client):
    c, _ = client

    for question in ["", "   "]:
        response = c.post("/ask", json={"question": question})
        assert response.status_code == 422, question


def test_ask_pregunta_demasiado_larga_422(client):
    c, _ = client

    response = c.post("/ask", json={"question": "a" * 501})

    assert response.status_code == 422


def test_ask_sin_contexto_responde_no_info_200(client):
    c, retriever = client
    retriever.chunks = []

    response = c.post("/ask", json={"question": "¿pregunta?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NO_INFO_MESSAGE
    assert body["sources"] == []
    assert body["used_fallback"] is True


def test_ask_sin_api_key_503():
    state = make_state(
        FakeRetriever(chunks=[]),
        cerebras_api_key="",
        cerebras_api_keys="",
    )
    with make_client(state) as c:
        response = c.post("/ask", json={"question": "¿pregunta?"})

    assert response.status_code == 503
    assert "CEREBRAS_API_KEY" in response.json()["detail"]


def test_ask_fallo_interno_500():
    state = make_state(ExplodingRetriever(chunks=[]))
    with make_client(state) as c:
        response = c.post("/ask", json={"question": "¿pregunta?"})

    assert response.status_code == 500
    assert "Error interno" in response.json()["detail"]


def test_health_reporta_chunks_y_device(client):
    c, _ = client

    response = c.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "index_chunks": 1284, "device": "cpu"}


def test_documents_agrega_chunks_por_documento(client):
    c, _ = client

    response = c.get("/documents")

    assert response.status_code == 200
    assert response.json() == [
        {"doc_name": "Res-CA-6744.md", "chunks": 2},
        {"doc_name": "Reso-CS-666.md", "chunks": 2},
    ]


def test_app_state_compartido_expone_reset(client):
    """El rebuild vía gRPC invalida el Retriever compartido (AppState)."""
    c, retriever = client

    state = c.app.state.rag
    state.on_index_rebuilt()

    assert retriever.resets == 1
    assert isinstance(state.config, Settings)
