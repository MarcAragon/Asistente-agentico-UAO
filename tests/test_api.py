"""Pruebas unitarias de la API REST pública (Fase 5).

Se valida el endpoint principal ``POST /ask`` utilizando una cadena RAG
simulada, evitando consumo de tokens o llamadas externas.

Las pruebas cubren:
- Flujo exitoso de consulta.
- Validaciones de entrada.
- Manejo de ausencia de API key.
- Errores internos del servicio.
- Endpoints de monitoreo ``/health`` y ``/documents``.
- Compartición del estado mediante ``AppState``.
- Validaciones adicionales del contrato HTTP.
"""


import pytest
from fastapi.testclient import TestClient

from asistente_agentico_uao.api.main import create_app
from asistente_agentico_uao.config import Settings
from asistente_agentico_uao.llm import NO_INFO_MESSAGE
from asistente_agentico_uao.retrieval import RetrievedChunk
from asistente_agentico_uao.service import AppState


def make_chunk(
    doc: str,
    section: str,
    text: str,
) -> RetrievedChunk:
    """Crea un fragmento recuperado simulado."""

    return RetrievedChunk(
        doc_name=doc,
        section=section,
        text=text,
        score=0.9,
        chunk_index=0,
    )


class FakeCollection:
    """Colección vectorial simulada para pruebas."""

    def __init__(
        self,
        count: int,
        doc_names: list[str],
    ):
        self._count = count
        self._doc_names = doc_names

    def count(self):
        """Retorna cantidad de chunks."""

        return self._count

    def get(self, include=None):
        """Retorna documentos simulados."""

        return {
            "metadatas": [
                {
                    "doc_name": doc,
                    "chunk_index": index,
                }
                for doc in self._doc_names
                for index in range(2)
            ]
        }


class FakeRetriever:
    """Retriever falso para evitar ChromaDB."""

    def __init__(
        self,
        chunks,
        collection=None,
    ):
        self.chunks = chunks
        self.collection = collection or FakeCollection(0, [])
        self.resets = 0
        self.questions: list[str] = []

    def retrieve(self, question):
        """Guarda preguntas recibidas."""

        self.questions.append(question)
        return self.chunks

    def reset(self):
        """Simula reinicio."""

        self.resets += 1


class FakeLLM:
    """Modelo falso para pruebas."""

    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, prompt_value):
        """Retorna respuesta fija."""

        return self.reply


class ExplodingRetriever(FakeRetriever):
    """Retriever que genera errores."""

    def retrieve(self, question):
        raise RuntimeError(
            "índice no disponible"
        )


def make_state(
    retriever,
    llm=None,
    **config_kwargs,
) -> AppState:
    """Construye estado simulado."""

    config = Settings(
        **{
            "cerebras_api_key": "k1",
            "grpc_enabled": False,
            "embedding_device": "cpu",
            **config_kwargs,
        }
    )

    return AppState(
        config=config,
        retriever=retriever,
        llm=llm or FakeLLM("ok"),
    )


def make_client(state: AppState) -> TestClient:
    """Crea cliente FastAPI."""

    app = create_app(
        state=state,
    )

    return TestClient(app)


@pytest.fixture
def client():
    """Cliente API con datos simulados."""

    retriever = FakeRetriever(
        chunks=[
            make_chunk(
                "Res-CA-6744.md",
                "Artículo 70º-2",
                "texto real.",
            )
        ],
        collection=FakeCollection(
            1284,
            [
                "Res-CA-6744.md",
                "Reso-CS-666.md",
            ],
        ),
    )

    llm = FakeLLM(
        "Respuesta con cita (Res-CA-6744.md, Artículo 70º-2)."
    )

    with make_client(
        make_state(
            retriever,
            llm,
        )
    ) as test_client:
        yield test_client, retriever


def test_ask_flujo_feliz_con_fuentes(client):
    """Verifica consulta exitosa."""

    c, retriever = client

    response = c.post(
        "/ask",
        json={
            "question": "¿cuántas repitencias?"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["model"] == "qwen-3.8-27b"
    assert body["used_fallback"] is False
    assert retriever.questions == [
        "¿cuántas repitencias?"
    ]


def test_ask_pregunta_vacia_o_blancos_422(client):
    """Verifica rechazo de preguntas vacías."""

    c, _ = client

    for question in ["", "   "]:
        response = c.post(
            "/ask",
            json={"question": question},
        )

        assert response.status_code == 422


def test_ask_pregunta_demasiado_larga_422(client):
    """Verifica límite máximo de caracteres."""

    c, _ = client

    response = c.post(
        "/ask",
        json={"question": "a" * 501},
    )

    assert response.status_code == 422


def test_ask_sin_contexto_responde_no_info_200(client):
    """Verifica respuesta sin documentos."""

    c, retriever = client

    retriever.chunks = []

    response = c.post(
        "/ask",
        json={"question": "¿pregunta?"},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["answer"] == NO_INFO_MESSAGE


def test_ask_sin_api_key_503():
    """Verifica ausencia de credencial."""

    state = make_state(
        FakeRetriever(chunks=[]),
        cerebras_api_key="",
        cerebras_api_keys="",
    )

    with make_client(state) as c:
        response = c.post(
            "/ask",
            json={"question": "¿pregunta?"},
        )

    assert response.status_code == 503


def test_ask_fallo_interno_500():
    """Verifica error interno."""

    state = make_state(
        ExplodingRetriever([])
    )

    with make_client(state) as c:
        response = c.post(
            "/ask",
            json={"question": "¿pregunta?"},
        )

    assert response.status_code == 500


def test_health_reporta_estado_correcto(client):
    """Verifica endpoint health."""

    c, _ = client

    response = c.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_documents_agrega_chunks_por_documento(client):
    """Verifica listado de documentos."""

    c, _ = client

    response = c.get("/documents")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_app_state_compartido_expone_reset(client):
    """Verifica reinicio del retriever."""

    c, retriever = client

    state = c.app.state.rag

    state.on_index_rebuilt()

    assert retriever.resets == 1


# ============================
# NUEVAS PRUEBAS DE COBERTURA
# ============================


def test_ask_rechaza_request_sin_question(client):
    """Verifica que el endpoint rechace payload incompleto."""

    c, _ = client

    response = c.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


def test_ask_rechaza_question_no_string(client):
    """Verifica validación de tipo de pregunta."""

    c, _ = client

    response = c.post(
        "/ask",
        json={
            "question": 123
        },
    )

    assert response.status_code == 422


def test_health_mantiene_device_cpu(client):
    """Verifica dispositivo configurado."""

    c, _ = client

    response = c.get("/health")

    assert response.json()["device"] == "cpu"


def test_documents_retorna_lista(client):
    """Verifica que documents siempre retorne lista."""

    c, _ = client

    response = c.get("/documents")

    assert isinstance(
        response.json(),
        list,
    )


def test_app_state_usa_settings(client):
    """Verifica configuración interna del estado."""

    c, _ = client

    state = c.app.state.rag

    assert isinstance(
        state.config,
        Settings,
    )