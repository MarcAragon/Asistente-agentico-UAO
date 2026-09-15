"""Pruebas unitarias de la Fase 4: ``llm.py`` (CerebrasLLM).

Cubre la integración del LLM sin llamadas reales a la API de Cerebras:

- ``collect_api_keys``: combinación de ``CEREBRAS_API_KEY`` +
  ``CEREBRAS_API_KEYS``, sin duplicados ni vacíos.
- ``CerebrasLLM.invoke``: rotación de claves ante 429/cuota o 401/403,
  backoff exponencial ante errores transitorios y relanzamiento de errores
  no recuperables.

``ChatCerebras`` se reemplaza por un stub inyectado en ``sys.modules`` para
no importar la librería real ni consumir tokens.
"""

import sys
import types
from types import SimpleNamespace
from typing import ClassVar

import pytest

from asistente_agentico_uao.config import Settings
from asistente_agentico_uao.llm import CerebrasLLM, collect_api_keys


class FakeHTTPError(Exception):
    """Imita los errores del SDK de OpenAI: exponen ``status_code``."""

    def __init__(self, status_code: int | None = None, message: str = "error"):
        super().__init__(message)
        self.status_code = status_code


class FakeChatCerebras:
    """Stub de ``ChatCerebras``: registra la clave usada y su comportamiento.

    ``behavior`` mapea api_key -> handler(messages); si no hay handler para
    la clave, responde OK. ``created`` acumula los clientes construidos para
    verificar que la rotación recrea el cliente con la nueva clave.
    """

    behavior: ClassVar[dict] = {}
    created: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.api_key = kwargs["api_key"]
        self.kwargs = kwargs
        FakeChatCerebras.created.append(self)

    def invoke(self, messages):
        handler = FakeChatCerebras.behavior.get(self.api_key)
        if handler is not None:
            return handler(messages)
        return SimpleNamespace(content=f"ok desde {self.api_key}")


def raise_error(exc: Exception):
    def _handler(messages):
        raise exc

    return _handler


@pytest.fixture
def llm_env(monkeypatch):
    """Stub de langchain_cerebras + ``time.sleep`` grabado (sin esperas)."""
    FakeChatCerebras.behavior = {}
    FakeChatCerebras.created = []
    sleep_calls: list[float] = []

    stub = types.ModuleType("langchain_cerebras")
    stub.ChatCerebras = FakeChatCerebras
    monkeypatch.setitem(sys.modules, "langchain_cerebras", stub)
    monkeypatch.setattr(
        "asistente_agentico_uao.llm.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )
    return SimpleNamespace(cerebras=FakeChatCerebras, sleeps=sleep_calls)


def make_config(
    api_key: str = "",
    api_keys: str = "",
    max_retries: int = 3,
) -> Settings:
    return Settings(
        cerebras_api_key=api_key, cerebras_api_keys=api_keys, llm_max_retries=max_retries
    )


# --- collect_api_keys -------------------------------------------------------


def test_collect_api_keys_combina_y_deduplica():
    """La principal va primero; las extra se parten por coma/;/espacio."""
    config = make_config(api_key="k1", api_keys="k1, k2;k3 k2")

    keys = collect_api_keys(config)

    assert keys == ["k1", "k2", "k3"]


# --- CerebrasLLM.invoke: rotación y reintentos ------------------------------


def test_sin_claves_mensaje_claro(llm_env):
    """Sin claves configuradas se lanza un error accionable."""
    llm = CerebrasLLM(make_config())

    with pytest.raises(RuntimeError, match="CEREBRAS_API_KEY"):
        llm.invoke("pregunta")

    assert llm_env.cerebras.created == []  # ni se intentó construir cliente


def test_429_rota_a_segunda_clave(llm_env):
    """Cuota agotada en k1: rota a k2 y responde con el nuevo cliente."""
    llm_env.cerebras.behavior["k1"] = raise_error(
        FakeHTTPError(429, "rate limit exceeded")
    )
    llm = CerebrasLLM(make_config(api_key="k1", api_keys="k2"))

    result = llm.invoke("pregunta")

    assert result.content == "ok desde k2"
    assert [c.api_key for c in llm_env.cerebras.created] == ["k1", "k2"]
    assert llm_env.sleeps == []  # la rotación es inmediata, sin backoff


def test_401_rota_y_la_segunda_funciona(llm_env):
    """Clave inválida (401) también dispara rotación inmediata."""
    llm_env.cerebras.behavior["k1"] = raise_error(FakeHTTPError(401))
    llm = CerebrasLLM(make_config(api_key="k1", api_keys="k2"))

    result = llm.invoke("pregunta")

    assert result.content == "ok desde k2"
    assert len(llm_env.cerebras.created) == 2


def test_401_con_una_sola_clave_se_relaza(llm_env):
    """401 sin más claves: se relanza; reintentar no lo corrige."""
    llm_env.cerebras.behavior["k1"] = raise_error(FakeHTTPError(401))
    llm = CerebrasLLM(make_config(api_key="k1", max_retries=3))

    with pytest.raises(FakeHTTPError):
        llm.invoke("pregunta")

    assert len(llm_env.cerebras.created) == 1  # no rotó ni recreó cliente
    assert llm_env.sleeps == []  # ni un solo backoff


def test_429_con_una_sola_clave_reintenta_con_backoff(llm_env):
    """429 sin más claves: backoff con la misma clave y éxito al 2º intento."""
    intentos = {"n": 0}

    def handler(messages):
        intentos["n"] += 1
        if intentos["n"] == 1:
            raise FakeHTTPError(429, "rate limit exceeded")
        return SimpleNamespace(content="ok")

    llm_env.cerebras.behavior["k1"] = handler
    llm = CerebrasLLM(make_config(api_key="k1", max_retries=3))

    result = llm.invoke("pregunta")

    assert result.content == "ok"
    assert len(llm_env.cerebras.created) == 1  # nunca rotó
    assert llm_env.sleeps == [1.0]  # primer backoff: 1s


def test_timeout_reintenta_misma_clave_sin_rotar(llm_env):
    """Error transitorio (408): backoff y reintento con la clave vigente."""
    intentos = {"n": 0}

    def handler(messages):
        intentos["n"] += 1
        if intentos["n"] == 1:
            raise FakeHTTPError(408, "request timeout")
        return SimpleNamespace(content="ok")

    llm_env.cerebras.behavior["k1"] = handler
    llm = CerebrasLLM(make_config(api_key="k1", max_retries=3))

    result = llm.invoke("pregunta")

    assert result.content == "ok"
    assert len(llm_env.cerebras.created) == 1
    assert llm_env.sleeps == [1.0]


def test_cuota_en_todas_las_claves_agota_y_relaza(llm_env):
    """429 en todas las claves: rota, agota los reintentos y se relanza."""
    llm_env.cerebras.behavior["k1"] = raise_error(FakeHTTPError(429))
    llm_env.cerebras.behavior["k2"] = raise_error(FakeHTTPError(429))
    llm = CerebrasLLM(make_config(api_key="k1", api_keys="k2", max_retries=3))

    with pytest.raises(FakeHTTPError):
        llm.invoke("pregunta")

    assert [c.api_key for c in llm_env.cerebras.created] == ["k1", "k2"]
    assert llm_env.sleeps == [1.0, 2.0]  # backoff exponencial creciente


def test_error_no_transitorio_se_relaza_sin_rotar(llm_env):
    """Un 400 (no transitorio, no de clave) se relanza de inmediato."""
    llm_env.cerebras.behavior["k1"] = raise_error(FakeHTTPError(400, "bad request"))
    llm = CerebrasLLM(make_config(api_key="k1", api_keys="k2"))

    with pytest.raises(FakeHTTPError):
        llm.invoke("pregunta")

    assert len(llm_env.cerebras.created) == 1  # no rotó a k2
    assert llm_env.sleeps == []


def test_collect_api_keys_vacio():
    """Sin claves configuradas la lista queda vacía."""
    assert collect_api_keys(make_config()) == []
