"""LLM de síntesis vía Cerebras con rotación de claves API (Fase 4).

``ChatCerebras`` (langchain-cerebras) está construido sobre el SDK de OpenAI
apuntando a la API de Cerebras: sus errores llegan como excepciones de
``openai`` con ``status_code`` (``RateLimitError`` 429, 401/403 de
autenticación, 5xx, timeouts de conexión).

``CerebrasLLM`` envuelve ``ChatCerebras`` con dos defensas:

- **Rotación de claves**: si la clave actual alcanza su límite (429/cuota) o
  es inválida (401/403), se pasa inmediatamente a la siguiente de la lista
  (``CEREBRAS_API_KEY`` + ``CEREBRAS_API_KEYS``, separadas por coma). Si solo
  hay una clave, un 429 se reintenta con backoff (suele ser un límite por
  segundo) y un 401/403 se relanza (reintentar no lo corrige).
- **Backoff exponencial** ante errores transitorios (timeout, 5xx) y ante
  429 cuando no hay más claves a las que rotar.

Aquí vive también ``NO_INFO_MESSAGE``: el mensaje exacto de no-información
que el prompt de la cadena (``rag/chain.py``) ordena copiar cuando el contexto
no alcanza. Centralizado para que prompt, cadena y post-proceso usen la
misma cadena literal.

⚠ Hallazgo de la Fase 3: las similitudes coseno de E5 son altas siempre
(≈0.81 incluso fuera de dominio), así que ``min_similarity`` NO discrimina
dominio. La defensa principal contra alucinaciones es este mensaje y las
reglas de solo-contexto del prompt, no el umbral de recuperación.
"""

from __future__ import annotations

import re
import time

from .config import Settings, settings

# Mensaje exacto de no-información: única respuesta permitida fuera del
# contexto. El prompt exige copiarlo literal; chain.py lo detecta para
# devolver sources=[] y used_fallback=True.
NO_INFO_MESSAGE = (
    "No tengo información suficiente en la normativa UAO para responder esa pregunta."
)

# Códigos HTTP transitorios del servicio (timeout de upstream, errores de
# servidor). El 429 se clasifica además como error de clave (rotación).
_RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504, 529}
_BASE_DELAY_S = 1.0

_llm = None  # singleton perezoso (carga única por proceso)


def _status_of(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _is_rate_limit(exc: BaseException) -> bool:
    """429 / cuota agotada: la clave llegó a su límite (rotar si se puede)."""
    if _status_of(exc) == 429:
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return "rate limit" in text or "quota" in text


def _is_auth_error(exc: BaseException) -> bool:
    """401/403: clave inválida o sin permisos (rotar si se puede)."""
    return _status_of(exc) in {401, 403}


def _is_retryable(exc: BaseException) -> bool:
    """True si la excepción es transitoria (429/timeout/conexión/5xx)."""
    if _status_of(exc) in _RETRYABLE_STATUSES:
        return True
    return type(exc).__name__ in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "TimeoutError",
        "ConnectionError",
    }


def collect_api_keys(config: Settings | None = None) -> list[str]:
    """Claves Cerebras disponibles, sin duplicados ni vacías.

    La principal es ``CEREBRAS_API_KEY``; ``CEREBRAS_API_KEYS`` aporta
    adicionales separadas por coma, punto y coma o espacio (para rotar ante
    límites de cuota por clave).
    """
    cfg = config or settings
    candidates = [
        cfg.cerebras_api_key.strip(),
        *(k.strip() for k in re.split(r"[,;\s]+", cfg.cerebras_api_keys)),
    ]
    seen: set[str] = set()
    keys: list[str] = []
    for key in candidates:
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


class CerebrasLLM:
    """ChatCerebras con rotación de claves API ante límites de cuota.

    ``.invoke(messages)`` es la interfaz que consume la cadena (LCEL vía
    ``RunnableLambda``). Política ante errores:

    1. Error de clave (429/cuota o 401/403) con más claves disponibles →
       rota inmediatamente a la siguiente (sin backoff) y reintenta.
    2. 429 sin más claves → backoff exponencial con la clave actual (los
       rate limits por segundo suelen ser transitorios).
    3. 401/403 sin más claves → se relanza (reintentar no lo corrige).
    4. Timeout/conexión/5xx → backoff exponencial con la clave actual.
    """

    def __init__(self, config: Settings | None = None):
        self.settings = config or settings
        self.api_keys = collect_api_keys(self.settings)
        self._index = 0
        self._client = None

    @property
    def client(self):
        """ChatCerebras de la clave vigente (se recrea al rotar)."""
        if self._client is None:
            from langchain_cerebras import ChatCerebras

            self._client = ChatCerebras(
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
                max_retries=0,  # el reintento lo gestiona esta clase
                timeout=60.0,
                # qwen-3.8 razona por defecto y puede agotar max_tokens en
                # thinking (content vacío); para síntesis solo-contexto se
                # desactiva (hallazgo F4).
                disable_reasoning=self.settings.llm_disable_reasoning,
                api_key=self.api_keys[self._index],
            )
        return self._client

    def rotate(self) -> bool:
        """Pasa a la siguiente clave; False si no hay más."""
        if self._index + 1 >= len(self.api_keys):
            return False
        self._index += 1
        self._client = None  # el cliente se recrea con la nueva clave
        print(
            f"[llm] límite/cuota en clave #{self._index}; rotando a "
            f"#{self._index + 1}/{len(self.api_keys)}"
        )
        return True

    def invoke(self, messages):
        """Invoca al LLM con rotación de claves y reintentos (ver clase)."""
        if not self.api_keys:
            raise RuntimeError(
                "CEREBRAS_API_KEY no configurada: define al menos una clave "
                "en .env (o CEREBRAS_API_KEYS para rotación)."
            )
        attempts = max(1, self.settings.llm_max_retries)  # por clave
        tries = 0  # intentos de backoff con la clave actual
        visited = 1  # claves distintas con las que se ha intentado
        while True:
            try:
                return self.client.invoke(messages)
            except Exception as exc:
                key_error = _is_rate_limit(exc) or _is_auth_error(exc)
                if key_error and visited < len(self.api_keys):
                    visited += 1
                    self.rotate()
                    tries = 0
                    continue
                if _is_auth_error(exc):
                    raise  # sin más claves: reintentar no lo corrige
                if not _is_retryable(exc):
                    raise
                tries += 1
                if tries >= attempts:
                    raise
                delay = _BASE_DELAY_S * (2 ** (tries - 1))
                print(
                    f"[llm] intento {tries}/{attempts} falló "
                    f"({type(exc).__name__}); reintentando en {delay:.0f}s"
                )
                time.sleep(delay)


def get_llm() -> CerebrasLLM:
    """Instancia (única por proceso) de CerebrasLLM con los Settings."""
    global _llm
    if _llm is None:
        _llm = CerebrasLLM()
    return _llm
