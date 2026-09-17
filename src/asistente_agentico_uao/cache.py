"""Cache semantico en Redis (Fase 7, tarea 7.5).

Reutiliza la respuesta de una pregunta ya contestada cuando llega otra
pregunta semanticamente muy parecida, evitando repetir la recuperacion y
la llamada al LLM. Se apoya en el modelo de embeddings que ya esta cargado
en memoria (``core/embeddings.py``, E5): cada pregunta nueva se compara por
similitud coseno (producto punto, los vectores ya vienen normalizados)
contra las preguntas cacheadas, y si la mas parecida supera
``settings.cache_similarity`` se reutiliza su respuesta.

Degradable a proposito: si Redis no esta disponible, o si
``UAO_RAG__CACHE_ENABLED=0``, ``buscar()`` siempre devuelve ``None`` (miss)
y ``guardar()`` no hace nada. La aplicacion sigue funcionando igual, solo
sin el atajo de velocidad.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import redis

from ..core.config import Settings, settings
from ..core.embeddings import embed_query
from .chain import RagAnswer, Source

PREFIJO = "uao_rag:cache"


class SemanticCache:
    """Cache semantico respaldado en Redis (inactivo si no hay conexion)."""

    def __init__(self, config: Settings | None = None, client=None):
        self.config = config or settings
        self._client = client
        self._intento_conexion = False

    def _get_client(self):
        """Cliente Redis perezoso; ``None`` si esta desactivado o no conecta."""
        if not self.config.cache_enabled:
            return None
        if self._client is not None:
            return self._client
        if self._intento_conexion:
            return None
        self._intento_conexion = True
        try:
            cliente = redis.Redis.from_url(
                self.config.redis_url, decode_responses=True, socket_timeout=1
            )
            cliente.ping()
        except redis.RedisError:
            return None
        self._client = cliente
        return cliente

    def buscar(self, pregunta: str) -> RagAnswer | None:
        cliente = self._get_client()
        if cliente is None:
            return None
        try:
            vector = embed_query(pregunta)
            mejor_hash = None
            mejor_similitud = -1.0
            for llave in cliente.keys(f"{PREFIJO}:embedding:*"):
                candidato = np.array(json.loads(cliente.get(llave)), dtype=np.float32)
                similitud = float(np.dot(vector, candidato))
                if similitud > mejor_similitud:
                    mejor_similitud = similitud
                    mejor_hash = llave.rsplit(":", 1)[-1]

            if mejor_hash is None or mejor_similitud < self.config.cache_similarity:
                cliente.incr(f"{PREFIJO}:stats:misses")
                return None

            datos = cliente.get(f"{PREFIJO}:answer:{mejor_hash}")
            if datos is None:
                cliente.incr(f"{PREFIJO}:stats:misses")
                return None

            cliente.incr(f"{PREFIJO}:stats:hits")
            return _deserializar(datos)
        except redis.RedisError:
            self._client = None  # fuerza reconexión en el próximo intento
            return None

    def guardar(self, pregunta: str, respuesta: RagAnswer) -> None:
        """Guarda la respuesta y su embedding, con vigencia de cache_ttl_seconds."""
        cliente = self._get_client()
        if cliente is None:
            return
        try:
            vector = embed_query(pregunta)
            hash_ = _hash_pregunta(pregunta)
            ttl = self.config.cache_ttl_seconds
            cliente.set(f"{PREFIJO}:answer:{hash_}", _serializar(respuesta), ex=ttl)
            cliente.set(
                f"{PREFIJO}:embedding:{hash_}", json.dumps(vector.tolist()), ex=ttl
            )
            cliente.set(f"{PREFIJO}:question:{hash_}", pregunta, ex=ttl)
        except redis.RedisError:
            self._client = None

    def invalidar_todo(self) -> None:
        """Borra todo lo cacheado (se llama tras un rebuild o prune del indice)."""
        cliente = self._get_client()
        if cliente is None:
            return
        try:
            llaves = [
                *cliente.keys(f"{PREFIJO}:answer:*"),
                *cliente.keys(f"{PREFIJO}:embedding:*"),
                *cliente.keys(f"{PREFIJO}:question:*"),
            ]
            if llaves:
                cliente.delete(*llaves)
        except redis.RedisError:
            self._client = None

    def estadisticas(self) -> dict[str, int | float]:
        """Contadores hit/miss/ratio (para exponer en /health)."""
        cliente = self._get_client()
        if cliente is None:
            return {"hits": 0, "misses": 0, "hit_ratio": 0.0}
        try:
            hits = int(cliente.get(f"{PREFIJO}:stats:hits") or 0)
            misses = int(cliente.get(f"{PREFIJO}:stats:misses") or 0)
            total = hits + misses
            return {
                "hits": hits,
                "misses": misses,
                "hit_ratio": round(hits / total, 3) if total else 0.0,
            }
        except redis.RedisError:
            self._client = None
            return {"hits": 0, "misses": 0, "hit_ratio": 0.0}


def _hash_pregunta(pregunta: str) -> str:
    normalizada = " ".join(pregunta.strip().lower().split())
    return hashlib.sha256(normalizada.encode("utf-8")).hexdigest()


def _serializar(respuesta: RagAnswer) -> str:
    return json.dumps(asdict(respuesta))


def _deserializar(datos: str) -> RagAnswer:
    bruto = json.loads(datos)
    fuentes = [Source(**f) for f in bruto.get("sources", [])]
    return RagAnswer(
        answer=bruto["answer"],
        sources=fuentes,
        model=bruto.get("model", ""),
        used_fallback=bruto.get("used_fallback", False),
    )
