"""Ingesta del corpus: chunking de markdown y pipeline markdown → Chroma.

- ``chunk.py``: fronteras duras (encabezados, artículos), empaquetado por
  tokens, overlap intra-sección y tablas atómicas.
- ``pipeline.py``: lógica compartida por el CLI ``scripts/ingest.py`` y el
  control plane gRPC ``IndexAdmin`` (Fase 5).
"""
