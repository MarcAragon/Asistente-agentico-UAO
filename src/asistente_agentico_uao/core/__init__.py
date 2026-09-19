"""Infraestructura y adaptadores externos del asistente.

Concentra lo que NO conoce el dominio RAG: configuración central
(``config``), el modelo de embeddings E5 (``embeddings``), la persistencia
vectorial ChromaDB (``vectorstore``) y el cliente del LLM Cerebras
(``llm``, con rotación de claves y ``NO_INFO_MESSAGE``).

Regla de dependencias: este paquete es la capa base; solo importa de
``ingestion`` (tipos de chunk) y nunca de ``rag``, ``api``, ``grpc_impl``
ni ``frontend``. Los símbolos no se re-exportan aquí a propósito: importar
``core`` no debe arrastrar cargas pesadas (Chroma, Cerebras) de forma
anticipada.
"""
