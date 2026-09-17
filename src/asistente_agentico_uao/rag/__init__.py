"""Dominio RAG: recuperación, síntesis, caché y estado de servicio.

Pipeline en cadena: ``retrieval`` (top-k con umbral) → ``chain``
(prompt + LLM + citas verificables) → ``cache`` (atajo semántico en Redis)
→ ``service`` (``AppState`` compartido por REST y gRPC).

Regla de dependencias: solo importa de ``core`` (config, embeddings,
vectorstore, llm) y ``ingestion``; nunca de ``api``, ``grpc_impl`` ni
``frontend``. Los símbolos no se re-exportan aquí para no forzar la carga
anticipada de Chroma/Redis/langchain al importar el paquete.
"""
