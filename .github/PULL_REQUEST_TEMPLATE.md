# 📋 Plantilla de Pull Request — Asistente Agéntico RAG UAO

<!--
Plantilla pre-llenada con el contexto de la PR de APIs de la FASE 5
(REST de consulta y gRPC de administración), rama feat--api_development.
Instrucciones:
- Completa TODAS las secciones. Si una no aplica, escribe "N/A" y justifica.
- El título de la PR debe seguir Convencional Commits:
  feat: | fix: | refactor: | docs: | chore: | test:
  Ej: "feat: API REST de consulta y gRPC de administración del índice (F5)"
- Elimina los comentarios HTML antes de abrir la PR.
-->

## 1. 🧩 Tipo de Pull Request

**Título sugerido:** `feat: API REST de consulta y gRPC de administración del índice (Fase 5)`

Marca **uno** como principal (y otros si esta PR es mixta):

- [x] `feat` — Nueva funcionalidad (pipeline, ingesta, recuperación, API…)
- [ ] `fix` — Corrección de un bug
- [ ] `refactor` — Cambio interno sin alterar comportamiento externo
- [ ] `docs` — Solo documentación
- [ ] `chore` — Infraestructura, dependencias, configuración, CI
- [x] `test` — Adición o corrección exclusiva de pruebas

---

## 2. 🧱 Módulo o Componente Afectado

Selecciona todas las áreas que toca esta PR:

- [ ] **Configuración central** (`src/asistente_agentico_uao/config.py`, `.env.example`, `.gitignore`)
- [ ] **Preprocesamiento de documentos** (`scripts/llama_cloud_parsing.py`, `Data/Documentos_MD/`)
- [ ] **Ingesta & Chunking** (índice de embeddings, Chroma)
- [ ] **Recuperación & Embeddings** (`sentence-transformers`, `chromadb`)
- [ ] **Generación / LLM** (`langchain-cerebras`, modelos de Cerebras)
- [x] **API / Servicio** (`fastapi`, `uvicorn`, `grpc`)
- [x] **Scripts & CLI** (`scripts/ingest_client.py`)
- [x] **Pruebas Unitarias** (`tests/test_api.py`, `tests/test_grpc.py`)
- [ ] **Infraestructura & Empaquetado** (`pyproject.toml`, `uv.lock`)
- [ ] **Documentación** (`README.md`, `plan-trabajo-tecnico.md`)

---

## 3. 📚 Descripción de los Cambios

**Tarea / fase asociada:** Fase 5 del `plan-trabajo-tecnico.md` (APIs: REST de consulta y gRPC de ingesta, tareas 5.1–5.5).

#### 🎯 Motivación y contexto
Tras validar la cadena RAG y la integración con el LLM de Cerebras en la Fase 4, el sistema requería interfaces estandarizadas para su consumo y administración. Se ha implementado una arquitectura de dos planos: un **plano de datos** (REST) optimizado para la consulta del asistente desde el frontend, y un **plano de control** (gRPC) para operaciones administrativas pesadas (ingesta masiva, reconstrucción del índice, limpieza), permitiendo una gestión robusta y desacoplada del conocimiento de la IA.

#### 📝 Resumen de cambios
- **`src/asistente_agentico_uao/api/` (Nuevo módulo REST)**:
  - `main.py`: Aplicación FastAPI con endpoints `/ask` (consulta RAG), `/health` (monitoreo) y `/documents` (trazabilidad). Gestiona el ciclo de vida del servidor gRPC embebido mediante `lifespan`.
  - `schemas.py`: Contratos Pydantic que validan preguntas (1-500 chars) y estructuran las respuestas con citas verificables.
- **`src/asistente_agentico_uao/grpc_impl/` (Nuevo módulo gRPC)**:
  - `server.py`: Servidor `IndexAdmin` que utiliza `grpc.aio`. Soporta modo embebido (compartiendo recursos con REST) o standalone.
  - `servicer.py`: Implementación de los RPCs `Ingest` (con streaming de progreso y métricas de tokens), `PruneIndex` y `IndexStatus`.
  - `stubs/`: Clientes y servidores generados desde el contrato Protobuf.
- **`src/asistente_agentico_uao/service.py`**: Introducción de `AppState` como contenedor inyectable de estado compartido (configuración, retriever, LLM).
- **Pruebas de interfaz**:
  - `tests/test_api.py` (9 pruebas): Cubre flujo feliz con fuentes, validaciones de esquema, manejo de errores 503 (sin API keys) y 500.
  - `tests/test_grpc.py` (6 pruebas): Verifica el streaming de ingesta, control de errores internos y reset de la colección tras rebuild.
#### 🔧 Detalles técnicos relevantes
- **Recursos Compartidos**: REST y gRPC corren en el mismo proceso (si se desea), compartiendo el `AppState`. Esto evita duplicar la carga del modelo de embeddings en VRAM/RAM y permite que un `rebuild` vía gRPC notifique instantáneamente al `Retriever` de la API REST.
- **Ingesta no bloqueante**: La operación `Ingest` ejecuta el pipeline en un hilo worker dedicado, informando el progreso en tiempo real al cliente vía streaming de gRPC.
- **Degradación Grácil**: Si el servidor gRPC no puede iniciar (ej. puerto ocupado), la API REST sigue funcionando para consultas.

---

## 4. ⚠️ Impacto y Compatibilidad

- [x] Esta PR **NO** rompe compatibilidad (cambios retrocompatibles)
- [ ] Esta PR introduce **cambios que rompen** (documentar abajo)

**Si hay breaking changes / migración, documenta:** N/A. Se han añadido nuevas interfaces. El código de la cadena RAG y el LLM se mantiene intacto, ahora consumido a través de `AppState`.

---

## 5. 🔐 Seguridad y Secretos

- [x] Ningún secreto (API keys, contraseñas, tokens) está escrito en el código ni en archivos rastreados
- [x] Los secretos viven únicamente en `.env` (gitignored) o en variables de entorno
- [x] `.env.example` se mantiene vigente
- [x] El servidor gRPC usa `insecure_port` por ser para tráfico interno/backend; el proxy TLS (Fase 7) protegerá la superficie pública.

---

## 6. ✅ Lista de Chequeo Pre-PR (Estándares del Curso UAO)

- [x] **Entorno de ejecución (`uv`):** Todo se ejecutó y probó con `uv` y Python 3.14.
- [x] **Sin warnings:** `uv run pytest` corre limpio; se gestionaron los imports asíncronos de gRPC.
- [x] **Lint (`ruff`):** `uv run ruff check src scripts tests` pasa al 100%.
- [x] **Control de exclusiones (`.gitignore`):** Se mantienen fuera los archivos de proto generados si así se decidió (o se incluyen los stubs para facilidad de despliegue).
- [x] **Pruebas unitarias (`pytest`):** `uv run pytest` pasa al 100 %: **62/62** (15 nuevas de API/gRPC + 47 previas).
- [x] **Clean Code:** Uso de fábricas para la app (`create_app`), inyección de dependencias (`AppState`) y separación clara de esquemas y lógica.
- [x] **Reproducibilidad:** `uv sync` instala las nuevas dependencias (fastapi, grpcio) sin problemas.

---

## 7. 🧪 Evidencia de Pruebas Ejecutadas

```
# Ejecución de tests de API y gRPC:
$ uv run pytest tests/test_api.py tests/test_grpc.py -v
tests/test_api.py::test_ask_flujo_feliz_con_fuentes PASSED                [  6%]
tests/test_api.py::test_ask_pregunta_vacia_o_blancos_422 PASSED           [ 13%]
...
tests/test_grpc.py::test_ingest_emite_progreso_en_streaming PASSED        [ 73%]
tests/test_grpc.py::test_index_status PASSED                              [100%]
============================== 15 passed in 0.85s ==============================

# Suite completa:
$ uv run pytest -q
..............................................................           [100%]
62 passed in 2.10s

# Lint:
$ uv run ruff check src scripts tests
All checks passed!
```

---

## 8. 👀 Notas para el Revisor

- Se ha priorizado el uso de `grpc.aio` para mantener la consistencia asíncrona con FastAPI.
- El `AppState` centraliza la lógica de "reinicio" del índice: cuando gRPC termina una ingesta con `rebuild=True`, llama a `state.on_index_rebuilt()`, lo que limpia la caché del `Retriever` compartido.
- La validación de Pydantic en `AskRequest` incluye un validador personalizado para evitar preguntas que sean solo espacios en blanco, algo que `min_length=1` no detecta por sí solo.

---

## 9. 🔗 Referencias

- `plan-trabajo-tecnico.md` — Fase 5 (APIs).
- `src/asistente_agentico_uao/api/main.py`
- `src/asistente_agentico_uao/grpc_impl/servicer.py`
- `tests/test_api.py`
- `tests/test_grpc.py`

