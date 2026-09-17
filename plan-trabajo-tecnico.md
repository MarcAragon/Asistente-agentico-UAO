# Plan de Trabajo Técnico — Asistente RAG UAO (Backend + APIs + Frontend)

> Proyecto: asistente conversacional RAG para normativa institucional UAO.
> Alcance de este plan: **backend + APIs (REST de consulta y gRPC de
> ingesta) + frontend web (Streamlit) + caché semántica (Redis) +
> contenerización (Docker)**. Documento base: `asistente-uao-rag.md`.
> Fecha: 2026-09-08.
> **Actualización 2026-09-09**: la extracción (Fase 1) migró de PyMuPDF+EasyOCR
> local a **LlamaCloud Parse** (servicio agentic, salida markdown). Ver §0.4
> (decisiones 7-8), Fase 1 y el registro de riesgos §6.
> **Actualización 2026-09-15**: Fase 5 completada (REST de consulta + gRPC de
> administración del índice) y **nueva Fase 7 — Frontend web (Streamlit),
> caché semántica (Redis) y proxy inverso TLS**, insertada antes de la
> contenerización, que pasa a ser **Fase 8**. Con esto el plan cubre el
> alcance del documento base (`asistente-uao-rag.md` §1.4 infraestructura,
> §2.2 objetivo 4 —interfaz interactiva y robusta—, §2.3 matriz de alcance,
> §4.4 stack y las tareas Kanban 008/010).

---

## 0. Estado actual y hechos validados

### 0.1 Entorno ya construido (Fase 0 completada)

| Componente | Versión instalada | Estado |
|---|---|---|
| Python (gestionado por `uv`) | 3.14.7 | ✅ Validado (el doc base §1.4 menciona 3.13; el repo fija `>=3.14` en `pyproject.toml` / `.python-version`) |
| `uv` | 0.12.2 | ✅ Lockfile con ~150 paquetes |
| `langchain` | 1.4.0 | ✅ Import OK |
| `langchain-cerebras` (`ChatCerebras`) | 0.8.2 | ✅ Import OK |
| `langchain-chroma` / `chromadb` | 1.1.0 / 1.5.9 | ✅ Import OK |
| `sentence-transformers` | 6.0.1 | ✅ Import OK |
| `torch` | 2.14.0+cu130 | ✅ CUDA disponible en el equipo |
| `llama-cloud` (SDK LlamaCloud Parse) | 2.16.0 | ✅ Parseo PDF → markdown |
| `fastapi` / `uvicorn[standard]` | 0.141.1 / 0.52.4 | ✅ Import OK |
| `pytest` / `ruff` (dev) | 9.1.1 / 0.16.6 | ✅ |

> **Cambio 2026-09-09**: se retiraron del proyecto `pymupdf` y `easyocr`
> (con sus transitivas: `torchvision`, etc.). El parseo de PDFs ya no es
> local: lo hace el servicio LlamaCloud Parse vía `llama-cloud`.

Comandos base del proyecto (nunca usar `pip`/`uv pip`):

```bash
uv add <paquete>        # dependencia de producción (actualiza pyproject + uv.lock)
uv add --dev <paquete>  # dependencia de desarrollo
uv sync                 # instalar/replicar entorno desde uv.lock
uv run <comando>        # ejecutar dentro del entorno
uv run pytest           # pruebas
uv run ruff check --fix # estilo
```

### 0.2 Diagnóstico real del corpus (`Data/Documentos/`, 20 PDFs)

Hallazgo crítico medido con PyMuPDF (no asumido): los PDFs de la UAO
presentan **tres perfiles de extracción**:

| Perfil | Cantidad | Archivos (ejemplos) | Estrategia |
|---|---|---|---|
| Texto digital legible | 12 | `Res-CS-515_2-1.pdf`, `Reso-CS-No.666...` | Extracción directa |
| Escaneados (solo imagen, 0 chars) | 5 | `Res-CA-6603.pdf`, `politicaspermanencia.pdf`, `reglamento-academico-doctorado-ingenieria.pdf`, `6-Resol-CS-557...`, `Reglamento-posgrados-Res-CA-6605.pdf` | OCR |
| Codificación de fuente rota (sin ToUnicode → caracteres de control) | 3 | `Res-CS-726-Código...pdf`, `res-ect-8338...pdf`, `res-rect-8339...pdf` | OCR |

> **Nota 2026-09-09**: este diagnóstico motivó el pipeline local
> PyMuPDF+EasyOCR, hoy retirado. Con **LlamaCloud Parse (tier agentic)** los
> tres perfiles se resuelven dentro del servicio (OCR y tablas incluidas);
> el único requisito es conexión a internet y créditos del plan.

### 0.3 Restricciones de hardware medidas

- GPU del equipo: **3.68 GiB totales** (NVIDIA, CUDA 13). Suficiente para
  los *embeddings* locales (modelo mpnet-base ≈ 1 GB, batches pequeños).
  La VRAM ya **no** es cuello de botella: desde la migración a LlamaCloud
  Parse no hay OCR local (antes EasyOCR hacía OOM en GPU).
- LLM: se consume vía API (Cerebras) → sin requisitos de VRAM locales.

### 0.4 Decisiones técnicas de diseño (lock de alcance)

1. **Orquestación**: LangChain 1.x con LCEL (cadenas declarativas
   `prompt | llm | parser`). No LlamaIndex.
2. **LLM**: `ChatCerebras(model="qwen-3.8-27b")` — modelo verificado en el
   catálogo de Cerebras (64k/128k contexto). API key por env var.
3. **Embeddings**: modelo local **E5 multilingüe**
   (`intfloat/multilingual-e5-base`, ventana de 512 tokens) con prefijos
   asimétricos (`query:` / `passage:`), resolución de device en runtime
   (`resolve_embedding_device()` en `src/asistente_agentico_uao/core/config.py`):
   CUDA (NVIDIA) → ROCm (AMD) → CPU. *(Actualizado en F2: mpnet-base se
   descartó porque su ventana de 128 tokens truncaba los chunks de ~400.)*
4. **Vector DB**: ChromaDB `PersistentClient` (directorio `Data/chroma/`),
   colección única con upsert determinista (ID = SHA-256 del chunk).
5. **API**: FastAPI + Pydantic v2. Carga perezosa del índice al arranque.
6. **Chunking**: 300–500 tokens con overlap, pero **separado de la
   limpieza**: primero se valida manualmente el texto limpio.
7. **Extracción de PDF (2026-09-09)**: servicio **LlamaCloud Parse**
   (`tier="agentic"`, `expand=["markdown_full"]`) vía SDK `llama-cloud`.
   Un `.md` por PDF en `Data/Documentos_MD/` — formato de entrada del RAG.
   PyMuPDF y EasyOCR quedaron fuera del proyecto.
8. **Limpieza (2026-09-09)**: reducida a una pasada ligera sobre el
   markdown de LlamaParse (ruido de layout: `logo:`/`signature:`,
   numeración de página huérfana, imágenes, espacios). Sin heurísticas de
   OCR (ligaduras, guiones de fin de línea, reconstrucción de párrafos).
9. **Frontend (2026-09-15)**: **Streamlit** (chat, `frontend/app.py`), que
   consume **solo la API REST `/ask`** por HTTP/JSON. No se llama gRPC desde
   el navegador (exigiría grpc-web): gRPC sigue siendo el plano de control
   interno. Fuera de alcance según `asistente-uao-rag.md` §2.3: historial
   multiturno con contexto, autenticación contra el directorio
   institucional e integración con SIA/Aula Virtual.
10. **Caché semántica (2026-09-15)**: **Redis** para reutilizar respuestas a
    preguntas frecuentes por similitud de la pregunta (embeddings E5), como
    pide el doc base §1.4. Es **opcional y degradable**: sin Redis la API
    responde igual (solo más lenta); se invalida al reindexar para no servir
    respuestas de un índice viejo.

---

## 1. Arquitectura objetivo

```text
                        ┌──────────────────────────────────────────────┐
                        │   FRONTEND WEB (Fase 7 · Streamlit)          │
 Estudiante ──────────►│   chat · respuesta + fuentes citadas         │
   (navegador)          │   └─ HTTP/JSON (solo REST /ask, nunca gRPC)  │
                        └───────────────────┬──────────────────────────┘
                                            │ proxy inverso TLS (:443)
                        ┌───────────────────▼──────────────────────────┐
                        │  CACHÉ SEMÁNTICA (Fase 7 · Redis)            │
                        │   hit por similitud de la pregunta → TTL     │
                        └───────────────────┬──────────────────────────┘
                                            ▼
                        ┌──────────────────────────────────────────────┐
                        │              BACKEND (este plan)             │
                        │                                              │
 Data/Documentos/*.pdf►│  INGESTA (offline, CLI)                      │
   20 PDFs oficiales    │  scripts/llama_cloud_parsing.py              │
                        │   └─ LlamaCloud Parse (agentic, API nube)    │
                        │      └─ Data/Documentos_MD/*.md (limpio)     │
                        │         └─ (chunk.py, Fase 2)                │
                        │  embeddings.py (sentence-transformers,       │
                        │      device CUDA/ROCm/CPU)                   │
                        │      ▼                                       │
                        │  vectorstore.py ─► ChromaDB persistente      │
                        │      Data/chroma/ (metadata: doc, sección)   │
                        │      ▲                                       │
                        │  gRPC :50051 (control plane, F5)             │
                        │   Ingest(stream) · PruneIndex · IndexStatus  │
                        │                                              │
 Frontend ───────────►  │  REST :8000 (data plane, F5)                 │
   POST /ask            │  retrieval.py ─► top-k + umbral similitud    │
                        │      ▼                                       │
                        │  cache.py (F7) ─► Redis (hit por similitud)  │
                        │      ▼                                       │
                        │  chain.py (LCEL): prompt ─► llm.py ─► parse  │
                        │      llm.py = ChatCerebras(qwen-3.8-27b)     │
                        │      ▼                                       │
 Respuesta ◄──────────  │  respuesta + fuentes[] (doc, sección, texto) │
                        └──────────────────────────────────────────────┘
```

Flujo de datos:

1. **Offline (ingesta)**: PDF → LlamaCloud Parse (API agentic, markdown) →
   limpieza ligera → `Data/Documentos_MD/*.md` → chunking por encabezados →
   embeddings → Chroma (upsert con metadatos de cita).
2. **Online (consulta)**: pregunta → embed query → búsqueda top-k en
   Chroma → filtro por umbral de similitud → prompt con contexto y regla
   de citación → LLM (Cerebras) → respuesta JSON con `sources[]`.
3. **Umbral de "no sé"**: si ningún fragmento supera
   `min_similarity`, la cadena ni siquiera llama al LLM: responde
   `{"answer": "No tengo información suficiente...", "sources": []}`.
4. **Frontend (online, F7)**: chat Streamlit → `POST /ask` (JSON) → dibuja
   la respuesta y abre las fuentes (`doc_name`, `section`, `score`,
   `excerpt`); el mensaje de no-información se muestra como estado
   explícito con `used_fallback=true` y **sin** fuentes.
5. **Caché semántica (F7)**: antes de llamar al LLM se consulta Redis con el
   embedding de la pregunta; un hit devuelve la respuesta guardada (con sus
   fuentes y su `model`) sin gastar tokens. Se invalida al reindexar.

---

## 2. Estructura del repositorio (convención `src`)

```text
Asistente-agentico-UAO/
├── Data/
│   ├── Documentos/              # Corpus original (20 PDFs) — solo lectura
│   ├── Documentos_MD/           # *.md limpios (salida LlamaCloud Parse; entrada del RAG)
│   └── chroma/                  # Persistencia ChromaDB (se crea en Fase 2; gitignored)
├── scripts/
│   ├── llama_cloud_parsing.py   # CLI PDFs → LlamaCloud Parse → Data/Documentos_MD/*.md
│   └── ingest.py                # FASE 2: CLI chunking+embeddings → Chroma
├── src/asistente_agentico_uao/
│   ├── __init__.py              # entrypoint de consola (pyproject: asistente-agentico-uao)
│   ├── core/                    # infraestructura y adaptadores externos (sin dominio RAG)
│   │   ├── config.py            # Settings (pydantic-settings) + device resolver
│   │   ├── embeddings.py        # FASE 2: SentenceTransformer + device + tokenizador
│   │   ├── vectorstore.py       # FASE 2: cliente Chroma + upsert + colección
│   │   └── llm.py               # FASE 4: ChatCerebras + rotación de claves + NO_INFO_MESSAGE
│   ├── rag/                     # dominio RAG (retrieval → chain → cache → service)
│   │   ├── retrieval.py         # FASE 3: retriever top-k + umbral
│   │   ├── chain.py             # FASE 4: cadena RAG LCEL completa
│   │   ├── cache.py             # FASE 7: caché semántica en Redis (opcional)
│   │   └── service.py           # FASE 5: AppState compartido REST/gRPC
│   ├── ingestion/               # FASE 2/5: chunk.py + pipeline.py (ingesta compartida)
│   ├── api/                     # FASE 5: main.py (app FastAPI) + schemas.py
│   ├── grpc_impl/               # FASE 5: proto + servicer + server (control de ingesta)
│   └── frontend/                # FASE 7: app.py (chat Streamlit) + client.py (cliente HTTP)

├── tests/                       # pytest (unit + integración) — se crea en Fase 2
├── asistente-uao-rag.md         # Documento base del proyecto
├── plan-trabajo-tecnico.md      # Este plan
├── pyproject.toml / uv.lock     # gestión con uv (nunca pip)
└── .env / .env.example          # claves API; .env es copia de .env.example (gitignored)
```

> **Nota 2026-09-15**: el paquete real es `src/asistente_agentico_uao/`
> (no `uao_rag` como se planeó al inicio) y el corpus vive bajo `Data/`
> (`Data/Documentos/` los PDFs, `Data/Documentos_MD/` el markdown
> parseado). Ya existen `Data/chroma/` (índice de 1284 chunks), `tests/`,
> `api/`, `grpc_impl/`, `frontend/` y `cache.py`.
>
> **Nota 2026-09-17 (refactor de estructura)**: los módulos que estaban
> sueltos en la raíz del paquete se agruparon por capa: `core/`
> (config, embeddings, vectorstore, llm — infraestructura y adaptadores) y
> `rag/` (retrieval, chain, cache, service — dominio RAG). Se añadió
> `ingestion/__init__.py`, que faltaba (era namespace package). Regla de
> dependencias: `core` no importa de `rag`; `rag` e `ingestion` solo
> importan de `core`; `api`, `grpc_impl` y `frontend` son la capa de
> entrada. Los imports (incluidos tests y scripts) se actualizaron a las
> rutas nuevas, p. ej. `asistente_agentico_uao.core.config` y
> `asistente_agentico_uao.rag.chain`; no se dejaron alias de
> compatibilidad a propósito, para que una ruta vieja falle de inmediato.

---

## 3. Contratos técnicos

### 3.1 Configuración (`src/asistente_agentico_uao/core/config.py`, prefijo `UAO_RAG__`)

| Variable | Default | Uso |
|---|---|---|
| `CEREBRAS_API_KEY` | — (obligatoria) | Autenticación API Cerebras |
| `CEREBRAS_API_KEYS` | — (opcional) | Claves extra separadas por coma; rotación ante límites de cuota (429) o clave inválida (401/403) |
| `LLAMA_CLOUD_API_KEY` | — (obligatoria para parsear) | Autenticación API LlamaCloud (solo Fase 1) |
| `UAO_RAG__DOCS_DIR` | `Data/Documentos` | Corpus PDF original |
| `UAO_RAG__MARKDOWN_DIR` | `Data/Documentos_MD` | Markdown limpio, entrada del RAG |
| `UAO_RAG__CHROMA_DIR` | `Data/chroma` | Persistencia vectorial |
| `UAO_RAG__TOP_K` | `5` | Fragmentos recuperados por consulta |
| `UAO_RAG__MIN_SIMILARITY` | `0.35` | Umbral para responder "no sé" |
| `UAO_RAG__EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Modelo de embeddings (E5 multilingüe, prefijos `query:`/`passage:`) |
| `UAO_RAG__EMBEDDING_DEVICE` | vacío (auto) | Forzar `cuda`/`cpu` si se desea |
| `UAO_RAG__LLM_MODEL` | `qwen-3.8-27b` | Modelo en Cerebras |
| `UAO_RAG__LLM_TEMPERATURE` | `0.1` | Temperatura de síntesis (F4) |
| `UAO_RAG__LLM_MAX_TOKENS` | `1024` | Techo de tokens de la respuesta (F4) |
| `UAO_RAG__LLM_DISABLE_REASONING` | `1` | Desactiva el thinking de qwen-3.8 (hallazgo F4: agotaba `max_tokens` y devolvía contenido vacío) |
| `UAO_RAG__LLM_MAX_RETRIES` | `3` | Reintentos ante 429/timeout con backoff exponencial (F4) |
| `UAO_RAG__GRPC_ENABLED` | `1` | Servicio gRPC de administración del índice embebido en el proceso de la API (F5) |
| `UAO_RAG__GRPC_PORT` | `50051` | Puerto del control plane gRPC (F5) |
| `UAO_RAG__REDIS_URL` | `redis://localhost:6379/0` | Conexión a Redis para el caché semántico (F7) |
| `UAO_RAG__CACHE_ENABLED` | `1` | Activa el caché semántico; `0` lo desactiva por completo (F7) |
| `UAO_RAG__CACHE_SIMILARITY` | `0.97` | Similitud mínima pregunta↔pregunta cacheada para reutilizar la respuesta (F7) |
| `UAO_RAG__CACHE_TTL_SECONDS` | `86400` | Vigencia de cada entrada de caché (24 h) (F7) |
| `UAO_RAG__API_BASE_URL` | `http://localhost:8000` | URL de la API que consume el frontend (solo frontend, F7) |
| `UAO_RAG__FEEDBACK_ENABLED` | `0` | Panel «¿fue útil?» (nice-to-have §2.3); si está en `0` no se muestra (F7) |

### 3.2 Esquema de la colección ChromaDB

- Colección: `uao_normativa`
- `id`: `sha256(doc_name + chunk_index)` (upsert idempotente)
- `document`: texto del chunk
- `metadata`:
  - `doc_name: str` — nombre del PDF original (fuente de la cita)
  - `section: str` — encabezado markdown más cercano aguas arriba del
    chunk (título, artículo o numeral)
  - `chunk_index: int` — posición dentro del documento
- **Cambio de contrato (2026-09-09)**: se eliminan `page`, `page_end` y
  `source` del esquema anterior: `markdown_full` de LlamaParse no conserva
  los límites de página del PDF. La citación pasa a ser **(documento,
  sección)**. Si más adelante se exige cita por página, se puede re-parsear
  con `expand=["pages"]` y guardar el índice de página por chunk.
- Embeddings: función propia (`SentenceTransformer.encode`) pasada a
  `chromadb.PersistentClient`, **no** `chromadb.utils.embedding_functions`
  (control total del device).

### 3.3 Contrato de la API (Fase 5)

`POST /ask`

```json
// Request
{ "question": "¿Cuál es el plazo para cancelar una asignatura?" }

// Response 200
{
  "answer": "Según el Reglamento ... (Art. 21) ...",
  "sources": [
    { "doc_name": "Reso-CS-No.666-modifica-reglamento-aca-pre.pdf",
      "section": "Artículo 21", "score": 0.71,
      "excerpt": "El estudiante podrá cancelar asignaturas hasta..." }
  ],
  "model": "qwen-3.8-27b",
  "used_fallback": false
}
```

`GET /health` → `{"status": "ok", "index_chunks": 1234, "device": "cuda"}`
`GET /documents` → lista de documentos indexados (para trazabilidad).

Reglas:
- `question` vacía o > 500 chars → `422` (validación Pydantic).
- Sin fragmentos sobre el umbral → `200` con respuesta de no-información y
  `sources: []` (el sistema nunca inventa).
- Sin `CEREBRAS_API_KEY` configurada → la app arranca pero `/ask`
  responde `503` con mensaje claro.
- CORS abierto (`*`) para el futuro frontend. ✅ **Consumido por el frontend
  de la Fase 7** (`frontend/app.py`): es el único contrato HTTP que usa; el
  contrato gRPC (`IndexAdmin`) es interno y no navegable.

---

## 4. Fases end-to-end (tareas, entregables, criterios de aceptación)

### Fase 0 — Entorno (✅ COMPLETADA)
`uv init`, `uv add` de todas las dependencias, `.env.example`, `.gitignore`,
validación de imports y de CUDA. Hecho (ver §0.1).

### Fase 1 — Ingesta con LlamaCloud Parse (extracción + limpieza) — EN CURSO

**Método nuevo (2026-09-09)**: el pipeline local PyMuPDF+EasyOCR y sus 6
etapas de limpieza se retiraron del proyecto (dependencias, `extract.py`,
`clean.py`, `scripts/clean_documents.py` y sus tests incluidos). La
extracción ahora es el servicio **LlamaCloud Parse** en modo agentic, que
devuelve markdown con ortografía, acentos, tablas (`<table>`) y encabezados
correctos incluso para los 8 PDFs problemáticos (5 escaneados + 3 con
fuente rota). La limpieza local se reduce a una pasada ligera.

| # | Tarea | Estado |
|---|---|---|
| 1.1 | `scripts/llama_cloud_parsing.py`: CLI generalizado — recorre `Data/Documentos/*.pdf`, sube cada PDF (`purpose="parse"`), llama `parsing.parse(tier="agentic", version="latest", expand=["markdown_full"])`, borra el archivo remoto tras parsear | ✅ |
| 1.2 | API key fuera del código: `LLAMA_CLOUD_API_KEY` en `.env` (gitignored), leída del entorno o del `.env`; documentada en `.env.example` | ✅ |
| 1.3 | Limpieza ligera del markdown (`clean_markdown`): quita `logo:`/`signature:`/`stamp:`, numeración huérfana `Página N (de M)`, imágenes `![](...)`, normaliza espacios y líneas en blanco; **conserva encabezados `##`/`###` y tablas** | ✅ |
| 1.4 | Salida: un `.md` por PDF en `Data/Documentos_MD/` (idempotente: salta ya parseados; `--redo` reprocesa; `--file` procesa uno) | ✅ |
| 1.5 | Tests: prueba temporal inline de `clean_markdown` ejecutada y eliminada (los tests permanentes se re-crean por fase según §5) | ✅ |
| 1.6 | **Ejecutar sobre los 20 PDFs y VALIDACIÓN MANUAL de los `.md`** | ✅ |

**Formato del texto limpio de entrada al RAG**: markdown UTF-8, un archivo
por documento oficial (`Data/Documentos_MD/<nombre-pdf>.md`), con la estructura
original preservada: encabezados `##`/`###` para título, artículos y
secciones; tablas HTML `<table>` para calendarios; párrafos separados por
línea en blanco; sin ruido de layout (logos, firmas gráficas, numeración
de página, imágenes). Este markdown es la entrada directa del chunking
(Fase 2), que usará los encabezados como fronteras duras de chunk.

**Criterio de aceptación F1**: el usuario aprueba el markdown de los 20
`.md` en `Data/Documentos_MD/` (fidelidad al original, tablas legibles, sin
basura). Iterar sobre `clean_markdown` según hallazgos.

### Fase 2 — Chunking + embeddings + indexación Chroma
| # | Tarea | Detalle técnico |
|---|---|---|
| 2.1 | `ingestion/chunk.py` | Entrada: `Data/Documentos_MD/*.md`. Chunking por párrafos a 300–500 tokens (`len//4` como proxy o tokenizador del modelo), overlap ~15%, con encabezados markdown (`##`/`###`) y límites de artículo/numeral como frontera dura; registra `section` (encabezado vigente) por chunk |
| 2.2 | `embeddings.py` | `SentenceTransformer(settings.embedding_model, device=resolve_embedding_device())`; `encode(..., batch_size=32, normalize_embeddings=True)` → similitud coseno |
| 2.3 | `vectorstore.py` | `PersistentClient(path=chroma_dir)`; `get_or_create_collection("uao_normativa", metadata={"hnsw:space": "cosine"})`; `upsert` con IDs SHA-256 |
| 2.4 | `scripts/ingest.py` | CLI: `Data/Documentos_MD/*.md` → chunks → Chroma; reporta nº de chunks/doc y tiempo; flag `--rebuild` (borra colección) |
| 2.5 | Tests | `chunk.py` con Chroma efímero (`EphemeralClient`) + modelo real de embeddings (test marcado `@pytest.mark.slow`) |

**Criterio de aceptación F2**: índice reproducible — correr el CLI dos
veces no duplica chunks (upsert); `collection.count()` estable; query de
humo retorna fragmentos relevantes con `page` correcto.

### Fase 3 — Motor de recuperación — ✅ COMPLETADA (2026-09-14)
| # | Tarea | Estado |
|---|---|---|
| 3.1 | `retrieval.py`: `Retriever.retrieve(question) -> list[RetrievedChunk]` — embed query (prefijo `query:` de E5), `collection.query(n_results=top_k*2)`, `score = 1 - distance` (coseno), descarte `score < min_similarity`, orden desc y corte a `top_k`; colección inyectable y carga perezosa; `[]` inmediato si la colección está vacía | ✅ |
| 3.2 | `format_context(chunks)`: bloques numerados `[N] (doc — sección): texto` para citación del LLM en F4 | ✅ |
| 3.3 | (Opcional) Re-ranking | Aplazado a F6 (según plan) |
| 3.4 | Tests | ✅ Verificados con suite temporal (unit + integración con EphemeralClient y embeddings E5 reales, 7/7 OK), **eliminada tras validar** a petición del usuario; se re-crean permanentes en F6 si se desea |
| — | `scripts/smoke_retrieval.py`: humo con banco de 10 preguntas + preguntas ad-hoc por argumento | ✅ |

**Criterio de aceptación F3**: ✅ humo sobre el índice real (1284 chunks):
en las 10 preguntas del banco el fragmento pertinente aparece en top-5 con
sim 0.81-0.88 (ej.: cancelaciones → Art. 31º/parágrafos del reglamento de
pregrado; tres repitencias → Art. 70º-2 de Res-CA-6744; calendario 2026-2 →
tablas de las Res. 8338/8339/8340).

**⚠ Hallazgo para F6**: las similitudes coseno de E5 son altas en general:
una pregunta claramente fuera de dominio ("receta de arepas") aún obtiene
sim ≈ 0.81, por lo que `min_similarity=0.35` **no** discrimina preguntas
fuera de dominio. Recalibrar el umbral (probablemente ≥ 0.85-0.90) con el
banco de preguntas de F6, o apoyar el "no sé" en la instrucción del prompt.

### Fase 4 — LLM Cerebras + cadena RAG — ✅ COMPLETADA (2026-09-15)
| # | Tarea | Estado |
|---|---|---|
| 4.1 | `llm.py`: `get_llm()` singleton con `ChatCerebras(model, temperature, max_tokens)`; `invoke_with_retry` con backoff exponencial (1s/2s/4s) ante 429/timeout/conexión/5xx; `NO_INFO_MESSAGE` centralizado | ✅ |
| 4.2 | Prompt de síntesis (ES) en `chain.py`: rol asistente normativa UAO; responder SOLO con contexto `[1]..[k]`; citar como `(Documento, sección)`; si el contexto no alcanza, copiar EXACTAMENTE `NO_INFO_MESSAGE`; no inventar; indicar qué falta si el contexto es parcial | ✅ |
| 4.3 | `chain.py`: LCEL `prompt | llm(con reintentos) | StrOutputParser`; post-proceso `build_sources` mapea citas `(Documento, sección)` verificadas a `Source` (doc_name, section, score, excerpt); citas no verificables se descartan (nunca se inventan fuentes); `RagAnswer` con `used_fallback` | ✅ |
| 4.4 | Umbral pre-LLM: `retrieve()` vacío → respuesta inmediata sin gastar tokens. **Mitigación del hallazgo F3**: como el umbral no discrimina dominio (sim ~0.81 fuera de dominio), la defensa principal es el prompt (reglas 2-3): el LLM juzga si el contexto responde; verificado de facto con 2 preguntas fuera de dominio → mensaje exacto sin alucinar | ✅ |
| 4.5 | `scripts/ask.py`: CLI end-to-end (pregunta ad-hoc o banco de 7 humos: 5 in-dominio + 2 fuera de dominio) | ✅ |
| 4.6 | Rotación de claves API: `CerebrasLLM` en `llm.py` envuelve `ChatCerebras` y rota a la siguiente clave de `CEREBRAS_API_KEY` + `CEREBRAS_API_KEYS` (coma) ante 429/cuota o 401/403, recreando el cliente; con una sola clave, 429 reintenta con backoff y 401/403 se relanza | ✅ |

**Criterio de aceptación F4**: ✅ Verificado con `uv run python scripts/ask.py`
sobre el índice real (1284 chunks): "tres repitencias" cita Art. 70º-2/70º-3
de Res-CA-6744; "requisitos magíster" cita Arts. 35º/13º de Res-CA-6605 e
indica explícitamente qué no está en el contexto; "transferencia interna"
cita Arts. 19°/17° (Reso-CS-666), 23º (Res-CA-6603) y 16º (Res. 7714). Las 2
fuera de dominio (arepas, Mundial 2022) responden el mensaje exacto de
no-información sin alucinar.

**⚠ Hallazgo F4 (2026-09-15)**: `qwen-3.8-27b` es un modelo de razonamiento:
por defecto gasta los `max_tokens` en tokens de thinking y devuelve
`content` vacío (`finish_reason=length`; ocurrió en 2 de 7 humos). Mitigado
con `disable_reasoning=True` vía `extra_body` (setting
`UAO_RAG__LLM_DISABLE_REASONING=1`) + guardia en `chain.py` que degrada
respuesta vacía a no-información. Efecto colateral positivo: latencia por
pregunta bajó de ~2-12 s a ~0.3-0.6 s.

**Notas para F6**: (a) recalibrar `min_similarity` con el banco (hallazgo
F3 sigue vigente); (b) en 2 preguntas in-dominio ("cancelaciones 2026-2",
"créditos mínimos posgrado") el LLM respondió no-información porque los
chunks recuperados eran de otro programa/periodo: evaluar recall del banco
y si las tablas de calendario contienen la fecha puntual.

### Fase 5 — API: REST (consulta) + gRPC (control de ingesta)

**Separación de funciones** (decisión 2026-09-15): cada protocolo cumple un
rol distinto y NO se duplican operaciones.

- **REST (FastAPI, :8000) — plano de consulta público (data plane)**: la
  puerta del frontend/usuario final. Solo LEE el índice y llama al LLM.
- **gRPC (:50051) — plano de control e ingesta (control plane)**: las
  operaciones de ESCRITURA/administración del índice (que antes solo
  existían como CLI F2), máquina-a-máquina y con progreso en streaming.

| # | Tarea | Detalle técnico |
|---|---|---|
| 5.1 | Contrato | `grpc_impl/protos/index_admin.proto` (`Ingest`→stream `IngestProgress`, `PruneIndex`, `IndexStatus`); stubs committeados + `scripts/gen_proto.py` |
| 5.2 | REST | `api/schemas.py` (Pydantic: `AskRequest(1..500)`, `AskResponse`…), `service.py` (`AppState` compartido por ambos protocolos), `api/main.py` (`POST /ask`, `GET /health`, `GET /documents`; CORS `*`; 422/503/500) |
| 5.3 | gRPC | `grpc_impl/servicer.py` (pipeline en hilo worker + streaming de progreso; INVALID_ARGUMENT/INTERNAL) y `grpc_impl/server.py` (embebido en el lifespan o standalone) |
| 5.4 | Pipeline compartido | `ingestion/pipeline.py`: lógica del CLI F2 extraída a funciones (CLI y gRPC ejecutan el mismo código); `scripts/ingest_client.py` como cliente de humo |
| 5.5 | Tests | `test_api.py` (TestClient: 422/200/503/500, health, documents), `test_grpc.py` (servidor aio in-proceso: streaming, códigos, prune, status), `test_pipeline.py` — sin tokens ni modelo real |
| 5.6 | Docs | README (ejecución de ambas APIs), `.env.example` (`UAO_RAG__GRPC_ENABLED/PORT`) |

**Criterio de aceptación F5**: ✅ `curl -X POST /ask` → 200 con `answer` +
`sources[]` verificables (humo real: `/health` 1284 chunks/cuda,
`/documents` 20 docs, 422 en pregunta en blanco); `IndexStatus` por gRPC →
1284 chunks/20 documentos; suite `uv run pytest` 68/68 y ruff en verde.

**Notas F5**: (a) el modelo de embeddings se carga una vez y lo comparten
REST y gRPC (mismo proceso); (b) tras un `rebuild` por gRPC se invalida la
colección cacheada del Retriever (`AppState.on_index_rebuilt`); (c) los
stubs gRPC están excluidos del lint (código generado); (d) streaming de
tokens del LLM queda como extensión futura (el pipeline actual es
invoke-completo).

### Fase 6 — Pruebas, evaluación y ajuste fino
| # | Tarea | Detalle técnico |
|---|---|---|
| 6.1 | Banco de preguntas | ~30 preguntas con respuesta conocida (doc + sección), cubriendo los 15 tipos de documento; incluye jerga/typos (simular preguntas reales) |
| 6.2 | Métricas de recuperación | Recall@5, MRR sobre el banco (script `scripts/evaluate_retrieval.py`) |
| 6.3 | Métricas de generación | RAGAS (`faithfulness`, `answer_relevancy`) o checklist manual si RAGAS no soporta Py3.13; exactitud de citas (% de citas que apuntan al doc/sección correctos) |
| 6.4 | Ajuste | Iterar: `top_k`, `min_similarity`, chunking, re-ranking (3.3), prompt |

**Criterio de aceptación F6**: reporte de métricas reproducible;
recall@5 ≥ 0.8 y citas exactas ≥ 80% como objetivo inicial.

### Fase 7 — Frontend web (Streamlit) + caché semántica (Redis) + proxy TLS

**Concepto (documento base)**: el objetivo específico 4 pide una «interfaz de
usuario interactiva y robusta»; el §4.4 fija **Streamlit** como interfaz tipo
chat; el §1.4 exige **Redis** para el caché semántico de consultas frecuentes y
un **proxy inverso con validación de certificados TLS** para exponer el
servicio de forma segura. Es la fase que convierte el backend en un producto
usable por el estudiante.

**Alcance**: chat de **una sola vuelta** (el historial multiturno con contexto
es «nice to have» en la matriz §2.3), sin autenticación contra el directorio
institucional ni integración con SIA/Aula Virtual (fuera de alcance). El
frontend consume **solo** `POST /ask` (REST); nunca gRPC.

| # | Tarea | Detalle técnico |
|---|---|---|
| 7.1 | `frontend/app.py` (chat) | Streamlit: `st.chat_input` + `st.chat_message`; cada pregunta viaja a `POST /ask` con `httpx` (`UAO_RAG__API_BASE_URL`, timeout explícito); `st.session_state` guarda el historial **de la sesión** solo para mostrarlo (no se reenvía como contexto al LLM: una vuelta) |
| 7.2 | Respuesta y fuentes | Respuesta en markdown + `st.expander("Fuentes (n)")` con `doc_name`, `section`, `score` (2 decimales) y `excerpt`; si `used_fallback=true` → aviso ámbar «no hay información suficiente en la normativa» y **cero fuentes**; el backend ya descarta citas no verificables, la UI nunca las inventa |
| 7.3 | UX y robustez | `st.spinner` con estado mientras responde el LLM (~0.3-0.6 s medidos, pero sin pantalla congelada); límite de 500 caracteres con contador (contrato §3.3); preguntas de ejemplo (incluye jerga/typos, doc base §1.5); errores mapeados a mensajes claros: `422` (pregunta vacía/larga), `503` (sin clave de LLM), `500` y API caída (timeout/conexión) |
| 7.4 | Aviso y privacidad | Aviso visible «respuestas orientativas con la fuente oficial citada; no reemplaza la asesoría de Secretaría Académica»; el chat no solicita ni almacena datos personales (Ley 1581 de 2012, doc base §1.4): la caché guarda preguntas y respuestas, nunca identidades |
| 7.5 | `cache.py` (caché semántica) | `redis-py`: índice de preguntas + hash del embedding de la pregunta normalizada; **hit** si la similitud ≥ `UAO_RAG__CACHE_SIMILARITY` (se reutiliza el modelo E5 ya cargado, sin modelo extra); valor = `RagAnswer` serializado (respuesta, fuentes, `model`, `used_fallback`); `TTL` = `UAO_RAG__CACHE_TTL_SECONDS`; contadores hit/miss/ratio visibles en `/health`; **degradable**: sin Redis o con `CACHE_ENABLED=0` la API responde igual |
| 7.6 | Integración e invalidación | El caché se consulta en `service.py` (capa compartida REST/gRPC) **antes** de `answer_question`, así la REST y futuras integraciones lo aprovechan; `Ingest(rebuild)` y `PruneIndex` del servicio gRPC invalidan el caché, y las claves se versionan con la huella del índice (`count()` + timestamp) para no servir respuestas de un corpus viejo |
| 7.7 | Proxy inverso TLS | `deploy/Caddyfile` (o `nginx.conf`): terminación TLS automática en `:443`, redirección HTTP→HTTPS y cabeceras de seguridad; la API (:8000), gRPC (:50051) y Redis (:6379) quedan **solo** en la red interna del compose, sin publicarse al host |
| 7.8 | Pruebas | `tests/test_cache.py` con `fakeredis`: hit por similitud, miss, expiración por TTL, degradación sin Redis e invalidación por reindexado; `tests/test_frontend.py` para las funciones puras del cliente (llamada a la API y mapeo de errores con `httpx` mockeado). Humo manual: `uv run streamlit run frontend/app.py` contra la API local |
| 7.9 | (nice-to-have §2.3) Retroalimentación | Botones «útil / no útil» por respuesta (contador en Redis o log JSON); se muestra solo con `UAO_RAG__FEEDBACK_ENABLED=1`; sin datos personales |

Nota 2026-09-17 (decisión 7.5): para el desarrollo local de cache.py se usa fakeredis (simula un Redis completo en memoria dentro del proceso de Python) en lugar de una instancia real de Redis, porque el entorno de desarrollo del frontend no tiene Docker instalado. La interfaz que expone redis-py (y que usa fakeredis para simularla) es la misma tanto en desarrollo como en producción, así que el paso a Fase 8 (docker-compose con Redis real) es solo un cambio de configuración —apuntar el cliente al contenedor real en vez de a la instancia simulada—, no un cambio de código en cache.py. Los tests de tests/test_cache.py corren igualmente sobre fakeredis, como ya estaba contemplado en el plan original.

**Criterio de aceptación F7**: `uv run streamlit run frontend/app.py` responde
preguntas reales mostrando la respuesta y sus fuentes; una pregunta fuera de
dominio muestra el estado de no-información **sin fuentes**; repetir una
pregunta frecuente (o su paráfrasis) se resuelve desde caché en ≲50 ms y se
contabiliza como `hit`; con Redis detenido la UI sigue operativa (solo más
lenta); el acceso público es HTTPS vía proxy y ni la API ni gRPC quedan
expuestos. Alineado con las tareas Kanban 008 (5 SP) y la parte de UI/Redis
de la 010.

### Fase 8 — Contenerización y documentación (solución completa)
| # | Tarea | Detalle técnico |
|---|---|---|
| 8.1 | `Dockerfile` backend | Multi-stage sobre `python:3.14-slim` (versión fijada en `.python-version`); instala `uv`; `uv sync --frozen`; el índice se pre-construye y se monta como volumen (`Data/chroma`) para no descargar el modelo de embeddings en cada build |
| 8.2 | `Dockerfile` frontend | Imagen ligera para Streamlit (`frontend/app.py`); sin dependencias de ML (es solo un cliente HTTP de la API) |
| 8.3 | `docker-compose.yml` | Servicios: `proxy` (TLS, único puerto expuesto), `frontend` (:8501 interno), `api` (:8000 interno + :50051 gRPC interno), `redis` (caché) y volumen `data/`; `env_file: .env`; `depends_on` con `healthcheck` por servicio; red interna |
| 8.4 | README técnico + `Makefile`/`justfile` | Instalación, ingesta, arranque, contrato API y CLI, arquitectura y operación (logs, rebuild del índice, invalidación de caché); atajos: `make ingest`, `make api`, `make frontend`, `make test`, `make lint`, `make up` |
| 8.5 | Guía de despliegue | Variables de `.env`, persistencia (`Data/chroma` y volumen de Redis), renovación de certificados TLS, respaldo del índice y verificación de que `:8000`/`:50051`/`:6379` no están publicados al host |

**Criterio de aceptación F8**: `docker compose up --build` levanta la solución
completa (frontend en HTTPS + API + gRPC + Redis) y el chat es consultable
desde el host; `docker compose down && docker compose up` conserva el índice;
la suite `uv run pytest` y `uv run ruff check src scripts tests` siguen en
verde.

---

## 5. Estrategia de pruebas (resumen)

> (2026-09-09) Los tests del pipeline anterior (`tests/test_clean.py`,
> 14 tests) se eliminaron junto con `extract.py`/`clean.py`; los tests se
> re-crean fase a fase. Durante la migración se usó una prueba temporal
> inline de `clean_markdown` (ya borrada).

- **Unitarias** (rápidas, sin red/GPU): limpieza de markdown, chunking,
  pipeline de ingesta, schemas, umbrales, formateo de contexto, citas/fuentes,
  API REST (`TestClient`), gRPC in-proceso y caché semántica (`fakeredis`).
  Mocks de LLM, embeddings y Redis.
- **Frontend (F7)**: funciones puras del cliente Streamlit (llamada a
  `POST /ask`, formateo de fuentes, mapeo de errores 422/503/500/conexión)
  con `httpx` mockeado; el render se valida con humo manual
  (`uv run streamlit run frontend/app.py` contra la API local).
- **Integración** (marcadas `slow`): embeddings reales + Chroma efímero;
  1 llamada real a Cerebras (opcional, tras `pytest -m "not slow"`).
- **Evaluación** (F6): banco de preguntas + métricas de recuperación y
  fidelidad; se ejecuta manualmente, no en CI.
- Comando base: `uv run pytest -m "not slow"` para el ciclo rápido.

---

## 6. Riesgos técnicos y mitigaciones (registro vivo)

| Riesgo | Evidencia | Mitigación implementada/prevista |
|---|---|---|
| PDFs escaneados o con fuentes sin ToUnicode (texto ilegible) | 8 PDFs medidos (5 escaneados + 3 con fuente rota) | Resueltos por LlamaCloud Parse agentic (OCR y tablas en el servicio) |
| Dependencia de servicio externo (LlamaCloud) para la ingesta | Nueva arquitectura 2026-09-09: el parseo requiere red y créditos del plan | El markdown resultante se persiste en `Data/Documentos_MD/`; se parsea una sola vez y el CLI es idempotente (`--redo` para reprocesar) |
| Privacidad: documentos institucionales suben a la nube de LlamaCloud | 20 PDFs oficiales enviados por API | El archivo remoto se borra tras el parseo (`files.delete`); el corpus es normativo público; confirmar autorización institucional si se incorporan documentos sensibles |
| Pérdida de numeración de página en `markdown_full` | Contrato de citas cambiado (doc + sección) | Citar por documento + sección/artículo; si se exige página, re-parsear con `expand=["pages"]` |
| Calidad de tablas (calendarios) | 2 calendarios cuatrimestrales/bimestrales son principalmente tablas | LlamaParse devuelve `<table>` HTML estructurado; validar manualmente los 3 calendarios en la revisión F1 |
| Rate limits del free tier de Cerebras | Doc. oficial de planes | Rotación de claves (`CEREBRAS_API_KEYS`) en `CerebrasLLM` + backoff exponencial (`invoke_with_retry` integrado en la clase); umbral pre-LLM evita llamadas inútiles |
| qwen-3.8-27b agota `max_tokens` en tokens de razonamiento (respuesta vacía) | Hallazgo F4 (2026-09-15): `finish_reason=length` con `reasoning_tokens=1024` en 2 de 7 humos | `disable_reasoning=True` vía `extra_body` (setting `UAO_RAG__LLM_DISABLE_REASONING=1`) + guardia en `chain.py` que degrada respuesta vacía a no-información |
| Alucinaciones | Riesgo inherente LLM | Solo-contexto + citas + umbral de similitud + mensaje de no-información |
| Nombres de archivo con Unicode NFD | Encontrado en `Data/Documentos/` | SIEMPRE usar `glob`/`Path`, nunca nombres hardcodeados con tildes |
| Caché semántica sirviendo respuestas de un índice viejo | Redis (F7) guarda respuestas; al reindexar el corpus quedan obsoletas | Invalidación en `Ingest(rebuild)`/`PruneIndex` y claves versionadas con la huella del índice (`count()` + timestamp); TTL de 24 h como red de seguridad |
| Dependencia de Redis en producción | Doc base §1.4 exige contenedor de Redis para el caché | El caché es **opcional y degradable**: sin Redis (o `UAO_RAG__CACHE_ENABLED=0`) la API responde igual; `healthcheck` en compose y contadores hit/miss en `/health` |
| Latencia percibida y pantallas congeladas en la UI | Doc base §1.5 (tiempo de procesamiento) | `st.spinner` con estado, timeout explícito del cliente HTTP, caché semántica para preguntas repetidas (latencia del LLM medida: 0.3-0.6 s) |
| Exposición pública del servicio y datos personales | Doc base §1.4 (proxy TLS + Ley 1581) | Proxy inverso con TLS automático como único puerto público; API/gRPC/Redis en red interna; el chat no pide datos personales y la caché guarda solo pregunta/respuesta |
| Preguntas reales con jerga/typos distintas del banco de pruebas | Doc base §1.5 (cuarta limitación) | Preguntas de ejemplo en la UI, banco F6 con jerga/typos y caché semántica que absorbe paráfrasis de la misma pregunta |

---

## 7. Cronograma estimado (ruta crítica en negrita)

| Fase | SP | Duración est. | Dependencia |
|---|---|---|---|
| F0 Entorno | 2 | ✅ hecho | — |
| **F1 Parseo LlamaParse + limpieza ligera (validación manual)** | **5** | **1 sesión** | F0 |
| **F2 Chunking+embeddings+Chroma** | **5** | **1-2 sesiones** | **F1** |
| **F3 Recuperación** | **3** | **1 sesión** | **F2** |
| **F4 LLM+Cadena RAG** | **5** | **1 sesión** | F3 |
| F5 APIs (REST consulta + gRPC ingesta) | 3 | ✅ hecho | F4 |
| F6 Evaluación+ajustes | 3 | 1-2 sesiones | F5 |
| **F7 Frontend Streamlit + caché Redis + proxy TLS** | **5** | **1-2 sesiones** | **F5 (puede solaparse con F6)** |
| F8 Docker+docs (backend+frontend+Redis+proxy) | 5 | 1 sesión | F7 |
| **Total** | **36 SP** | **~9-12 sesiones** | |

La **ruta crítica** es F1 → F2 → F3 → F4: cualquier retrabajo en la
validación manual del texto limpio (F1.9) desplaza todo lo demás, por eso
es el checkpoint bloqueante actual. F7 solo depende de F5 (ya completada), así
que puede ejecutarse en paralelo con F6; F8 cierra el proyecto y depende de F7.
La carga total (36 SP) cuadra con el Kanban del documento base (38 SP, tareas
008 «interfaz en Streamlit» y 010 «Docker + Redis + Proxy»).

---

## 8. Próximo paso inmediato

~~Fase 4~~ ✅ COMPLETADA (2026-09-15): `llm.py` (ChatCerebras con
`disable_reasoning` + reintentos con backoff), `chain.py` (LCEL +
`build_sources` + umbral pre-LLM + guardia de respuesta vacía) y
`scripts/ask.py` (humo end-to-end, 7 preguntas). Adicionalmente (4.6):
rotación de claves API ante límites de cuota (`CEREBRAS_API_KEYS`). Mitigación
del hallazgo F3
verificada: las 2 preguntas fuera de dominio responden el mensaje exacto de
no-información sin alucinar (la defensa es el prompt, no el umbral).

**Siguiente: Fase 6** — Pruebas, evaluación y ajuste fino (banco de ~30
preguntas con jerga/typos, recall@5 y MRR, fidelidad/exactitud de citas y
recalibración de `min_similarity`). La Fase 5 de APIs ya está implementada y
verificada: REST de consulta en `api/` + gRPC de ingesta en `grpc_impl/`.

**Después — Fase 7 (frontend)**: `frontend/app.py` en Streamlit consumiendo
`POST /ask`, caché semántica en Redis (`cache.py`) y proxy inverso TLS
(`deploy/`); puede arrancar en paralelo con F6 porque solo depende de la API
de la Fase 5. La **Fase 8** cierra con la contenerización de la solución
completa (backend + frontend + Redis + proxy) y la documentación de
despliegue.
