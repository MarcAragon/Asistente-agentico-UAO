# Plan de Trabajo Técnico — Asistente RAG UAO (Backend + APIs + Frontend)

> Proyecto: asistente conversacional RAG para normativa institucional UAO.
> Alcance de este plan: **backend + APIs (REST de consulta y gRPC de
> ingesta) + frontend web (Streamlit) + caché semántica (Redis) +
> contenerización (Docker) + observabilidad del LLM (MLflow)**.
> Documento base: `asistente-uao-rag.md`.
> Fecha: 2026-09-08. Desde el 2026-09-17 este documento vive en `docs/`
> (movido junto a `asistente-uao-rag.md` y `guia-despliegue.md`).
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
> **Actualización 2026-09-17 (rama `feat--refactor+docker`)**: dos frentes.
> (a) **Refactor de estructura** del paquete en capas: `core/` (config,
> embeddings, vectorstore, llm) y `rag/` (retrieval, chain, cache, service)
> + `ingestion/__init__.py` que faltaba — ver §2 y la nota del mismo día.
> (b) **Fase 8 completada**: contenerización de la solución completa
> (`docker/Dockerfile*`, `docker-compose.yml`, `docker/Caddyfile`,
> `Makefile`), README técnico y guía de despliegue. Dependencias F7/F8
> añadidas a la tabla §0.1, variables de despliegue (compose/Caddy)
> documentadas en §3.1 y pruebas de despliegue registradas en §5.
> **Actualización 2026-09-18 (rama `feat--ml_flow`)**: **Fase 9 —
> Observabilidad del LLM con MLflow** (tracking server + dashboard de trazas),
> que el documento base no contemplaba y por eso **no estaba en este plan**.
> Ver la dependencia en §0.1, la decisión 11 de §0.4, la fase completa en §4,
> las variables en §3.1, las pruebas en §5 y los riesgos en §6. El servicio
> `mlflow` se suma a los 4 servicios de F8 **sin publicar puertos** (el
> dashboard se sirve por el proxy Caddy).

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
| `fakeredis` / `pyyaml` (dev) | 2.38.0 / 6.0.3 | ✅ fakeredis simula Redis en desarrollo y tests del caché (F7); `pyyaml` parsea `docker-compose.yml` en `tests/test_deployment.py` (F8) |
| `streamlit` (F7) | 1.64.0 | ✅ Chat web (`frontend/app.py`); el Dockerfile.frontend fija `==1.64.0` y `httpx==0.28.1` |
| `httpx` (F7) | 0.28.1 | ✅ Cliente HTTP del frontend hacia `POST /ask` |
| `redis` (F7) | 8.1.0 | ✅ Cliente del caché semántico (fakeredis en desarrollo, Redis real en compose) |
| `mlflow` (F9) | 3.16.1 | ✅ Cliente del *tracking* de trazas del LLM (`mlflow.openai.autolog()` en `api/main.py`); el compose usa la imagen oficial `ghcr.io/mlflow/mlflow` |
| `grpcio` / `grpcio-tools` / `protobuf` (F5) | 1.83.1 / 1.83.1 / 7.36.1 | ✅ Control plane de ingesta y stubs generados (`scripts/gen_proto.py`) |

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
11. **Observabilidad del LLM (2026-09-18, F9)**: **MLflow** como *tracking
    server* de las llamadas al LLM. La API se instrumenta con
    `mlflow.openai.autolog()` —funciona con Cerebras porque `ChatCerebras`
    hereda de `BaseChatOpenAI` y construye clientes `openai.OpenAI` apuntados
    a `api.cerebras.ai`—, **solo si el entorno define `MLFLOW_TRACKING_URI`**
    (lo hace `docker-compose.yml`): es opcional y degradable, y en local no se
    importa `mlflow`. El dashboard se sirve por el proxy
    (`https://mlflow.<SITE_ADDRESS>`) y el servicio no publica puertos. El
    documento base no menciona observabilidad del modelo, por eso se registra
    como **Fase 9** (fuera del alcance original, ver §4).

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
 Data/Documentos/*.pdf►│  INGESTA (offline: CLI o gRPC IndexAdmin)    │
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
2. **Online (consulta)**: caché semántica (F7: hit por similitud de la
   pregunta → respuesta guardada con sus fuentes) → si no hay hit,
   embed query → búsqueda top-k en Chroma → filtro por umbral de
   similitud → prompt con contexto y regla de citación → LLM (Cerebras) →
   respuesta JSON con `sources[]` (que se guarda en caché).
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
6. **Observabilidad (F9)**: cada llamada al LLM queda trazada en MLflow
   (prompt con contexto, respuesta, tokens y latencia) dentro del experimento
   `asistente-uao`; el dashboard se consulta por el proxy y el *tracking
   server* nunca se expone (red interna + volumen `mlflow_data`).

---

## 2. Estructura del repositorio (convención `src`)

```text
Asistente-agentico-UAO/
├── Data/
│   ├── Documentos/              # Corpus original (20 PDFs) — solo lectura
│   ├── Documentos_MD/           # *.md limpios (salida LlamaCloud Parse; entrada del RAG)
│   └── chroma/                  # Persistencia ChromaDB (se crea en Fase 2; gitignored)
├── scripts/
│   ├── llama_cloud_parsing.py   # FASE 1: CLI PDFs → LlamaCloud Parse → Data/Documentos_MD/*.md
│   ├── ingest.py                # FASE 2: CLI chunking+embeddings → Chroma
│   ├── smoke_retrieval.py       # FASE 3: humo del retriever (banco de 10 preguntas + ad-hoc)
│   ├── ask.py                   # FASE 4: CLI end-to-end de la cadena RAG (banco de 7 humos)
│   ├── ingest_client.py         # FASE 5: cliente gRPC de humo (Ingest/PruneIndex/IndexStatus)
│   └── gen_proto.py             # FASE 5: regenera los stubs gRPC desde el .proto
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
│   ├── api/                     # FASE 5/9: main.py (app FastAPI + autolog de MLflow) + schemas.py
│   ├── grpc_impl/               # FASE 5: proto + servicer + server (control de ingesta)
│   └── frontend/                # FASE 7: app.py (chat Streamlit) + client.py (cliente HTTP)

├── docker/                      # FASE 8: Dockerfile (backend), Dockerfile.frontend
│                                #         y Caddyfile (proxy inverso TLS + dashboard MLflow)
├── docker-compose.yml           # FASE 8/9: redis + api + frontend + proxy + mlflow (solo 80/443)
├── .dockerignore                # FASE 8: sin .env, .venv, cachés ni Data/ en el build
├── tests/                       # pytest (unit + integración + despliegue estático)
├── Makefile                     # atajos: dev, pipeline, docker, operación y respaldos
├── README.md                    # README técnico (arquitectura, contratos, comandos)
├── .streamlit/config.toml       # tema del chat Streamlit (base oscura, #2E5EAA)
├── .github/PULL_REQUEST_TEMPLATE.md  # plantilla de PR (Conventional Commits)
├── docs/
│   ├── asistente-uao-rag.md     # Documento base del proyecto
│   ├── plan-trabajo-tecnico.md  # Este plan (movido a docs/ el 2026-09-17)
│   └── guia-despliegue.md       # FASE 8: despliegue, TLS, respaldos y troubleshooting
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
En F8, `docker/Dockerfile.frontend` copia el paquete a `/app/frontend/` y
arranca `streamlit run frontend/app.py`: Streamlit ejecuta el script con su
propio directorio en `sys.path`, así que `app.py` sigue importando
`from client import preguntar_api` sin hacks de `PYTHONPATH` ni depender del
paquete instalado (la imagen del frontend solo instala `streamlit` y
`httpx`).

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
| `UAO_RAG__EMBEDDING_BATCH_SIZE` | `16` | Tamaño de lote del `encode` (16 en contenedor; 32 era el valor inicial local) |
| `UAO_RAG__CHUNK_SIZE_TOKENS` | `400` | Objetivo de tokens del chunk final (F2) |
| `UAO_RAG__CHUNK_MAX_TOKENS` | `450` | Techo duro del chunk; bajo `max_seq_length` de E5 (512) |
| `UAO_RAG__CHUNK_OVERLAP_TOKENS` | `70` | Oraciones finales repetidas en el chunk siguiente (solo intra-sección) |
| `UAO_RAG__CHUNK_MIN_TOKENS` | `80` | Chunks menores se fusionan con el anterior de la misma sección |
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

Variables de despliegue (F8/F9, las consume el compose/Caddy; NO llevan el
prefijo `UAO_RAG__` porque no son ajustes de `Settings`):

| Variable | Default | Uso |
|---|---|---|
| `SITE_ADDRESS` | `localhost` | Dominio del proxy Caddy; con dominio real, ACME emite y renueva el certificado |
| `TLS_DIRECTIVE` | `internal` | `internal` = CA propia de Caddy (demo/intranet) o correo de contacto para ACME |
| `IMAGE_TAG` | `0.1.0` | Etiqueta de las imágenes construidas por el compose |
| `APP_UID` / `APP_GID` | uid/gid del host | Exportados por el `Makefile` (`id -u`/`id -g`) y pasados como `build.args` para que el bind `./Data` sea escribible sin privilegios |
| `HF_HOME` | `/models` (en contenedor) | Caché del modelo de embeddings en el volumen `models_cache` (fijado también en el Dockerfile) |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` (en compose) | Endpoint del *tracking server* que consume la API; **si no se define, la instrumentación queda apagada** (F9) |
| `MLFLOW_EXPERIMENT_NAME` | `asistente-uao` (en compose) | Experimento donde MLflow agrupa las trazas del LLM (F9) |

> **F9 — dashboard de observabilidad**: se consume en
> `https://mlflow.<SITE_ADDRESS>` (subdominio servido por el proxy Caddy, como
> el chat). El servicio `mlflow` no publica puertos (`expose: "5000"`), guarda
> su backend SQLite y sus artefactos en el volumen `mlflow_data` y la API le
> envía las trazas por la red interna.

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

// Con el caché activado (F7), un hit por similitud devuelve exactamente
// este esquema: la respuesta guardada con sus fuentes y su "model"
// original, sin llamada al LLM (válido también para used_fallback=true).
```

`GET /health` → `{"status": "ok", "index_chunks": 1284, "device": "cuda",
"cache_hits": 0, "cache_misses": 0, "cache_hit_ratio": 0.0}` (los tres
últimos, implementados en F7: contadores del caché semántico).
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
- **En producción (F8)**: la REST se publica bajo el prefijo `/api` del proxy
  Caddy (`handle_path` elimina el prefijo; uvicorn arranca con `--root-path
  /api` para que la documentación OpenAPI quede en `https://<host>/api/docs`);
  dentro de la red interna el frontend habla directamente con
  `http://api:8000`.

---

## 4. Fases end-to-end (tareas, entregables, criterios de aceptación)

### Fase 0 — Entorno (✅ COMPLETADA)
`uv init`, `uv add` de todas las dependencias, `.env.example`, `.gitignore`,
validación de imports y de CUDA. Hecho (ver §0.1).

### Fase 1 — Ingesta con LlamaCloud Parse (extracción + limpieza) — ✅ COMPLETADA (2026-09-09)

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

### Fase 2 — Chunking + embeddings + indexación Chroma — ✅ COMPLETADA
| # | Tarea | Detalle técnico |
|---|---|---|
| 2.1 | `ingestion/chunk.py` | Entrada: `Data/Documentos_MD/*.md`. Chunking por párrafos a 300–500 tokens (`len//4` como proxy o tokenizador del modelo), overlap ~15%, con encabezados markdown (`##`/`###`) y límites de artículo/numeral como frontera dura; registra `section` (encabezado vigente) por chunk. *Implementado*: prefijo contextual `doc — section` incluido en el conteo, `chunk_max_tokens=450` como techo (por debajo del `max_seq_length` de E5), overlap solo intra-sección y fusión de chunks menores a `chunk_min_tokens=80`; parámetros en §3.1 |
| 2.2 | `embeddings.py` | `SentenceTransformer(settings.embedding_model, device=resolve_embedding_device())`; `encode(..., batch_size=32, normalize_embeddings=True)` → similitud coseno. *Implementado*: normalización y prefijos `query:`/`passage:` de E5; batch configurable (`UAO_RAG__EMBEDDING_BATCH_SIZE=16` en `.env.example`) |
| 2.3 | `vectorstore.py` | `PersistentClient(path=chroma_dir)`; `get_or_create_collection("uao_normativa", metadata={"hnsw:space": "cosine"})`; `upsert` con IDs SHA-256 |
| 2.4 | `scripts/ingest.py` | CLI: `Data/Documentos_MD/*.md` → chunks → Chroma; reporta nº de chunks/doc y tiempo; flag `--rebuild` (borra colección) |
| 2.5 | Tests | `chunk.py` con Chroma efímero (`EphemeralClient`) + modelo real de embeddings (test marcado `@pytest.mark.slow`) |

> **Nota (2026-09-17)**: los detalles de implementación del chunking
> (prefijo contextual, techo de 450 tokens, overlap intra-sección y fusión de
> chunks cortos) no se habían asentado en este plan; se documentan ahora con
> los valores reales del código. Las métricas del índice (1284 chunks / 20
> documentos, ~24 MB en `Data/chroma/`) son consistentes con lo registrado
> en F3 y F5.

**Criterio de aceptación F2**: ✅ COMPLETADA — índice reproducible (correr
el CLI dos veces no duplica chunks: upsert con ID SHA-256);
`collection.count()` estable en **1284 chunks / 20 documentos** (~24 MB en
`Data/chroma/`); humo con `scripts/smoke_retrieval.py` (banco de 10
preguntas + `--pregunta` ad-hoc). Las citas son (documento, sección): el
campo `page` desapareció con la migración a LlamaParse (§3.2).

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

**Notas F4 (complementarias, 2026-09-17)**: (a) la rotación de claves
(`CEREBRAS_API_KEYS`) cubre 429 **y también 401/403** (clave inválida); el
catálogo vigente según `.env.example` es `qwen-3.8-27b` (default) y
`gpt-oss-120b` como alternativa; (b) `collect_api_keys` en `core/llm.py`
centraliza la lectura de claves y la API responde 503 cuando no hay
ninguna configurada.

### Fase 5 — API: REST (consulta) + gRPC (control de ingesta) — ✅ COMPLETADA (2026-09-15)

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
| 5.6 | Docs | README (ejecución de ambas APIs), `.env.example` (`UAO_RAG__GRPC_ENABLED/PORT`); la plantilla de PR (`.github/PULL_REQUEST_TEMPLATE.md`, convención de Conventional Commits) se creó en esta fase y se sigue usando |

**Criterio de aceptación F5**: ✅ `curl -X POST /ask` → 200 con `answer` +
`sources[]` verificables (humo real: `/health` 1284 chunks/cuda,
`/documents` 20 docs, 422 en pregunta en blanco); `IndexStatus` por gRPC →
1284 chunks/20 documentos; suite `uv run pytest` 68/68 y ruff en verde
(95/95 desde F8, con los tests de caché/frontend/despliegue añadidos).
`GET /health` expone además los contadores del caché (`cache_hits`,
`cache_misses`, `cache_hit_ratio`).

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
| 7.1 | `src/asistente_agentico_uao/frontend/app.py` (chat) | Streamlit: `st.chat_input` + `st.chat_message`; cada pregunta viaja a `POST /ask` con `httpx` (`UAO_RAG__API_BASE_URL`, timeout explícito); `st.session_state` guarda el historial **de la sesión** solo para mostrarlo (no se reenvía como contexto al LLM: una vuelta); tema en `.streamlit/config.toml` (base oscura, color institucional `#2E5EAA`) |
| 7.2 | Respuesta y fuentes | Respuesta en markdown + `st.expander("Fuentes (n)")` con `doc_name`, `section`, `score` (2 decimales) y `excerpt`; si `used_fallback=true` → aviso ámbar «no hay información suficiente en la normativa» y **cero fuentes**; el backend ya descarta citas no verificables, la UI nunca las inventa |
| 7.3 | UX y robustez | `st.spinner` con estado mientras responde el LLM (~0.3-0.6 s medidos, pero sin pantalla congelada); límite de 500 caracteres con contador (contrato §3.3); preguntas de ejemplo (incluye jerga/typos, doc base §1.5); errores mapeados a mensajes claros: `422` (pregunta vacía/larga), `503` (sin clave de LLM), `500` y API caída (timeout/conexión) |
| 7.4 | Aviso y privacidad | Aviso visible «respuestas orientativas con la fuente oficial citada; no reemplaza la asesoría de Secretaría Académica»; el chat no solicita ni almacena datos personales (Ley 1581 de 2012, doc base §1.4): la caché guarda preguntas y respuestas, nunca identidades |
| 7.5 | `cache.py` (caché semántica) | `redis-py`: índice de preguntas + hash del embedding de la pregunta normalizada; **hit** si la similitud ≥ `UAO_RAG__CACHE_SIMILARITY` (se reutiliza el modelo E5 ya cargado, sin modelo extra); valor = `RagAnswer` serializado (respuesta, fuentes, `model`, `used_fallback`); `TTL` = `UAO_RAG__CACHE_TTL_SECONDS`; contadores hit/miss/ratio visibles en `/health`; **degradable**: sin Redis o con `CACHE_ENABLED=0` la API responde igual |
| 7.6 | Integración e invalidación | El caché se consulta en `service.py` (capa compartida REST/gRPC) **antes** de `answer_question`, así la REST y futuras integraciones lo aprovechan; `Ingest(rebuild)` y `PruneIndex` del servicio gRPC invalidan el caché, y las claves se versionan con la huella del índice (`count()` + timestamp) para no servir respuestas de un corpus viejo |
| 7.7 | Proxy inverso TLS | ✅ `docker/Caddyfile` (F8, 2026-09-17): terminación TLS automática en `:443` (`tls {$TLS_DIRECTIVE:internal}`, CA local o ACME), redirección HTTP→HTTPS y cabeceras de seguridad; `/` → Streamlit, `/api/*` → REST, `/healthz` → proxy; la API (:8000), gRPC (:50051) y Redis (:6379) quedan **solo** en la red interna del compose, sin publicarse al host |
| 7.8 | Pruebas | `tests/test_cache.py` con `fakeredis`: hit por similitud, miss, expiración por TTL, degradación sin Redis e invalidación por reindexado; `tests/test_frontend.py` para las funciones puras del cliente (llamada a la API y mapeo de errores con `httpx` mockeado). Humo manual: `uv run streamlit run
src/asistente_agentico_uao/frontend/app.py` contra la API local |
| 7.9 | (nice-to-have §2.3) Retroalimentación | Botones «útil / no útil» por respuesta (contador en Redis o log JSON); se muestra solo con `UAO_RAG__FEEDBACK_ENABLED=1`; sin datos personales. **Estado**: pendiente (nice-to-have); la infraestructura de contadores ya existe en `rag/cache.py` (hits/misses del caché) |

Nota 2026-09-17 (decisión 7.5): para el desarrollo local de cache.py se usa fakeredis (simula un Redis completo en memoria dentro del proceso de Python) en lugar de una instancia real de Redis, porque el entorno de desarrollo del frontend no tiene Docker instalado. La interfaz que expone redis-py (y que usa fakeredis para simularla) es la misma tanto en desarrollo como en producción, así que el paso a Fase 8 (docker-compose con Redis real) es solo un cambio de configuración —apuntar el cliente al contenedor real en vez de a la instancia simulada—, no un cambio de código en cache.py. Los tests de tests/test_cache.py corren igualmente sobre fakeredis, como ya estaba contemplado en el plan original.

**Criterio de aceptación F7**: `uv run streamlit run
src/asistente_agentico_uao/frontend/app.py` responde
preguntas reales mostrando la respuesta y sus fuentes; una pregunta fuera de
dominio muestra el estado de no-información **sin fuentes**; repetir una
pregunta frecuente (o su paráfrasis) se resuelve desde caché en ≲50 ms y se
contabiliza como `hit`; con Redis detenido la UI sigue operativa (solo más
lenta); el acceso público es HTTPS vía proxy y ni la API ni gRPC quedan
expuestos. Alineado con las tareas Kanban 008 (5 SP) y la parte de UI/Redis
de la 010.

### Fase 8 — Contenerización y documentación (solución completa) — ✅ COMPLETADA (2026-09-17)

| # | Tarea | Estado / artefacto |
|---|---|---|
| 8.1 | `Dockerfile` backend | ✅ `docker/Dockerfile`: multi-stage sobre `python:3.14-slim` (ARG `PYTHON_VERSION`, alineado con `.python-version`); `uv` copiado de la imagen oficial (`ghcr.io/astral-sh/uv:0.12.2`, nada de `pip`); capa de dependencias (`uv sync --frozen --no-dev --no-install-project`) separada de la del código (`uv sync --frozen --no-dev`); runtime con `libgomp1` (OpenMP de torch/onnxruntime), `ca-certificates`, usuario **sin privilegios** `app` (uid/gid del host vía `ARG APP_UID/APP_GID`) y `HEALTHCHECK` sobre `/health`. El índice **no se construye en el build**: `./Data` (índice + corpus + markdown) se monta desde el host y el modelo E5 se cachea en el volumen `models_cache` (`HF_HOME=/models`). Se fija `PYTHONPATH=/app/src` porque `core/config.py` deriva `PROJECT_ROOT` de `parents[3]`: importado desde `site-packages` apuntaría a `.venv/lib` |
| 8.2 | `Dockerfile` frontend | ✅ `docker/Dockerfile.frontend`: venv propio con `uv venv` + `uv pip install "streamlit==1.64.0" "httpx==0.28.1"` (versiones del `uv.lock`, sin torch/chromadb/sentence-transformers), usuario `app`, `HEALTHCHECK` sobre `/_stcore/health`, arranque `streamlit run frontend/app.py --server.headless=true` (con `WORKDIR /app`, el paquete se copia como `/app/frontend/`) |
| 8.3 | `docker-compose.yml` | ✅ servicios `proxy` (Caddy, **único** con puertos publicados 80/443), `frontend` (:8501 `expose`), `api` (:8000 REST + :50051 gRPC `expose`), `redis` (:6379 `expose`) — F9 añade el 5.º servicio `mlflow` (`:5000` `expose`); `env_file: .env` con `required: false`; `healthcheck` en los 4 servicios y `depends_on: condition: service_healthy` (api→redis, frontend→api, proxy→frontend); red `internal`; volúmenes `redis_data`, `models_cache`, `caddy_data`, `caddy_config` + bind `./Data:/app/Data`. El Redis real sustituye al `fakeredis` de desarrollo apuntando `UAO_RAG__REDIS_URL` a `redis://redis:6379/0` (solo configuración: `cache.py` no cambia) |
| 8.4 | README técnico + `Makefile` | ✅ `README.md` (arquitectura, requisitos, puesta en marcha, contratos REST/gRPC, estructura, pruebas, operación, seguridad) y `Makefile` reescrito autodocumentado (`make help` con secciones vía `##@`/`##`): entorno y calidad (`install`, `env-init`, `lint`, `format`, `check`, `test[-fast|-slow]`, `proto`), pipeline (`parse*`, `ingest*`), desarrollo (`api`, `grpc`, `frontend`, `ask-cli`, `smoke`), Docker (`config`, `build`, `up`, `down`, `restart`, `ps`, `logs*`, `shell`, `redis-cli`, `models-prefetch`, `health`, `ask`, `urls`, `grpc-url`), operación (`ingest-docker`, `grpc-status`, `cache-flush`, `cache-stats`) y respaldos/limpieza (`index-backup`, `index-restore`, `clean`, `clean-volumes`, `clean-images`, `disk`), conservando los alias previos (`docker-up`, `docker-down`, `redis-test`…) |
| 8.5 | Guía de despliegue | ✅ `docs/guia-despliegue.md` (13 secciones): arquitectura desplegada, requisitos, preparación, variables (app y compose), despliegue paso a paso con verificación, persistencia y **respaldos** (índice, certificados con advertencia sobre claves privadas, caché), **TLS** (CA interna de Caddy para local y ACME con renovación automática para dominio real), operación (actualizar, reindexar, invalidar caché, cron), **GPU opcional**, seguridad/privacidad (verificación de que `:8000`/`:50051`/`:6379`/`:8501` no se publican), solución de problemas (Docker y aplicación) y checklist de despliegue |

**Criterio de aceptación F8**:
- `uv run pytest`: ✅ **95/95** (80 previos + 15 nuevos de `tests/test_deployment.py`).
- `uv run ruff check src scripts tests` y `ruff format --check`: ✅ en verde.
- `docker compose config` (`make config`): ✅ válido; la interpolación confirma que
  **solo** `proxy` publica 80/443 y que api/gRPC/redis no salen del host.
- `docker compose up --build`: ⏳ **pendiente de ejecución por el usuario** — el
  socket de Docker del entorno de esta sesión no era accesible (el usuario no
  pertenece al grupo `docker` y `sudo` pide contraseña), así que el *build* no
  se pudo correr aquí. La configuración se validó de forma estática (compose,
  Dockerfiles, Caddyfile y `.dockerignore`) y con pruebas automatizadas; la
  ejecución real se cierra con `make up` (guía §5) sin cambios pendientes.

**Decisiones y notas F8 (2026-09-17)**:

1. **Proxy: Caddy en vez de nginx** (el plan §7.7 admitía `Caddyfile` o
   `nginx.conf`). Motivos: TLS **automático** (`tls {$TLS_DIRECTIVE:internal}`
   para la CA local, o el correo de ACME con renovación automática), soporte de
   *websockets* de Streamlit sin configuración extra y **cero certificados en el
   repositorio** (el intento previo versionaba `server.crt`). Se conserva la
   redirección `:80`→`:443` y se añaden cabeceras de seguridad (HSTS,
   `nosniff`, `X-Frame-Options`, `Referrer-Policy`).
2. **Rutas del proxy**: `/` → Streamlit, `/api/*` → REST (`handle_path` elimina
   el prefijo) y `/healthz` → 200 del proxy. La API arranca con `--root-path
   /api` para que la documentación OpenAPI funcione tras el prefijo. El
   healthcheck del contenedor usa un sitio interno `:8080` sin TLS, para evitar
   SNI y redirecciones.
3. **Índice por bind mount** (`./Data:/app/Data`) en lugar de copiarlo a la
   imagen: reutiliza el índice pre-construido, cumple «`down && up` conserva el
   índice» y permite reindexar dentro del contenedor (`make ingest-docker`). La
   guía documenta la variante con volumen nombrado (§6.4).
4. **Usuario sin privilegios con uid/gid del host**: evita los problemas de
   permisos del bind mount; el `Makefile` exporta `APP_UID`/`APP_GID`
   (`id -u`/`id -g`) y el compose los pasa como `build.args`.
5. **`make up` = build + healthchecks + precarga del modelo**: `docker compose
   up -d --build --wait` seguido de `make models-prefetch` (descarga el E5 al
   volumen `models_cache`, ~1,2 GB) para que la primera pregunta del usuario no
   pague la descarga ni el *timeout* de 30 s del cliente HTTP del frontend.
6. **Pruebas de despliegue** (`tests/test_deployment.py`, 15 tests, sin Docker
   ni red): servicios y red interna, ausencia de puertos internos publicados,
   healthchecks y dependencias, bind mount del índice + caché del modelo,
   conexiones internas (`redis://redis:6379/0`, `http://api:8000`), `uv sync
   --frozen`/`PYTHONPATH`/usuario sin privilegios en los Dockerfiles,
   `.dockerignore` (`.env`, `.venv`, `Data/chroma`), Caddyfile (TLS, rutas,
   healthcheck), `.env.example`, targets del `Makefile`, secciones de la guía,
   **ausencia de claves reales** (`csk-…`/`llx-…`) en los archivos versionados
   y `.env` ignorado por git y por el contexto de build. Añade `pyyaml` y
   `fakeredis` como dependencias de desarrollo (`uv add --dev`).
7. **Tamaño de imagen (riesgo asumido)**: `uv.lock` fija `torch` con wheels
   CUDA de PyPI (~4-5 GB de imagen) y `uv sync` no permite elegir el backend de
   torch (`--torch-backend` existe solo en la interfaz `uv pip`); el contenedor
   corre en CPU igualmente (`resolve_embedding_device` detecta que no hay GPU).
   La variante CPU-only exigiría añadir un índice de PyTorch al
   `pyproject.toml` y re-lockear (optimización futura, ver §6).
8. **Nota de operación**: no ejecutar la app local y el contenedor a la vez
   sobre el mismo `Data/chroma` (Chroma/SQLite bloquea la base); documentado en
   la guía junto con el resto de la solución de problemas.

---

### Fase 9 — Observabilidad del LLM con MLflow (fuera del alcance del doc base) — ✅ COMPLETADA (2026-09-18)

**Por qué no estaba en el plan**: `asistente-uao-rag.md` no contempla
observabilidad del modelo (trazas/experimentos); la Fase 8 cerró el despliegue
con 4 servicios y el invariante «solo el proxy publica puertos». Esta fase, ya
implementada en la rama `feat--ml_flow`, se documenta aquí **a posteriori** para
dejar el registro actualizado (el código existía sin sección en el plan).

**Concepto**: el RAG ya cita fuentes y los humos miden latencia, pero sin trazas
por llamada no se puede auditar qué prompt/contexto produjo cada respuesta.
MLflow da ese registro (prompt, respuesta, modelo, parámetros, tokens y
latencia) y un dashboard para comparar modelos y detectar regresiones. Es la
pieza de observabilidad de un sistema agéntico, no una funcionalidad del
estudiante: **la API responde igual si MLflow no está**.

| # | Tarea | Estado / artefacto |
|---|---|---|
| 9.1 | Dependencia `mlflow` | ✅ `pyproject.toml` / `uv.lock` (cliente 3.16.1, `uv add mlflow`) |
| 9.2 | Instrumentación de la API | ✅ `api/main.py`: si el entorno define `MLFLOW_TRACKING_URI`, se ejecuta `mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)` + `mlflow.openai.autolog()`. Funciona con Cerebras porque `ChatCerebras` hereda de `BaseChatOpenAI` y crea clientes `openai.OpenAI`/`AsyncOpenAI` hacia `api.cerebras.ai`. En local, sin la variable, ni siquiera se importa `mlflow` |
| 9.3 | Servicio de *tracking* | ✅ `docker-compose.yml`, servicio `mlflow` (`ghcr.io/mlflow/mlflow`): `mlflow server` con backend SQLite (`--backend-store-uri sqlite:////mlflow/mlflow.db`, `--workers 1` porque SQLite no admite varios escritores), artefactos en `/mlflow/artifacts`, `--allowed-hosts "*"`, `expose: "5000"` (red interna, **sin `ports`**) y `healthcheck` sobre `/health` con `depends_on: service_healthy` desde `api` |
| 9.4 | Dashboard por el proxy | ✅ `docker/Caddyfile`: sitio `mlflow.{$SITE_ADDRESS:localhost}` con `tls {$TLS_DIRECTIVE:internal}` y `reverse_proxy mlflow:5000`; se conserva el invariante de F8 (solo el proxy publica 80/443) |
| 9.5 | Persistencia | ✅ volumen `mlflow_data` (`/mlflow`): experimentos + trazas SQLite + artefactos; sobrevive a `down`/`up` |
| 9.6 | Operación | ✅ `Makefile`: `logs-mlflow` y `urls` (imprime el dashboard); `.env.example` documenta que el compose fija `MLFLOW_TRACKING_URI`/`MLFLOW_EXPERIMENT_NAME` |
| 9.7 | Pruebas | ✅ `tests/test_deployment.py` (+1 prueba, 16 en total): 5 servicios, `mlflow` sin puertos publicados, `expose 5000`, volumen `mlflow_data`, SQLite, healthcheck, `MLFLOW_TRACKING_URI=http://mlflow:5000` y `depends_on: service_healthy` en `api`, y la ruta `mlflow.{$SITE_ADDRESS:localhost}` del Caddyfile |
| 9.8 | Documentación | ✅ `README.md` (§1 observabilidad, §3.2 URLs, §3.3 atajos, §6 pruebas, §7 operación, §8 privacidad), `docs/guia-despliegue.md` (§4.2 variables, §5.1/§5.2 verificación, §6 respaldos, §10 seguridad y §11 troubleshooting) y esta fase del plan |

**Criterio de aceptación F9**:
- `uv run pytest`: ✅ **96/96** (95 previas + 1 nueva de MLflow) y `ruff` en verde.
- `docker compose config`: ✅ válido; la interpolación confirma que **solo**
  `proxy` publica puertos (`mlflow` queda con `expose: "5000"`).
- `https://mlflow.<SITE_ADDRESS>` abre el dashboard con el experimento
  `asistente-uao` y una traza por `POST /ask` (mensajes, tokens y latencia).
- Sin `MLFLOW_TRACKING_URI` (desarrollo local), la API responde igual:
  **degradable**.

**Decisiones y notas F9 (2026-09-18)**:

1. **`mlflow.openai.autolog()` y no `mlflow.langchain.autolog()`**: el efecto
   es equivalente porque `ChatCerebras` es un `BaseChatOpenAI` que construye el
   cliente `openai.OpenAI`; instrumentar el SDK captura también las llamadas
   reintentadas y la rotación de claves de `core/llm.py`. Si una versión futura
   de `langchain-cerebras` dejara de usar el SDK de OpenAI, la alternativa es
   `mlflow.langchain.autolog()` (traza el *Runnable* de LCEL).
2. **Opt-in por variable de entorno**: `create_app()` solo importa `mlflow` si
   existe `MLFLOW_TRACKING_URI`; en desarrollo local (`uv run uvicorn`) no hay
   trazado ni dependencia de red, y por eso los 96 tests no necesitan MLflow.
3. **La API espera a `mlflow`** (`depends_on: condition: service_healthy`): el
   precio de no perder las primeras trazas. Como la instrumentación es
   degradable, si se prioriza la disponibilidad sobre la observabilidad basta
   con relajar ese `depends_on` (quitarlo o usar `service_started`), sin tocar
   `main.py`. Queda registrado como riesgo en §6.
4. **El `ports: 127.0.0.1:5000:5000` se retiró**: contradecía el comentario del
   propio servicio («no publica puertos») y el invariante verificado de F8, que
   la suite comprueba. El dashboard se sirve por el proxy
   (`https://mlflow.<SITE_ADDRESS>`); si en desarrollo se quiere acceso directo
   sin TLS, re-añadir el mapeo **solo en loopback** y actualizar la prueba.
5. **`image: latest` todavía sin pin**: la convención de F8 es fijar versiones
   (`python:3.14-slim`, `uv:0.12.2`, `streamlit==1.64.0`). Aquí se dejó `latest`
   porque desde el entorno de la sesión no se pudo verificar el tag publicado
   del servidor (`ghcr.io/mlflow/mlflow:<versión>`); el pin a la versión del
   cliente (3.16.1) queda como pendiente en §6.
6. **Privacidad**: las trazas contienen la pregunta del estudiante y el
   contexto recuperado (normativa pública), pero ningún dato personal; quedan
   en el volumen `mlflow_data` del host y el servicio no sale de la red interna.
   Son independientes de la caché semántica (Redis no guarda trazas).

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
  (`uv run streamlit run src/asistente_agentico_uao/frontend/app.py` contra la
  API local).
- **Integración** (marcadas `slow`): embeddings reales + Chroma efímero;
  1 llamada real a Cerebras (opcional, tras `pytest -m "not slow"`).
- **Despliegue (F8/F9)**: `tests/test_deployment.py` (16 pruebas estáticas, sin
  Docker ni red) que parsean `docker-compose.yml` con `pyyaml` e inspeccionan
  Dockerfiles, Caddyfile, `.dockerignore`, `.env.example` y `Makefile`:
  los 5 servicios (api, frontend, proxy, redis y mlflow) y la red interna, que
  api/frontend/redis/mlflow no publiquen puertos, healthchecks y
  `depends_on: service_healthy`, bind `./Data` + `models_cache`, `uv sync
  --frozen` / `PYTHONPATH` / usuario sin privilegios, rutas TLS del proxy
  (incluido `mlflow.<SITE_ADDRESS>`), volumen `mlflow_data` y ausencia de claves
  reales versionadas.
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
| Imagen del backend pesada (~4-5 GB) por las wheels CUDA de `torch` fijadas en `uv.lock` | Verificado al preparar F8: `uv sync` no permite elegir backend de torch (`--torch-backend` solo existe en `uv pip`) | El contenedor funciona en CPU (`resolve_embedding_device` detecta que no hay GPU) y el modelo se cachea en el volumen `models_cache`; multi-stage evita la caché de uv en la imagen final; `make disk` para vigilar el consumo. Optimización futura: índice PyTorch CPU en `pyproject.toml` + re-lock |
| Regresión de configuración de despliegue (puerto interno publicado, secreto versionado, servicio sin healthcheck) | Riesgo detectado al sistematizar F8 | `tests/test_deployment.py` (16 pruebas estáticas) + `make config`; validan servicios, red interna, ausencia de `ports` en api/frontend/redis/mlflow, healthchecks, `.dockerignore` y ausencia de claves reales en archivos versionados |
| Observabilidad acoplada a la disponibilidad del core | F9: `api` declara `depends_on: service_healthy` de `mlflow`, así que un *tracking server* caído impide arrancar la API aunque la instrumentación sea opcional | Decisión consciente para no perder las primeras trazas; alternativa documentada (F9 nota 3): quitar el `depends_on` o usar `service_started`, ya que el código es degradable sin `MLFLOW_TRACKING_URI` |
| Imagen `ghcr.io/mlflow/mlflow:latest` sin pin | F9: la convención del proyecto es fijar versiones; `latest` puede cambiar el esquema del backend SQLite entre builds | Pin a la versión del cliente (3.16.1) tras verificar el tag con `make up`; respaldar el volumen `mlflow_data` antes de actualizar (guía §6) |
| Trazas del LLM (preguntas y contexto) persistidas en disco | F9: el volumen `mlflow_data` guarda prompt/respuesta de cada `/ask` | Servicio solo en la red interna y consumido por el proxy; sin datos personales (el chat no los pide) y desactivable no definiendo `MLFLOW_TRACKING_URI`; respaldo opcional (guía §6) |
| Discrepancia entre el entorno local y el contenedor en las rutas de datos | `core/config.py` deriva `PROJECT_ROOT` de `parents[3]`; importado desde `site-packages` apuntaría a `.venv/lib` | `PYTHONPATH=/app/src` en la imagen + `UAO_RAG__DOCS_DIR`/`MARKDOWN_DIR`/`CHROMA_DIR` explícitas en el compose, con bind `./Data:/app/Data` |

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
| F8 Docker+docs (backend+frontend+Redis+proxy) | 5 | ✅ hecho (2026-09-17) | F7 |
| F9 Observabilidad con MLflow (fuera del doc base) | 2 | ✅ hecho (2026-09-18) | F8 |
| **Total** | **38 SP** | **~9-12 sesiones** | |

La **ruta crítica** es F1 → F2 → F3 → F4: cualquier retrabajo en la
validación manual del texto limpio (F1.9) desplaza todo lo demás, por eso
es el checkpoint bloqueante actual. F7 solo depende de F5 (ya completada), así
que puede ejecutarse en paralelo con F6; F8 cierra el proyecto y depende de F7.
La carga del alcance del curso (36 SP) cuadra con el Kanban del documento base
(38 SP, tareas 008 «interfaz en Streamlit» y 010 «Docker + Redis + Proxy»);
F9 (2 SP) es un extra de observabilidad **fuera de ese alcance**, porque el doc
base no menciona MLflow. Con F7, F8 y F9 completadas, **queda pendiente F6**
(pruebas, evaluación y ajuste fino).

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
recalibración de `min_similarity`). Las Fases 5, 7 y 8 ya están implementadas y
verificadas: REST de consulta en `api/`, gRPC de ingesta en `grpc_impl/`, chat
Streamlit + caché semántica en `rag/cache.py` y la solución completa
contenerizada (`docker/Dockerfile*`, `docker-compose.yml`, `docker/Caddyfile`)
con su guía de despliegue (`docs/guia-despliegue.md`).

**Fase 8 ✅ COMPLETADA (2026-09-17)**: contenerización end-to-end y
documentación. `make up` levanta los servicios (proxy TLS con Caddy,
frontend Streamlit, API REST + gRPC y Redis; y desde F9, el *tracking server*
de MLflow) publicando solo 80/443, con
healthchecks encadenados, precarga del modelo de embeddings e índice por bind
mount (persistente entre `down`/`up`). Verificación disponible en esta sesión:
`docker compose config` válido, `uv run pytest` 95/95 y `ruff` en verde; el
`docker compose up --build` real queda por ejecutar por el usuario (el socket de
Docker no era accesible desde el entorno de la sesión). El siguiente paso
recomendado es `make up` + §5 de la guía, y **después F6** para cerrar la
calidad del RAG (métricas y recalibración del umbral).

**Ajustes estructurales y documentación (2026-09-17, misma rama)**: además de
la contenerización, esta rama movió la documentación a `docs/`
(`plan-trabajo-tecnico.md` y `asistente-uao-rag.md` viven ahora junto a
`guia-despliegue.md`) y consolidó el paquete en capas (`core/`, `rag/`,
`ingestion/`, `api/`, `grpc_impl/`, `frontend/`), sin dejar alias de
compatibilidad a propósito: una ruta vieja falla de inmediato en lugar de
romper en silencio en runtime. En F8 se actualizó el README técnico y se
añadió la guía de despliegue.

**Fase 9 ✅ COMPLETADA (2026-09-18, rama `feat--ml_flow`)**: observabilidad del
LLM con MLflow, que **no estaba en el plan** porque el documento base no la
contempla. La API instrumenta las llamadas al LLM
(`mlflow.openai.autolog()`, activo solo con `MLFLOW_TRACKING_URI`), el compose
añade el *tracking server* (`mlflow`, SQLite + volumen `mlflow_data`, sin
puertos publicados) y el dashboard se sirve por el proxy en
`https://mlflow.<SITE_ADDRESS>`. Verificación de esta sesión: `uv run pytest`
**96/96**, `ruff` en verde y `docker compose config` válido (solo `proxy`
publica 80/443; `mlflow` queda en `expose: "5000"`). Quedan como pendientes de
la fase el pin de la imagen del servidor y la decisión sobre el `depends_on`
de `api` (F9 notas 3 y 5, §6). El siguiente paso **sigue siendo F6**:
métricas de recuperación/generación y recalibración del umbral.
