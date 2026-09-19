# Asistente agéntico RAG — Asistente RAG UAO

Asistente conversacional que responde preguntas sobre la **normativa
institucional de la Universidad Autónoma de Occidente** (reglamentos,
resoluciones, calendarios, políticas) usando **RAG** (retrieval-augmented
generation): recupera los fragmentos pertinentes de los documentos oficiales
y sintetiza una respuesta **citando siempre la fuente** (documento + sección).
Cuando la normativa indexada no responde la pregunta, el asistente responde
explícitamente que no tiene información suficiente: **nunca inventa**.

Proyecto del curso de *Sistemas Agénticos* (UAO). El documento base del
alcance es [`asistente-uao-rag.md`](asistente-uao-rag.md) y el plan técnico
por fases es [`plan-trabajo-tecnico.md`](plan-trabajo-tecnico.md).

---

## 1. Arquitectura

```text
                        ┌──────────────────────────────────────────────┐
 Estudiante ──────────►│   FRONTEND WEB (Streamlit)                    │
   (navegador)          │   chat · respuesta + fuentes citadas         │
                        │   └─ HTTP/JSON (solo REST /ask, nunca gRPC)  │
                        └───────────────────┬──────────────────────────┘
                                            │ proxy inverso TLS (:443)
                        ┌───────────────────▼──────────────────────────┐
                        │  CACHÉ SEMÁNTICA (Redis)                     │
                        │   hit por similitud de la pregunta → TTL     │
                        └───────────────────┬──────────────────────────┘
                                            ▼
                        ┌──────────────────────────────────────────────┐
                        │              BACKEND                         │
 Data/Documentos/*.pdf►│  INGESTA (offline, CLI o gRPC)               │
   20 PDFs oficiales    │  scripts/llama_cloud_parsing.py              │
                        │   └─ LlamaCloud Parse (agentic, API nube)    │
                        │      └─ Data/Documentos_MD/*.md (limpio)     │
                        │         └─ ingestion/chunk.py                │
                        │  core/embeddings.py (E5 multilingüe, CPU/GPU)│
                        │      ▼                                       │
                        │  core/vectorstore.py ─► ChromaDB persistente │
                        │      Data/chroma/ (metadata: doc, sección)   │
                        │      ▲                                       │
                        │  gRPC :50051 (control plane)                 │
                        │   Ingest(stream) · PruneIndex · IndexStatus  │
                        │                                              │
 Frontend ───────────►  │  REST :8000 (data plane)                     │
   POST /ask            │  rag/retrieval.py ─► top-k + umbral          │
                        │      ▼                                       │
                        │  rag/cache.py ─► Redis (hit por similitud)   │
                        │      ▼                                       │
                        │  rag/chain.py (LCEL): prompt ─► llm.py ─► …  │
                        │      llm.py = ChatCerebras(qwen-3.8-27b)     │
                        │      ▼                                       │
 Respuesta ◄──────────  │  respuesta + fuentes[] (doc, sección, texto) │
                        └──────────────────────────────────────────────┘
```

**Dos planos de servicio, sin duplicar funciones:**

| Plano | Protocolo | Puerto | Rol |
|---|---|---|---|
| Datos (consulta) | REST/HTTP | `:8000` | `POST /ask`, `GET /health`, `GET /documents`. Solo **lee** el índice y llama al LLM. Es lo único que consume el chat. |
| Control (ingesta) | gRPC | `:50051` | `Ingest` (progreso en streaming), `PruneIndex`, `IndexStatus`. Máquina a máquina; **nunca** se expone al navegador. |

**Capas del paquete** (`src/asistente_agentico_uao/`) con regla de
dependencias `core ← rag/ingestion ← api/grpc_impl/frontend`:

| Capa | Módulos | Responsabilidad |
|---|---|---|
| `core/` | `config.py`, `embeddings.py`, `vectorstore.py`, `llm.py` | Infraestructura y adaptadores externos (settings, E5, Chroma, Cerebras). |
| `rag/` | `retrieval.py`, `chain.py`, `cache.py`, `service.py` | Dominio RAG: recuperación, cadena LCEL, caché semántica, estado compartido. |
| `ingestion/` | `chunk.py`, `pipeline.py` | Chunking por encabezados y pipeline de indexación (compartido por CLI y gRPC). |
| `api/`, `grpc_impl/`, `frontend/` | `main.py`, `servicer.py`, `app.py`… | Capa de entrada (REST, gRPC, UI Streamlit). |
| `scripts/` | `llama_cloud_parsing.py`, `ingest.py`, `ask.py`… | CLIs de operación y humo. |

---

## 2. Requisitos

| Herramienta | Versión | Uso |
|---|---|---|
| [`uv`](https://docs.astral.sh/uv/) | ≥ 0.12 | Gestión del entorno y de dependencias (**nunca `pip`/`uv pip`**). |
| Python | 3.14 (fijado en `.python-version`) | El entorno lo instala/gestiona `uv`. |
| Docker Engine + Compose v2+ | ≥ 24 / ≥ 2.20 (`--wait`) | Despliegue de la Fase 8 (`docker-compose.yml`). |
| GPU NVIDIA/AMD | opcional | Acelera los *embeddings* E5; sin GPU se usa CPU automáticamente. |

Claves de API necesarias en `.env` (copia de `.env.example`, **gitignored**):

- `CEREBRAS_API_KEY` — generación de la respuesta (`qwen-3.8-27b`). Sin ella la
  API arranca pero `/ask` responde `503`. Se admiten claves extra separadas por
  coma en `CEREBRAS_API_KEYS` con rotación automática ante cuota agotada.
- `LLAMA_CLOUD_API_KEY` — solo para parsear PDFs nuevos (Fase 1).
---

## 3. Puesta en marcha

### 3.1 Desarrollo local (con `uv`)

```bash
make env-init        # crea .env desde .env.example (completa las API keys)
make install         # uv sync: replica el entorno desde uv.lock
make test            # suite completa de pytest
make api             # API REST en http://localhost:8000 (gRPC embebido :50051)
make frontend        # chat Streamlit en http://localhost:8501 (otra terminal)
```

Flujo de datos (una sola vez, cuando cambie el corpus):

```bash
make parse           # PDFs → LlamaCloud Parse → Data/Documentos_MD/*.md
make ingest          # markdown → chunks → embeddings E5 → Data/chroma
```

### 3.2 Solución completa con Docker (recomendado para desplegar)

```bash
make env-init        # .env con las claves (se inyecta por env_file, no se copia a la imagen)
make up              # build + healthchecks + precarga del modelo de embeddings
make urls            # URLs de acceso
make down            # apaga (conserva índice, caché, certificados y modelos)
```

- Chat: **https**://localhost/ · API: https://localhost/api/health ·
  documentación OpenAPI: https://localhost/api/docs
- Únicos puertos publicados al host: **80** (redirección HTTP→HTTPS y ACME) y
  **443** (TLS). La API, gRPC, Streamlit y Redis quedan **solo** en la red
  interna del compose.
- Certificado TLS: por defecto Caddy usa su **CA interna** (válido para
  `localhost`/intranet, el navegador avisa una vez). Para un dominio real,
  define `SITE_ADDRESS` y `TLS_DIRECTIVE` en `.env` y Caddy emite y renueva el
  certificado por ACME automáticamente.

Detalles, verificación paso a paso, respaldos y solución de problemas:
**[`docs/guia-despliegue.md`](docs/guia-despliegue.md)**.

### 3.3 Atajos del `Makefile`

`make` (o `make help`) lista todos los targets agrupados:

| Grupo | Targets |
|---|---|
| Entorno y calidad | `install`, `sync`, `env-init`, `lint`, `format`, `check`, `test`, `test-fast`, `test-slow`, `proto` |
| Pipeline de datos | `parse`, `parse-redo`, `parse-file FILE=…`, `ingest`, `ingest-rebuild`, `ingest-prune` |
| Desarrollo local | `api`, `grpc`, `frontend`, `ask-cli`, `smoke` |
| Docker | `config`, `build`, `up`, `down`, `restart`, `ps`, `logs[-api|-frontend|-proxy|-redis]`, `shell`, `redis-cli`, `models-prefetch`, `health`, `ask`, `urls` |
| Operación | `ingest-docker`, `ingest-rebuild-docker`, `grpc-status`, `cache-flush`, `cache-stats` |
| Respaldos y limpieza | `index-backup`, `index-restore FILE=…`, `clean`, `clean-volumes`, `clean-images`, `disk` |

---

## 4. Contratos

### 4.1 API REST (data plane, `:8000`)

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

| Endpoint | Descripción | Códigos |
|---|---|---|
| `POST /ask` | Pregunta (1..500 caracteres) → respuesta + fuentes verificables. | `200`, `422` (validación), `503` (sin `CEREBRAS_API_KEY`), `500` |
| `GET /health` | Estado, chunks indexados, `device` y contadores de caché (hits/misses/ratio). | `200` |
| `GET /documents` | Documentos indexados con su número de chunks (trazabilidad). | `200` |

Reglas: sin fragmentos sobre el umbral, o si el LLM juzga que el contexto no
responde, la respuesta es el mensaje de no-información con `sources: []` y
`used_fallback: true`. Las citas que no se pueden verificar contra el contexto
recuperado se descartan: el sistema **no inventa fuentes**.

### 4.2 gRPC `IndexAdmin` (control plane, `:50051`)

| RPC | Efecto |
|---|---|
| `Ingest(IngestRequest) → stream IngestProgress` | Indexa markdown (opcionalmente `rebuild`), emitiendo progreso por documento (chunks, tokens, segundos). Idempotente. |
| `PruneIndex(Empty) → PruneResult` | Borra los chunks de documentos que ya no están en `Data/Documentos_MD`. |
| `IndexStatus(Empty) → IndexStatusResponse` | Chunks, documentos, modelo de embeddings y `device`. |

Contrato en `src/asistente_agentico_uao/grpc_impl/protos/index_admin.proto`;
stubs committeados (se regeneran con `make proto`). Cliente de humo:
`python scripts/ingest_client.py [--status|--rebuild|--prune|--file X]`.

### 4.3 Configuración

Todas las variables de la app usan el prefijo `UAO_RAG__` y se documentan en
`.env.example` (tabla completa en
[`plan-trabajo-tecnico.md`](plan-trabajo-tecnico.md) §3.1). Las variables sin
prefijo (`SITE_ADDRESS`, `TLS_DIRECTIVE`, `IMAGE_TAG`, `APP_UID`) son del
`docker compose`, no de la aplicación.

---

## 5. Estructura del repositorio

```text
Asistente-agentico-UAO/
├── Data/
│   ├── Documentos/              # Corpus original (20 PDFs) — solo lectura
│   ├── Documentos_MD/           # *.md limpios (salida LlamaCloud Parse)
│   └── chroma/                  # Índice vectorial ChromaDB (gitignored, regenerable)
├── docker/
│   ├── Dockerfile               # Imagen del backend (multi-stage, uv, no-root)
│   ├── Dockerfile.frontend      # Imagen ligera de Streamlit (sin ML)
│   └── Caddyfile                # Proxy inverso TLS, websockets y /api/*
├── docker-compose.yml           # redis + api + frontend + proxy (solo 80/443 publicados)
├── .dockerignore                # Excluye .env, .venv, cachés y datos del contexto de build
├── docs/
│   └── guia-despliegue.md       # Guía de despliegue y operación (Fase 8)
├── scripts/                     # CLIs: parseo, ingesta, humo, proto, cliente gRPC
├── src/asistente_agentico_uao/  # Paquete (core · rag · ingestion · api · grpc_impl · frontend)
├── tests/                       # pytest: unitarias, API/gRPC, caché, frontend y despliegue
├── Makefile                     # Atajos de desarrollo, Docker y operación
├── pyproject.toml / uv.lock     # Dependencias gestionadas con uv
├── .env / .env.example          # Secretos locales (gitignored) / plantilla
├── asistente-uao-rag.md         # Documento base del alcance
└── plan-trabajo-tecnico.md      # Plan de trabajo por fases
```

---

## 6. Pruebas y estilo

```bash
make test            # uv run pytest  → suite completa (rápida, sin red ni GPU)
make test-fast       # uv run pytest -m "not slow"
make test-slow       # integración real: embeddings E5 + Chroma efímero
make lint            # uv run ruff check src scripts tests
make format          # uv run ruff format + --fix
make check           # lint + test
```

Cobertura de la suite: chunking, limpieza de markdown, configuración, API REST
(`TestClient`), gRPC en proceso, caché semántica (`fakeredis`), cliente HTTP del
frontend, cadena RAG (con LLM/embeddings mockeados) y **validación estática del
despliegue** (`tests/test_deployment.py`: compose, Dockerfiles, `.dockerignore`
y ausencia de secretos en los archivos versionados).

---

## 7. Operación

| Necesidad | Comando |
|---|---|
| Ver estado y salud | `make ps`, `make health`, `make cache-stats` |
| Ver logs de un servicio | `make logs-api` / `logs-frontend` / `logs-proxy` / `logs-redis` |
| Reindexar el corpus | `make ingest-docker` (incremental) o `make ingest-rebuild-docker` (limpio) |
| Invalidar el caché semántico | `make cache-flush` (también se invalida tras un rebuild vía gRPC) |
| Estado del índice por gRPC | `make grpc-status` |
| Respaldar / restaurar el índice | `make index-backup`, `make index-restore FILE=…` |
| Entrar al contenedor | `make shell` (API), `make redis-cli` (caché) |

La guía de despliegue detalla persistencia, respaldos, renovación de
certificados y verificación de que `:8000`, `:50051` y `:6379` no quedan
publicados al host.

---

## 8. Seguridad y privacidad

- Los secretos viven solo en `.env` (gitignored) o en variables de entorno;
  nunca en el repositorio, las imágenes ni los logs.
- Superficie pública mínima: solo el proxy TLS (80/443). API, gRPC y Redis sin
  puertos publicados.
- El chat no solicita ni almacena datos personales (Ley 1581 de 2012): la caché
  guarda preguntas y respuestas anonimizadas, nunca identidades.
- Las respuestas son orientativas, citan la fuente oficial y no reemplazan la
  asesoría de Secretaría Académica o Bienestar Universitario.
- Al usar LlamaCloud Parse, los PDFs institucionales (normativa pública) se
  suben por API y el archivo remoto se elimina tras el parseo.

---

## 9. Referencias

- [`asistente-uao-rag.md`](asistente-uao-rag.md) — documento base (alcance, objetivos, stack).
- [`plan-trabajo-tecnico.md`](plan-trabajo-tecnico.md) — plan por fases, contratos y riesgos.
- [`docs/guia-despliegue.md`](docs/guia-despliegue.md) — despliegue, TLS, respaldos y troubleshooting.

