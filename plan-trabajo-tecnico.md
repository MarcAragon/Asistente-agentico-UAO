# Plan de Trabajo Técnico — Asistente RAG UAO (Backend + API)

> Proyecto: asistente conversacional RAG para normativa institucional UAO.
> Alcance de este plan: **backend + API REST de preguntas** (sin frontend).
> Documento base: `asistente-uao-rag.md`. Fecha: 2026-09-08.
> **Actualización 2026-09-09**: la extracción (Fase 1) migró de PyMuPDF+EasyOCR
> local a **LlamaCloud Parse** (servicio agentic, salida markdown). Ver §0.4
> (decisiones 7-8), Fase 1 y el registro de riesgos §6.

---

## 0. Estado actual y hechos validados

### 0.1 Entorno ya construido (Fase 0 completada)

| Componente | Versión instalada | Estado |
|---|---|---|
| Python (gestionado por `uv`) | 3.13.7 | ✅ Validado |
| `uv` | 0.12.2 | ✅ Lockfile con 151 paquetes |
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
   (`resolve_embedding_device()` en `src/asistente_agentico_uao/config.py`):
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

---

## 1. Arquitectura objetivo

```text
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
                        │                                              │
 Petición HTTP ──────►  │  API (online, FastAPI)                       │
   POST /ask            │  retrieval.py ─► top-k + umbral similitud    │
                        │      ▼                                       │
                        │  chain.py (LCEL): prompt ─► llm.py ─► parse  │
                        │      llm.py = ChatCerebras(qwen-3.8-27b)     │
                        │      ▼                                       │
 Petición ◄───────────  │  respuesta + fuentes[] (doc, sección, texto) │
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
│   ├── __init__.py
│   ├── config.py                # Settings (pydantic-settings) + device resolver
│   ├── ingestion/               # FASE 2: chunk.py (chunking markdown 300-500 tokens)
│   ├── embeddings.py            # FASE 2: SentenceTransformer + device
│   ├── vectorstore.py           # FASE 2: cliente Chroma + upsert + colección
│   ├── retrieval.py             # FASE 3: retriever top-k + umbral
│   ├── llm.py                   # FASE 4: ChatCerebras + prompt de síntesis
│   ├── chain.py                 # FASE 4: cadena RAG LCEL completa
│   └── api/                     # FASE 5: main.py (app FastAPI) + schemas.py
├── tests/                       # pytest (unit + integración) — se crea en Fase 2
├── asistente-uao-rag.md         # Documento base del proyecto
├── plan-trabajo-tecnico.md      # Este plan
├── pyproject.toml / uv.lock     # gestión con uv (nunca pip)
└── .env / .env.example          # claves API; .env es copia de .env.example (gitignored)
```

> **Nota 2026-09-09**: el paquete real es `src/asistente_agentico_uao/`
> (no `uao_rag` como se planeó al inicio) y el corpus vive bajo `Data/`
> (`Data/Documentos/` los PDFs, `Data/Documentos_MD/` el markdown
> parseado). `Data/chroma/` y `tests/` aún no existen: se crean en Fase 2.

---

## 3. Contratos técnicos

### 3.1 Configuración (`src/asistente_agentico_uao/config.py`, prefijo `UAO_RAG__`)

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
- CORS abierto (`*`) para el futuro frontend.

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

### Fase 5 — API REST (FastAPI)
| # | Tarea | Detalle técnico |
|---|---|---|
| 5.1 | `api/schemas.py` | `AskRequest(question: str 1..500)`, `Source(doc_name, section, score, excerpt)`, `AskResponse(answer, sources, model, used_fallback)` |
| 5.2 | `api/main.py` | Lifespan: cargar `Retriever` + cadena una vez (no por request); `POST /ask`, `GET /health`, `GET /documents`; CORS `*`; manejo de errores 503/500 |
| 5.3 | Ejecución | `uv run uvicorn asistente_agentico_uao.api.main:app --host 0.0.0.0 --port 8000` |
| 5.4 | Tests | `fastapi.testclient.TestClient`: 422 validación, /health, /ask con cadena mockeada (sin gastar tokens) |

**Criterio de aceptación F5**: `curl -X POST /ask -d '{"question": "..."}'`
devuelve 200 con `answer` + `sources[]` verificables; validaciones y
errores según contrato §3.3.

### Fase 6 — Pruebas, evaluación y ajuste fino
| # | Tarea | Detalle técnico |
|---|---|---|
| 6.1 | Banco de preguntas | ~30 preguntas con respuesta conocida (doc + sección), cubriendo los 15 tipos de documento; incluye jerga/typos (simular preguntas reales) |
| 6.2 | Métricas de recuperación | Recall@5, MRR sobre el banco (script `scripts/evaluate_retrieval.py`) |
| 6.3 | Métricas de generación | RAGAS (`faithfulness`, `answer_relevancy`) o checklist manual si RAGAS no soporta Py3.13; exactitud de citas (% de citas que apuntan al doc/sección correctos) |
| 6.4 | Ajuste | Iterar: `top_k`, `min_similarity`, chunking, re-ranking (3.3), prompt |

**Criterio de aceptación F6**: reporte de métricas reproducible;
recall@5 ≥ 0.8 y citas exactas ≥ 80% como objetivo inicial.

### Fase 7 — Contenerización y documentación (alcance backend)
| # | Tarea | Detalle técnico |
|---|---|---|
| 7.1 | `Dockerfile` | Multi-stage sobre `python:3.13-slim`; instala `uv`; `uv sync --frozen`; el índice se pre-construye y se monta como volumen (`Data/chroma`) para no descargar el modelo de embeddings en cada build |
| 7.2 | `docker-compose.yml` | Servicio `api` (puerto 8000) + volumen `data/` + `env_file: .env`. (Redis/proxy quedan para fase de frontend) |
| 7.3 | README técnico | Instalación, ingesta, arranque, contrato API, arquitectura |
| 7.4 | `Makefile` o `justfile` | Atajos: `make ingest`, `make api`, `make test`, `make lint` |

**Criterio de aceptación F7**: `docker compose up --build` levanta la API
funcional consultable desde el host.

---

## 5. Estrategia de pruebas (resumen)

> (2026-09-09) Los tests del pipeline anterior (`tests/test_clean.py`,
> 14 tests) se eliminaron junto con `extract.py`/`clean.py`; los tests se
> re-crean fase a fase. Durante la migración se usó una prueba temporal
> inline de `clean_markdown` (ya borrada).

- **Unitarias** (rápidas, sin red/GPU): limpieza de markdown, chunking,
  schemas, umbrales, formateo de contexto. Mocks de LLM y embeddings.
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

---

## 7. Cronograma estimado (ruta crítica en negrita)

| Fase | SP | Duración est. | Dependencia |
|---|---|---|---|
| F0 Entorno | 2 | ✅ hecho | — |
| **F1 Parseo LlamaParse + limpieza ligera (validación manual)** | **5** | **1 sesión** | F0 |
| **F2 Chunking+embeddings+Chroma** | **5** | **1-2 sesiones** | **F1** |
| **F3 Recuperación** | **3** | **1 sesión** | **F2** |
| **F4 LLM+Cadena RAG** | **5** | **1 sesión** | F3 |
| F5 API FastAPI | 3 | 1 sesión | F4 |
| F6 Evaluación+ajustes | 3 | 1-2 sesiones | F5 |
| F7 Docker+docs | 3 | 1 sesión | F6 |
| **Total** | **29 SP** | **~7-9 sesiones** | |

La **ruta crítica** es F1 → F2 → F3 → F4: cualquier retrabajo en la
validación manual del texto limpio (F1.9) desplaza todo lo demás, por eso
es el checkpoint bloqueante actual.

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

**Siguiente: Fase 5** — API REST (FastAPI):
1. `api/schemas.py`: `AskRequest(question: str 1..500)`, `Source(doc_name,
   section, score, excerpt)` (ya existe como dataclass en `chain.py`;
   definir la versión Pydantic), `AskResponse(answer, sources, model,
   used_fallback)`.
2. `api/main.py`: lifespan que carga `Retriever` + LLM una vez; `POST /ask`
   (200 con `RagAnswer`, 503 sin `CEREBRAS_API_KEY`), `GET /health`
   (`index_chunks`, `device`), `GET /documents`; CORS `*`.
3. Ejecutar con `uv run uvicorn asistente_agentico_uao.api.main:app
   --host 0.0.0.0 --port 8000` y validar con `curl`.
