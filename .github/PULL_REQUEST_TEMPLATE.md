# 📋 Plantilla de Pull Request — Asistente Agéntico RAG UAO

<!--
Plantilla pre-llenada con el contexto de la rama feat--refactor+docker
(F8 — Contenerización y documentación + refactor de estructura en capas).
Instrucciones:
- Completa TODAS las secciones. Si una no aplica, escribe "N/A" y justifica.
- El título de la PR debe seguir Conventional Commits:
  feat: | fix: | refactor: | docs: | chore: | test:
  Ej: "refactor: estructura en capas del paquete + contenerización Docker (F8)"
- Elimina los comentarios HTML antes de abrir la PR.
-->

## 1. 🧩 Tipo de Pull Request

**Título sugerido:** `refactor: estructura en capas del paquete y contenerización Docker de la solución completa (Fase 8)`

Marca **uno** como principal (y otros si esta PR es mixta):

- [ ] `feat` — Nueva funcionalidad (pipeline, ingesta, recuperación, API…)
- [ ] `fix` — Corrección de un bug
- [x] `refactor` — Cambio interno sin alterar comportamiento externo (principal)
- [ ] `docs` — Solo documentación
- [x] `chore` — Infraestructura, dependencias, configuración (Docker, Makefile, empaquetado)
- [x] `test` — Adición de pruebas (`tests/test_deployment.py`)

---

## 2. 🧱 Módulo o Componente Afectado

Selecciona todas las áreas que toca esta PR:

- [x] **Configuración central** (`src/asistente_agentico_uao/core/config.py`, `.env.example`, `.gitignore`)
- [ ] **Preprocesamiento de documentos** (`scripts/llama_cloud_parsing.py`, `Data/Documentos_MD/`)
- [ ] **Ingesta & Chunking** (índice de embeddings, Chroma)
- [ ] **Recuperación & Embeddings** (`sentence-transformers`, `chromadb`)
- [ ] **Generación / LLM** (`langchain-cerebras`, modelos de Cerebras)
- [x] **API / Servicio** (`fastapi`, `uvicorn`, `grpc`) — mismos endpoints; solo cambian imports por el refactor
- [x] **Scripts & CLI** (`scripts/`) — ajustes de imports (sin cambios de comportamiento)
- [x] **Pruebas Unitarias** (`tests/`, incl. la nueva `tests/test_deployment.py`)
- [x] **Infraestructura & Empaquetado** (`docker/`, `docker-compose.yml`, `Makefile`, `.dockerignore`, `pyproject.toml`, `uv.lock`)
- [x] **Documentación** (`README.md`, `docs/plan-trabajo-tecnico.md`, `docs/guia-despliegue.md`)

---

## 3. 📚 Descripción de los Cambios

**Tarea / fase asociada:** Fase 8 del `docs/plan-trabajo-tecnico.md`
(contenerización y documentación, tareas 8.1–8.5) + ajuste estructural del
paquete en capas (commits 9e87335/f3b32f3/e78ee54, con revert intermedio
863114b).

#### 🎯 Motivación y contexto
Con las Fases 5 (APIs REST/gRPC) y 7 (frontend Streamlit, caché semántica en
Redis y proxy TLS) implementadas y verificadas, faltaba empaquetar la
solución completa para su despliegue: imágenes reproducibles y orquestación
de los 4 servicios (proxy TLS, frontend, API+gRPC, Redis) con una única
superficie pública (80/443). Antes de contenerizar se consolidó la estructura
del paquete en capas, para que las Dockerfiles copien módulos con límites
claros de dependencia.

#### 📝 Resumen de cambios

**A. Refactor de estructura en capas** (sin cambio de comportamiento):
- `src/asistente_agentico_uao/core/`: `config.py`, `embeddings.py`, `llm.py`, `vectorstore.py` (infraestructura y adaptadores externos).
- `src/asistente_agentico_uao/rag/`: `retrieval.py`, `chain.py`, `cache.py`, `service.py` (dominio RAG).
- `src/asistente_agentico_uao/ingestion/`: `chunk.py` + `pipeline.py`, ahora con el `__init__.py` que faltaba (era namespace package).
- Regla de dependencias: `core` no importa de `rag`; `rag` e `ingestion` solo importan de `core`; `api`, `grpc_impl` y `frontend` son la capa de entrada.
- Imports actualizados en `src/`, `scripts/` y `tests/` (p. ej. `asistente_agentico_uao.core.config`, `asistente_agentico_uao.rag.chain`). **Sin alias de compatibilidad a propósito**: una ruta vieja falla de inmediato en lugar de romper en silencio en runtime.
- `ruff format` aplicado a todo el repo (43 archivos).

**B. Contenerización (F8)**:
- **`docker/Dockerfile` (backend)**: multi-stage sobre `python:3.14-slim`; `uv` copiado de la imagen oficial (`ghcr.io/astral-sh/uv`, nada de `pip`); `uv sync --frozen` por capas (dependencias separadas del código); runtime con `libgomp1` (OpenMP de torch/onnxruntime) y `ca-certificates`; usuario sin privilegios `app` (uid/gid del host vía `ARG APP_UID/APP_GID`); `HEALTHCHECK` sobre `/health`; el índice NO se construye en el build (bind mount `./Data`); `HF_HOME=/models` para cachear el modelo E5 en volumen; `PYTHONPATH=/app/src`; uvicorn con `--root-path /api` (OpenAPI funcional tras el prefijo del proxy).
- **`docker/Dockerfile.frontend`**: venv propio con solo `streamlit==1.64.0` + `httpx==0.28.1` (sin torch/chromadb/sentence-transformers), usuario `app`, `HEALTHCHECK` sobre `/_stcore/health`, arranque headless.
- **`docker-compose.yml`**: `redis` (AOF activado, healthcheck), `api` (REST :8000 + gRPC :50051 con `expose`, bind `./Data:/app/Data`, volumen `models_cache`, `UAO_RAG__REDIS_URL=redis://redis:6379/0`), `frontend` (:8501 `expose`, solo habla con `http://api:8000`) y `proxy` (Caddy, ÚNICO servicio con `ports` 80/443); red `internal`; `env_file` opcional (`required: false` para `make config` en un clon limpio); healthchecks encadenados con `depends_on: condition: service_healthy` (api→redis, frontend→api, proxy→frontend).
- **`docker/Caddyfile`**: TLS automático (`tls {$TLS_DIRECTIVE:internal}` — CA local de Caddy o ACME con renovación automática), redirección HTTP→HTTPS, `/` → Streamlit (websockets sin config extra), `/api/*` → REST (sin el prefijo), `/healthz`; sitio interno `:8080` sin TLS para el healthcheck del contenedor; cabeceras de seguridad (HSTS, nosniff, X-Frame-Options, Referrer-Policy). Cero certificados versionados.
- **`.dockerignore`**: excluye `.env` (los secretos solo entran por `env_file` en runtime), `.venv`, cachés y `Data/` (se monta en runtime).
- **`Makefile`** reescrito y autodocumentado (`make help` con secciones `##@`): entorno y calidad (`install`, `env-init`, `lint`, `format`, `check`, `test[-fast|-slow]`, `proto`), pipeline (`parse*`, `ingest*`), desarrollo (`api`, `grpc`, `frontend`, `ask-cli`, `smoke`), Docker (`config`, `build`, `up`, `down`, `logs*`, `shell`, `models-prefetch`, `health`, `ask`, `urls`), operación (`ingest-docker`, `grpc-status`, `cache-flush`, `cache-stats`) y respaldos/limpieza (`index-backup`, `index-restore`, `clean*`, `disk`).
- **`tests/test_deployment.py`** (15 pruebas estáticas, **sin Docker ni red**): compose (servicios, red interna, ausencia de puertos internos publicados, healthchecks y dependencias), Dockerfiles (`uv sync --frozen`, `PYTHONPATH`, usuario sin privilegios), `.dockerignore`, Caddyfile (TLS y rutas), `.env.example` y ausencia de claves reales (`csk-…`/`llx-…`) versionadas.
- **`pyproject.toml`/`uv.lock`**: `pyyaml` y `fakeredis` añadidos como dependencias de desarrollo (parseo del compose en los tests).
- **`README.md`** técnico (arquitectura, requisitos, contratos REST/gRPC, estructura, pruebas, operación y seguridad) + **`docs/guia-despliegue.md`** (13 secciones: despliegue paso a paso, TLS, respaldos, GPU opcional, seguridad y solución de problemas).
- Movimiento de la documentación a `docs/` (`plan-trabajo-tecnico.md`, `asistente-uao-rag.md`).

#### 🔧 Detalles técnicos relevantes
- **Una sola superficie pública**: solo el proxy Caddy publica 80/443; API (:8000), gRPC (:50051) y Redis (:6379) quedan en la red interna del compose (`expose`, nunca `ports`) — verificado por pruebas automatizadas.
- **Índice por bind mount**: `down && up` conserva el índice y permite reindexar dentro del contenedor (`make ingest-docker`); el modelo E5 se descarga una sola vez al volumen `models_cache` (`make up` = build + `--wait` + `models-prefetch`).
- **No romper el desarrollo local**: `fakeredis` (misma interfaz de redis-py) en desarrollo vs. Redis real en compose; el paso es solo configuración (`UAO_RAG__REDIS_URL`), sin tocar `rag/cache.py`.
- **Tamaño de imagen (riesgo asumido)**: `uv.lock` fija wheels CUDA de `torch` (~4-5 GB); el contenedor corre en CPU (`resolve_embedding_device` detecta que no hay GPU). La variante CPU-only (índice PyTorch en `pyproject.toml` + re-lock) queda como optimización futura (§6 del plan).
- El healthcheck del proxy usa un sitio interno `:8080` sin TLS para evitar SNI y redirecciones en la sonda.

---

## 4. ⚠️ Impacto y Compatibilidad

- [x] Esta PR **NO** rompe compatibilidad (cambios retrocompatibles)
- [ ] Esta PR introduce **cambios que rompen** (documentar abajo)

**Si hay breaking changes / migración, documenta:** N/A a nivel de
comportamiento: los endpoints REST/gRPC, sus contratos y el resultado de las
pruebas se mantienen idénticos. **Nota de rutas internas (consciente)**: los
imports cambian (`asistente_agentico_uao.config` →
`...core.config`; `...chain` → `...rag.chain`; etc.) y NO se dejaron alias de
compatibilidad a propósito; afecta solo a código externo que importara el
paquete directamente (los tests y scripts del repo ya están actualizados).
Operativamente, `make up` sustituye al arranque manual de
uvicorn/streamlit.

---

## 5. 🔐 Seguridad y Secretos

- [x] Ningún secreto (API keys, contraseñas, tokens) está escrito en el código ni en archivos rastreados
- [x] Los secretos viven únicamente en `.env` (gitignored) o en variables de entorno; además `.env` está excluido del contexto de build (`.dockerignore`) y entra al contenedor solo por `env_file` en runtime
- [x] `.env.example` se mantiene vigente (incluye las variables de despliegue `SITE_ADDRESS`, `TLS_DIRECTIVE`, `IMAGE_TAG`)
- [x] Solo el proxy Caddy publica puertos al host (80/443); API (:8000), gRPC (:50051) y Redis (:6379) quedan en la red interna — verificado por `tests/test_deployment.py`
- [x] TLS automático en :443 (CA interna de Caddy para demo/intranet, o ACME con renovación automática para dominio real) + cabeceras de seguridad; cero certificados versionados
- [x] `tests/test_deployment.py` verifica la ausencia de claves reales (`csk-…`/`llx-…`) en archivos versionados y que `.env` sea ignorado por git y por el contexto de build

---

## 6. ✅ Lista de Chequeo Pre-PR (Estándares del Curso UAO)

- [x] **Entorno de ejecución (`uv`):** Todo se ejecutó y probó con `uv` y Python 3.14; las imágenes se construyen con la imagen oficial de `uv` (nunca `pip`).
- [x] **Sin warnings:** `uv run pytest` corre limpio (95/95 en ~10,5 s; 1 warning de deprecación interna de starlette/anyio, fuera del alcance de esta PR).
- [x] **Lint (`ruff`):** `uv run ruff check src scripts tests` y `uv run ruff format --check` pasan al 100 % (43 archivos formateados).
- [x] **Control de exclusiones (`.dockerignore`/`.gitignore`):** `.env`, `.venv`, cachés y `Data/` fuera del contexto de build; docs y artefactos fuera de git.
- [x] **Pruebas unitarias (`pytest`):** `uv run pytest` pasa al 100 %: **95/95** (80 previas + 15 nuevas de `tests/test_deployment.py`).
- [x] **Validación de compose:** `docker compose config` (`make config`) válido; la interpolación confirma que solo `proxy` publica 80/443.
- [x] **Clean Code:** capas con reglas de dependencia (`core`/`rag`/`ingestion`/capas de entrada), `make help` autodocumentado y comentarios de intención en Dockerfiles/compose/Caddyfile.
- [x] **Reproducibilidad:** build con `uv sync --frozen` sobre `uv.lock` versionado; `make env-init` crea `.env` desde `.env.example` en el primer arranque.
- [ ] **Build real de contenedores (`docker compose up --build`):** ⏳ pendiente de ejecución por el usuario — el socket de Docker no era accesible desde el entorno de esta sesión (usuario fuera del grupo `docker`). La configuración quedó validada estáticamente (`make config` + 15 pruebas); cerrar con `make up` (guía de despliegue §5).

---

## 7. 🧪 Evidencia de Pruebas Ejecutadas

```
# Suite completa:
$ uv run pytest -q
...............................................................           [100%]
95 passed, 1 warning in 10.53s

# Lint y formato:
$ uv run ruff check src scripts tests
All checks passed!
$ uv run ruff format --check src scripts tests
43 files already formatted

# Validación estática del compose (interpolación incluida):
$ docker compose config --quiet && echo COMPOSE_CONFIG_OK
COMPOSE_CONFIG_OK
```

---

## 8. 👀 Notas para el Revisor

- El refactor de capas se hizo **antes** de contenerizar (commits 9e87335 → f3b32f3 → e78ee54, con revert intermedio 863114b), de modo que las Dockerfiles copian la estructura final del paquete.
- La imagen del backend es grande (~4-5 GB) por las wheels CUDA de `torch` fijadas en `uv.lock`: decisión consciente, registrada como riesgo con optimización futura (índice PyTorch CPU + re-lock) en §6 del plan.
- `make up` no solo levanta: espera los healthchecks (`docker compose up -d --build --wait`) y precarga el modelo E5 al volumen `models_cache` para que la primera pregunta del usuario no pague la descarga (~1,2 GB) ni el timeout de 30 s del cliente del frontend.
- `Dockerfile.frontend` instala solo `streamlit`+`httpx`: la imagen del chat es ligera. `app.py` importa `from client import preguntar_api` y funciona en el contenedor porque Streamlit ejecuta el script con su propio directorio en `sys.path` (no hace falta `PYTHONPATH` extra).
- La plantilla de PR de la fase anterior (F5) se reutiliza como base: solo se rellenó con el contexto actual; la estructura de secciones del curso se conserva.
- `docker/Dockerfile.frontend` fue reescrito respecto al intento previo (commit 3526df0, revertido en 927de8f): ya no versiona certificados y usa `COPY --from=uv` con ARG global en `FROM`.

---

## 9. 🔗 Referencias

- `docs/plan-trabajo-tecnico.md` — Fase 8 (contenerización y documentación) y §2 (estructura en capas).
- `docs/guia-despliegue.md` — despliegue paso a paso, TLS, respaldos y troubleshooting.
- `docker/Dockerfile`, `docker/Dockerfile.frontend`, `docker/Caddyfile`, `docker-compose.yml`, `.dockerignore`
- `Makefile`, `README.md`
- `tests/test_deployment.py`




