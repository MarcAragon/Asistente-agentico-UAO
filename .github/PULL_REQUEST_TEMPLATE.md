# 📋 Plantilla de Pull Request — Asistente Agéntico RAG UAO

<!--
Plantilla pre-llenada con el contexto de la rama feat--ml_flow
(F9 — Observabilidad del LLM con MLflow: tracking de trazas + dashboard).
Instrucciones:
- Completa TODAS las secciones. Si una no aplica, escribe "N/A" y justifica.
- El título de la PR debe seguir Conventional Commits:
  feat: | fix: | refactor: | docs: | chore: | test:
  Ej: "feat: observabilidad del LLM con MLflow (tracking + dashboard) — Fase 9"
- Elimina los comentarios HTML antes de abrir la PR.
-->

## 1. 🧩 Tipo de Pull Request

**Título sugerido:** `feat: observabilidad del LLM con MLflow (tracking server + dashboard) — Fase 9`

Marca **uno** como principal (y otros si esta PR es mixta):

- [x] `feat` — Nueva funcionalidad (observabilidad del LLM: trazas + dashboard) (principal)
- [ ] `fix` — Corrección de un bug
- [ ] `refactor` — Cambio interno sin alterar comportamiento externo
- [x] `docs` — Documentación (la sección **Fase 9** que faltaba en el plan, README y guía)
- [x] `chore` — Infraestructura, dependencias, configuración (compose, Caddyfile, Makefile, `mlflow` en `pyproject.toml`/`uv.lock`)
- [x] `test` — Adición de prueba (`tests/test_deployment.py`, +1)

---

## 2. 🧱 Módulo o Componente Afectado

Selecciona todas las áreas que toca esta PR:

- [ ] **Configuración central** (`src/asistente_agentico_uao/core/config.py`)
- [ ] **Preprocesamiento de documentos** (`scripts/llama_cloud_parsing.py`, `Data/Documentos_MD/`)
- [ ] **Ingesta & Chunking** (índice de embeddings, Chroma)
- [ ] **Recuperación & Embeddings** (`sentence-transformers`, `chromadb`)
- [x] **Generación / LLM** (`langchain-cerebras`, Cerebras) — se instrumentan sus llamadas (`mlflow.openai.autolog()`); el modelo no cambia
- [x] **API / Servicio** (`api/main.py`) — instrumentación opcional en la fábrica `create_app()`; **mismos endpoints y contratos**
- [ ] **Scripts & CLI** (`scripts/`)
- [x] **Pruebas Unitarias** (`tests/test_deployment.py`)
- [x] **Infraestructura & Empaquetado** (`docker-compose.yml`, `docker/Caddyfile`, `Makefile`, `.env.example`, `pyproject.toml`, `uv.lock`)
- [x] **Observabilidad** (nuevo servicio `mlflow`, volumen `mlflow_data`, dashboard por el proxy)
- [x] **Documentación** (`docs/plan-trabajo-tecnico.md`, `README.md`, `docs/guia-despliegue.md`)

---

## 3. 📚 Descripción de los Cambios

**Tarea / fase asociada:** **Fase 9 — Observabilidad del LLM con MLflow**, una
fase que **no existía en `docs/plan-trabajo-tecnico.md`** (el documento base
`asistente-uao-rag.md` no contempla observabilidad del modelo). Esta PR añade
además esa sección al plan, porque el código ya estaba implementado en la rama
`feat--ml_flow` pero **no quedó documentado**.

#### 🎯 Motivación y contexto
El RAG ya cita fuentes y los humos miden latencia, pero no había registro por
llamada: no se podía auditar qué prompt/contexto produjo cada respuesta ni
comparar modelos o detectar regresiones. MLflow aporta ese *tracking* (prompt
con contexto, respuesta, modelo, parámetros, tokens y latencia) y un dashboard
para el equipo. Requisito de diseño: **la observabilidad no puede condicionar el
servicio** — si MLflow no está, la API responde igual.

#### 📝 Resumen de cambios

**A. Instrumentación de la API (opt-in, `src/asistente_agentico_uao/api/main.py`)**
- Si el entorno define `MLFLOW_TRACKING_URI` (lo hace el compose), se ejecuta
  `mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)` y `mlflow.openai.autolog()`.
- Funciona con Cerebras porque `ChatCerebras` hereda de `BaseChatOpenAI` y
  construye clientes `openai.OpenAI`/`AsyncOpenAI` apuntados a
  `api.cerebras.ai`: instrumentar el SDK captura cada
  `chat.completions.create()` (incluidos los reintentos y la rotación de claves
  de `core/llm.py`).
- **Degradable**: sin la variable ni siquiera se importa `mlflow`; en local y en
  la suite de tests no hay trazado ni dependencia de red.

**B. Stack de observabilidad (`docker-compose.yml`)**
- Nuevo servicio `mlflow` (`ghcr.io/mlflow/mlflow`): `mlflow server` con backend
  SQLite (`--backend-store-uri sqlite:////mlflow/mlflow.db`, `--workers 1`
  porque SQLite no admite varios escritores), artefactos en `/mlflow/artifacts`,
  `--allowed-hosts "*"`, `expose: "5000"` y `healthcheck` sobre `/health`.
- La API recibe `MLFLOW_TRACKING_URI=http://mlflow:5000` y
  `MLFLOW_EXPERIMENT_NAME=asistente-uao`, y espera al servicio con
  `depends_on: condition: service_healthy`.
- Nuevo volumen `mlflow_data` (`/mlflow`): experimentos + trazas + artefactos,
  persistente entre `down`/`up`.
- **Se retiró el `ports: 127.0.0.1:5000:5000`** que traía el borrador:
  contradecía el comentario del propio servicio («no publica puertos») y el
  invariante verificado de F8 (solo el proxy publica 80/443). El dashboard se
  sirve por el proxy.

**C. Dashboard por el proxy (`docker/Caddyfile`)**
- Sitio `mlflow.{$SITE_ADDRESS:localhost}` con `tls {$TLS_DIRECTIVE:internal}` y
  `reverse_proxy mlflow:5000` (mismos defaults que el sitio principal, para que
  el Caddyfile también valide fuera del compose).

**D. Operación y configuración**
- `Makefile`: nuevo target `logs-mlflow`, variable `MLFLOW_URL` y `make urls`
  imprime el dashboard; se documenta en `make help`.
- `.env.example`: bloque de documentación (comentarios) que aclara que el
  compose fija `MLFLOW_TRACKING_URI`/`MLFLOW_EXPERIMENT_NAME` y que definirlas
  en `.env` no las sobreescribe.
- `pyproject.toml`/`uv.lock`: `mlflow` (cliente 3.16.1) como dependencia vía
  `uv add` (nunca `pip`).

**E. Pruebas (`tests/test_deployment.py` → 16 pruebas)**
- `SERVICIOS` pasa a 5 (`api`, `frontend`, `proxy`, `redis`, `mlflow`).
- Nueva `test_mlflow_es_solo_interno_y_se_sirve_por_el_proxy`: sin `ports`,
  `expose 5000`, volumen `mlflow_data`, backend SQLite, healthcheck,
  `MLFLOW_TRACKING_URI=http://mlflow:5000` + `depends_on: service_healthy` en
  `api`, y ruta `mlflow.{$SITE_ADDRESS:localhost}` del Caddyfile.
- El invariante de F8 sigue intacto: **solo el proxy publica puertos**.

**F. Documentación (cierre del hueco)**
- `docs/plan-trabajo-tecnico.md`: **nueva sección Fase 9** (motivación, tabla de
  tareas 9.1–9.8, criterio de aceptación y decisiones 1–6), más el registro en
  la cabecera, §0.1 (dependencia), §0.4 (decisión 11), §1 (flujo), §2
  (estructura), §3.1 (variables), §5 (pruebas), §6 (3 riesgos nuevos), §7
  (cronograma, 38 SP) y §8 (próximo paso).
- `README.md` (§1 arquitectura/observabilidad, §3.2 URLs, §3.3 atajos, §6
  pruebas, §7 operación, §8 seguridad y privacidad).
- `docs/guia-despliegue.md` (§1 topología, §4.2 variables, §5.1/§5.2
  verificación, §6 respaldos, §10 seguridad y §11 solución de problemas).

#### 🔧 Detalles técnicos relevantes
- **`mlflow.openai.autolog()` y no `mlflow.langchain.autolog()`**: equivalente en
  efecto (ChatCerebras es un `BaseChatOpenAI` construido sobre el SDK de
  OpenAI); si una versión futura de `langchain-cerebras` dejara de usarlo, la
  alternativa documentada es `mlflow.langchain.autolog()` (traza el Runnable de
  LCEL). Queda como decisión 1 de la Fase 9.
- **Trazas locales y en red interna**: el volumen `mlflow_data` guarda pregunta
  y contexto recuperado (normativa pública), sin datos personales; es
  independiente de la caché semántica de Redis.
- **Riesgo asumido (documentado, no oculto)**: `api` depende de `mlflow` healthy,
  así que un *tracking server* caído bloquea el arranque de la API aunque la
  observabilidad sea opcional. Está registrado en §6 con la alternativa
  (quitar el `depends_on` o usar `service_started`).
- **Pendiente de la fase**: pin de `ghcr.io/mlflow/mlflow` (hoy `latest`, no
  reproducible entre builds) a la versión del cliente (3.16.1), tras verificar
  el tag publicado con `make up`.

---

## 4. ⚠️ Impacto y Compatibilidad

- [x] Esta PR **NO** rompe compatibilidad (cambios retrocompatibles)
- [ ] Esta PR introduce **cambios que rompen** (documentar abajo)

**N/A a nivel de contratos**: `POST /ask`, `GET /health`, `GET /documents` y el
gRPC `IndexAdmin` no cambian; el comportamiento del RAG (respuesta + fuentes) es
idéntico. Detalles operativos a tener en cuenta:

- **Nuevo servicio y volumen**: `docker compose up` arranca ahora 5 servicios
  (`mlflow` se suma a proxy, frontend, api y redis) y usa el volumen
  `mlflow_data`; `down`/`up` lo conserva.
- **`api` espera a `mlflow`** (`depends_on: service_healthy`): si el tracking
  server no arranca, la API no arranca tampoco. Es una decisión consciente
  (documentada como riesgo en §6); la alternativa es quitar ese `depends_on`.
- **Se retiró un puerto publicado**: el borrador exponía `127.0.0.1:5000:5000`;
  ahora el dashboard solo es accesible por `https://mlflow.<SITE_ADDRESS>`. Si
  alguien usaba ese atajo de desarrollo, debe cambiar al subdominio del proxy
  (o re-añadir el mapeo en loopback y actualizar la prueba).
- **Desarrollo local intacto**: sin `MLFLOW_TRACKING_URI` no se importa `mlflow`;
  `uv run pytest`, `make api`, `make frontend` y los tests siguen igual.

---

## 5. 🔐 Seguridad y Secretos

- [x] Ningún secreto (API keys, contraseñas, tokens) está escrito en el código ni en archivos rastreados
- [x] `MLFLOW_TRACKING_URI`/`MLFLOW_EXPERIMENT_NAME` no son secretos: el tracking server vive en la red interna del compose y no requiere autenticación
- [x] Solo el proxy Caddy publica puertos al host (80/443); `mlflow` (`:5000`) queda en `expose` dentro de la red interna — verificado por `tests/test_deployment.py`
- [x] El dashboard se sirve por el proxy con TLS (`https://mlflow.<SITE_ADDRESS>`) y hereda las cabeceras de seguridad del sitio principal
- [x] Las trazas (pregunta + contexto recuperado de normativa pública) se guardan en el volumen `mlflow_data` del host, nunca en un servicio público; el chat no solicita datos personales (Ley 1581 de 2012)
- [x] La observabilidad es **opt-in**: sin `MLFLOW_TRACKING_URI` no hay trazado, así que puede desactivarse sin tocar código
- [x] `tests/test_deployment.py` sigue verificando la ausencia de claves reales (`csk-…`/`llx-…`) en archivos versionados y que `.env` esté ignorado por git y por el contexto de build

---

## 6. ✅ Lista de Chequeo Pre-PR (Estándares del Curso UAO)

- [x] **Entorno de ejecución (`uv`):** probado con `uv` y Python 3.14; la dependencia se añadió con `uv add mlflow` (nunca `pip`).
- [x] **Sin warnings:** `uv run pytest` corre limpio (96/96; 1 warning de deprecación interna de starlette/anyio, ajeno a esta PR).
- [x] **Lint (`ruff`):** `uv run ruff check src scripts tests` y `uv run ruff format --check` pasan al 100 % (se corrigió el orden de imports y el formato de `api/main.py`).
- [x] **Control de exclusiones (`.dockerignore`/`.gitignore`):** sin cambios; `.env` y `Data/` siguen fuera del contexto de build.
- [x] **Pruebas unitarias (`pytest`):** `uv run pytest` pasa al 100 %: **96/96** (95 previas + 1 nueva de MLflow en `tests/test_deployment.py`).
- [x] **Validación de compose:** `docker compose config` válido; la interpolación confirma que **solo** `proxy` publica 80/443 y que `mlflow` queda en `expose: "5000"`.
- [x] **Clean Code:** la instrumentación es una guarda de 8 líneas en `create_app()`, con comentarios de intención; el servicio y el volumen siguen la convención del compose de F8 (healthcheck, red `internal`, sin puertos).
- [x] **Reproducibilidad:** `uv.lock` actualizado con `mlflow` 3.16.1; el build sigue con `uv sync --frozen`.
- [ ] **Build real de contenedores (`docker compose up --build`):** ⏳ pendiente de ejecución por el usuario — el socket de Docker no era accesible desde el entorno de esta sesión. La configuración se validó de forma estática (`docker compose config` + 16 pruebas); cerrar con `make up` y comprobar el dashboard (guía §5.1). **En particular, no se verificó aún que `mlflow.openai.autolog()` registre la primera traza real** (requiere el stack levantado con la `CEREBRAS_API_KEY`).

---

## 7. 🧪 Evidencia de Pruebas Ejecutadas

```
# Suite completa (incluye la nueva prueba de MLflow):
$ uv run pytest -q
........................................................................ [ 75%]
........................                                                 [100%]
96 passed, 1 warning in 10.11s

# Lint y formato (se corrigió api/main.py: orden de imports + formato):
$ uv run ruff check src scripts tests
All checks passed!
$ uv run ruff format --check src scripts tests
43 files already formatted

# Validación estática del compose (interpolación incluida):
$ docker compose config --quiet && echo COMPOSE_CONFIG_OK
COMPOSE_CONFIG_OK

# Reparto de puertos por servicio (solo el proxy publica):
api       ports=None   expose=['8000', '50051']
frontend  ports=None   expose=['8501']
mlflow    ports=None   expose=['5000']
proxy     ports=['80:80', '443:443']
redis     ports=None   expose=['6379']

# Cliente de MLflow fijado en el entorno:
$ uv run python -c "import mlflow; print(mlflow.__version__)"
3.16.1

# Alcance del cambio:
$ git diff HEAD --stat
 .env.example                           |   8 +
 Makefile                               |  14 +-
 README.md                              |  35 +-
 docker-compose.yml                     |  53 ++-
 docker/Caddyfile                       |   9 +
 docs/guia-despliegue.md                |  61 ++-
 docs/plan-trabajo-tecnico.md           | 155 +++++-
 pyproject.toml                         |   1 +
 src/asistente_agentico_uao/api/main.py |  18 +
 tests/test_deployment.py               |  42 +-
 uv.lock                                | 828 +++++++++++++++++++++++++++++++++
 11 files changed, 1169 insertions(+), 55 deletions(-)
```

---

## 8. 👀 Notas para el Revisor

- **Hueco de documentación (lo que motivó esta PR):** la **Fase 9 no existía** en
  `docs/plan-trabajo-tecnico.md` porque el documento base no contempla
  observabilidad del modelo. El código ya estaba en la rama `feat--ml_flow` pero
  sin sección en el plan: aquí se añade la sección completa (motivación, tareas
  9.1–9.8, criterio de aceptación y decisiones 1–6) y su registro en cabecera,
  §0.1, §0.4, §1, §2, §3.1, §5, §6, §7 y §8.
- **Se cerraron dos defectos que bloqueaban la PR** (encontrados al validar esta
  rama): (1) `ruff check`/`ruff format` fallaban en `api/main.py` por el
  `import os` fuera de lugar y una línea en blanco de más; (2) dos pruebas de
  `tests/test_deployment.py` fallaban porque el servicio `mlflow` no estaba en
  `SERVICIOS` y publicaba `127.0.0.1:5000`. Ahora la suite está en **96/96**.
- **Puerto retirado:** el `ports: 127.0.0.1:5000:5000` del borrador contradecía
  el comentario del propio servicio y el invariante de F8 (verificado por tests).
  El dashboard se sirve por el proxy. Si se prefiere el atajo de desarrollo,
  re-añadirlo **solo en loopback** y relajar la prueba correspondiente.
- **`mlflow.openai.autolog()` (y no `mlflow.langchain.autolog()`):** se eligió
  porque `ChatCerebras` extiende `BaseChatOpenAI` y crea clientes
  `openai.OpenAI`/`AsyncOpenAI` (verificado en el código de
  `langchain-cerebras`). Queda como decisión 1 de la Fase 9 por si el wrapper
  cambia de base en el futuro.
- **Riesgo asumido y registrado:** `api` tiene `depends_on: service_healthy` de
  `mlflow`, de modo que un tracking server caído impide arrancar la API aunque
  la instrumentación sea opcional. Alternativa en §6: quitar el `depends_on` o
  usar `service_started` (sin tocar `main.py`).
- **Pendiente de la fase:** pin de `ghcr.io/mlflow/mlflow` (`latest` hoy) a la
  versión del cliente (3.16.1) tras verificar el tag publicado con `make up`.
- **Verificación manual pendiente (no ejecutable en esta sesión):** no se pudo
  hacer `docker compose up --build` (socket de Docker inaccesible), así que la
  primera traza real y el dashboard no se comprobaron en vivo; la configuración
  sí quedó validada de forma estática. Cerrar con `make up`, `make urls` y una
  pregunta real (guía §5.1).
- La plantilla de PR de la fase anterior (F8) se reutiliza como base y se
  rellena con el contexto actual: la estructura de secciones del curso se
  conserva (se añade una casilla de **Observabilidad** en §2).

---

## 9. 🔗 Referencias

- `docs/plan-trabajo-tecnico.md` — **§4 Fase 9 (nueva)**, decisión 11 de §0.4,
  dependencia en §0.1, variables en §3.1, pruebas en §5, riesgos en §6 y
  cronograma en §7.
- `docs/guia-despliegue.md` — §1 arquitectura desplegada, §4.2 variables,
  §5.1/§5.2 verificación, §6 respaldos, §10 seguridad y §11 solución de problemas.
- `src/asistente_agentico_uao/api/main.py` — instrumentación opt-in (`create_app`).
- `docker-compose.yml` (servicio `mlflow` + volumen `mlflow_data`),
  `docker/Caddyfile` (subdominio del dashboard), `Makefile` (`logs-mlflow`,
  `MLFLOW_URL`, `urls`), `.env.example` (bloque de observabilidad).
- `tests/test_deployment.py` — 16 pruebas estáticas del despliegue.
- `README.md` — §1 (observabilidad), §3.2, §3.3, §6, §7 y §8.
- MLflow — [Tracking](https://mlflow.org/docs/latest/tracking.html) ·
  [`mlflow.openai` autolog](https://mlflow.org/docs/latest/llms/openai/autologging.html).
- PR previa: #13 (`feat--refactor+docker`, F8 contenerización) — base de esta rama.
