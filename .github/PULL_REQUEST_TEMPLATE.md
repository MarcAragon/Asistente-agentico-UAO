# 📋 Plantilla de Pull Request — Asistente Agéntico RAG UAO

<!--
Plantilla pre-llenada con el contexto de la FASE 4 (LLM Cerebras + cadena RAG),
rama feat--lmm_integration. Al reutilizarla para otra fase, reponer los
checkbox y sustituir el contenido de las secciones 3-9.
Instrucciones:
- Completa TODAS las secciones. Si una no aplica, escribe "N/A" y justifica.
- El título de la PR debe seguir Convencional Commits:
  feat: | fix: | refactor: | docs: | chore: | test:
  Ej: "feat: cadena RAG con LLM Cerebras y rotación de claves API"
- Elimina los comentarios HTML antes de abrir la PR.
-->

## 1. 🧩 Tipo de Pull Request

**Título sugerido:** `feat: cadena RAG con LLM Cerebras y rotación de claves API (Fase 4)`

Marca **uno** como principal (y otros si esta PR es mixta):

- [x] `feat` — Nueva funcionalidad (pipeline, ingesta, recuperación, API…)
- [ ] `fix` — Corrección de un bug
- [ ] `refactor` — Cambio interno sin alterar comportamiento externo
- [ ] `docs` — Solo documentación
- [ ] `chore` — Infraestructura, dependencias, configuración, CI
- [ ] `test` — Adición o corrección exclusiva de pruebas

---

## 2. 🧱 Módulo o Componente Afectado

Selecciona todas las áreas que toca esta PR:

- [x] **Configuración central** (`src/asistente_agentico_uao/config.py`, `.env.example`, `.gitignore`)
- [ ] **Preprocesamiento de documentos** (`scripts/llama_cloud_parsing.py`, `Data/Documentos_MD/`)
- [ ] **Ingesta & Chunking** (índice de embeddings, Chroma — fases futuras)
- [ ] **Recuperación & Embeddings** (`sentence-transformers`, `chromadb`)
- [x] **Generación / LLM** (`langchain-cerebras`, modelos de Cerebras)
- [ ] **API / Servicio** (`fastapi`, `uvicorn`)
- [x] **Scripts & CLI** (`scripts/`)
- [ ] **Pruebas Unitarias** (`tests/`, `pytest`)
- [ ] **Infraestructura & Empaquetado** (`pyproject.toml`, `uv.lock`, `uv_build`, `.gitignore`)
- [x] **Documentación** (`README.md`, `plan-trabajo-tecnico.md`, licencia)

---

## 3. 📚 Descripción de los Cambios

**Tarea / fase asociada:** Fase 4 del `plan-trabajo-tecnico.md`: LLM Cerebras + cadena RAG (tareas 4.1–4.6).
<!-- Issue relacionado, si existe: Closes #NN -->

#### 🎯 Motivación y contexto
Con la Fase 3 completa (recuperación top-5 validada sobre el índice real de 1284 chunks), faltaba la capa de generación: convertir los fragmentos recuperados en respuestas con citas verificables sin alucinar. Además, el hallazgo de la Fase 3 obligaba a replantear el "no sé": las similitudes coseno de E5 son altas siempre (~0.81 incluso para preguntas fuera de dominio como "receta de arepas"), por lo que el umbral `min_similarity=0.35` no discrimina dominio y subirlo a ciegas excluiría chunks válidos (los in-dominio puntúan 0.81–0.88). La defensa se trasladó al prompt: el LLM juzga si el contexto responde; el umbral queda como piso duro y su recalibración se pospone a F6 con el banco de preguntas. Se descartó el re-ranking y el LCEL con ramificación pura (`RunnableBranch`) por legibilidad/testabilidad: la rama "no gastar tokens" es explícita en Python y el LCEL se reserva para la parte generativa.

#### 📝 Resumen de cambios
- **`llm.py` (nuevo)**: `CerebrasLLM` envuelve `ChatCerebras` con rotación de claves API y reintentos con backoff; `NO_INFO_MESSAGE` centralizado; `get_llm()` singleton.
- **`chain.py` (nuevo)**: cadena RAG LCEL (`prompt | llm | StrOutputParser`) + orquestación `answer_question() -> RagAnswer` con umbral pre-LLM, post-proceso de citas (`build_sources`) y guardia de respuesta vacía.
- **`scripts/ask.py` (nuevo)**: CLI end-to-end (pregunta ad-hoc o banco de 7 humos: 5 in-dominio + 2 fuera de dominio).
- **`config.py` / `.env.example`**: nuevos parámetros del LLM (`llm_temperature`, `llm_max_tokens`, `llm_disable_reasoning`, `llm_max_retries`) y `cerebras_api_keys` para rotación.
- **`plan-trabajo-tecnico.md`**: Fase 4 marcada completada, hallazgos F4 documentados, riesgo nuevo en §6, §8 apuntando a Fase 5.

#### 🔧 Detalles técnicos relevantes
- **Rotación de claves API (4.6)**: ante 429/cuota o 401/403, `CerebrasLLM` rota inmediatamente a la siguiente clave de `CEREBRAS_API_KEY` + `CEREBRAS_API_KEYS` (separadas por coma/`;`/espacio, deduplicadas) recreando el cliente. Con una sola clave: 429 reintenta con backoff exponencial (1s/2s/4s, los límites por segundo son transitorios) y 401/403 se relanza (reintentar no lo corrige). Timeout/conexión/5xx siempre reintentan con backoff sin rotar.
- **Hallazgo F4 — modelo de razonamiento**: `qwen-3.8-27b` gasta por defecto los `max_tokens` en tokens de thinking y devuelve `content` vacío (`finish_reason=length`, `reasoning_tokens=1024`; ocurrió en 2 de 7 humos). Mitigado con `disable_reasoning=True` vía `extra_body` (configurable con `UAO_RAG__LLM_DISABLE_REASONING=1`) + guardia en `chain.py` que degrada respuesta vacía a no-información. Efecto colateral: latencia bajó de ~2-12 s a ~0.3-0.6 s por pregunta.
- **Prompt de síntesis (ES)**: rol de asistente de normativa UAO con reglas: responder SOLO con el contexto `[1]..[k]`, citar como `(Documento, sección)`, copiar literalmente `NO_INFO_MESSAGE` si el contexto no alcanza, no inventar y declarar qué falta si el contexto es parcial.
- **Post-proceso de citas**: regex tolerante extrae `(Documento, sección)` y las mapea a los chunks recuperados (coincidencia exacta, luego flexible por sección contenida, luego por documento); las citas no verificables se descartan (nunca se inventan fuentes); si la respuesta no es de no-información y no hay citas verificables, se devuelven todos los chunks recuperados por trazabilidad.
- **Umbral pre-LLM (4.4)**: si `retrieve()` devuelve `[]` no se llama al LLM. Con el hallazgo de F3 casi nunca dispara; la defensa real contra fuera de dominio es el prompt (verificada de facto: 2/2 preguntas fuera de dominio responden el mensaje exacto sin alucinar).
- Sin dependencias nuevas: `langchain-cerebras` ya estaba en `pyproject.toml`.

---

## 4. ⚠️ Impacto y Compatibilidad

- [x] Esta PR **NO** rompe compatibilidad (cambios retrocompatibles)
- [ ] Esta PR introduce **cambios que rompen** (documentar abajo)

**Si hay breaking changes / migración, documenta:**
- Variables de entorno nuevas (todas opcionales, con default funcional):
  - `UAO_RAG__LLM_TEMPERATURE=0.1` — temperatura de síntesis.
  - `UAO_RAG__LLM_MAX_TOKENS=1024` — techo de tokens de la respuesta.
  - `UAO_RAG__LLM_DISABLE_REASONING=1` — desactiva el thinking de qwen-3.8 (sin esto, respuestas vacías).
  - `UAO_RAG__LLM_MAX_RETRIES=3` — reintentos con backoff por clave.
  - `CEREBRAS_API_KEYS=clave2,clave3` — claves extra para rotación ante límites de cuota.
- Pasos de migración para quien tenga el repo clonado: ninguno obligatorio (`uv sync` + `.env` con `CEREBRAS_API_KEY` basta). Opcional: añadir `CEREBRAS_API_KEYS` para redundancia ante límites del free tier.
- Archivos/datos generados incluidos: ninguno nuevo (el índice `Data/chroma/` y `Data/Documentos_MD/` son de fases anteriores, gitignored).

---

## 5. 🔐 Seguridad y Secretos

- [x] Ningún secreto (API keys, contraseñas, tokens) está escrito en el código ni en archivos rastreados
- [x] Los secretos viven únicamente en `.env` (gitignored) o en variables de entorno
- [x] `.env.example` está actualizado y **rastreado** como plantilla documental
- [x] Las claves leídas en código tienen valores por defecto vacíos y validación en tiempo de ejecución (`CerebrasLLM.invoke` lanza `RuntimeError` con mensaje claro si no hay claves)

Variables que esta PR introduce:

| Variable | Para qué | Dónde obtenerla |
|---|---|---|
| `CEREBRAS_API_KEY` (ya existía) | Clave principal de Cerebras | https://cloud.cerebras.ai |
| `CEREBRAS_API_KEYS` | Claves adicionales (rotación ante 429/401/403) | https://cloud.cerebras.ai (una por cuenta) |
| `UAO_RAG__LLM_*` | Parámetros del LLM (ver §4) | N/A (defaults funcionales) |

---

## 6. ✅ Lista de Chequeo Pre-PR (Estándares del Curso UAO)

- [x] **Entorno de ejecución (`uv`):** Todo se ejecutó y probó exclusivamente con **`uv`** y la versión de Python fijada en `.python-version` / `pyproject.toml` (≥ 3.14).
- [x] **Sin warnings:** `uv run python scripts/ask.py` corre limpio (solo avisos de terceros: token opcional del HF Hub y barra de progreso de carga del modelo).
- [x] **Lint (`ruff`):** `uv run ruff check src scripts` pasa sin errores ni advertencias.
- [x] **Control de exclusiones (`.gitignore`):** `.env`, `.venv`, `__pycache__/`, modelos pesados y datos regenerables voluminosos **NO** están rastreados por Git. `.env.example` **SÍ** lo está.
- [x] **Estructura del código:** Fuente en `src/asistente_agentico_uao/` (`llm.py`, `chain.py`), utilidades/CLI en `scripts/` (`ask.py`), pruebas en `tests/` (cuando existan).
- [x] **Pruebas unitarias (`pytest`):** `uv run pytest` pasa al 100 % (21/21: los tests permanentes de fases previas, sin regresiones). **Justificación sin tests nuevos permanentes:** decisión explícita del usuario para esta fase; se validó con pruebas temporales (9 de rotación de claves + 9 de cadena, ejecutadas y borradas) y con humo real end-to-end. Los tests permanentes de la cadena se añadirán en F6 con el banco de evaluación.
- [x] **Clean Code & Refactorización:** Funciones cohesivas con responsabilidad única, nombres descriptivos, docstrings en módulos y funciones públicas, sin código muerto ni comentado.
- [x] **Un solo punto de configuración:** Los valores configurables se leen vía `Settings` (`config.py`); ningún módulo parsea `.env` por su cuenta ni hardcodea rutas o claves.
- [x] **Reproducibilidad:** `uv sync` en un clon limpio instala todo lo necesario para ejecutar esta PR (sin dependencias nuevas).

---

## 7. 🧪 Evidencia de Pruebas Ejecutadas

<!-- Pega la salida real (recortada si es larga) de los comandos que ejecutaste. No describas de memoria: pega lo que la terminal imprimió. -->

```
# Comando 1 (suite completa + lint):
$ uv run pytest -q
21 passed in 1.10s
$ uv run ruff check src scripts tests
All checks passed!

# Comando 2 (pruebas temporales de rotación de claves — ejecutadas y luego borradas):
$ uv run pytest tests/test_rotacion_temporal.py -q
9 passed in 1.02s
# Cubre: 429→rota a clave 2; 401→rota; cuota en todas las claves→backoff
# sobre la última y agotamiento acotado; 401 con una clave→se relanza;
# timeout→reintenta la misma clave; 400→se relanza sin rotar; sin
# claves→RuntimeError con mensaje claro.

# Comando 3 (humo end-to-end sobre el índice real, 1284 chunks):
$ uv run python scripts/ask.py
Pregunta: ¿Que pasa si repruebo tres veces una misma asignatura?
Respuesta (qwen-3.8-27b):
Si repruebas tres veces una misma asignatura, ingresas a prueba académica
por repitencia (Res-CA-6744-Modifica-Reglamento-de-Pregrado.pdf,
ARTÍCULO 70º-2. INGRESO A PRUEBA ACADÉMICA POR REPITENCIA:)...
Fuentes (2):
  1. sim=0.882  Res-CA-6744-Modifica-Reglamento-de-Pregrado.pdf — ARTÍCULO 70º-2...
  2. sim=0.864  Res-CA-6744-Modifica-Reglamento-de-Pregrado.pdf — ARTÍCULO 70º-3...

Pregunta: ¿Cuál es la receta traditional de las arepas antioqueñas?
Respuesta (qwen-3.8-27b):
No tengo información suficiente en la normativa UAO para responder esa pregunta.
  [fallback: sin llamada al LLM o respuesta de no-información]
Fuentes (0):
```

Resultado del criterio de aceptación F4 (7 humos): "tres repitencias" cita Art. 70º-2/70º-3 de Res-CA-6744; "requisitos magíster" cita Arts. 35º/13º de Res-CA-6605 e indica explícitamente qué no está en el contexto; "transferencia interna" cita Arts. 19°/17° (Reso-CS-666), 23º (Res-CA-6603) y 16º (Res. 7714), mapeados a 4 fuentes; las 2 fuera de dominio (arepas, Mundial 2022) responden el mensaje exacto de no-información sin alucinar.

#### 🔄 Pasos para replicar / probar manualmente
1. `git clone ... && cd Asistente-agentico-UAO && uv sync`
2. `cp .env.example .env` y completar `CEREBRAS_API_KEY` (opcional: `CEREBRAS_API_KEYS` para rotación).
3. Verificar que el índice existe: `uv run python scripts/smoke_retrieval.py` (o regenerarlo con `uv run python scripts/ingest.py`).
4. Humo end-to-end: `uv run python scripts/ask.py` (banco de 7) o `uv run python scripts/ask.py "¿pregunta ad-hoc?"`.

#### 🖼️ Capturas / salida visual (opcional)
Salida completa del humo F4 registrada en la sesión de trabajo: 7 preguntas con tiempos de 0.3–0.6 s por respuesta (tras desactivar el razonamiento) y fuentes con similitud 0.83–0.88.

---

## 8. 👀 Notas para el Revisor

- **Umbral `min_similarity` NO se recalibró** (sigue en 0.35): deliberado. Con similitudes E5 de ~0.81 fuera de dominio, cualquier umbral absoluto erraría en ambos sentidos; la recalibración se hace en F6 con el banco de ~30 preguntas. La mitigación actual es el prompt de solo-contexto, ya verificada con 2 preguntas fuera de dominio.
- **2 preguntas in-dominio responden "no sé"** ("cancelaciones 2026-2", "créditos mínimos posgrado"): el LLM juzgó correctamente que los chunks recuperados eran de otro programa/periodo. Anotado para F6 (evaluar recall del banco y si las tablas de calendario contienen la fecha puntual); no es un defecto de la cadena.
- **`disable_reasoning=True` por defecto**: decisión discutible si se quisiera razonamiento para preguntas complejas; externalizada en `UAO_RAG__LLM_DISABLE_REASONING` para revertirla sin tocar código.
- **Rotación de claves por proceso**: el índice de clave vigente vive en el singleton `CerebrasLLM` (sin estado compartido entre procesos). Suficiente para F5 (un proceso uvicorn); con múltiples workers cada uno rota independientemente.
- **Deuda aceptada**: sin tests permanentes de la cadena/rotación (decisión del usuario para esta fase); cubierto con humo manual y quedará en F6.
- La extracción de citas por regex asume el formato `(Documento, sección)` que el prompt exige; citas malformadas se descartan y caen al fallback de trazabilidad (todos los chunks recuperados).

---

## 9. 🔗 Referencias

- `plan-trabajo-tecnico.md` — Fase 4 (tareas 4.1–4.6), hallazgos F3/F4, §6 riesgos, §8 próximo paso.
- `asistente-uao-rag.md` — documento base del proyecto.
- Documentación externa: [langchain-cerebras (ChatCerebras)](https://python.langchain.com/docs/integrations/chat/cerebras/), [Cerebras Inference API](https://inference-docs.cerebras.ai/) (parámetro `disable_reasoning` vía `extra_body`), LangChain LCEL (`RunnableLambda`, `StrOutputParser`).
- Fases previas de esta rama: F1 LlamaCloud Parse, F2 chunking+embeddings+Chroma, F3 recuperación (`retrieval.py`, `scripts/smoke_retrieval.py`).
