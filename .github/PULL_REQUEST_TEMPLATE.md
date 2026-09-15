# 📋 Plantilla de Pull Request — Asistente Agéntico RAG UAO

<!--
Plantilla pre-llenada con el contexto de la PR de TESTS de la FASE 4
(pruebas permanentes de la integración LLM), rama test. Al reutilizarla
para otra fase, reponer los checkbox y sustituir el contenido de las
secciones 3-9.
Instrucciones:
- Completa TODAS las secciones. Si una no aplica, escribe "N/A" y justifica.
- El título de la PR debe seguir Convencional Commits:
  feat: | fix: | refactor: | docs: | chore: | test:
  Ej: "test: pruebas permanentes de la integración LLM y cadena RAG (F4)"
- Elimina los comentarios HTML antes de abrir la PR.
-->

## 1. 🧩 Tipo de Pull Request

**Título sugerido:** `test: pruebas permanentes de la integración LLM y cadena RAG (Fase 4)`

Marca **uno** como principal (y otros si esta PR es mixta):

- [ ] `feat` — Nueva funcionalidad (pipeline, ingesta, recuperación, API…)
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
- [ ] **Ingesta & Chunking** (índice de embeddings, Chroma — fases futuras)
- [ ] **Recuperación & Embeddings** (`sentence-transformers`, `chromadb`)
- [ ] **Generación / LLM** (`langchain-cerebras`, modelos de Cerebras)
- [ ] **API / Servicio** (`fastapi`, `uvicorn`)
- [ ] **Scripts & CLI** (`scripts/`)
- [x] **Pruebas Unitarias** (`tests/`, `pytest`)
- [ ] **Infraestructura & Empaquetado** (`pyproject.toml`, `uv.lock`, `uv_build`, `.gitignore`)
- [ ] **Documentación** (`README.md`, `plan-trabajo-tecnico.md`, licencia)

---

## 3. 📚 Descripción de los Cambios

**Tarea / fase asociada:** cierre de la deuda de pruebas de la Fase 4 del `plan-trabajo-tecnico.md` (LLM Cerebras + cadena RAG, tareas 4.1–4.6). No modifica código de producción: solo añade pruebas.
<!-- Issue relacionado, si existe: Closes #NN -->

#### 🎯 Motivación y contexto
La PR de la Fase 4 registró una **deuda aceptada**: sin tests permanentes de la cadena RAG ni de la rotación de claves (decisión del usuario en esa fase; cubierto con humo manual y 18 pruebas temporales ejecutadas y borradas: 9 de rotación + 9 de cadena). Antes de construir la Fase 5 (API REST sobre `answer_question`) y la Fase 6 (evaluación con banco de preguntas), esa lógica crítica —rotación de claves ante límites de cuota, mapeo de citas a fuentes, "no gastar tokens" sin contexto y degradación a no-información— necesita red de seguridad permanente. Esta PR la salda con 23 pruebas unitarias que reproducen y amplían la cobertura de las pruebas temporales de F4.

#### 📝 Resumen de cambios
- **`tests/test_llm.py` (nuevo, 10 pruebas)**: `collect_api_keys` (combinación/deduplicación/vacío) y `CerebrasLLM.invoke` (rotación 429/401 entre claves, backoff ante 429/timeout con una sola clave, agotamiento acotado, relanzamiento de errores no recuperables y de 401 sin más claves, `RuntimeError` sin claves).
- **`tests/test_chain.py` (nuevo, 13 pruebas)**: `_excerpt`, `extract_citations` (deduplicación/espacios), `build_sources` (mapeo de citas verificables, descarte de citas no verificables con trazabilidad, coincidencia flexible por sección, `sources=[]` en no-información) y `answer_question` (umbral pre-LLM sin gastar tokens, `used_fallback`, degradación de respuesta vacía, flujo feliz con citas y verificación del prompt renderizado).
- **Código de producción**: sin cambios (`llm.py`, `chain.py`, `config.py` intactos). Sin dependencias nuevas.

#### 🔧 Detalles técnicos relevantes
- **Cero llamadas reales a la API de Cerebras**: `ChatCerebras` se reemplaza por un stub (`FakeChatCerebras`) inyectado en `sys.modules` como módulo falso `langchain_cerebras`. Esto además evita importar la librería real en los tests y permite verificar un comportamiento clave: que la rotación **recrea el cliente** con la nueva clave (se registra la `api_key` de cada cliente construido).
- **Backoff sin esperas**: `time.sleep` se sustituye por un grabador (`llm_env.sleeps`), de modo que los tests verifican los retardos exactos del backoff exponencial (1s, 2s) sin ralentizar la suite (0.24 s para las 23 pruebas).
- **Errores falsos con `status_code`**: `FakeHTTPError` imita el contrato del SDK de OpenAI (429/401/403/408/400) que `llm.py` usa para clasificar.
- **Cadena completa con dobles de prueba**: `answer_question` se ejercita con `FakeRetriever`, `FakeLLM` (que captura el prompt renderizado con `to_string()`) y `GuardLLM` (falla el test si el LLM llega a invocarse sin contexto: garantiza la tarea 4.4 "no gastar tokens").
- **Configuración real en los tests de rotación**: se construyen `Settings` con kwargs explícitos (claves ficticias `k1`/`k2`, `llm_max_retries` controlado), sin tocar el `.env` ni variables de entorno.
- Convenciones del repo respetadas: nombres y docstrings en español, Arrange/Act/Assert, `ClassVar` en atributos de clase del stub (limpieza `ruff RUF012`).

---

## 4. ⚠️ Impacto y Compatibilidad

- [x] Esta PR **NO** rompe compatibilidad (cambios retrocompatibles)
- [ ] Esta PR introduce **cambios que rompen** (documentar abajo)

**Si hay breaking changes / migración, documenta:** N/A. La PR solo añade dos archivos bajo `tests/`; no toca código de producción, configuración, dependencias ni el contrato de `answer_question`/`CerebrasLLM`. No introduce variables de entorno nuevas y no requiere pasos de migración (`uv sync` basta). No genera archivos ni datos nuevos (los dobles de prueba son efímeros y en memoria).

---

## 5. 🔐 Seguridad y Secretos

- [x] Ningún secreto (API keys, contraseñas, tokens) está escrito en el código ni en archivos rastreados
- [x] Los secretos viven únicamente en `.env` (gitignored) o en variables de entorno
- [x] `.env.example` no requirió cambios en esta PR (ya estaba actualizado en la Fase 4)
- [x] Las claves usadas en los tests son ficticias (`"k1"`, `"k2"`) y pasan solo por dobles de prueba: los tests **nunca** construyen un cliente real ni tocan la red

Variables que esta PR introduce: **ninguna** (no hay cambios de configuración). Las pruebas de rotación usan claves de mentira inyectadas vía `Settings(...)` en memoria, sin leer el `.env` real ni variables del sistema.

---

## 6. ✅ Lista de Chequeo Pre-PR (Estándares del Curso UAO)

- [x] **Entorno de ejecución (`uv`):** Todo se ejecutó y probó exclusivamente con **`uv`** y la versión de Python fijada en `.python-version` / `pyproject.toml` (≥ 3.14).
- [x] **Sin warnings:** `uv run pytest` corre limpio (sin warnings de pytest ni deprecations); los dobles de prueba evitan importar `sentence-transformers`/`langchain-cerebras` en los tests nuevos.
- [x] **Lint (`ruff`):** `uv run ruff check src scripts tests` pasa sin errores ni advertencias (se corrigió `RUF012` con `ClassVar` en el stub).
- [x] **Control de exclusiones (`.gitignore`):** Sin cambios; `.env`, `.venv`, `__pycache__/`, `.pytest_cache/` y datos regenerables siguen fuera del rastreo. Solo se añaden `tests/test_llm.py` y `tests/test_chain.py`.
- [x] **Estructura del código:** Pruebas en `tests/` siguiendo las convenciones de los tests previos (fakes inyectados, `monkeypatch`, docstrings en español, Arrange/Act/Assert). Código de producción intacto en `src/asistente_agentico_uao/`.
- [x] **Pruebas unitarias (`pytest`):** `uv run pytest` pasa al 100 %: **47/47** (23 nuevas + 24 previas, sin regresiones). Las pruebas nuevas no consumen tokens ni llaman a la API de Cerebras (stub de `ChatCerebras` + `time.sleep` grabado).
- [x] **Clean Code & Refactorización:** Cada prueba cubre un comportamiento único con nombre descriptivo en español; helpers pequeños (`make_chunk`, `make_config`, `raise_error`); sin código muerto ni comentado.
- [x] **Un solo punto de configuración:** Los tests de rotación usan `Settings` real con kwargs explícitos; ningún test parsea `.env` por su cuenta ni hardcodea rutas del proyecto.
- [x] **Reproducibilidad:** `uv sync` en un clon limpio instala todo lo necesario para ejecutar esta PR (sin dependencias nuevas; `pytest` ya estaba en el grupo `dev`).

---

## 7. 🧪 Evidencia de Pruebas Ejecutadas

<!-- Pega la salida real (recortada si es larga) de los comandos que ejecutaste. No describas de memoria: pega lo que la terminal imprimió. -->

```
# Comando 1 (suite completa de tests, incluidas las 23 nuevas):
$ uv run pytest -q
...............................................                          [100%]
47 passed in 1.23s

# Comando 2 (solo los tests nuevos de Fase 4):
$ uv run pytest tests/test_llm.py tests/test_chain.py -v
tests/test_llm.py::test_collect_api_keys_combina_y_deduplica PASSED      [  4%]
tests/test_llm.py::test_sin_claves_mensaje_claro PASSED                  [  8%]
tests/test_llm.py::test_429_rota_a_segunda_clave PASSED                  [ 13%]
tests/test_llm.py::test_401_rota_y_la_segunda_funciona PASSED            [ 17%]
tests/test_llm.py::test_401_con_una_sola_clave_se_relaza PASSED          [ 21%]
tests/test_llm.py::test_429_con_una_sola_clave_reintenta_con_backoff PASSED [ 26%]
tests/test_llm.py::test_timeout_reintenta_misma_clave_sin_rotar PASSED   [ 30%]
tests/test_llm.py::test_cuota_en_todas_las_claves_agota_y_relaza PASSED  [ 34%]
tests/test_llm.py::test_error_no_transitorio_se_relaza_sin_rotar PASSED  [ 39%]
tests/test_llm.py::test_collect_api_keys_vacio PASSED                    [ 43%]
tests/test_chain.py::test_excerpt_colapsa_saltos_y_recorta PASSED        [ 47%]
tests/test_chain.py::test_extract_citations_deduplica_y_tolera_espacios PASSED [ 52%]
tests/test_chain.py::test_extract_citations_sin_citas PASSED             [ 56%]
tests/test_chain.py::test_citas_se_mapean_a_fuentes PASSED               [ 60%]
tests/test_chain.py::test_cita_no_verificable_se_descarta PASSED         [ 65%]
tests/test_chain.py::test_cita_con_variacion_de_seccion_se_verifica PASSED [ 69%]
tests/test_chain.py::test_no_info_retorna_sources_vacios PASSED          [ 73%]
tests/test_chain.py::test_chunks_vacios_retorna_sources_vacios PASSED    [ 78%]
tests/test_chain.py::test_umbral_pre_llm_no_gasta_tokens PASSED          [ 82%]
tests/test_chain.py::test_llm_no_info_marca_fallback PASSED              [ 86%]
tests/test_chain.py::test_respuesta_vacia_se_degrada_a_no_info PASSED    [ 91%]
tests/test_chain.py::test_respuesta_con_citas_genera_sources PASSED      [ 95%]
tests/test_chain.py::test_prompt_incluye_pregunta_contexto_y_no_info PASSED [100%]
============================== 23 passed in 0.24s ==============================

# Comando 3 (lint sobre todo el repo):
$ uv run ruff check src scripts tests
All checks passed!
```

Criterio de aceptación de esta PR: la deuda de pruebas registrada en la Fase 4 queda saldada — la rotación de claves, el post-proceso de citas y la orquestación `answer_question` tienen cobertura permanente que corre en ~1.2 s sin red ni tokens.

#### 🔄 Pasos para replicar / probar manualmente
1. `git clone ... && cd Asistente-agentico-UAO && uv sync` (no se necesita `CEREBRAS_API_KEY` ni índice de Chroma para los tests nuevos).
2. Ejecutar solo las pruebas nuevas: `uv run pytest tests/test_llm.py tests/test_chain.py -v`.
3. Ejecutar la suite completa con lint: `uv run pytest -q && uv run ruff check src scripts tests`.
4. (Opcional) Verificar que nada cambió en producción: `git diff <rama-base> -- src/ scripts/` debe estar vacío.

#### 🖼️ Capturas / salida visual (opcional)
N/A: son pruebas unitarias sin salida visual; la evidencia es la salida de pytest/ruff pegada arriba.

---

## 8. 👀 Notas para el Revisor

- **Cobertura deliberadamente unitaria**: los tests usan dobles de prueba (stub de `ChatCerebras`, `FakeRetriever`, `FakeLLM`); la validación end-to-end real contra el índice (1284 chunks) ya se hizo en la Fase 4 con `scripts/ask.py` y no se repite aquí para no gastar tokens. La evaluación con banco de ~30 preguntas sigue siendo tarea de F6.
- **El stub reemplaza `langchain_cerebras` en `sys.modules`** (módulo falso), no se hace monkeypatch sobre la librería real: los tests no importan la dependencia y siguen verificando que la rotación recrea el cliente con la nueva clave (`FakeChatCerebras.created` registra cada cliente construido y su `api_key`).
- **Los retardos de backoff no se esperan**: `time.sleep` se graba en una lista; los tests afirman los valores exactos (1s, 2s), así la suite completa sigue en ~1.2 s.
- **`Settings` real con claves ficticias**: los tests de rotación construyen `Settings(cerebras_api_key="k1", ...)` con kwargs explícitos, que tienen prioridad sobre `.env` y variables de entorno; por eso el test es determinista aunque la máquina tenga un `.env` con claves reales.
- **`GuardLLM`** garantiza la tarea 4.4 por construcción: si `answer_question` invocara al LLM sin contexto, el test falla.
- **No se cubre** (quedan para F6): exactitud semántica de las respuestas, recall del banco de preguntas, recalibración de `min_similarity`, ni pruebas de la API (F5 añadirá sus propios tests con `TestClient`).
- Los hallazgos y decisiones de la Fase 4 (umbral en 0.35, `disable_reasoning=True`, rotación por proceso) siguen vigentes y sin cambios; esta PR no altera ninguna.

---

## 9. 🔗 Referencias

- PR anterior de la Fase 4 (`feat: cadena RAG con LLM Cerebras y rotación de claves API`): sección 6 registró la deuda de pruebas que esta PR salda.
- `plan-trabajo-tecnico.md` — Fase 4 (tareas 4.1–4.6) y notas para F6.
- Código bajo prueba: `src/asistente_agentico_uao/llm.py` (`CerebrasLLM`, `collect_api_keys`, `NO_INFO_MESSAGE`) y `src/asistente_agentico_uao/chain.py` (`answer_question`, `build_sources`, `extract_citations`, `_excerpt`).
- Documentación externa: [pytest monkeypatch](https://docs.pytest.org/en/stable/reference/reference.html#monkeypatch), [unittest.mock / dobles de prueba](https://docs.python.org/3/library/unittest.mock.html), [LangChain Runnable (to_string)](https://python.langchain.com/docs/concepts/runnables/).
- Fases previas de esta rama: F1 LlamaCloud Parse, F2 chunking+embeddings+Chroma, F3 recuperación.
