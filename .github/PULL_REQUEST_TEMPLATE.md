# 📋 Plantilla de Pull Request — Asistente Agéntico RAG UAO

<!--
Instrucciones de uso:
- Completa TODAS las secciones. Si una sección no aplica, escribe "N/A" y justifica brevemente.
- El título de la PR debe seguir Convencional Commits:
  feat: | fix: | refactor: | docs: | chore: | test:
  Ej: "feat: pipeline de preprocesamiento de documentos con LlamaCloud Parse"
- Elimina los comentarios HTML antes de abrir la PR.
-->

## 1. 🧩 Tipo de Pull Request

Marca **uno** como principal (y otros si esta PR es mixta):

- [ ] `feat` — Nueva funcionalidad (pipeline, ingesta, recuperación, API…)
- [ ] `fix` — Corrección de un bug
- [ ] `refactor` — Cambio interno sin alterar comportamiento externo
- [ ] `docs` — Solo documentación
- [ ] `chore` — Infraestructura, dependencias, configuración, CI
- [ ] `test` — Adición o corrección exclusiva de pruebas

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
- [ ] **Pruebas Unitarias** (`tests/`, `pytest`)
- [ ] **Infraestructura & Empaquetado** (`pyproject.toml`, `uv.lock`, `uv_build`, `.gitignore`)
- [ ] **Documentación** (`README.md`, `plan-trabajo-tecnico.md`, licencia)

---

## 3. 📚 Descripción de los Cambios

**Tarea / fase asociada:** <!-- Ej. "Fase 1 del plan-trabajo-tecnico.md: preprocesamiento de documentos" -->
<!-- Issue relacionado, si existe: Closes #NN -->

#### 🎯 Motivación y contexto
<!-- ¿Qué problema resuelve esta PR? ¿Por qué es necesaria ahora? ¿Qué alternativa se descartó y por qué? -->

#### 📝 Resumen de cambios
<!-- Lista concisa, en viñetas, de QUÉ cambió (no cómo). -->

#### 🔧 Detalles técnicos relevantes
<!-- El "cómo": decisiones de diseño, patrones usados, dependencias nuevas (con versión), trade-offs, y por qué la solución elegida es la correcta. Incluye fragmentos de código solo si son esenciales. -->

---

## 4. ⚠️ Impacto y Compatibilidad

- [ ] Esta PR **NO** rompe compatibilidad (cambios retrocompatibles)
- [ ] Esta PR introduce **cambios que rompen** (documentar abajo)

**Si hay breaking changes / migración, documenta:**
- Variables de entorno nuevas o renombradas: <!-- Ej. `LLAMA_CLOUD_API_KEY` ahora se lee vía `Settings` -->
- Pasos de migración para quien tenga el repo clonado: <!-- Ej. "copiar .env.example a .env y completar claves" -->
- Archivos/datos generados incluidos: <!-- Ej. "21 markdown en Data/Documentos_MD — regenerables con el script" -->

---

## 5. 🔐 Seguridad y Secretos

- [ ] Ningún secreto (API keys, contraseñas, tokens) está escrito en el código ni en archivos rastreados
- [ ] Los secretos viven únicamente en `.env` (gitignored) o en variables de entorno
- [ ] `.env.example` está actualizado y **rastreado** como plantilla documental
- [ ] Las claves leídas en código tienen valores por defecto vacíos y validación en tiempo de ejecución

<!-- Enumera las variables que esta PR introduce: nombre, para qué sirve y dónde obtenerla. -->

---

## 6. ✅ Lista de Chequeo Pre-PR (Estándares del Curso UAO)

- [ ] **Entorno de ejecución (`uv`):** Todo se ejecutó y probó exclusivamente con **`uv`** y la versión de Python fijada en `.python-version` / `pyproject.toml` (≥ 3.14).
- [ ] **Sin warnings:** `uv run python scripts/...` y los comandos de verificación corren limpios, sin advertencias.
- [ ] **Lint (`ruff`):** `uv run ruff check src scripts` pasa sin errores ni advertencias.
- [ ] **Control de exclusiones (`.gitignore`):** `.env`, `.venv`, `__pycache__/`, modelos pesados y datos regenerables voluminosos **NO** están rastreados por Git. `.env.example` **SÍ** lo está.
- [ ] **Estructura del código:** Fuente en `src/asistente_agentico_uao/`, utilidades/CLI en `scripts/`, pruebas en `tests/` (cuando existan).
- [ ] **Pruebas unitarias (`pytest`):** `uv run pytest` pasa al 100 %. Si la PR no incluye tests, justificar aquí por qué (fase del proyecto, alcance, plan para añadirlos).
- [ ] **Clean Code & Refactorización:** Funciones cohesivas con responsabilidad única, nombres descriptivos, docstrings en módulos y funciones públicas, sin código muerto ni comentado.
- [ ] **Un solo punto de configuración:** Los valores configurables se leen vía `Settings` (`config.py`); ningún módulo parsea `.env` por su cuenta ni hardcodea rutas o claves.
- [ ] **Reproducibilidad:** `uv sync` en un clon limpio instala todo lo necesario para ejecutar esta PR.

---

## 7. 🧪 Evidencia de Pruebas Ejecutadas

<!-- Pega la salida real (recortada si es larga) de los comandos que ejecutaste. No describas de memoria: pega lo que la terminal imprimió. -->

```
# Comando 1:
$ uv run ...
<salida>

# Comando 2:
$ uv run ruff check src scripts
<salida>
```

#### 🔄 Pasos para replicar / probar manualmente
1. <!-- Ej. `git clone ... && cd ... && uv sync` -->
2. <!-- Ej. `cp .env.example .env` y completar `LLAMA_CLOUD_API_KEY` -->
3. <!-- Ej. `uv run python scripts/llama_cloud_parsing.py --file <nombre-parcial>` -->

#### 🖼️ Capturas / salida visual (opcional)
<!-- Capturas de terminal, o del comportamiento de la API/UI si aplica. -->

---

## 8. 👀 Notas para el Revisor

<!-- Puntos específicos que quieres que el revisor mire con lupa, dudas abiertas, decisiones discutibles, o deudas técnicas aceptadas conscientemente. -->

---

## 9. 🔗 Referencias

- <!-- plan-trabajo-tecnico.md, fase X -->
- <!-- Documentación externa: pydantic-settings, LlamaCloud Parse, Cerebras… -->
- <!-- Issues o PRs relacionadas -->
