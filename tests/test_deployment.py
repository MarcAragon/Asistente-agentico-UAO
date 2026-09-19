"""Pruebas del despliegue con Docker Compose (Fase 8).

Validan **estáticamente** los artefactos de contenerización —no necesitan
Docker ni la red— para que una regresión de configuración (un puerto publicado
de más, un servicio sin healthcheck, un secreto en un archivo versionado) se
detecte en `uv run pytest`:

- `docker-compose.yml`: servicios (api, frontend, proxy, redis y mlflow),
  red interna y ausencia de puertos internos.
- `docker/Dockerfile*`: uv con lock congelado, usuario sin privilegios, `.env`
  fuera del contexto de build.
- `docker/Caddyfile`: TLS, rutas `/api/*` y `/`, y healthcheck interno.
- `.env.example` y `docs/guia-despliegue.md`: documentación vigente.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
COMPOSE_PATH = RAIZ / "docker-compose.yml"

SERVICIOS = {"api", "frontend", "proxy", "redis", "mlflow"}
PUERTOS_PUBLICOS = {"80", "443"}

# Formatos reales de las claves del proyecto (Cerebras y LlamaCloud): si
# aparecen en un archivo versionado, se filtraron secretos al repositorio.
PATRONES_SECRETO = (
    re.compile(r"csk-[A-Za-z0-9]{20,}"),
    re.compile(r"llx-[A-Za-z0-9]{20,}"),
)
ARCHIVOS_VERSIONADOS = (
    "docker-compose.yml",
    ".dockerignore",
    ".env.example",
    "Makefile",
    "README.md",
    "docs/plan-trabajo-tecnico.md",
    "docs/asistente-uao-rag.md",
    "docker/Dockerfile",
    "docker/Dockerfile.frontend",
    "docker/Caddyfile",
    "docs/guia-despliegue.md",
)


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def _leer(ruta_relativa: str) -> str:
    return (RAIZ / ruta_relativa).read_text(encoding="utf-8")


def test_compose_define_los_cinco_servicios(compose):
    assert set(compose["services"]) == SERVICIOS


def test_solo_el_proxy_publica_puertos(compose):
    """API, gRPC, Streamlit, Redis y MLflow viven en la red interna (§7.7/§8.5)."""
    servicios = compose["services"]
    for nombre, servicio in servicios.items():
        if nombre == "proxy":
            continue
        assert "ports" not in servicio, f"{nombre} no debe publicar puertos al host"
        assert servicio.get("expose"), f"{nombre} debe declarar puertos internos"

    publicados = {str(puerto).split(":")[0] for puerto in servicios["proxy"]["ports"]}
    assert publicados == PUERTOS_PUBLICOS


def test_mlflow_es_solo_interno_y_se_sirve_por_el_proxy(compose):
    """Observabilidad (Fase 9): el tracking server no publica puertos.

    El dashboard se consume vía Caddy (`https://mlflow.<SITE_ADDRESS>`), igual
    que el resto de la solución: publicar el ``:5000`` al host rompería el
    invariante de F8. La API le envía las trazas por la red interna.
    """
    mlflow = compose["services"]["mlflow"]
    assert "ports" not in mlflow, "el dashboard se sirve por el proxy, no al host"
    assert "5000" in mlflow["expose"]
    assert "mlflow_data:/mlflow" in mlflow["volumes"]
    assert "sqlite:////mlflow/mlflow.db" in mlflow["command"]
    assert mlflow["healthcheck"]["test"], "sin healthcheck no hay depends_on fiable"

    # La API instrumenta el LLM solo si el entorno define MLFLOW_TRACKING_URI
    # (mlflow.openai.autolog, ver api/main.py) y espera al tracking server.
    api = compose["services"]["api"]
    assert api["environment"]["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert api["depends_on"]["mlflow"]["condition"] == "service_healthy"

    # El dashboard se publica mediante el proxy (misma regla que frontend/api).
    caddyfile = _leer("docker/Caddyfile")
    assert "mlflow.{$SITE_ADDRESS:localhost}" in caddyfile
    assert "reverse_proxy mlflow:5000" in caddyfile


def test_todos_los_servicios_usan_la_red_interna(compose):
    assert "internal" in compose["networks"]
    for nombre, servicio in compose["services"].items():
        assert servicio["networks"] == ["internal"], f"{nombre} sin red interna"


def test_healthchecks_y_dependencias(compose):
    servicios = compose["services"]
    for nombre, servicio in servicios.items():
        assert "healthcheck" in servicio, f"{nombre} sin healthcheck"

    assert servicios["api"]["depends_on"]["redis"]["condition"] == "service_healthy"
    assert servicios["frontend"]["depends_on"]["api"]["condition"] == "service_healthy"
    assert (
        servicios["proxy"]["depends_on"]["frontend"]["condition"] == "service_healthy"
    )


def test_api_comparte_el_indice_y_cachea_el_modelo(compose):
    api = compose["services"]["api"]
    volumenes = api["volumes"]
    assert "./Data:/app/Data" in volumenes, "el índice debe montarse desde el host"
    assert "models_cache:/models" in volumenes
    assert api["environment"]["HF_HOME"] == "/models"
    assert set(compose["volumes"]) >= {
        "redis_data",
        "models_cache",
        "caddy_data",
        "mlflow_data",
    }


def test_conexiones_internas_entre_servicios(compose):
    servicios = compose["services"]
    assert (
        servicios["api"]["environment"]["UAO_RAG__REDIS_URL"] == "redis://redis:6379/0"
    )
    assert (
        servicios["frontend"]["environment"]["UAO_RAG__API_BASE_URL"]
        == "http://api:8000"
    )
    # Rutas explícitas: el paquete se importa desde /app/src en el contenedor.
    assert servicios["api"]["environment"]["UAO_RAG__CHROMA_DIR"] == "/app/Data/chroma"


def test_dockerfile_backend_reproducible_y_sin_privilegios():
    contenido = _leer("docker/Dockerfile")
    assert "uv sync --frozen --no-dev" in contenido
    assert "uv sync --frozen --no-dev --no-install-project" in contenido
    assert "PYTHONPATH=/app/src" in contenido
    # El uid/gid llega por build-arg (APP_UID/APP_GID del host); la directiva
    # final de usuario sin privilegios es USER ${APP_UID}:${APP_GID}.
    assert "USER ${APP_UID}" in contenido
    assert "HEALTHCHECK" in contenido
    assert "pip install" not in contenido
    assert "ARG PYTHON_VERSION=3.14" in contenido


def test_dockerfile_frontend_es_ligero():
    contenido = _leer("docker/Dockerfile.frontend")
    assert "streamlit==" in contenido and "httpx==" in contenido
    assert "uv pip install" in contenido
    # Sin dependencias de ML: el chat solo consume la API por HTTP. Se revisan
    # las instrucciones (sin comentarios, que sí pueden nombrarlas).
    instrucciones = "\n".join(
        linea for linea in contenido.splitlines() if not linea.strip().startswith("#")
    )
    for prohibido in ("torch", "chromadb", "sentence-transformers"):
        assert prohibido not in instrucciones
    assert "USER app" in contenido


def test_dockerignore_excluye_secretos_y_datos():
    lineas = _leer(".dockerignore").splitlines()
    for patron in (".env", ".venv", "Data/chroma", "!.env.example"):
        assert patron in lineas, f"falta {patron} en .dockerignore"


def test_caddyfile_tiene_tls_rutas_y_healthcheck():
    contenido = _leer("docker/Caddyfile")
    assert "tls {$TLS_DIRECTIVE:internal}" in contenido
    assert "handle_path /api/*" in contenido
    assert "reverse_proxy api:8000" in contenido
    assert "reverse_proxy frontend:8501" in contenido
    assert "handle /healthz" in contenido
    assert ":8080" in contenido, "sitio interno para el healthcheck del contenedor"
    assert "Strict-Transport-Security" in contenido


def test_env_example_documenta_el_despliegue():
    contenido = _leer(".env.example")
    for variable in (
        "SITE_ADDRESS=",
        "TLS_DIRECTIVE=",
        "UAO_RAG__REDIS_URL=",
        "UAO_RAG__CACHE_ENABLED=",
        "UAO_RAG__API_BASE_URL=",
    ):
        assert variable in contenido, f"{variable} no está documentada"


def test_makefile_expone_los_atajos_de_la_fase_8():
    contenido = _leer("Makefile")
    for target in (
        "up: ##",
        "down: ##",
        "models-prefetch: ##",
        "index-backup: ##",
        "index-restore: ##",
        "cache-flush: ##",
        "grpc-status: ##",
    ):
        assert target in contenido, f"falta el target {target!r} en el Makefile"


def test_guia_de_despliegue_cubre_los_puntos_clave():
    contenido = _leer("docs/guia-despliegue.md")
    for seccion in (
        "## 4. Variables de entorno",
        "## 5. Despliegue paso a paso",
        "## 6. Persistencia y respaldos",
        "## 7. TLS y certificados",
        "## 11. Solución de problemas",
        "## 12. Checklist de despliegue",
    ):
        assert seccion in contenido, f"falta la sección {seccion!r}"


def test_sin_secretos_en_archivos_versionados():
    """Ninguna clave real (Cerebras/LlamaCloud) en archivos del repositorio."""
    for ruta in ARCHIVOS_VERSIONADOS:
        contenido = _leer(ruta)
        for patron in PATRONES_SECRETO:
            assert not patron.search(contenido), f"posible secreto en {ruta}"


def test_env_local_esta_ignorado():
    """.env (con las claves reales) no se versiona ni entra al contexto de build."""
    lineas_git = {linea.strip() for linea in _leer(".gitignore").splitlines()}
    lineas_docker = {linea.strip() for linea in _leer(".dockerignore").splitlines()}
    assert ".env" in lineas_git, ".env debe estar en .gitignore"
    assert ".env" in lineas_docker, ".env debe estar en .dockerignore"
