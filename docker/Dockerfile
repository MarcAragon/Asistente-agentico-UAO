# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.12.2

# Nueva etapa: solo referencia la imagen oficial de uv, usando el ARG global.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ------------------------------------------------------------------ builder --
FROM python:${PYTHON_VERSION}-slim AS builder

# Ahora --from apunta a la etapa "uv" (nombre local), no a una imagen remota.
COPY --from=uv /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    VIRTUAL_ENV=/app/.venv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# ------------------------------------------------------------------ runtime --
FROM python:${PYTHON_VERSION}-slim AS runtime

# libgomp1: OpenMP que requieren torch y onnxruntime (chromadb) en slim.
# ca-certificates: HTTPS hacia la API de Cerebras y hacia Hugging Face.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios. Por defecto uid/gid 1000 (el usuario típico del
# host) para que el bind mount ./Data y los volúmenes sean escribibles; se
# puede ajustar en el build: --build-arg APP_UID=$(id -u).
ARG APP_UID=1000
ARG APP_GID=1000
RUN set -eu; \
    if getent group "${APP_GID}" >/dev/null; then \
        GROUP_NAME="$(getent group "${APP_GID}" | cut -d: -f1)"; \
    else \
        GROUP_NAME=app; \
        groupadd --gid "${APP_GID}" "${GROUP_NAME}"; \
    fi; \
    if getent passwd "${APP_UID}" >/dev/null; then \
        echo "UID ${APP_UID} ya existe, se reutiliza"; \
    else \
        useradd --uid "${APP_UID}" --gid "${APP_GID}" \
            --no-create-home --shell /usr/sbin/nologin app; \
    fi

WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV=/app/.venv \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/app \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false

COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts

# Puntos de montaje de runtime (plan 8.3): ./Data (corpus, markdown e índice
# ya construido) y /models (caché del modelo de embeddings). Se crean aquí
# con el propietario correcto para que Docker inicialice los volúmenes
# nombrados heredando ese dueño y el usuario `app` pueda escribir.
RUN mkdir -p /app/Data /models && chown -R "${APP_UID}:${APP_GID}" /app /models

USER ${APP_UID}:${APP_GID}

# 8000 = REST de consulta (data plane) · 50051 = gRPC IndexAdmin (control
# plane). Ninguno se publica al host: solo la red interna del compose.
EXPOSE 8000 50051

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

# --root-path /api: el proxy TLS sirve la API bajo https://<host>/api/*;
# este flag solo corrige las URLs generadas en la documentación OpenAPI
# (https://<host>/api/docs) y no afecta a los endpoints.
CMD ["uvicorn", "asistente_agentico_uao.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--root-path", "/api"]