# ==============================================================================
#  Asistente RAG UAO — Makefile
#
#  Atajos de desarrollo, operación y despliegue. Todo el proyecto se ejecuta
#  con `uv` (nunca pip) y la solución completa se levanta con Docker Compose
#  (Fase 8 del plan-trabajo-tecnico.md).
#
#  `make` o `make help`  → lista todos los targets disponibles.
# ==============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Parámetros sobrescribibles desde la línea de comandos,
# p. ej.: make ask QUESTION="¿hasta cuándo puedo cancelar?"
# PROXY_URL: superficie pública (proxy TLS) · API_URL: API local sin proxy.
# MLFLOW_URL: dashboard de MLflow por el proxy (subdominio mlflow.<SITE_ADDRESS>).
# FILE: filtro parcial de documento para parse/ingest · ARGS: extra para CLIs.
# Nota: sin comentarios al final de estas líneas (Make los deja como espacios).
COMPOSE   ?= docker compose
PROXY_URL ?= https://localhost
MLFLOW_URL ?= https://mlflow.localhost
API_URL   ?= http://localhost:8000
QUESTION  ?= ¿Qué pasa si repruebo tres veces una misma asignatura?
FILE      ?=
ARGS      ?=

# uid/gid del host para que los bind mounts (./Data) sean escribibles desde
# el contenedor sin privilegios (docker/Dockerfile, ARG APP_UID/APP_GID).
APP_UID := $(shell id -u)
APP_GID := $(shell id -g)
export APP_UID APP_GID

CYAN := \033[36m
BOLD := \033[1m
RESET := \033[0m

# -------------------------------------------------------------------- ayuda --
.PHONY: help
##@ Ayuda

help: ## Muestra esta ayuda (targets documentados con `##`)
	@printf "\n$(BOLD)Asistente RAG UAO$(RESET) — Universidad Autónoma de Occidente\n"
	@printf "Atajos: make <target> · documentación: README.md y docs/guia-despliegue.md\n"
	@awk 'BEGIN {FS = ":.*?## "} \
		/^##@/ {printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5)} \
		/^[a-zA-Z0-9_-]+:.*?## / {printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)
	@printf "\n"

# -------------------------------------------------- entorno y calidad (dev) --
##@ Entorno y calidad

.PHONY: install sync env-init lint format check test test-fast test-slow proto
install: ## Instala/replica el entorno desde uv.lock (uv sync)
	uv sync

sync: install ## Alias de `install`

env-init: ## Crea .env desde .env.example si no existe (secretos locales)
	@if [ -f .env ]; then \
		echo "  .env ya existe (no se toca)"; \
	else \
		cp .env.example .env; \
		echo "  .env creado. Completa CEREBRAS_API_KEY y LLAMA_CLOUD_API_KEY."; \
	fi

lint: ## Revisa estilo y posibles errores con ruff
	uv run ruff check src scripts tests

format: ## Formatea el código con ruff
	uv run ruff format src scripts tests
	uv run ruff check --fix src scripts tests

check: lint test ## Pipeline de verificación completo (lint + tests)

test: ## Ejecuta toda la suite de pytest
	uv run pytest

test-fast: ## Suite rápida (sin las pruebas marcadas como `slow`)
	uv run pytest -m "not slow"

test-slow: ## Solo las pruebas de integración (embeddings reales + Chroma)
	uv run pytest -m slow

proto: ## Regenera los stubs gRPC desde grpc_impl/protos/index_admin.proto
	uv run python scripts/gen_proto.py
# ------------------------------------------------------ pipeline de datos --
##@ Pipeline de datos (parseo e ingesta)

.PHONY: parse parse-redo parse-file ingest ingest-rebuild ingest-prune
parse: ## Parsea los PDFs con LlamaCloud Parse (omite los ya parseados)
	uv run python scripts/llama_cloud_parsing.py

parse-redo: ## Re-parsea TODOS los PDFs (sobrescribe Data/Documentos_MD/*.md)
	uv run python scripts/llama_cloud_parsing.py --redo

parse-file: ## Parsea un PDF por coincidencia parcial (FILE=666)
	uv run python scripts/llama_cloud_parsing.py --file "$(FILE)"

ingest: ## Indexa el markdown en Chroma (chunking + embeddings E5)
	uv run python scripts/ingest.py

ingest-rebuild: ## Reconstruye el índice desde cero (borra la colección)
	uv run python scripts/ingest.py --rebuild

ingest-prune: ## Elimina del índice los documentos ya ausentes
	uv run python scripts/ingest.py --prune

# ------------------------------------------------------- desarrollo local --
##@ Desarrollo local (sin Docker)

.PHONY: api grpc frontend ask-cli smoke
api: ## Arranca la API REST en :8000 (gRPC embebido en :50051)
	uv run uvicorn asistente_agentico_uao.api.main:app --host 0.0.0.0 --port 8000

grpc: ## Arranca SOLO el control plane gRPC (standalone, :50051)
	uv run python -m asistente_agentico_uao.grpc_impl.server

frontend: ## Arranca el chat Streamlit contra la API local
	uv run streamlit run src/asistente_agentico_uao/frontend/app.py

ask-cli: ## Prueba la cadena RAG por CLI (QUESTION=... o banco de humo)
	uv run python scripts/ask.py "$(QUESTION)" $(ARGS)

smoke: ## Humo del motor de recuperación sobre el índice real
	uv run python scripts/smoke_retrieval.py $(ARGS)
# ------------------------------------------------------------------ Docker --
##@ Docker Compose (Fase 8)

.PHONY: config build up down restart ps logs logs-api logs-frontend logs-proxy
.PHONY: logs-redis logs-mlflow shell redis-cli models-prefetch health ask urls

config: ## Valida docker-compose.yml y la interpolación de variables
	$(COMPOSE) config -q && echo "  docker-compose.yml válido"

build: ## Construye las imágenes de backend y frontend
	$(COMPOSE) build

up: ## Levanta la solución completa (build + healthchecks + precarga del modelo)
	$(COMPOSE) up -d --build --wait --wait-timeout 300
	@$(MAKE) --no-print-directory models-prefetch
	@$(MAKE) --no-print-directory urls

down: ## Detiene y elimina los contenedores (CONSERVA índice y volúmenes)
	$(COMPOSE) down

restart: ## Reinicia los servicios ya construidos
	$(COMPOSE) restart

ps: ## Estado y salud de los contenedores
	$(COMPOSE) ps

logs: ## Sigue los logs de todos los servicios (Ctrl-C para salir)
	$(COMPOSE) logs -f --tail=100

logs-api: ## Logs de la API (REST + gRPC)
	$(COMPOSE) logs -f --tail=100 api

logs-frontend: ## Logs del chat Streamlit
	$(COMPOSE) logs -f --tail=100 frontend

logs-proxy: ## Logs del proxy TLS (Caddy)
	$(COMPOSE) logs -f --tail=100 proxy

logs-redis: ## Logs de Redis (caché semántico)
	$(COMPOSE) logs -f --tail=100 redis

logs-mlflow: ## Logs del tracking server de MLflow (observabilidad, Fase 9)
	$(COMPOSE) logs -f --tail=100 mlflow

shell: ## Shell interactiva dentro del contenedor de la API
	$(COMPOSE) exec api bash

redis-cli: ## Cliente redis-cli contra el Redis del compose
	$(COMPOSE) exec redis redis-cli

models-prefetch: ## Descarga el modelo de embeddings al volumen (solo la 1ª vez)
	@echo "  Precargando el modelo de embeddings (se cachea en el volumen /models)..."
	@$(COMPOSE) exec -T api python -c "\
from asistente_agentico_uao.core.embeddings import get_model; \
print('  Modelo listo en device:', get_model().device)"

health: ## Estado de la API a través del proxy TLS (chunks, device, caché)
	@curl -sk $(PROXY_URL)/api/health | python3 -m json.tool

ask: ## Pregunta al asistente a través del proxy TLS (QUESTION=...)
	@curl -sk -X POST $(PROXY_URL)/api/ask \
		-H 'Content-Type: application/json' \
		--data '{"question": "$(QUESTION)"}' | python3 -m json.tool

urls: ## Muestra las URLs de acceso y los puertos publicados
	@echo "  Chat (HTTPS):   $(PROXY_URL)/"
	@echo "  API (HTTPS):    $(PROXY_URL)/api/health   ·   docs en /api/docs"
	@echo "  MLflow (HTTPS): $(MLFLOW_URL)/   ·   experimento 'asistente-uao'"
	@echo "  Publicados al host: 80 y 443 (solo el proxy)"
	@echo "  Internos (no publicados): api:8000, gRPC:50051, frontend:8501, redis:6379, mlflow:5000"
# -------------------------------------- operación dentro del contenedor --
##@ Operación en contenedores (ingesta, estado, caché)

.PHONY: ingest-docker ingest-rebuild-docker grpc-status cache-flush cache-stats
ingest-docker: ## Indexa dentro del contenedor (usa el ./Data montado)
	$(COMPOSE) exec -T api python scripts/ingest.py

ingest-rebuild-docker: ## Reconstruye el índice dentro del contenedor
	$(COMPOSE) exec -T api python scripts/ingest.py --rebuild

grpc-status: ## Estado del índice por gRPC (control plane interno, :50051)
	$(COMPOSE) exec -T api python scripts/ingest_client.py --status

cache-flush: ## Invalida el caché semántico (tras reindexar el corpus)
	$(COMPOSE) exec -T redis sh -c \
		"redis-cli --scan --pattern 'uao_rag:cache:*' | xargs -r redis-cli del"

cache-stats: ## Muestra los contadores hit/miss del caché semántico
	$(COMPOSE) exec -T redis sh -c \
		"echo -n 'hits: ';   redis-cli get uao_rag:cache:stats:hits; \
		 echo -n 'misses: '; redis-cli get uao_rag:cache:stats:misses"

# -------------------------------------------------- respaldos y limpieza --
##@ Respaldos y limpieza

.PHONY: index-backup index-restore clean clean-volumes clean-images disk
index-backup: ## Respalda el índice Chroma en Data/chroma_backup/ (tar.gz)
	@mkdir -p Data/chroma_backup
	@tar czf "Data/chroma_backup/chroma-$$(date +%Y%m%d-%H%M%S).tar.gz" -C Data chroma
	@ls -1t Data/chroma_backup/*.tar.gz | head -1 | xargs -I{} echo "  Respaldo creado: {}"

index-restore: ## Restaura el índice desde FILE=<tar.gz> (respalda el actual antes)
	@test -n "$(FILE)" || { \
		echo "  Uso: make index-restore FILE=Data/chroma_backup/chroma-....tar.gz"; \
		echo "  Respaldos disponibles:"; \
		ls -1t Data/chroma_backup/*.tar.gz 2>/dev/null || echo "    (ninguno)"; \
		exit 1; }
	@test -f "$(FILE)" || { echo "  No existe $(FILE)"; exit 1; }
	@echo "  Detén los servicios antes de restaurar: make down"
	@mkdir -p Data/chroma_backup
	@tar czf "Data/chroma_backup/chroma-pre-restore-$$(date +%Y%m%d-%H%M%S).tar.gz" -C Data chroma
	@rm -rf Data/chroma && tar xzf "$(FILE)" -C Data
	@echo "  Índice restaurado desde $(FILE)"

clean: ## Detiene los contenedores (conserva índice, caché y certificados)
	$(COMPOSE) down --remove-orphans

clean-volumes: ## Elimina además los volúmenes (caché, modelo y certificados)
	@echo "  Aviso: borra caché semántico, modelo de embeddings y certificados TLS."
	$(COMPOSE) down -v --remove-orphans

clean-images: ## Elimina las imágenes construidas localmente
	-$(COMPOSE) down --rmi local --remove-orphans

disk: ## Uso de disco de Docker (imágenes, volúmenes, caché de build)
	docker system df

# ------------------------------- compatibilidad con los atajos previos --
.PHONY: docker-build docker-up docker-down docker-logs redis-test
docker-build: build ## Alias de `build`
docker-up: up ## Alias de `up`
docker-down: down ## Alias de `down`
docker-logs: logs ## Alias de `logs`
redis-test: ## Comprueba la conectividad con el Redis del compose
	$(COMPOSE) exec -T api python -c "\
import os, redis; print('Redis:', redis.Redis.from_url(os.environ['UAO_RAG__REDIS_URL']).ping())"
