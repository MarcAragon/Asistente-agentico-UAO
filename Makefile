.PHONY: install test ingest ingest-rebuild ingest-prune api frontend docker-build docker-up docker-down docker-logs redis-test clean

install:
	uv sync

test:
	uv run pytest

ingest:
	uv run python scripts/ingest.py

ingest-rebuild:
	uv run python scripts/ingest.py --rebuild

ingest-prune:
	uv run python scripts/ingest.py --prune

api:
	uv run uvicorn asistente_agentico_uao.api.main:app --host 0.0.0.0 --port 8000

frontend:
	uv run streamlit run src/asistente_agentico_uao/frontend/app.py

docker-build:
	docker compose build

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

redis-test:
	docker compose exec api uv run python -c "import redis; r=redis.Redis.from_url('redis://redis:6379/0'); print('Redis:', r.ping())"

clean:
	docker compose down
