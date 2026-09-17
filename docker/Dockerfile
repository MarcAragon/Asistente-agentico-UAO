FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY Data ./Data

RUN pip install --no-cache-dir uv \
    && uv sync --frozen

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "asistente_agentico_uao.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
