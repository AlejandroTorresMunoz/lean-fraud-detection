# syntax=docker/dockerfile:1
#
# Multi-stage image built on uv. One image serves the whole runtime: the FastAPI scorer, the stream
# producer, and the stream consumer (the CMD picks which). Deps are resolved from uv.lock (--frozen)
# for a reproducible build; the trained model + data are NOT baked in (they are git-ignored and
# produced by training) — they are bind-mounted at run time (see docker-compose.yml).

# ---- builder: install deps + project into a venv with uv ----
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1) Deps only (cached layer): resolve from the lockfile without the project source. `--no-group dev`
#    drops the test/lint tooling; the default `agent` group stays (the consumer imports it).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-group dev

# 2) Project source, then install the project itself into the venv.
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-group dev

# ---- runtime: slim image with just the venv + code ----
FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

EXPOSE 8000
# Default role: the FastAPI scorer. The producer/consumer services override this CMD.
CMD ["uvicorn", "lean_fraud.serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
