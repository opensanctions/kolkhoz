FROM python:3.13-slim

LABEL org.opencontainers.image.title="Kolkhoz"
LABEL org.opencontainers.image.source="https://github.com/opensanctions/kolkhoz"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/

RUN groupadd --gid 10023 app \
    && useradd --uid 10023 --gid 10023 --create-home --shell /bin/bash app

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY kolkhoz ./kolkhoz
COPY evaluate.py README.md ./
RUN uv sync --frozen --no-dev

COPY datasets ./datasets
COPY fixtures ./fixtures

RUN chown -R app:app /app /opt/venv
USER app

# INPUT_BASE_PATH points at the CSVs baked into the image; OUTPUT_BASE_PATH
# and the PRAVDA_* variables must be supplied at runtime (e.g. by the k8s
# CronJob spec).
ENV INPUT_BASE_PATH=/app/datasets

ENTRYPOINT ["kolkhoz"]
CMD ["run"]
