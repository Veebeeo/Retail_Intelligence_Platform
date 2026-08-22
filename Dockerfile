# syntax=docker/dockerfile:1
#
# Serving image. Deliberately excludes the training stack: the API loads a
# pickled champion bundle produced by the training job, so it needs numpy,
# pandas and statsmodels but not mlflow, xgboost or prophet.

# --- build stage ----------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-deps .

# --- runtime stage --------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# libpq5 is the runtime library; libpq-dev (headers, compiler) stays in the
# build stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser pipelines ./pipelines
COPY --chown=appuser:appuser models ./models

# Run unprivileged. The original image ran everything as root.
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# --- dashboard stage ------------------------------------------------------
# Separate target so the API image does not carry Streamlit and its transitive
# tree, and so the dashboard does not pip-install on every container start.
FROM runtime AS dashboard

USER root
COPY requirements-dash.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements-dash.txt \
    && rm requirements-dash.txt
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/dashboard.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--server.headless", "true"]
