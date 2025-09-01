FROM python:3.12-slim AS builder

ENV POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir poetry==${POETRY_VERSION}

# Copy project definition and install dependencies first (no source yet)
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

# Now add sources and install the package itself
COPY reactions_generator ./reactions_generator
COPY README.md ./
RUN poetry install --only main


FROM python:3.12-slim AS runtime

# System deps: ffmpeg for video/audio processing (Debian-based python image)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# Bring the virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy runtime assets used by the app at execution time
COPY example ./example

# Default entrypoint to the CLI defined in pyproject [tool.poetry.scripts]
ENTRYPOINT ["main"]
CMD ["--help"]
