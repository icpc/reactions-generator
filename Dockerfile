FROM python:3.12-slim

# System deps: ffmpeg for video/audio processing (Debian-based Python image)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    FONTS_DIR=/app/fonts

WORKDIR /app

# Install Poetry and base tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel poetry

# Install dependencies first (cached layer) without installing the project itself
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root

# Now copy sources and install the package
COPY reactions_generator ./reactions_generator
COPY README.md ./
COPY example ./example
COPY fonts ./fonts
RUN poetry install --only main

# Default entrypoint to the CLI defined in pyproject [tool.poetry.scripts]
ENTRYPOINT ["main"]
CMD ["--help"]
