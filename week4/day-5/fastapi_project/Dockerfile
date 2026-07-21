# syntax=docker/dockerfile:1

# ---------- Stage 1: builder ----------
# Installs dependencies into an isolated location so the final image
# doesn't carry build tools, caches, or compilers.
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# ---------- Stage 2: runtime ----------
# Minimal, non-root final image.
FROM python:3.12-slim AS runtime

# Metadata
LABEL org.opencontainers.image.title="documents-api" \
      org.opencontainers.image.description="Production-ready FastAPI documents service"

# Create an unprivileged user to run the app as.
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

# Bring in the packages installed in the builder stage.
COPY --from=builder /root/.local /home/app/.local

# Copy application source.
COPY app ./app

# Writable location for the SQLite file (see DATABASE_URL in config).
RUN mkdir -p /app/data && chown -R app:app /app

ENV PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
