# Documents API

A production-ready FastAPI service exposing CRUD endpoints for a `documents`
resource, backed by SQLAlchemy, with health/readiness probes, centralized
Pydantic Settings configuration, and a multi-stage Docker build.

## Project layout

```
app/
  db/
    database.py     # SQLAlchemy engine, session, Base, get_db dependency
    models.py        # ORM models
  routers/
    documents.py      # /documents CRUD endpoints
    health.py         # /health and /ready endpoints
  schemas/
    document.py        # Pydantic request/response models
    health.py           # Pydantic response models for health/ready
  services/
    document_service.py  # Business logic, separate from HTTP layer
  config.py            # Pydantic Settings (all config lives here)
  main.py               # App factory, startup/shutdown, router wiring
  middleware.py         # Request logging middleware
tests/                    # pytest suite (uses an isolated in-memory DB)
Dockerfile                 # Multi-stage production build
requirements.txt            # Runtime dependencies
requirements-dev.txt         # + test/lint/type-check dependencies
```

## Configuration

All configuration is defined in `app/config.py` via `pydantic-settings` and
can be supplied through environment variables or a `.env` file. See
`.env.example` for the full list of options (app name/version, environment,
host/port, database URL, CORS origins, log level).

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API docs.

## Running tests

```bash
pytest
```

## Endpoints

| Method | Path              | Description                     |
|--------|-------------------|----------------------------------|
| GET    | `/health`         | Liveness probe                  |
| GET    | `/ready`          | Readiness probe (checks DB)     |
| POST   | `/documents`      | Create a document                |
| GET    | `/documents`      | List documents (paginated)       |
| GET    | `/documents/{id}` | Get a single document            |
| PATCH  | `/documents/{id}` | Partially update a document      |
| DELETE | `/documents/{id}` | Delete a document                |

## Docker

Build:

```bash
docker build -t documents-api .
```

Run:

```bash
docker run -p 8000:8000 documents-api
```

Then check:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

The image uses a multi-stage build (a `builder` stage that installs
dependencies with build tools, and a slim `runtime` stage that only contains
the installed packages and app source), runs as a non-root user, and ships
a container-level `HEALTHCHECK` that calls `/health`.
