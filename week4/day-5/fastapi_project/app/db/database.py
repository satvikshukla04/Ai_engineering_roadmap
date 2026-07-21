"""SQLAlchemy engine, session factory, and FastAPI dependency."""
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()

# `check_same_thread` is only needed for SQLite; harmless to set conditionally.
connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

# For file-based SQLite URLs (e.g. sqlite:///./data/documents.db), make sure
# the parent directory exists so the very first connection doesn't fail.
if settings.database_url.startswith("sqlite:///") and settings.database_url not in (
    "sqlite:///:memory:",
    "sqlite://",
):
    _db_path = Path(settings.database_url.removeprefix("sqlite:///"))
    if str(_db_path) != ":memory:":
        _db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Called once at application startup."""
    Base.metadata.create_all(bind=engine)


def check_db_connection() -> bool:
    """Used by the /ready endpoint to verify the DB is reachable."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - readiness check must never raise
        return False
