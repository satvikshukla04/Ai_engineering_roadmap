"""SQLite engine + a get_session dependency for FastAPI routes.

Using SQLite because this is a learning project and I want zero setup
(no docker, no external DB) to run it or the tests. Swapping the URL for
Postgres later would just mean changing DATABASE_URL.
"""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlmodel import Session, SQLModel, create_engine

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./genai_chat_api.db")

# check_same_thread=False is needed because FastAPI can use the connection
# from a different thread than the one that created it. SQLite is fine with
# this as long as we're not sharing a single Session across threads, which
# we aren't (get_session opens a fresh one per request).
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)


def init_db() -> None:
    """Create tables if they don't exist yet. Called once on app startup."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session and closes it after."""
    with Session(engine) as session:
        yield session
