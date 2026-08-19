"""
Database engine, session, and initialization for the recruitment platform.

This module is the single source of truth for how we connect to SQLite in
development. It intentionally knows nothing about agents, LLMs, or the
LangGraph workflow -- it is a pure persistence layer that the rest of the
application depends on, not the other way around.
"""

import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base

# Default to a local SQLite file at the project root. Override with
# DATABASE_URL in .env if you ever need to point elsewhere (e.g. a
# throwaway file for tests, or later a Postgres URL in production).
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "screening.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

# check_same_thread=False is required for SQLite when the same connection
# might be touched from different async tasks/threads (the future FastAPI
# layer will do this). Each request/script still gets its own Session from
# SessionLocal, so this does not introduce a correctness issue on its own.
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)


def init_db() -> None:
    """Create all tables if they don't already exist.

    Safe to call multiple times -- SQLAlchemy only creates what's missing.
    This does NOT handle schema migrations (e.g. adding a column to an
    existing table); that's an Alembic concern for later, not something
    we're solving in this phase.
    """
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session():
    """Context-manager session for scripts/tests:

        with get_session() as db:
            db.add(some_object)
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db():
    """Generator-style dependency, ready for FastAPI's Depends(get_db)
    once the API layer exists. Does not commit automatically -- callers
    are responsible for committing within the request/handler.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
