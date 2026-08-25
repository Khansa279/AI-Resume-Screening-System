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
    This does NOT handle general schema migrations (e.g. renaming or
    dropping a column); that's still an Alembic concern for later. It DOES
    run `_apply_lightweight_column_migrations`, a tiny, dependency-free
    "add this column if it's missing" step -- just enough to let an
    existing dev screening.db pick up new nullable columns (like
    Resume.content_hash) without forcing a DB wipe every time the schema
    grows by one column.
    """
    Base.metadata.create_all(bind=engine)
    _apply_lightweight_column_migrations()


def _apply_lightweight_column_migrations() -> None:
    """Add newly-introduced nullable columns to existing SQLite tables.

    `Base.metadata.create_all` only creates tables that don't exist yet --
    it never alters an existing table, so a column added to a model after
    the DB file was first created would otherwise silently not show up,
    and every read/write against it would raise "no such column".

    This is intentionally minimal (SQLite `ALTER TABLE ... ADD COLUMN`
    only, nullable columns only) rather than pulling in Alembic, per the
    project's "no unnecessary new dependencies" constraint. Add an entry
    here whenever a column is added to an existing table.
    """
    if not DATABASE_URL.startswith("sqlite"):
        return

    # (table_name, column_name, column_ddl_type)
    pending_columns = [
        ("resumes", "content_hash", "VARCHAR"),
        ("skill_matches", "soft_skill_status", "TEXT"),
    ]

    with engine.connect() as conn:
        for table_name, column_name, column_type in pending_columns:
            existing_columns = {
                row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table_name})")
            }
            if column_name not in existing_columns:
                conn.exec_driver_sql(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                )
        conn.commit()


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
