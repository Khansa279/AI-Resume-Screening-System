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

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .models import Base, Resume

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

    Also applies the one additive schema change this redesign needs
    (resumes.content_hash) on databases created before that column existed,
    and backfills hashes for existing rows. Does not delete duplicate
    development records.
    """
    Base.metadata.create_all(bind=engine)
    _ensure_resume_content_hash_column()
    _backfill_resume_content_hashes()


def _ensure_resume_content_hash_column() -> None:
    """SQLite create_all will not ADD a column to an existing table."""
    if not DATABASE_URL.startswith("sqlite"):
        return
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(resumes)")).fetchall()
        col_names = {row[1] for row in rows}
        if "content_hash" not in col_names:
            conn.execute(text("ALTER TABLE resumes ADD COLUMN content_hash VARCHAR"))
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_resumes_content_hash ON resumes (content_hash)")
        )


def _backfill_resume_content_hashes() -> None:
    """Fill content_hash for existing Resume rows that have raw_text."""
    from sqlalchemy import or_, select

    from ..document_parser import resume_content_hash

    db = SessionLocal()
    try:
        rows = list(
            db.scalars(
                select(Resume).where(
                    or_(Resume.content_hash.is_(None), Resume.content_hash == "")
                )
            )
        )
        updated = False
        for row in rows:
            if row.raw_text:
                row.content_hash = resume_content_hash(row.raw_text)
                updated = True
        if updated:
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


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
