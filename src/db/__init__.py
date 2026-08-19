"""Persistence layer for the recruitment platform (SQLite via SQLAlchemy).

This package is intentionally isolated from src/agents and src/workflow --
it knows nothing about LLMs or LangGraph. A future service layer will be
what wires screening results into these repository functions.
"""

from .database import DATABASE_URL, engine, SessionLocal, init_db, get_session, get_db
from . import models
from . import repository

__all__ = [
    "DATABASE_URL",
    "engine",
    "SessionLocal",
    "init_db",
    "get_session",
    "get_db",
    "models",
    "repository",
]
