"""Service layer: bridges the LangGraph agent workflow to persistence.

This is the piece the docstrings in src/db/database.py and
src/db/repository.py kept pointing at -- the only place that knows about
BOTH src/workflow.py (agents) and src/db (SQLite), so those two layers
stay decoupled from each other.
"""

from .screening_service import screen_candidate, screen_position

__all__ = ["screen_candidate", "screen_position"]