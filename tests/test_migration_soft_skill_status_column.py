"""Regression test for the skill_matches.soft_skill_status column migration.

REAL BUG: Batch 7 added `soft_skill_status` to the SQLAlchemy `SkillMatch`
model (src/db/models.py) so soft-skill requirements (e.g. "Good
communication skills") could be distinguished from ordinary technical
gaps. `Base.metadata.create_all()` only creates NEW tables -- it never
alters an EXISTING table -- so any screening.db created before this
column existed kept the old 8-column `skill_matches` table. The very
first query that touched `SkillMatchResult.matches` after upgrading then
raised `sqlite3.OperationalError: no such column:
skill_matches.soft_skill_status`, even though `init_db()` had already run.

Fix (src/db/database.py): `_apply_lightweight_column_migrations()` -- the
same lightweight "ALTER TABLE ADD COLUMN if missing" mechanism already
used for `resumes.content_hash` -- now also covers
`("skill_matches", "soft_skill_status", "TEXT")`.

This test builds a SQLite file with the OLD (pre-migration) 8-column
`skill_matches` schema by hand (mirroring exactly what a real pre-Batch-7
screening.db looks like), runs `init_db()` against it, and asserts:
  1. The column is added.
  2. Existing rows are preserved and read back with soft_skill_status=None
     (not fabricated -- we don't know retroactively whether an old row
     was a soft-skill match).
  3. Calling init_db() again (simulating a second app start) is a no-op,
     not an error -- ALTER TABLE ADD COLUMN is not naturally idempotent
     in SQLite, so this specifically exercises the "skip if already
     present" guard.
"""

import importlib
import sqlite3

import pytest


# Columns skill_matches had BEFORE the soft_skill_status column existed
# (see the models.py history prior to the "Fix role relevance and soft
# skill screening logic" change) -- deliberately hand-built here (not
# imported from models.py) so this test still catches a regression even
# if a future change further evolves the current schema.
_OLD_SKILL_MATCHES_DDL = """
CREATE TABLE skill_matches (
    id INTEGER NOT NULL PRIMARY KEY,
    skill_match_result_id INTEGER NOT NULL,
    requirement VARCHAR,
    matched BOOLEAN,
    matched_skill VARCHAR,
    match_quality VARCHAR,
    confidence FLOAT,
    notes TEXT
)
"""


@pytest.fixture()
def old_schema_db(tmp_path, monkeypatch):
    """Fresh SQLite file with the pre-Batch-7 skill_matches shape, plus one
    real row, then reload src.db.database bound to it (DATABASE_URL is
    read at import time, so a plain monkeypatch of the env var alone
    wouldn't affect the already-imported module)."""
    db_path = tmp_path / "old_screening.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_OLD_SKILL_MATCHES_DDL)
    conn.execute(
        "INSERT INTO skill_matches "
        "(id, skill_match_result_id, requirement, matched, matched_skill, "
        " match_quality, confidence, notes) VALUES (1, 1, 'Python', 1, 'Python', 'exact', 1.0, '')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    import src.db.database as db_module
    importlib.reload(db_module)
    yield db_module, db_path

    # Leave the module in a consistent state for any test that runs after
    # this one and imports src.db.database expecting the real default DB.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    importlib.reload(db_module)


def _columns(db_path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(skill_matches)")}
    finally:
        conn.close()


def test_init_db_adds_missing_soft_skill_status_column(old_schema_db):
    db_module, db_path = old_schema_db
    assert "soft_skill_status" not in _columns(db_path)

    db_module.init_db()

    assert "soft_skill_status" in _columns(db_path)


def test_init_db_preserves_existing_rows_with_null_status(old_schema_db):
    db_module, db_path = old_schema_db
    db_module.init_db()

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT id, requirement, matched_skill, soft_skill_status FROM skill_matches WHERE id = 1"
        ).fetchone()
    finally:
        conn.close()

    assert row == (1, "Python", "Python", None)


def test_init_db_is_idempotent_on_the_new_column(old_schema_db):
    """ALTER TABLE ADD COLUMN raises if the column already exists -- this
    is what proves the migration checks PRAGMA table_info before altering,
    rather than blindly running ALTER TABLE on every startup."""
    db_module, db_path = old_schema_db
    db_module.init_db()
    db_module.init_db()
    db_module.init_db()

    assert "soft_skill_status" in _columns(db_path)


def test_orm_read_of_matches_no_longer_raises_operational_error(old_schema_db):
    """End-to-end reproduction of the exact reported error: reading
    SkillMatchResult.matches (which SELECTs skill_matches.soft_skill_status
    by column name) must succeed after init_db(), not raise
    'no such column: skill_matches.soft_skill_status'."""
    db_module, db_path = old_schema_db
    db_module.init_db()

    from sqlalchemy.orm import sessionmaker
    from src.db import models

    Session = sessionmaker(bind=db_module.engine)
    with Session() as session:
        matches = session.query(models.SkillMatch).all()
        assert len(matches) == 1
        assert matches[0].requirement == "Python"
        assert matches[0].soft_skill_status is None
