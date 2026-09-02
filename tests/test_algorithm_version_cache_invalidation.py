"""Regression tests for algorithm-version-aware screening cache invalidation.

REAL BUG (test_real_data.py staleness, tracked as an open item alongside
the role-relevance/soft-skill work in Batches 7-10): screen_candidate()'s
"reuse a completed screening" short-circuit (see
repository.get_completed_screening_for_resume_and_jd) is keyed ONLY on
(resume content_hash, job_description_id). That proves the RESUME TEXT
for a given (resume, JD) pair hasn't changed -- it says nothing about
whether the SCORING CODE that evaluated it is still today's code. Once a
resume+JD pair had been screened and cached, every scoring fix landed
since (Batches 7-10: role-relevance stopwords, JD requirement
decomposition, soft-skill tristate, title-score Jaccard, semantic role
families) would silently never get applied to that pair again -- the
stale pre-fix score would be served forever as "the" result, even while
every OTHER candidate was scored correctly under the fixed code.

FIX:
  - src/db/models.py: Screening gained a nullable `algorithm_version`
    column (migrated via the existing SQLite lightweight-column
    mechanism in src/db/database.py, same pattern as
    skill_matches.soft_skill_status).
  - src/db/repository.py: save_screening_result() accepts an optional
    `algorithm_version` and stamps it onto the Screening row when a
    result is saved.
  - src/services/screening_service.py: a new module-level
    CURRENT_ALGORITHM_VERSION constant. screen_candidate() only takes
    the 0-LLM-call reuse path when the cached Screening's stamped
    version matches this constant; otherwise it invalidates JUST that
    one Screening (via the existing, already-tested
    repository.invalidate_completed_screening -- the same targeted
    mechanism demo_screening.py's "re-screen" mode already uses) and
    falls through to re-run the real pipeline, reusing the same
    Screening row rather than creating a duplicate.

This intentionally does NOT touch scoring weights, the skill taxonomy,
DecisionSynthesizerAgent's formula, or any other candidate's/JD's data --
see test_invalidating_one_stale_screening_does_not_touch_other_rows below.

All tests here run with NO API key / NO LLM calls: job_requirements is
pre-seeded directly (mirroring src/scripts/test_batch1_resume_reuse.py's
pattern), and every agent on the normal screening path
(ResumeParserAgent, SkillExtractorAgent, SkillsMatcherAgent,
ExperienceEvaluatorAgent, DecisionSynthesizerAgent) is deterministic as
of Batch 3+ -- see test_deterministic_pipeline.py.
"""

import asyncio
import importlib
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base
from src.db import repository as repo
from src.services import screening_service
from src.services.screening_service import CURRENT_ALGORITHM_VERSION, screen_candidate

RESUME_TEXT = Path("sample_data/resumes/junior_dev.txt").read_text(encoding="utf-8")
JD_TEXT = Path("sample_data/job_descriptions/backend_engineer.txt").read_text(encoding="utf-8")


# ============================================================================
# Helpers: an isolated, fully-current-schema temp SQLite DB per test (via
# Base.metadata.create_all -- NOT the lightweight-migration path, which is
# covered separately below), with a pre-seeded position/JD/requirements so
# no LLM call is ever needed.
# ============================================================================

@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = Session()
    try:
        yield session
        session.commit()
    finally:
        session.close()


def _seed_position_and_jd(db) -> int:
    org = repo.create_organization(db, "Test Org")
    dept = repo.create_department(db, org.id, "Engineering")
    position = repo.create_position(db, dept.id, "Backend Engineer - Python")
    jd = repo.create_job_description(db, position.id, raw_text=JD_TEXT)
    repo.save_job_requirements(
        db, jd.id, title="Backend Software Engineer", min_years_experience=3,
        required_skills=["Python", "Django", "PostgreSQL"],
        preferred_skills=["AWS", "Docker"], parsing_confidence=0.9,
    )
    db.flush()
    return position.id


def _write_resume_file(tmp_path) -> str:
    p = tmp_path / "resume.txt"
    p.write_text(RESUME_TEXT, encoding="utf-8")
    return str(p)


# ============================================================================
# Behavioral tests
# ============================================================================

def test_screen_candidate_stamps_current_algorithm_version(db_session, tmp_path):
    position_id = _seed_position_and_jd(db_session)
    resume_path = _write_resume_file(tmp_path)

    screening = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_path)
    )
    db_session.flush()

    assert screening.algorithm_version == CURRENT_ALGORITHM_VERSION


def test_screen_candidate_reuses_result_when_version_matches(db_session, tmp_path):
    position_id = _seed_position_and_jd(db_session)
    resume_path = _write_resume_file(tmp_path)

    first = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_path)
    )
    db_session.flush()
    first_id = first.id
    first_score = first.result.match_score

    second = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_path)
    )
    db_session.flush()

    assert second.id == first_id, "same (resume, JD) pair with a current-version cached result must reuse the row"
    assert second.result.match_score == first_score
    assert second.algorithm_version == CURRENT_ALGORITHM_VERSION


def test_screen_candidate_invalidates_and_rescoures_when_version_is_stale(db_session, tmp_path):
    """Core regression: a completed screening stamped with an OLDER
    algorithm_version must be re-scored (not silently served), and must
    end up re-stamped with the current version -- reusing the same
    Screening row rather than creating a duplicate."""
    position_id = _seed_position_and_jd(db_session)
    resume_path = _write_resume_file(tmp_path)

    first = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_path)
    )
    db_session.flush()
    first_id = first.id

    # Simulate a screening completed under an OLDER pipeline version.
    stale = repo.get_screening(db_session, first_id)
    stale.algorithm_version = "some-old-pre-batch7-version"
    db_session.flush()

    second = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_path)
    )
    db_session.flush()

    assert second.id == first_id, "must reuse the SAME screening row, not create a duplicate"
    assert second.algorithm_version == CURRENT_ALGORITHM_VERSION
    assert second.status == "completed"
    # Recomputation actually happened for real -- child rows exist again
    # after invalidate_completed_screening() cleared them.
    assert second.result is not None
    assert second.skill_match_result is not None
    assert second.experience_evaluation is not None
    assert second.explanation is not None


def test_screen_candidate_invalidates_when_version_is_none(db_session, tmp_path):
    """A completed screening with algorithm_version=None simulates a row
    persisted before this column existed (pre-migration) -- must also be
    treated as stale, not silently trusted."""
    position_id = _seed_position_and_jd(db_session)
    resume_path = _write_resume_file(tmp_path)

    first = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_path)
    )
    db_session.flush()
    first_id = first.id

    stale = repo.get_screening(db_session, first_id)
    assert stale.algorithm_version == CURRENT_ALGORITHM_VERSION  # sanity: it WAS stamped
    stale.algorithm_version = None
    db_session.flush()

    second = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_path)
    )
    db_session.flush()

    assert second.id == first_id
    assert second.algorithm_version == CURRENT_ALGORITHM_VERSION


def test_screening_stamped_with_prior_production_version_is_invalidated_and_rescored(db_session, tmp_path):
    """Regression for the embedding/role-relevance version bump.

    Unlike test_screen_candidate_invalidates_and_rescoures_when_version_is_stale
    above (which uses an arbitrary synthetic old-version string), this
    pins the exact PREVIOUS PRODUCTION value of CURRENT_ALGORITHM_VERSION
    ("2026-08-role-relevance-batch10" -- the version in place before the
    embedding/role-relevance changes) so a future accidental revert of
    the bump, or a second bump that forgets to change this literal, is
    caught directly rather than only via a synthetic placeholder string.

    Simulates exactly the reported symptom: a Screening completed and
    cached under the old batch10 pipeline (old match_score/role_relevance/
    years_relevant numbers) must NOT be served as-is once
    CURRENT_ALGORITHM_VERSION has moved past that string -- it must be
    invalidated (child rows cleared) and recomputed, reusing the same
    Screening row.
    """
    PRIOR_PRODUCTION_VERSION = "2026-08-role-relevance-batch10"
    assert CURRENT_ALGORITHM_VERSION != PRIOR_PRODUCTION_VERSION, (
        "This test assumes CURRENT_ALGORITHM_VERSION has been bumped past "
        "the prior production version it's meant to invalidate -- if this "
        "fails, the version bump was reverted or never applied."
    )

    position_id = _seed_position_and_jd(db_session)
    resume_path = _write_resume_file(tmp_path)

    first = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_path)
    )
    db_session.flush()
    first_id = first.id

    # Simulate a screening that was completed and cached BEFORE the
    # embedding/role-relevance change landed (the exact reported bug:
    # an existing Alex-Morgan-style Screening stamped under batch10).
    stale = repo.get_screening(db_session, first_id)
    stale.algorithm_version = PRIOR_PRODUCTION_VERSION
    db_session.flush()

    second = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_path)
    )
    db_session.flush()

    # Same row reused (no duplicate Screening created), but recomputed
    # and re-stamped with the current version -- not silently served.
    assert second.id == first_id, "must reuse the SAME screening row, not create a duplicate"
    assert second.algorithm_version == CURRENT_ALGORITHM_VERSION
    assert second.algorithm_version != PRIOR_PRODUCTION_VERSION
    assert second.status == "completed"
    assert second.result is not None
    assert second.skill_match_result is not None
    assert second.experience_evaluation is not None
    assert second.explanation is not None


def test_screening_stamped_with_current_version_is_still_reused_unchanged(db_session, tmp_path):
    """Sanity/non-regression check for requirement #4: existing reuse
    behavior when stored version == CURRENT_ALGORITHM_VERSION must be
    completely unaffected by the bump -- same row, same score, still
    zero recomputation (no new completed_at timestamp)."""
    position_id = _seed_position_and_jd(db_session)
    resume_path = _write_resume_file(tmp_path)

    first = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_path)
    )
    db_session.flush()
    assert first.algorithm_version == CURRENT_ALGORITHM_VERSION
    first_completed_at = first.completed_at
    first_score = first.result.match_score

    second = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_path)
    )
    db_session.flush()

    assert second.id == first.id
    assert second.algorithm_version == CURRENT_ALGORITHM_VERSION
    assert second.result.match_score == first_score
    assert second.completed_at == first_completed_at, (
        "reuse path must not re-run the pipeline -- completed_at should be untouched"
    )


def test_invalidating_one_stale_screening_does_not_touch_other_rows(db_session, tmp_path):
    """Scope discipline: re-scoring ONE stale (resume, JD) pair must not
    disturb any OTHER candidate's Screening for the same JD."""
    position_id = _seed_position_and_jd(db_session)

    resume_a_path = _write_resume_file(tmp_path)
    resume_b_dir = tmp_path / "b"
    resume_b_dir.mkdir()
    resume_b_path = resume_b_dir / "resume.txt"
    resume_b_path.write_text(
        Path("sample_data/resumes/senior_python_dev.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    screening_a = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_a_path)
    )
    screening_b = asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=str(resume_b_path))
    )
    db_session.flush()

    b_score_before = screening_b.result.match_score
    b_completed_at_before = screening_b.completed_at

    # Mark ONLY A as stale.
    stale_a = repo.get_screening(db_session, screening_a.id)
    stale_a.algorithm_version = "old-version"
    db_session.flush()

    asyncio.run(
        screen_candidate(db_session, position_id=position_id, resume_file_path=resume_a_path)
    )
    db_session.flush()

    b_after = repo.get_screening(db_session, screening_b.id)
    assert b_after.algorithm_version == CURRENT_ALGORITHM_VERSION
    assert b_after.result.match_score == b_score_before
    assert b_after.completed_at == b_completed_at_before, "candidate B's screening must be untouched"


def test_save_screening_result_defaults_to_none_for_existing_callers(db_session, tmp_path):
    """Backward compatibility: callers that don't pass algorithm_version
    (e.g. seed_demo.py-style direct persistence, or existing tests) must
    keep working -- the column is optional and defaults to None."""
    position_id = _seed_position_and_jd(db_session)
    candidate = repo.get_or_create_candidate(db_session, name="Direct Caller", email="direct@example.com")
    resume = repo.create_resume(db_session, candidate_id=candidate.id, position_id=position_id, raw_text="r")
    jd = repo.get_current_job_description(db_session, position_id)
    screening = repo.create_screening(db_session, resume.id, jd.id)

    result = repo.save_screening_result(
        db_session, screening.id,
        match_score=0.5, recommendation="Test", requires_human=False,
        confidence=0.5, reasoning_summary="test",
    )
    db_session.flush()

    assert result is not None
    refreshed = repo.get_screening(db_session, screening.id)
    assert refreshed.algorithm_version is None


# ============================================================================
# Migration test: mirrors test_migration_soft_skill_status_column.py --
# an existing screenings table WITHOUT algorithm_version must gain it via
# the lightweight ALTER TABLE mechanism, idempotently, without losing data.
# ============================================================================

_OLD_SCREENINGS_DDL = """
CREATE TABLE screenings (
    id INTEGER NOT NULL PRIMARY KEY,
    resume_id INTEGER NOT NULL,
    job_description_id INTEGER NOT NULL,
    status VARCHAR,
    started_at DATETIME,
    completed_at DATETIME,
    created_at DATETIME NOT NULL
)
"""


@pytest.fixture()
def old_schema_db(tmp_path, monkeypatch):
    db_path = tmp_path / "old_screening.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(_OLD_SCREENINGS_DDL)
    conn.execute(
        "INSERT INTO screenings (id, resume_id, job_description_id, status, created_at) "
        "VALUES (1, 1, 1, 'completed', '2026-01-01 00:00:00')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    import src.db.database as db_module
    importlib.reload(db_module)
    yield db_module, db_path
    monkeypatch.delenv("DATABASE_URL", raising=False)
    importlib.reload(db_module)


def _screening_columns(db_path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(screenings)")}
    finally:
        conn.close()


def test_init_db_adds_missing_algorithm_version_column(old_schema_db):
    db_module, db_path = old_schema_db
    assert "algorithm_version" not in _screening_columns(db_path)

    db_module.init_db()

    assert "algorithm_version" in _screening_columns(db_path)


def test_init_db_preserves_existing_screening_rows(old_schema_db):
    db_module, db_path = old_schema_db
    db_module.init_db()

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT id, status, algorithm_version FROM screenings WHERE id = 1"
        ).fetchone()
    finally:
        conn.close()

    assert row == (1, "completed", None)


def test_init_db_is_idempotent_on_algorithm_version_column(old_schema_db):
    db_module, db_path = old_schema_db
    db_module.init_db()
    db_module.init_db()
    db_module.init_db()
    assert "algorithm_version" in _screening_columns(db_path)
