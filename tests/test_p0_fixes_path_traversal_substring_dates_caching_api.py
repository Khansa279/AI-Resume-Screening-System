"""Regression tests for five issues investigated per user request, each
independently confirmed against the actual current code before being
fixed (none of these existed as fixes anywhere in the repository prior
to this change -- see the investigation history for how that was
verified).

F-01: path traversal in the resume-upload API route
F-02: substring skill-matching false positives (Java/JavaScript, C/C++)
F-03: "YYYY-MM" date format silently defaulting every month to January
F-07: a failed JobAnalyzerAgent run could permanently cache an empty
      JobRequirements row for a JD version
F-08: the screening API response was missing fields the frontend's
      CandidateCard.jsx component already has UI for
"""

import asyncio
from pathlib import Path

import pytest

from src.agents.skills_matcher import SkillsMatcherAgent
from src.agents.experience_eval import _parse_one_date
from src.models import Skill


# ---------------------------------------------------------------------------
# F-01: path traversal
# ---------------------------------------------------------------------------

def test_upload_route_never_uses_client_filename_for_disk_path():
    """Static guard: the untrusted client-supplied filename must never
    be used directly to build the on-disk destination path in the
    upload route. This is checked at the source level (rather than via
    an HTTP-level traversal attempt) because the route also depends on
    a live DB position/JD, which is out of scope for this test module."""
    src = Path("src/api/routes/screening.py").read_text(encoding="utf-8")
    assert "dest = tmp_dir / filename" not in src, (
        "the upload route appears to build a destination path directly "
        "from the untrusted client filename again -- this is the exact "
        "path-traversal regression this test guards against"
    )
    assert "dest = tmp_dir / safe_name" in src and "uuid.uuid4()" in src, (
        "expected the upload route to generate a random on-disk filename "
        "rather than trusting the client-supplied one"
    )


# ---------------------------------------------------------------------------
# F-02: substring skill-matching false positives
# ---------------------------------------------------------------------------

def _score(requirement: str, skill_name: str):
    agent = SkillsMatcherAgent()
    return agent._score_requirement(requirement, [Skill(name=skill_name, category="language")])


def test_java_does_not_falsely_satisfy_javascript_requirement():
    m = _score("JavaScript", "Java")
    assert m.matched is False, (
        f"'Java' incorrectly satisfied requirement 'JavaScript' via naive "
        f"substring containment: {m}"
    )


def test_cplusplus_does_not_falsely_satisfy_c_requirement():
    m = _score("C", "C++")
    assert m.matched is False, f"'C++' incorrectly satisfied requirement 'C': {m}"


def test_genuine_substring_match_still_works():
    """Regression guard: a real, whole-word substring relationship must
    still result in a match. (It's caught by the earlier taxonomy-based
    semantic-match step for a recognized skill like "Python" -- that's
    fine and unrelated to this fix; the point of this test is just that
    matching still happens at all.)"""
    m = _score("Python programming experience", "Python")
    assert m.matched is True


def test_genuine_substring_match_still_works_reverse_direction():
    m = _score("REST", "REST APIs")
    assert m.matched is True


# ---------------------------------------------------------------------------
# F-03: YYYY-MM date parsing
# ---------------------------------------------------------------------------

def test_yyyy_mm_format_parses_the_correct_month():
    assert _parse_one_date("2022-06") == __import__("datetime").date(2022, 6, 1)
    assert _parse_one_date("2023-11") == __import__("datetime").date(2023, 11, 1)
    assert _parse_one_date("2022-01") == __import__("datetime").date(2022, 1, 1)


def test_mm_yyyy_format_still_works_unchanged():
    """Regression guard: the pre-existing MM-YYYY (month first) format
    must be completely unaffected."""
    assert _parse_one_date("06-2022") == __import__("datetime").date(2022, 6, 1)


def test_month_name_format_still_works_unchanged():
    assert _parse_one_date("June 2022") == __import__("datetime").date(2022, 6, 1)


def test_job_years_reflects_correct_month_for_yyyy_mm_dates():
    """End-to-end: a WorkExperience using YYYY-MM dates must compute a
    duration reflecting the REAL months, not silently treat every start
    date as January."""
    from src.agents.experience_eval import job_years
    from src.models import WorkExperience

    exp = WorkExperience(title="X", company="Y", start_date="2022-06", end_date="2022-12")
    years = job_years(exp)
    # ~6 months (June to December), not ~12 months (January to December,
    # the pre-fix silently-wrong result for a mis-parsed start date).
    assert 0.4 <= years <= 0.6, f"expected ~0.5 years (6 months), got {years}"


# ---------------------------------------------------------------------------
# F-07: failed JD analysis caching
# ---------------------------------------------------------------------------

def test_job_requirements_row_usability_check():
    from src.services.screening_service import _job_requirements_row_is_usable

    class FakeSkill:
        pass

    class FakeRow:
        def __init__(self, parsing_confidence, skills):
            self.parsing_confidence = parsing_confidence
            self.skills = skills

    # The exact failure signature JobAnalyzerAgent._parse_response()
    # produces on a failed API call or unparseable JSON response.
    failed_row = FakeRow(parsing_confidence=0.3, skills=[])
    assert _job_requirements_row_is_usable(failed_row) is False

    # A real, successfully-parsed JD with skills -- must be treated as
    # usable regardless of confidence value.
    real_row = FakeRow(parsing_confidence=0.3, skills=[FakeSkill()])
    assert _job_requirements_row_is_usable(real_row) is True

    real_low_confidence_row = FakeRow(parsing_confidence=0.4, skills=[])
    assert _job_requirements_row_is_usable(real_low_confidence_row) is True

    normal_row = FakeRow(parsing_confidence=0.9, skills=[FakeSkill(), FakeSkill()])
    assert _job_requirements_row_is_usable(normal_row) is True


def test_ensure_job_requirements_retries_after_a_failed_cached_row(tmp_path, monkeypatch):
    """End-to-end: a JD version whose ONLY cached JobRequirements row
    looks like a failed parse must be re-analyzed (not permanently
    served as-is) the next time a candidate is screened against it."""
    import importlib
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    import src.db.database as db_module
    importlib.reload(db_module)
    from src.db import repository as repo
    from src.agents.job_analyzer import JobAnalyzerAgent
    from src.services import screening_service
    importlib.reload(screening_service)

    db_module.init_db()

    with db_module.get_session() as db:
        org = repo.create_organization(db, "T")
        dept = repo.create_department(db, org.id, "E")
        position = repo.create_position(db, dept.id, "P")
        jd = repo.create_job_description(db, position.id, raw_text="Some JD text")
        # Simulate the exact failure signature: a row with the
        # failure-fallback confidence and zero skills.
        repo.save_job_requirements(
            db, jd.id, title="", min_years_experience=0,
            required_skills=[], preferred_skills=[], parsing_confidence=0.3,
        )
        jd_id = jd.id

    call_log = []

    async def fake_process(self, state):
        call_log.append(1)
        from src.models import JobRequirements as JRModel
        return {"job_requirements": JRModel(
            title="Real Title", required_skills=["Python"], preferred_skills=[],
            min_years_experience=0, parsing_confidence=0.9,
        )}

    monkeypatch.setattr(JobAnalyzerAgent, "process", fake_process)

    with db_module.get_session() as db:
        jr = asyncio.run(screening_service.ensure_job_requirements(db, jd_id))
        assert jr.title == "Real Title"
        assert len(jr.skills) == 1

    assert call_log == [1], "JobAnalyzerAgent should have been called exactly once to re-parse the failed row"


# ---------------------------------------------------------------------------
# F-08: API response field expansion
# ---------------------------------------------------------------------------

def test_screening_candidate_result_schema_has_expanded_fields():
    from src.api.schemas import ScreeningCandidateResult
    fields = ScreeningCandidateResult.model_fields
    for expected in ("matching_skills", "skill_gaps", "explanation", "candidate_email", "candidate_phone"):
        assert expected in fields, f"expected {expected!r} on ScreeningCandidateResult"


def test_screening_candidate_result_defaults_are_safe():
    """Fields must default to empty/None so existing callers that don't
    pass them (or screenings with no Explanation row) don't break."""
    from src.api.schemas import ScreeningCandidateResult
    result = ScreeningCandidateResult(
        screening_id=1, resume_id=1, candidate_name="Test",
        match_score=0.5, recommendation="x", requires_human=False, confidence=0.5,
    )
    assert result.matching_skills == []
    assert result.skill_gaps == []
    assert result.explanation is None
    assert result.candidate_email is None
    assert result.candidate_phone is None
