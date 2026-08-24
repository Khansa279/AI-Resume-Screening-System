"""Batch 7 regression tests.

Covers three confirmed, targeted fixes from the role-relevance /
skill-gap investigation. Each corresponds 1:1 to a root cause verified
against the REAL deterministic pipeline (see diagnostic session) using
John Doe's real resume (sample_data/resumes/senior_python_dev.txt).

1. TITLE STOPWORD BUG (src/agents/experience_eval.py)
   `_STOP` (used for general text tokenization) included role/seniority
   words -- "engineer", "developer", "software", "senior", "junior",
   "lead", "associate", "intern" -- and job_relevance() used it for TITLE
   comparison too. Almost every real job title is built entirely out of
   those words, so `_tokens(title)` came back as an EMPTY set for titles
   like "Senior Software Engineer", and `_overlap()` short-circuits to
   0.0 whenever either side is empty. That zeroed title_score (45% of
   role_relevance's weight) for most conventional titles.
   Fix: a separate, smaller `_TITLE_STOP` (grammatical filler only) and
   `_title_tokens()`, used ONLY for title comparison. General `_tokens`/
   `_STOP` (used for responsibility-text overlap) are untouched.

2. technologies=[] BUG (src/agents/resume_parser.py)
   `_parse_experience()` hardcoded `technologies=[]` on every
   WorkExperience -- the deterministic resume parser never populated it.
   `job_relevance()`'s skill_hit component (40% of role_relevance's
   weight) needs `exp.technologies` to be non-empty to do anything beyond
   a weak literal-substring fallback.
   Fix: `_technologies_for_job()` recognizes technologies/tools via the
   shared taxonomy from each job's title + responsibility bullets.

3. LONG JD REQUIREMENT PHRASES NOT DECOMPOSED (experience_eval.py)
   `job_relevance()` built `jd_skills` from `canonical_skill_key()` on
   the WHOLE requirement string. For a bare skill name ("Python") that's
   fine; for a realistic JobAnalyzer requirement like "REST API
   development (Django REST Framework or FastAPI)" the whole sentence
   just becomes one opaque, never-matching key.
   Fix: `skill_taxonomy.normalize_requirement_skills()` decomposes a
   requirement into the individual canonical skills it references
   (mirrors what SkillsMatcherAgent already did correctly), used to build
   `jd_skills` and the responsibility-mention fallback.

4. SOFT SKILLS TREATED AS BINARY KEYWORD GAPS (skills_matcher.py,
   screening_service.py)
   A purely soft-skill requirement (e.g. "Good communication skills")
   with no resume evidence either way was scored and reported exactly
   like a missing technical skill -- reducing overall_score and showing
   up in skill_gaps -- even though absence of the word "communication"
   does not mean the candidate lacks the trait.
   Fix: SkillMatch.soft_skill_status distinguishes "demonstrated" (found,
   explicit or inferred) from "not_mentioned" (no evidence either way --
   excluded from scoring denominators AND from the skill_gaps list) for
   requirements that decompose entirely to soft-skill taxonomy concepts.
   Ordinary/technical/mixed requirements are completely unaffected
   (soft_skill_status stays None, existing gap behavior unchanged).

These tests intentionally do NOT touch SkillsMatcherAgent's CREDIT_BY_
QUALITY weights, DecisionSynthesizerAgent's scoring formula, or the DB
schema.
"""

from pathlib import Path

import asyncio

from src.agents.experience_eval import evaluate_experience, job_relevance, _title_tokens
from src.agents.resume_parser import parse_resume_text
from src.agents.skills_matcher import SkillsMatcherAgent
from src.models import (
    ExperienceEvaluation,
    JobRequirements,
    ResumeData,
    ScreeningOutput,
    Skill,
    WorkExperience,
)
from src.skill_taxonomy import normalize_requirement_skills, requirement_is_pure_soft_skill
from src.services.screening_service import _build_explanation


SAMPLE_RESUME = Path("sample_data/resumes/senior_python_dev.txt")  # John Doe


# ---------------------------------------------------------------------------
# Fix 1: title stopword bug
# ---------------------------------------------------------------------------

def test_title_tokens_no_longer_empty_for_ordinary_titles():
    """This is the exact bug: under the old shared _STOP, ALL of these
    tokenized to an empty set, because every word in them was a stopword."""
    assert _title_tokens("Senior Software Engineer") == {"senior", "software", "engineer"}
    assert _title_tokens("Software Engineer") == {"software", "engineer"}
    assert _title_tokens("Junior Developer") == {"junior", "developer"}
    assert _title_tokens("Backend Engineer - Python") == {"backend", "engineer", "python"}


def test_job_relevance_no_longer_zero_for_generic_title_with_matching_skills():
    """A candidate whose title has no 'backend'-style distinctive word but
    whose technologies genuinely match the JD must not score ~0."""
    reqs = JobRequirements(
        title="Backend Engineer - Python",
        required_skills=["Python", "Django", "PostgreSQL"],
        preferred_skills=["Docker"],
    )
    exp = WorkExperience(
        title="Senior Software Engineer",  # no "backend"/"python" in the title itself
        company="TechCorp",
        technologies=["Python", "Django", "PostgreSQL", "Docker"],
        responsibilities=["Designed RESTful APIs using Python and Django"],
    )
    relevance = job_relevance(exp, reqs)
    # Before the fix this was ~0.02-0.1 (title contributed nothing, and
    # technologies would have been ignored by the parser anyway). 0.4 is
    # a conservative bar -- well above the old near-zero floor -- without
    # asserting an exact value tied to the current 0.45/0.4/0.15 weights.
    assert relevance >= 0.4, f"expected a strong match, got {relevance}"


def test_resume_parser_populates_technologies_for_john_doe():
    """Fix 2: technologies must no longer be hardcoded to []."""
    text = SAMPLE_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)
    assert data.work_experience, "fixture resume should have parsed work experience"
    for job in data.work_experience:
        assert job.technologies, f"expected non-empty technologies for {job.title!r}"

    senior_role = data.work_experience[0]
    assert senior_role.title == "Senior Software Engineer"
    # Drawn directly from that job's real bullet text ("...using Python and
    # Django", "...PostgreSQL database queries...", "...Docker").
    for expected in ("Python", "Django", "PostgreSQL", "Docker"):
        assert expected in senior_role.technologies, senior_role.technologies


def test_role_relevance_for_john_doe_vs_backend_python_jd_is_no_longer_near_zero():
    """End-to-end regression reproducing the reported bug: a strong
    backend candidate's role_relevance against a realistic Backend
    Engineer - Python JD was 0.02 before the fix. It must now be
    substantially higher."""
    text = SAMPLE_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)
    reqs = JobRequirements(
        title="Backend Engineer - Python",
        required_skills=[
            "2+ years of professional experience with Python",
            "Experience building REST APIs (Django REST Framework or FastAPI preferred)",
            "Strong SQL skills and experience with relational databases (PostgreSQL)",
            "Familiarity with Git version control",
            "Understanding of software development best practices",
            "Good communication skills in English",
        ],
        preferred_skills=[
            "Experience with Django or FastAPI frameworks",
            "Familiarity with Docker and containerization",
            "Exposure to message queues (Redis, RabbitMQ, Celery)",
            "Basic AWS/cloud experience (EC2, S3, RDS)",
            "Experience with CI/CD pipelines",
        ],
        min_years_experience=2,
        responsibilities=[
            "Design, build, and maintain RESTful APIs using Python",
            "Write clean, well-tested, production-quality code",
        ],
    )
    ev = evaluate_experience(data, reqs)
    assert ev.role_relevance >= 0.2, (
        f"role_relevance={ev.role_relevance} is still near-zero for a strong backend "
        f"candidate against a matching JD"
    )
    assert ev.role_relevance > 0.02  # the exact figure measured before the fix


def test_experience_evaluation_is_still_deterministic_after_fix():
    """Guard against accidentally introducing nondeterminism."""
    text = SAMPLE_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)
    reqs = JobRequirements(
        title="Backend Software Engineer",
        required_skills=["Python", "Django", "SQL"],
        preferred_skills=["Docker"],
        min_years_experience=3,
    )
    e1 = evaluate_experience(data, reqs)
    e2 = evaluate_experience(data, reqs)
    assert e1.model_dump() == e2.model_dump()


# ---------------------------------------------------------------------------
# Fix 3 (part of role relevance): long JD requirement phrases decompose
# into canonical skills instead of being treated as one opaque string.
# ---------------------------------------------------------------------------

def test_normalize_requirement_skills_decomposes_long_phrases():
    keys = normalize_requirement_skills(
        "REST API development (Django REST Framework or FastAPI)"
    )
    assert "django" in keys
    assert "fast api" in keys or "fastapi" in keys
    assert "rest apis" in keys or "rest api" in keys

    keys2 = normalize_requirement_skills("Message queues (Redis, RabbitMQ, Celery)")
    assert "redis" in keys2
    assert "rabbitmq" in keys2


def test_normalize_requirement_skills_falls_back_for_unrecognized_text():
    """A phrase with nothing in the taxonomy must not vanish -- it keeps
    behaving like the old whole-phrase key (still exact-matchable)."""
    keys = normalize_requirement_skills("Willingness to work weekends occasionally")
    assert keys  # non-empty, not silently dropped


def test_job_relevance_credits_decomposed_long_requirement_phrases():
    """Direct reproduction of the user-reported examples: a candidate
    whose technologies exactly satisfy a long descriptive requirement
    must get skill_hit credit for it, not zero."""
    reqs = JobRequirements(
        title="Senior Backend Engineer - Fintech",
        required_skills=[
            "REST API development (Django REST Framework or FastAPI)",
            "Message queues (Redis, RabbitMQ, Celery)",
        ],
        preferred_skills=[],
    )
    exp = WorkExperience(
        title="Backend Engineer", company="X",
        technologies=["Django", "Redis"], responsibilities=[],
    )
    relevance = job_relevance(exp, reqs)
    assert relevance >= 0.4, f"expected strong skill_hit credit, got {relevance}"


# ---------------------------------------------------------------------------
# Fix 4: soft skills -- DEMONSTRATED / NOT_MENTIONED, never a false gap
# ---------------------------------------------------------------------------

def _run_matcher(skills, reqs):
    agent = SkillsMatcherAgent()
    return asyncio.run(agent.process({"extracted_skills": skills, "job_requirements": reqs}))["skills_match"]


def test_unmentioned_soft_skill_is_not_a_gap_and_does_not_reduce_score():
    reqs = JobRequirements(
        required_skills=["Python", "Good communication skills in English"],
        preferred_skills=[],
    )
    skills = [Skill(name="Python", category="language", source="explicit")]
    result = _run_matcher(skills, reqs)

    comm_match = next(m for m in result.matches if "communication" in m.requirement.lower())
    assert comm_match.matched is False
    assert comm_match.soft_skill_status == "not_mentioned"

    # Excluded from the denominator entirely: only "Python" (met) counts.
    assert result.required_skills_total == 1
    assert result.required_skills_met == 1
    assert result.overall_score == 1.0  # not penalized for the unmentioned soft skill

    explanation = _build_explanation(
        result,
        ExperienceEvaluation(years_relevant=1.0, role_relevance=0.5),
        ScreeningOutput(
            match_score=1.0, recommendation="Proceed", requires_human=False,
            confidence=0.9, reasoning_summary="ok",
        ),
    )
    assert not any("communication" in gap.lower() for gap in explanation["skill_gaps"])


def test_explicitly_mentioned_soft_skill_is_demonstrated():
    reqs = JobRequirements(required_skills=["Communication"], preferred_skills=[])
    skills = [Skill(name="Communication", category="soft_skill", source="explicit", confidence=0.95)]
    result = _run_matcher(skills, reqs)
    m = result.matches[0]
    assert m.matched is True
    assert m.soft_skill_status == "demonstrated"
    assert result.required_skills_total == 1
    assert result.required_skills_met == 1


def test_inferred_soft_skill_evidence_counts_as_demonstrated():
    """'Mentored junior developers' in a resume bullet -> Leadership,
    inferred, lower confidence than an explicit mention -- but it still
    counts as real evidence, not a gap. Uses a phrased JD requirement
    (not an exact name match) so the semantic/alias path -- where the
    inferred provenance note is attached -- is actually exercised."""
    reqs = JobRequirements(required_skills=["Strong leadership skills"], preferred_skills=[])
    skills = [Skill(name="Leadership", category="soft_skill", source="inferred", confidence=0.7)]
    result = _run_matcher(skills, reqs)
    m = result.matches[0]
    assert m.matched is True
    assert m.match_quality == "semantic"
    assert m.soft_skill_status == "demonstrated"
    assert "inferred" in m.notes.lower()


def test_exact_name_match_still_wins_over_semantic_for_explicit_soft_skill():
    """An explicit, exactly-named skill should still take the fast exact
    path (unaffected by the soft-skill provenance note logic)."""
    reqs = JobRequirements(required_skills=["Leadership"], preferred_skills=[])
    skills = [Skill(name="Leadership", category="soft_skill", source="explicit", confidence=0.95)]
    result = _run_matcher(skills, reqs)
    m = result.matches[0]
    assert m.matched is True
    assert m.match_quality == "exact"
    assert m.soft_skill_status == "demonstrated"


def test_technical_requirement_gap_behavior_is_unchanged():
    """Regression guard: ordinary technical gaps must still count exactly
    as before -- only PURE soft-skill requirements get the new treatment."""
    reqs = JobRequirements(required_skills=["Kubernetes"], preferred_skills=[])
    skills = [Skill(name="Python", category="language", source="explicit")]
    result = _run_matcher(skills, reqs)
    m = result.matches[0]
    assert m.matched is False
    assert m.soft_skill_status is None  # not a soft skill -- ordinary gap
    assert result.required_skills_total == 1
    assert result.required_skills_met == 0
    assert result.overall_score == 0.0


def test_mixed_requirement_with_soft_and_technical_terms_is_not_exempted():
    """A requirement referencing BOTH a technical and a soft-skill concept
    must not be silently exempted from scoring just because part of it is
    soft-skill flavored."""
    assert requirement_is_pure_soft_skill("Communication skills and Python") is False
    assert requirement_is_pure_soft_skill("Good communication skills in English") is True
    assert requirement_is_pure_soft_skill("Python") is False


def test_soft_skill_matcher_is_still_fully_deterministic():
    reqs = JobRequirements(
        required_skills=["Teamwork", "Python", "Good communication skills"],
        preferred_skills=["Leadership"],
    )
    skills = [Skill(name="Python", category="language", source="explicit")]
    r1 = _run_matcher(skills, reqs)
    r2 = _run_matcher(skills, reqs)
    assert r1.model_dump() == r2.model_dump()


# ---------------------------------------------------------------------------
# Regression guard: the two SkillMatch classes (src/models.py Pydantic vs
# src/db/models.py SQLAlchemy) must stay in sync, or repository.py's
# `models.SkillMatch(**pydantic_match.model_dump())` call breaks at
# runtime for every real screening, even though agent-level tests above
# (which never touch the DB) keep passing. See demo_screening.py ->
# screen_candidate() -> repository.save_skill_match_result().
# ---------------------------------------------------------------------------

def test_pydantic_and_orm_skill_match_fields_are_in_sync():
    from src.db import models as db_models
    from src.models import SkillMatch as PydanticSkillMatch

    orm_columns = set(db_models.SkillMatch.__table__.columns.keys()) - {"id", "skill_match_result_id"}
    pydantic_fields = set(PydanticSkillMatch.model_fields.keys())
    missing_from_orm = pydantic_fields - orm_columns
    assert not missing_from_orm, (
        f"src/models.py::SkillMatch has fields the SQLAlchemy SkillMatch "
        f"table doesn't: {missing_from_orm}. repository.save_skill_match_result() "
        f"will raise 'invalid keyword argument' the next time a real "
        f"screening tries to persist one of these matches."
    )


def test_save_skill_match_result_accepts_soft_skill_status(tmp_path, monkeypatch):
    """Reproduces the exact failing call path from screen_candidate():
    repository.save_skill_match_result(matches=[pydantic_match.model_dump(), ...])
    with a real soft_skill_status value present."""
    import os
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'verify.db'}")

    # database.py reads DATABASE_URL at import time, so this needs a fresh
    # import bound to the temp DB rather than reusing the already-imported
    # src.db module (which may be bound to a different engine).
    import importlib
    from src.db import database as db_module
    importlib.reload(db_module)
    from src.db import repository as repo

    db_module.init_db()
    with db_module.get_session() as db:
        org = repo.create_organization(db, "T")
        dept = repo.create_department(db, org.id, "E")
        pos = repo.create_position(db, dept.id, "P")
        jd = repo.create_job_description(db, pos.id, raw_text="jd")
        cand = repo.get_or_create_candidate(db, name="C", email="c@example.com")
        resume = repo.create_resume(db, candidate_id=cand.id, position_id=pos.id, raw_text="r")
        screening = repo.create_screening(db, resume.id, jd.id)

        result = repo.save_skill_match_result(
            db, screening.id,
            required_skills_met=1, required_skills_total=1,
            preferred_skills_met=0, preferred_skills_total=0,
            overall_score=1.0, confidence=0.95, reasoning="test",
            matches=[
                {"requirement": "Python", "matched": True, "matched_skill": "Python",
                 "match_quality": "exact", "confidence": 1.0, "notes": "",
                 "soft_skill_status": None},
                {"requirement": "Good communication skills", "matched": False,
                 "matched_skill": "", "match_quality": "none", "confidence": 0.9,
                 "notes": "", "soft_skill_status": "not_mentioned"},
            ],
        )
        assert len(result.matches) == 2
        statuses = {m.requirement: m.soft_skill_status for m in result.matches}
        assert statuses["Good communication skills"] == "not_mentioned"
        assert statuses["Python"] is None
