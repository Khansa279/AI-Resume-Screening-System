"""Priority 6 / Batch 11 regression tests.

REAL BUG (confirmed via src/scripts/diagnose_years_relevant.py against the
actual sample_data resumes/JDs, using the real, unmodified evaluate_experience()):

    years_relevant = sum(years * (0.35 + 0.65 * rel) for years, rel in per_job)

applied a flat 35% MINIMUM credit to every work-experience entry's
duration, no matter how low job_relevance() scored it -- there was no
lower gate at all. A job scored at rel=0.01 (near-zero relevance) still
counted 35%+ of its duration as "relevant" experience, identical in kind
to a job with real, if lexically weak, relevance.

This produced exactly the reported anomalies when traced against the
real pipeline:
  - Ananya Patel: a 4.07-year, 1%-relevant Frontend Developer role alone
    contributed ~1.45 "relevant" years (35% floor), making years_relevant
    (2.21) wildly inconsistent with role_relevance (7%).
  - Vikram Singh: a genuinely unrelated Mechanical Engineer role (33%
    relevance) plus several resume-parsing artifacts mis-parsed as
    "jobs" (a career-break bracket, an "EDUCATION & TRAINING" section
    header, a university line -- see
    tests/test_bullet_marker_normalization.py for the parsing issue
    itself, which is a SEPARATE, already-partially-mitigated bug) each
    received a guaranteed floor of credit purely for having some
    duration, inflating years_relevant to 5.0 against a 33% role_relevance.

Fix (src/agents/experience_eval.py):
  _years_relevant_credit(rel) now only applies the pre-existing
  0.35 + 0.65*rel hedge (which exists because job_relevance() is a
  lexical/taxonomy approximation that CAN under-score genuinely related
  work -- see the Batch 8/9/10 history in that file) to jobs at/above
  _YEARS_RELEVANT_FLOOR_THRESHOLD (0.2, matching the same rel < 0.2 cutoff
  this module already uses to flag a job as "Limited relevance" a few
  lines below). Below that threshold, credit ramps linearly from 0 at
  rel=0 up to the existing formula's value AT the threshold -- continuous,
  no cliff at the boundary -- so a genuinely irrelevant job now
  contributes years_relevant near 0 instead of a guaranteed 35%+.

Scope discipline: this does NOT touch job_relevance()/job_years()
themselves, role_relevance's own (separate) weighted-average formula,
_experience_score_for_years(), the resume-parsing bug that produces
spurious WorkExperience entries, or any DB/schema/API surface.
"""

from pathlib import Path

from src.agents.experience_eval import (
    evaluate_experience,
    job_relevance,
    job_years,
    _years_relevant_credit,
    _YEARS_RELEVANT_FLOOR_THRESHOLD,
    _YEARS_RELEVANT_FLOOR_CREDIT,
)
from src.agents.resume_parser import parse_resume_text
from src.document_parser import parse_document
from src.models import JobRequirements, ResumeData, WorkExperience


# ---------------------------------------------------------------------------
# 1. _years_relevant_credit itself: no floor below threshold, continuous,
#    unchanged behavior at/above threshold.
# ---------------------------------------------------------------------------

def test_credit_is_zero_at_zero_relevance():
    """The core fix: previously even rel=0.0 got 35% credit. Now it's 0."""
    assert _years_relevant_credit(0.0) == 0.0


def test_credit_ramps_linearly_below_threshold():
    """Credit must scale down toward 0 as relevance drops, not just be a
    smaller flat floor."""
    low = _years_relevant_credit(0.05)
    lower = _years_relevant_credit(0.01)
    assert 0.0 < lower < low < _years_relevant_credit(_YEARS_RELEVANT_FLOOR_THRESHOLD)


def test_credit_is_continuous_at_the_threshold_boundary():
    """No cliff: credit just below the threshold must be close to credit
    just at/above it (within the linear ramp's own granularity), not a
    sudden jump."""
    just_below = _years_relevant_credit(_YEARS_RELEVANT_FLOOR_THRESHOLD - 0.001)
    at = _years_relevant_credit(_YEARS_RELEVANT_FLOOR_THRESHOLD)
    assert abs(at - just_below) < 0.01


def test_credit_above_threshold_matches_original_unchanged_formula():
    """At/above the threshold, behavior must be EXACTLY the original
    0.35 + 0.65*rel hedge -- this is what keeps John Doe (whose jobs were
    all >= 0.2 relevance) completely unaffected by this fix."""
    for rel in (0.2, 0.33, 0.5, 0.6, 1.0):
        expected = _YEARS_RELEVANT_FLOOR_CREDIT + (1 - _YEARS_RELEVANT_FLOOR_CREDIT) * rel
        assert _years_relevant_credit(rel) == expected


def test_credit_never_exceeds_one():
    assert _years_relevant_credit(1.0) == 1.0


# ---------------------------------------------------------------------------
# 2. Direct reproduction of the reported anomalies via evaluate_experience()
# ---------------------------------------------------------------------------

BACKEND_PYTHON_JD = JobRequirements(
    title="Backend Engineer - Python",
    required_skills=[
        "2+ years of professional experience with Python",
        "Experience building REST APIs (Django REST Framework or FastAPI preferred)",
        "Strong SQL skills and experience with relational databases (PostgreSQL)",
        "Familiarity with Git version control",
    ],
    preferred_skills=["Experience with Django or FastAPI frameworks", "Docker"],
    min_years_experience=2,
    responsibilities=["Design, build, and maintain RESTful APIs using Python"],
)

FINTECH_SENIOR_JD = JobRequirements(
    title="Senior Backend Engineer - Fintech",
    required_skills=[
        "4+ years building backend systems in Python",
        "2+ years experience in fintech/payments domain",
        "Expert-level Django or FastAPI knowledge",
        "Deep PostgreSQL expertise",
    ],
    preferred_skills=["Previous experience at payment gateways"],
    min_years_experience=4,
    responsibilities=["Architect and build payment processing microservices"],
)


def test_ananya_patel_years_relevant_now_consistent_with_low_role_relevance():
    """Before the fix: years_relevant=2.21 against role_relevance=0.07 --
    a 4+ year, near-zero-relevance Frontend Developer role alone
    contributed the majority of that via the 35% floor. After the fix,
    years_relevant must drop sharply and be genuinely small, in line with
    what a 7% role_relevance implies."""
    parsed = parse_document("sample_data/resumes/resume_03_ananya_patel.pdf")
    assert parsed.success
    data = parse_resume_text(parsed.text)

    ev = evaluate_experience(data, BACKEND_PYTHON_JD)
    assert ev.role_relevance < 0.1  # confirms this is really the low-relevance case
    assert ev.years_relevant < 1.5, (
        f"years_relevant={ev.years_relevant} is still implausibly high for "
        f"role_relevance={ev.role_relevance}"
    )
    # Sanity: must have dropped meaningfully from the pre-fix 2.21 figure.
    assert ev.years_relevant < 2.0


def test_vikram_singh_years_relevant_drops_when_artifact_and_unrelated_jobs_present():
    """Before the fix: years_relevant=5.0 against role_relevance=0.33,
    inflated by resume-parsing artifacts (career break bracket, education
    section header, university line -- each near-zero relevance) each
    getting a guaranteed 35% floor. After the fix these near-zero-relevance
    entries contribute close to nothing, so the total must be measurably
    lower even though the genuinely relevant jobs (Freelance Backend
    Developer, and the ambiguous Mechanical Engineer role which stays
    above the threshold) are unaffected."""
    parsed = parse_document("sample_data/resumes/resume_04_vikram_singh.pdf")
    assert parsed.success
    data = parse_resume_text(parsed.text)

    ev = evaluate_experience(data, BACKEND_PYTHON_JD)
    assert ev.years_relevant < 5.0, (
        f"years_relevant={ev.years_relevant} should have dropped from the "
        f"pre-fix 5.0 once near-zero-relevance entries lost their floor credit"
    )


def test_john_doe_unaffected_since_all_his_jobs_are_above_threshold():
    """Regression guard for THIS fix specifically (the years_relevant
    floor/ramp in _years_relevant_credit): regardless of how job_relevance()
    itself computes a given job's relevance (that's a separate concern --
    see the generalized, non-hardcoded role-relevance replacement in
    src/agents/experience_eval.py, which can legitimately shift individual
    job_relevance() values up or down from what an earlier hardcoded
    role-family taxonomy produced), evaluate_experience()'s years_relevant
    must always exactly match summing job_years(e) * _years_relevant_credit(rel)
    over each job -- i.e. the credit formula itself (this fix's actual
    subject) is applied consistently, whatever the underlying relevance
    numbers happen to be."""
    text = Path("sample_data/resumes/senior_python_dev.txt").read_text(encoding="utf-8")
    data = parse_resume_text(text)

    for jd in (BACKEND_PYTHON_JD, FINTECH_SENIOR_JD):
        ev = evaluate_experience(data, jd)
        expected = round(
            sum(job_years(e) * _years_relevant_credit(job_relevance(e, jd)) for e in data.work_experience), 2,
        )
        assert ev.years_relevant == expected


# ---------------------------------------------------------------------------
# 3. Synthetic edge cases: a job with EXACTLY zero relevance must
#    contribute nothing to years_relevant, regardless of its duration.
# ---------------------------------------------------------------------------

def test_completely_unrelated_long_job_contributes_near_zero_years():
    """A long-tenured but completely unrelated role (e.g. hospitality,
    which shares no taxonomy skills or title tokens with a backend JD)
    must no longer inflate years_relevant just because it lasted a long
    time."""
    data = ResumeData(
        work_experience=[
            WorkExperience(
                title="Restaurant Shift Manager", company="Cafe Co",
                start_date="2015-01", end_date="2023-01",
                responsibilities=["Managed front-of-house staff and inventory"],
                technologies=[],
            ),
        ],
    )
    reqs = JobRequirements(
        title="Backend Engineer - Python",
        required_skills=["Python", "Django", "PostgreSQL"],
        min_years_experience=2,
    )
    ev = evaluate_experience(data, reqs)
    rel = job_relevance(data.work_experience[0], reqs)
    assert rel < 0.05
    assert ev.years_relevant < 0.5, (
        f"years_relevant={ev.years_relevant} for an 8-year, near-zero-relevance "
        f"role is still implausibly high"
    )


def test_mixed_relevant_and_unrelated_jobs_only_relevant_one_counts_substantially():
    data = ResumeData(
        work_experience=[
            WorkExperience(
                title="Backend Engineer", company="TechCo",
                start_date="2022-01", end_date="2023-01",
                technologies=["Python", "Django", "PostgreSQL"],
                responsibilities=["Built REST APIs with Django and PostgreSQL"],
            ),
            WorkExperience(
                title="Restaurant Shift Manager", company="Cafe Co",
                start_date="2015-01", end_date="2021-01",
                responsibilities=["Managed front-of-house staff"],
                technologies=[],
            ),
        ],
    )
    reqs = JobRequirements(
        title="Backend Engineer - Python",
        required_skills=["Python", "Django", "PostgreSQL"],
        min_years_experience=2,
    )
    ev = evaluate_experience(data, reqs)
    # ~1 year of genuinely relevant backend work, plus near-nothing from
    # the 6-year unrelated role -- must land close to the relevant job's
    # own duration, not be inflated by the unrelated one.
    assert 0.5 < ev.years_relevant < 2.0


# ---------------------------------------------------------------------------
# 4. Determinism preserved
# ---------------------------------------------------------------------------

def test_years_relevant_credit_fix_is_still_deterministic():
    text = Path("sample_data/resumes/senior_python_dev.txt").read_text(encoding="utf-8")
    data = parse_resume_text(text)
    e1 = evaluate_experience(data, BACKEND_PYTHON_JD)
    e2 = evaluate_experience(data, BACKEND_PYTHON_JD)
    assert e1.model_dump() == e2.model_dump()
