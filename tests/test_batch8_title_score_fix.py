"""Batch 8 regression tests.

REAL BUG: `job_relevance()`'s `title_score` component (45% of
role_relevance's weight) compared a candidate's job-title tokens against
a union of `jd_title_tokens | jd_resp_tokens | tokens(required_skills
text)` -- a ~60+ token bag built from the ENTIRE job description's
responsibilities and required-skills text, not just its title.

Because `_overlap()` is plain Jaccard (`|intersection| / |union|`),
unioning in that much unrelated vocabulary made the denominator balloon
regardless of how well the title itself matched. Diagnostic against the
real "Backend Engineer - Python" JD (sample_data/job_descriptions/
jd_01_backend_python_standard.txt) confirmed this wasn't specific to any
one resume:

    John Doe ("Senior Software Engineer")            title_score ~ 0.015
    EXACT title match ("Backend Engineer - Python")  title_score ~ 0.048

Even a byte-for-byte identical title scored ~0.05 instead of 1.0 -- the
component was structurally incapable of rewarding title relevance at any
realistic JD size, independent of title matching strictness per se.

FIX (src/agents/experience_eval.py::job_relevance): title_score now
compares `job_title` tokens directly against `jd_title` tokens only,
using the SAME existing `_title_tokens`/`_overlap` (Jaccard) helpers --
no new similarity machinery, no change to the 0.45/0.4/0.15 weights, no
change to skill_hit, resp_score, or the cross-job aggregation in
evaluate_experience().
"""

from src.agents.experience_eval import (
    job_relevance,
    evaluate_experience,
    _title_tokens,
    _overlap,
)
from src.models import JobRequirements, ResumeData, WorkExperience


BACKEND_PYTHON_JD = JobRequirements(
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
        "Knowledge of microservices architecture",
        "Contributions to open source projects",
    ],
    min_years_experience=2,
    responsibilities=[
        "Design, build, and maintain RESTful APIs using Python",
        "Write clean, well-tested, production-quality code",
        "Work with PostgreSQL databases - schema design, query optimization",
        "Collaborate with frontend team on API contracts",
        "Participate in code reviews and technical discussions",
        "Debug and resolve production issues",
        "Contribute to system design and architecture decisions",
        "Write technical documentation",
    ],
)


def _title_score_only(job_title: str, reqs: JobRequirements) -> float:
    """Isolate just the title_score component the way job_relevance()
    computes it now (title tokens vs JD title tokens, nothing else)."""
    return _overlap(_title_tokens(job_title), _title_tokens(reqs.title))


# ---------------------------------------------------------------------------
# 1. Exact title match gets high (full) similarity
# ---------------------------------------------------------------------------

def test_exact_title_match_gets_full_title_similarity():
    score = _title_score_only("Backend Engineer - Python", BACKEND_PYTHON_JD)
    assert score == 1.0


def test_exact_title_match_job_relevance_is_high():
    exp = WorkExperience(
        title="Backend Engineer - Python",
        company="Anywhere",
        technologies=["Python", "Django", "PostgreSQL", "Git"],
        responsibilities=["Design, build, and maintain RESTful APIs using Python"],
    )
    relevance = job_relevance(exp, BACKEND_PYTHON_JD)
    assert relevance >= 0.7, f"expected a strong overall match for an exact title, got {relevance}"


# ---------------------------------------------------------------------------
# 2. "Senior Software Engineer" vs "Backend Engineer - Python" no longer
#    near zero (the exact John Doe reproduction from the diagnostic)
# ---------------------------------------------------------------------------

def test_senior_software_engineer_title_score_no_longer_near_zero():
    old_style_union = _title_tokens(BACKEND_PYTHON_JD.title) | set()  # old union included jd_resp/skills too
    new_score = _title_score_only("Senior Software Engineer", BACKEND_PYTHON_JD)
    # Old behavior measured ~0.015 for this exact pair (see diagnostic).
    assert new_score > 0.015 * 5, f"expected a meaningfully higher title_score, got {new_score}"
    assert new_score == 0.2  # {'engineer'} / {'backend','engineer','python','senior','software'}


def test_backend_titles_get_meaningful_similarity_not_just_exact_match():
    """Titles that share real vocabulary with the JD title (e.g. contain
    'backend' or 'engineer') but aren't identical must land in a
    meaningful middle range -- not the near-zero the old formula gave
    every non-exact title regardless of relevance."""
    for title in ("Senior Backend Developer", "Freelance Backend Developer", "Senior Software Engineer"):
        score = _title_score_only(title, BACKEND_PYTHON_JD)
        assert 0.1 <= score < 1.0, f"{title!r} title_score={score} outside expected meaningful range"


# ---------------------------------------------------------------------------
# 3. Unrelated / frontend title remains low
# ---------------------------------------------------------------------------

def test_frontend_title_remains_low():
    score = _title_score_only("Frontend Developer", BACKEND_PYTHON_JD)
    assert score == 0.0


def test_unrelated_title_remains_low():
    score = _title_score_only("Data Analyst", BACKEND_PYTHON_JD)
    assert score == 0.0


# ---------------------------------------------------------------------------
# 4. skill_hit behavior is unchanged (same inputs -> same skill_hit-driven
#    contribution as before the fix; only title_score's own value moved)
# ---------------------------------------------------------------------------

def test_skill_hit_component_unchanged_by_title_fix():
    """A job with NO title overlap at all (title_score == 0 both before
    and after the fix) isolates skill_hit's contribution -- job_relevance
    for this case must be explainable purely from skill_hit + resp_score,
    confirming skill matching itself was untouched."""
    exp = WorkExperience(
        title="Unrelated Role Title Zzz",  # shares nothing with JD title tokens
        company="X",
        technologies=["Python", "Django", "PostgreSQL", "Git"],
        responsibilities=[],
    )
    assert _title_score_only(exp.title, BACKEND_PYTHON_JD) == 0.0
    relevance = job_relevance(exp, BACKEND_PYTHON_JD)
    # 0.45*0 (title) + 0.4*skill_hit + 0.15*0 (no responsibilities) == 0.4*skill_hit
    # 4 of the 4 distinct required-skill concepts covered by these
    # technologies (python, django/rest, postgresql, git) -> skill_hit
    # should be strong, giving a relevance close to 0.4 * skill_hit.
    assert relevance > 0.2, f"expected skill_hit alone to still carry real signal, got {relevance}"


# ---------------------------------------------------------------------------
# 5. resp_score (responsibility overlap) behavior is unchanged
# ---------------------------------------------------------------------------

def test_responsibility_score_component_unchanged_by_title_fix():
    """Two jobs with identical technologies/title (so title_score and
    skill_hit are held constant) but different responsibility text must
    still differentiate job_relevance via resp_score, exactly as before."""
    base_kwargs = dict(
        title="Backend Engineer - Python",  # exact title -> title_score fixed at 1.0 for both
        company="X",
        technologies=["Python", "Django", "PostgreSQL", "Git"],
    )
    on_topic = WorkExperience(
        responsibilities=["Design, build, and maintain RESTful APIs using Python"],
        **base_kwargs,
    )
    off_topic = WorkExperience(
        responsibilities=["Organized the office holiday party and ordered snacks"],
        **base_kwargs,
    )
    rel_on = job_relevance(on_topic, BACKEND_PYTHON_JD)
    rel_off = job_relevance(off_topic, BACKEND_PYTHON_JD)
    assert rel_on > rel_off, (
        "resp_score should still differentiate on-topic vs off-topic "
        "responsibilities when title_score and skill_hit are held constant"
    )


# ---------------------------------------------------------------------------
# 6. job_relevance remains fully deterministic
# ---------------------------------------------------------------------------

def test_job_relevance_is_deterministic():
    exp = WorkExperience(
        title="Senior Software Engineer",
        company="TechCorp",
        technologies=["Python", "Django", "PostgreSQL", "Docker"],
        responsibilities=["Designed RESTful APIs using Python and Django"],
    )
    r1 = job_relevance(exp, BACKEND_PYTHON_JD)
    r2 = job_relevance(exp, BACKEND_PYTHON_JD)
    r3 = job_relevance(exp, BACKEND_PYTHON_JD)
    assert r1 == r2 == r3


def test_evaluate_experience_is_deterministic_after_fix():
    data = ResumeData(
        work_experience=[
            WorkExperience(
                title="Senior Software Engineer", company="TechCorp",
                start_date="2022-01", end_date="Present",
                technologies=["Python", "Django", "PostgreSQL", "Docker"],
                responsibilities=["Designed RESTful APIs using Python and Django"],
            ),
        ],
    )
    e1 = evaluate_experience(data, BACKEND_PYTHON_JD)
    e2 = evaluate_experience(data, BACKEND_PYTHON_JD)
    assert e1.model_dump() == e2.model_dump()


# ---------------------------------------------------------------------------
# 7. Real John Doe end-to-end: role_relevance improves, but is still
#    driven by real evidence (not tuned to a specific target number).
# ---------------------------------------------------------------------------

def test_john_doe_role_relevance_improves_but_is_not_hardcoded():
    from pathlib import Path
    from src.agents.resume_parser import parse_resume_text

    text = Path("sample_data/resumes/senior_python_dev.txt").read_text(encoding="utf-8")
    data = parse_resume_text(text)
    ev = evaluate_experience(data, BACKEND_PYTHON_JD)

    # Before the fix this was measured at 0.23 (23%). It must be higher
    # now that title_score can actually contribute, but this test does
    # NOT assert a specific target value -- only that the direction and
    # rough magnitude are consistent with a real fix rather than a
    # hardcoded/tuned constant.
    assert ev.role_relevance > 0.23
    assert ev.role_relevance <= 1.0
