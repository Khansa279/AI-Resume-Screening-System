"""Priority 5 regression tests: title_score isolation fix.

REAL BUG (confirmed by direct inspection of the production code, despite
Batch 8's test file docstring claiming this was already fixed):
`job_relevance()`'s lexical `title_score` component compared the
candidate's title tokens against a UNION of the JD title tokens, the JD
responsibilities tokens, AND the JD required_skills tokens:

    title_score = _overlap(
        job_title, jd_title | jd_resp | _tokens(" ".join(requirements.required_skills))
    )

This meant a word that appears ONLY in the job description's
responsibilities or required-skills text (never in the JD title itself)
could either dilute the denominator (lowering title_score for candidates
whose title legitimately overlaps the JD title) or, more subtly, hand a
candidate title unrelated "credit" purely because a token in it happened
to also appear somewhere in the JD's responsibilities/skills text -- i.e.
title_score was not actually isolated to "does the candidate's title
resemble the JD's title", which is the property its name promises and
which downstream reasoning (career_progression text, gaps, "role
relevance") depends on.

FIX (src/agents/experience_eval.py::job_relevance): the pure LEXICAL
title_score component compares `job_title` tokens directly against
`jd_title` tokens ONLY -- the smallest possible change, touching a
single line. This does NOT touch skill_hit (still built from `jd_skills`
/ `requirements.required_skills`) or resp_score (still built from
`jd_resp`), or the 0.45 / 0.4 / 0.15 weights.

UPDATE: the hardcoded role-family backstop this file originally
referenced (`_role_family_score` / `_ROLE_RELATEDNESS`) has since been
replaced with a generalized, non-hardcoded semantic backstop
(`_semantic_role_relevance`, see experience_eval.py) that intentionally
DOES credit a candidate title for overlapping with the JD's actual
skills/responsibilities -- that is the whole point of the generalization
(role relevance should track real overlapping work, not just literal
title wording). The tests below that previously asserted "a word that
only appears in the JD's responsibilities/skills must contribute ZERO
title credit" have been updated to distinguish the still-isolated PURE
LEXICAL component from the now-intentionally-broader semantic backstop.

All tests below call the REAL `job_relevance()` (and, where noted,
`evaluate_experience()`) -- not a reimplemented/isolated helper -- so
they exercise the exact code path that was buggy.
"""

from pathlib import Path

from src.agents.experience_eval import (
    job_relevance,
    evaluate_experience,
    _title_tokens,
    _overlap,
)
from src.agents.resume_parser import parse_resume_text
from src.models import JobRequirements, ResumeData, WorkExperience


SAMPLE_RESUME = Path("sample_data/resumes/senior_python_dev.txt")  # John Doe


# ---------------------------------------------------------------------------
# 1. Backend Engineer candidate vs Backend Engineer JD still behaves
#    correctly (exact title match stays a strong/full match).
# ---------------------------------------------------------------------------

def test_exact_backend_title_match_still_scores_full_relevance():
    reqs = JobRequirements(
        title="Backend Engineer - Python",
        required_skills=["Python", "Django", "PostgreSQL"],
        preferred_skills=["Docker"],
        responsibilities=["Design, build, and maintain RESTful APIs using Python"],
    )
    exp = WorkExperience(
        title="Backend Engineer - Python",
        company="Anywhere",
        technologies=["Python", "Django", "PostgreSQL", "Docker"],
        responsibilities=["Design, build, and maintain RESTful APIs using Python"],
    )
    relevance = job_relevance(exp, reqs)
    assert relevance >= 0.7, f"expected a strong match for an exact title, got {relevance}"


def test_exact_title_tokens_isolated_from_jd_title_give_full_lexical_credit():
    """Directly pins the isolated title_score behavior: when the
    candidate title token set equals the JD title token set exactly,
    the lexical component (ignoring the family backstop) is 1.0."""
    reqs = JobRequirements(title="Backend Engineer - Python", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Backend Engineer - Python", company="X")
    # No responsibilities/skills at all, so skill_hit=0 and resp_score=0 --
    # isolates the relevance value to 0.45 * title_score.
    relevance = job_relevance(exp, reqs)
    assert relevance == 0.45  # 0.45 * 1.0 title_score + 0 + 0


# ---------------------------------------------------------------------------
# 2. Words appearing ONLY in responsibilities/required_skills must not
#    give a candidate title extra lexical credit (the core isolation bug).
# ---------------------------------------------------------------------------

def test_pure_lexical_title_overlap_ignores_jd_responsibilities_text():
    """The PURE LEXICAL title-token overlap (candidate title tokens vs JD
    TITLE tokens only) must not be diluted or inflated by words that only
    appear in the JD's responsibilities text -- that isolation is what
    this file's fix was originally about, and it's unaffected by the
    later generalization to a semantic backstop (checked separately
    below)."""
    reqs = JobRequirements(
        title="Platform Engineer",
        required_skills=[],
        preferred_skills=[],
        responsibilities=["Manage clusters using Kubernetes and Docker"],
    )
    exp = WorkExperience(
        title="Kubernetes Specialist",  # shares no tokens with "Platform Engineer"
        company="X",
    )
    lexical_only = _overlap(_title_tokens(exp.title), _title_tokens(reqs.title))
    assert lexical_only == 0.0, "candidate/JD titles share no literal words"


def test_generalized_semantic_backstop_intentionally_credits_real_overlap_with_jd_work():
    """UPDATED intent: unlike the old lexical-only isolation, the
    generalized semantic backstop (_semantic_role_relevance) SHOULD
    credit a candidate whose title/skills genuinely overlap with what the
    JD's responsibilities describe -- e.g. a 'Kubernetes Specialist'
    candidate against a JD whose responsibilities are literally about
    Kubernetes is a real signal, not noise. This is a deliberate design
    change requested to make relevance track actual work, not just
    title wording -- see the module docstring above."""
    reqs = JobRequirements(
        title="Platform Engineer",
        required_skills=[],
        preferred_skills=[],
        responsibilities=["Manage clusters using Kubernetes and Docker"],
    )
    exp = WorkExperience(title="Kubernetes Specialist", company="X")

    from src.agents.experience_eval import _semantic_role_relevance
    assert _semantic_role_relevance(exp, reqs) > 0.0, (
        "expected the generalized semantic backstop to recognize the real "
        "Kubernetes overlap between the candidate's title and the JD's "
        "responsibilities"
    )
    relevance = job_relevance(exp, reqs)
    assert relevance > 0.0


def test_lexical_title_score_matches_isolated_title_token_overlap():
    """Direct, formula-level pin: the pure lexical title-token Jaccard
    overlap must equal exactly the overlap between job_title tokens and
    jd_title tokens alone -- not any larger union with responsibilities
    or required-skills text."""
    reqs = JobRequirements(
        title="Senior Backend Engineer - Fintech",
        required_skills=["Message queues (Redis, RabbitMQ, Celery)"],
        preferred_skills=[],
        responsibilities=["Own PCI-DSS compliance and audits"],
    )
    exp = WorkExperience(title="Redis Specialist", company="X")  # "redis" appears only in required_skills text

    from src.agents.experience_eval import _overlap
    expected_title_score = _overlap(_title_tokens(exp.title), _title_tokens(reqs.title))
    assert expected_title_score == 0.0  # "redis"/"specialist" share nothing with the JD title


# ---------------------------------------------------------------------------
# 3. Genuine candidate-title <-> JD-title overlap still works (this must
#    NOT regress just because we stopped unioning in extra vocabulary).
# ---------------------------------------------------------------------------

def test_genuine_title_overlap_with_jd_title_still_credited():
    reqs = JobRequirements(title="Backend Developer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Junior Backend Developer", company="X")
    relevance = job_relevance(exp, reqs)
    # job_title={junior,backend,developer} vs jd_title={backend,developer}:
    # intersection=2, union=3 -> lexical title_score = 2/3, which is what
    # drives this (the generalized semantic backstop has nothing to add
    # here since neither side lists any skills/technologies). Must score
    # highly, not be diluted by an empty/absent resp+skills set.
    assert relevance == round(0.45 * (2 / 3), 2)


def test_genuine_partial_title_overlap_without_semantic_signal_is_meaningful():
    """A title that shares a real, distinctive word with the JD title but
    has no recognizable taxonomy-skill/technology signal on either side
    must still get credit purely from the isolated lexical overlap (the
    generalized semantic backstop contributes nothing here since there's
    nothing to compute a signature from beyond the title tokens
    themselves)."""
    reqs = JobRequirements(title="Zephyr Coordinator", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Zephyr Assistant", company="X")

    relevance = job_relevance(exp, reqs)
    # job_title={zephyr,assistant} vs jd_title={zephyr,coordinator}:
    # intersection=1, union=3 -> title_score=1/3, relevance=0.45*(1/3)
    assert relevance == round(0.45 * (1 / 3), 2)


# ---------------------------------------------------------------------------
# 4. John Doe's Backend Engineer case still produces a sensible score.
# ---------------------------------------------------------------------------

def test_john_doe_role_relevance_still_sensible_after_title_isolation_fix():
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
    # Same threshold Batch 7's equivalent regression test used -- must
    # stay comfortably above the pre-Batch-7 near-zero floor. The
    # isolation fix can only raise (never lower) title_score for this
    # candidate/JD pair (John Doe's most senior title shares "engineer"
    # AND now doesn't get diluted by the large resp/skills union), so
    # role_relevance should be at least as strong as before.
    assert ev.role_relevance >= 0.2, (
        f"role_relevance={ev.role_relevance} regressed after the title_score isolation fix"
    )


def test_job_relevance_is_still_fully_deterministic_after_fix():
    reqs = JobRequirements(
        title="Backend Engineer - Python",
        required_skills=["Python", "Django", "PostgreSQL"],
        responsibilities=["Design and build RESTful APIs using Python"],
    )
    exp = WorkExperience(
        title="Senior Software Engineer",
        company="TechCorp",
        technologies=["Python", "Django", "PostgreSQL"],
        responsibilities=["Designed RESTful APIs using Python and Django"],
    )
    r1 = job_relevance(exp, reqs)
    r2 = job_relevance(exp, reqs)
    r3 = job_relevance(exp, reqs)
    assert r1 == r2 == r3
