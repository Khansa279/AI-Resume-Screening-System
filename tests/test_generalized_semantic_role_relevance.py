"""Regression tests for the generalized, non-hardcoded role/relevance
replacement in src/agents/experience_eval.py.

BACKGROUND: job_relevance() previously fell back on a hand-curated
"role family" taxonomy (_ROLE_FAMILY_KEYWORDS / _ROLE_RELATEDNESS /
_primary_role_family / _role_family_score) whenever two job titles
shared no literal words. That taxonomy was a FIXED, hardcoded list of
role names (backend/frontend/data/devops/mobile/qa/ai_ml/generic_swe)
with a hand-written relatedness matrix between them -- every unseen role
either fell through to a vague "generic_swe" bucket or scored a flat
0.0, and extending it to a new discipline meant editing this file by
hand (the "ai_ml" family itself was added exactly that way).

FIX: that taxonomy has been replaced by `_semantic_role_relevance()`,
which measures overlap between a candidate's ACTUAL demonstrated skills/
responsibilities and what the JD's title/skills/responsibilities
describe, using the same general-purpose skill taxonomy
(src/skill_taxonomy.py) already used elsewhere in this pipeline for
skill matching. Skill taxonomy entries are technologies/tools/concepts,
not job titles, so this generalizes to any title -- seen or unseen --
without a hardcoded role list. An OPT-IN LLM hint
(`_llm_role_relevance_hint`) exists for genuinely ambiguous cases, is
disabled by default, and always degrades gracefully to the
deterministic score when unavailable.

This file covers the specific scenarios requested for this change:
  1. Related but differently-named roles -> high relevance
  2. Unseen job titles -> the system still works (no crash, sensible score)
  3. Clearly unrelated roles -> low relevance
  4. Existing functionality does not regress (exact/lexical matches,
     determinism, no accidental LLM calls by default)
  5. LLM unavailable/disabled -> deterministic fallback still works
"""

import os

import pytest

from src.agents.experience_eval import (
    job_relevance,
    evaluate_experience,
    _semantic_role_relevance,
    _role_signature,
    _llm_role_relevance_hint,
)
from src.models import JobRequirements, ResumeData, WorkExperience


# ---------------------------------------------------------------------------
# 1. Related but differently-named roles must score HIGH relevance --
#    the core example from the request: an ML Engineer JD vs a Data
#    Scientist candidate with NLP/PyTorch experience.
# ---------------------------------------------------------------------------

ML_ENGINEER_JD = JobRequirements(
    title="Machine Learning Engineer",
    required_skills=["Python", "PyTorch", "Machine Learning", "NLP"],
    preferred_skills=["Docker", "AWS"],
    responsibilities=["Build and deploy machine learning models for production NLP systems"],
)


def test_data_scientist_with_nlp_pytorch_is_highly_relevant_to_ml_engineer_jd():
    """The exact example from the request: titles differ completely
    ("Data Scientist" vs "Machine Learning Engineer" share no words), but
    the candidate's actual skills/responsibilities overlap heavily with
    the JD, so relevance must be HIGH -- not driven by any hardcoded
    role-name mapping."""
    exp = WorkExperience(
        title="Data Scientist",
        company="Acme Analytics",
        technologies=["Python", "PyTorch", "NLP", "Machine Learning"],
        responsibilities=["Built NLP models using PyTorch for text classification"],
    )
    relevance = job_relevance(exp, ML_ENGINEER_JD)
    assert relevance >= 0.4, f"expected high relevance for a genuinely overlapping role, got {relevance}"
    # And it must be well above a genuinely unrelated role's score (see
    # test_unrelated_role_scores_low_relevance below), not just "nonzero".
    assert relevance > 0.3


def test_semantic_signature_overlap_drives_the_high_relevance():
    """Confirms the HIGH relevance above is attributable to the
    generalized signature overlap, not a coincidence of some other
    component."""
    exp = WorkExperience(
        title="Data Scientist",
        company="Acme Analytics",
        technologies=["Python", "PyTorch", "NLP", "Machine Learning"],
        responsibilities=["Built NLP models using PyTorch for text classification"],
    )
    score = _semantic_role_relevance(exp, ML_ENGINEER_JD)
    assert score >= 0.3, f"expected strong semantic signature overlap, got {score}"


def test_differently_worded_titles_with_shared_skills_still_match_for_a_second_domain():
    """A second, unrelated-to-ML example so this isn't a single
    coincidence: a 'Site Reliability Engineer' candidate with
    Kubernetes/Docker/AWS experience against a JD titled 'Platform
    Engineer' asking for the same tooling -- titles share only generic
    words, but the actual work overlaps heavily."""
    jd = JobRequirements(
        title="Platform Engineer",
        required_skills=["Kubernetes", "Docker", "AWS", "Terraform"],
        responsibilities=["Operate and scale Kubernetes clusters across multiple AWS regions"],
    )
    exp = WorkExperience(
        title="Site Reliability Engineer",
        company="CloudCo",
        technologies=["Kubernetes", "Docker", "AWS", "Terraform"],
        responsibilities=["Managed Kubernetes clusters and CI/CD pipelines on AWS"],
    )
    relevance = job_relevance(exp, jd)
    assert relevance >= 0.55, f"expected high relevance from real tooling overlap, got {relevance}"


# ---------------------------------------------------------------------------
# 2. Unseen job titles -- titles that exist in no hardcoded list at all --
#    must not crash and must still produce a sensible, real-signal-driven
#    score.
# ---------------------------------------------------------------------------

def test_completely_novel_titles_do_not_crash_and_score_from_real_skill_overlap():
    """Neither title below is a common role name, let alone one that
    would ever appear in a hand-curated mapping -- the system must not
    error out, and must still find the real overlap in skills."""
    jd = JobRequirements(
        title="Quantum Sensing Systems Integrator",
        required_skills=["Python", "MATLAB", "Signal Processing"],
        responsibilities=["Integrate quantum sensor arrays with control software written in Python"],
    )
    exp = WorkExperience(
        title="Cryogenic Instrumentation Technologist",
        company="LabCo",
        technologies=["Python", "MATLAB"],
        responsibilities=["Wrote Python and MATLAB control software for lab instrumentation"],
    )
    relevance = job_relevance(exp, jd)  # must not raise
    assert 0.0 <= relevance <= 1.0
    assert relevance > 0.0, "real shared skills (Python, MATLAB) should register some relevance"


def test_unseen_title_with_no_recognizable_skills_degrades_to_zero_not_an_error():
    """An unseen title with genuinely no overlapping skill/taxonomy
    signal and no shared title words must gracefully settle at 0.0 --
    not raise, and not silently fall back into some default family
    bucket the way the old taxonomy's 'generic_swe' catch-all did."""
    jd = JobRequirements(title="Widget Optimization Strategist", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Glacial Cartography Fellow", company="X")
    relevance = job_relevance(exp, jd)
    assert relevance == 0.0


def test_role_signature_handles_missing_data_gracefully():
    """_role_signature must not crash on None/empty technologies or
    responsibilities -- a common real-world case for a resume entry with
    a bare title and nothing else."""
    sig = _role_signature("Some Never-Before-Seen Title", None, None)
    assert isinstance(sig, set)


# ---------------------------------------------------------------------------
# 3. Clearly unrelated roles must score LOW relevance.
# ---------------------------------------------------------------------------

def test_unrelated_role_scores_low_relevance():
    jd = JobRequirements(
        title="Backend Engineer",
        required_skills=["Python", "Django", "PostgreSQL"],
        responsibilities=["Build and maintain REST APIs using Python and Django"],
    )
    exp = WorkExperience(
        title="Restaurant Shift Manager",
        company="Diner Co",
        technologies=[],
        responsibilities=["Managed front-of-house staff and inventory", "Handled customer complaints"],
    )
    relevance = job_relevance(exp, jd)
    assert relevance < 0.15, f"expected low relevance for a genuinely unrelated role, got {relevance}"


def test_unrelated_role_with_ambiguous_common_word_still_scores_low():
    """A title that happens to share a common English word with the JD
    title (e.g. 'Server') but has none of the JD's real skills/tooling
    must not be inflated purely from that coincidence."""
    jd = JobRequirements(title="Backend Server Engineer", required_skills=["Python", "Kubernetes"])
    exp = WorkExperience(
        title="Server",  # restaurant waiter
        company="Diner Co",
        responsibilities=["Took customer orders", "Served food and drinks"],
    )
    relevance = job_relevance(exp, jd)
    assert relevance < 0.2, f"expected low relevance despite the shared word 'server', got {relevance}"


# ---------------------------------------------------------------------------
# 4. Existing functionality does not regress.
# ---------------------------------------------------------------------------

def test_exact_title_match_still_scores_full_lexical_relevance():
    jd = JobRequirements(title="Backend Engineer - Python", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Backend Engineer - Python", company="X")
    assert job_relevance(exp, jd) == 0.45  # 0.45 * 1.0 title_score + 0 skill_hit + 0 resp_score


def test_job_relevance_remains_fully_deterministic_by_default():
    """With the (default-disabled) LLM hint off, repeated calls with
    identical input must be byte-identical -- no nondeterminism was
    introduced by the generalized signature approach."""
    jd = ML_ENGINEER_JD
    exp = WorkExperience(
        title="Data Scientist", company="Acme",
        technologies=["Python", "PyTorch", "NLP"],
        responsibilities=["Built NLP models"],
    )
    results = {job_relevance(exp, jd) for _ in range(5)}
    assert len(results) == 1


def test_evaluate_experience_end_to_end_is_unaffected_in_shape():
    """evaluate_experience()'s existing output shape/keys/behavior must
    be completely unaffected -- only the internal title-relevance
    computation changed."""
    data = ResumeData(work_experience=[
        WorkExperience(
            title="Senior Software Engineer", company="TechCorp",
            start_date="2022-01", end_date="Present",
            technologies=["Python", "Django", "PostgreSQL"],
            responsibilities=["Designed RESTful APIs using Python and Django"],
        ),
    ])
    reqs = JobRequirements(
        title="Backend Software Engineer",
        required_skills=["Python", "Django", "SQL"],
        min_years_experience=2,
    )
    ev = evaluate_experience(data, reqs)
    assert ev.years_relevant > 0
    assert 0.0 <= ev.role_relevance <= 1.0
    assert 0.0 <= ev.experience_score <= 1.0


# ---------------------------------------------------------------------------
# 5. LLM unavailable / disabled -> deterministic fallback still works.
# ---------------------------------------------------------------------------

def test_llm_hint_is_disabled_by_default():
    """Without ENABLE_LLM_ROLE_RELEVANCE set, the hint function must
    return None immediately -- no LLM/config access is even attempted."""
    os.environ.pop("ENABLE_LLM_ROLE_RELEVANCE", None)
    assert _llm_role_relevance_hint("Data Scientist", "PyTorch NLP", "ML Engineer", "PyTorch NLP") is None


def test_llm_hint_falls_back_to_none_when_enabled_but_no_provider_configured(monkeypatch):
    """Even if the feature flag is turned on, a missing/invalid LLM
    configuration must be caught and degrade to None, never raise."""
    monkeypatch.setenv("ENABLE_LLM_ROLE_RELEVANCE", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from src import config as config_module
    config_module.reset_config()
    try:
        result = _llm_role_relevance_hint("Data Scientist", "PyTorch", "ML Engineer", "PyTorch")
        assert result is None
    finally:
        monkeypatch.delenv("ENABLE_LLM_ROLE_RELEVANCE", raising=False)
        config_module.reset_config()


def test_job_relevance_works_end_to_end_with_llm_feature_flag_on_but_unconfigured(monkeypatch):
    """End-to-end guard: turning the opt-in flag on, with no LLM provider
    configured (the CI/test default), must not break job_relevance() at
    all -- it silently falls back to the deterministic score."""
    monkeypatch.setenv("ENABLE_LLM_ROLE_RELEVANCE", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from src import config as config_module
    config_module.reset_config()
    try:
        jd = JobRequirements(title="Data Analyst", required_skills=["SQL"])
        exp = WorkExperience(title="Reporting Specialist", company="X", responsibilities=["Wrote SQL reports"])
        relevance = job_relevance(exp, jd)  # must not raise
        assert 0.0 <= relevance <= 1.0
    finally:
        monkeypatch.delenv("ENABLE_LLM_ROLE_RELEVANCE", raising=False)
        config_module.reset_config()


def test_llm_hint_can_only_ever_raise_never_lower_title_score(monkeypatch):
    """Even a well-behaved (mocked) LLM hint must never be allowed to
    pull title_score below what the deterministic signature already
    established -- job_relevance() combines it via max()."""
    monkeypatch.setenv("ENABLE_LLM_ROLE_RELEVANCE", "true")

    def fake_hint(*args, **kwargs):
        return 0.0  # a deliberately LOW hint

    monkeypatch.setattr("src.agents.experience_eval._llm_role_relevance_hint", fake_hint)
    try:
        jd = JobRequirements(title="Backend Engineer - Python", required_skills=[], preferred_skills=[])
        exp = WorkExperience(title="Backend Engineer - Python", company="X")
        # Exact lexical match already gives title_score=1.0; a low LLM
        # hint must not be able to drag it back down.
        assert job_relevance(exp, jd) == 0.45
    finally:
        monkeypatch.delenv("ENABLE_LLM_ROLE_RELEVANCE", raising=False)
