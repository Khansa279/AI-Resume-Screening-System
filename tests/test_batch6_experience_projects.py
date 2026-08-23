"""Batch 6 regression tests.

REAL BUG: candidates with zero listed work_experience (the standard shape
for a fresh-graduate / entry-level "Associate AI Engineer, 0-1 years"
candidate) had experience_score hard-coded to 0.0 in
ExperienceEvaluatorAgent.evaluate_experience(), regardless of how many
years the role actually required. The function already had a
years-required-aware fallback (0.7 for a 0-years-required role with zero
relevant experience) further down -- but that logic was unreachable for
anyone with an empty work_experience list, because the function returned
early before ever getting there.

Separately, resume_data.projects (a flat list[str] of academic/personal
project blurbs -- e.g. "Flood Severity Prediction -- built an ML model
using Python and scikit-learn") was never read by this agent at all, even
though jd_05_associate_ai_engineer.txt itself explicitly lists "academic
or personal AI/ML projects" as a preferred qualification for exactly this
population.

Fix (src/agents/experience_eval.py):
  - _experience_score_for_years() is now shared between the "has work
    experience" and "no work experience" code paths, so an entry-level
    candidate against a 0-1-year-required role gets experience_score=0.7
    instead of 0.0.
  - project_relevance() / _best_project() score resume_data.projects
    against the job requirements using the same taxonomy-driven skill
    matching (extract_canonical_skills) and token-overlap approach as
    job_relevance() uses for real WorkExperience entries, and feed a
    relevant project into role_relevance and strengths.
  - Projects are NEVER added to years_relevant -- see
    test_projects_do_not_inflate_years_relevant below.

These tests intentionally do NOT touch the skills matcher, taxonomy,
JobAnalyzer, or DecisionSynthesizer scoring weights.
"""

from src.agents.experience_eval import (
    evaluate_experience,
    project_relevance,
    _best_project,
    _experience_score_for_years,
)
from src.models import Education, JobRequirements, ResumeData, WorkExperience


ENTRY_LEVEL_JD = JobRequirements(
    title="Associate AI Engineer",
    required_skills=["Python", "Machine Learning", "Deep Learning", "Artificial Intelligence"],
    preferred_skills=["Data Structures", "Algorithms", "SQL", "Git"],
    min_years_experience=0,
    responsibilities=["Assist in designing, developing, and deploying AI and ML solutions"],
)


def _entry_level_candidate(projects: list[str] | None = None) -> ResumeData:
    return ResumeData(
        education=[Education(degree="BS", field="Computer Science", institution="XYZ University")],
        projects=projects or [],
    )


# ---------------------------------------------------------------------------
# (a) Entry-level candidate with no work experience
# ---------------------------------------------------------------------------

def test_a_entry_level_no_work_experience_not_zeroed_for_zero_years_required():
    """Core fix: a 0-1-year role should not floor a fresh graduate's
    experience_score to 0.0 just because work_experience is empty."""
    data = _entry_level_candidate()
    ev = evaluate_experience(data, ENTRY_LEVEL_JD)

    assert ev.years_relevant == 0.0
    assert ev.experience_score == 0.7  # matches the years_required<=0 fallback used elsewhere
    assert "No work experience listed" in ev.gaps_identified


def test_a_experience_score_still_reflects_years_required_when_nonzero():
    """The fix must not blanket-boost every empty-work-experience
    candidate -- a role that genuinely requires years should still score
    a no-experience candidate low."""
    data = _entry_level_candidate()
    reqs = ENTRY_LEVEL_JD.model_copy(update={"min_years_experience": 3})
    ev = evaluate_experience(data, reqs)

    assert ev.years_relevant == 0.0
    assert ev.experience_score == 0.0
    assert any("requires 3y" in g for g in ev.gaps_identified)


def test_a_shared_helper_matches_both_code_paths():
    """_experience_score_for_years is the single source of truth now --
    both the empty- and non-empty-work-experience paths must agree for
    the same (years_relevant, years_required) pair."""
    assert _experience_score_for_years(0.0, 0) == 0.7
    assert _experience_score_for_years(0.0, 2) == 0.0
    assert _experience_score_for_years(2.0, 2) == 1.0


# ---------------------------------------------------------------------------
# (b) Relevant ML/AI project improving relevance
# ---------------------------------------------------------------------------

def test_b_relevant_ml_project_boosts_role_relevance_and_strengths():
    data = _entry_level_candidate(projects=[
        "Flood Severity Prediction -- built a Machine Learning model in "
        "Python using scikit-learn to predict flood severity from weather data",
    ])
    ev = evaluate_experience(data, ENTRY_LEVEL_JD)

    baseline = evaluate_experience(_entry_level_candidate(), ENTRY_LEVEL_JD)
    assert ev.role_relevance > baseline.role_relevance
    assert any("Flood Severity Prediction" in s for s in ev.strengths)


def test_b_project_relevance_scores_higher_for_on_topic_project():
    on_topic = project_relevance(
        "Flood Severity Prediction -- built a Machine Learning model in "
        "Python using scikit-learn",
        ENTRY_LEVEL_JD,
    )
    off_topic = project_relevance(
        "Personal Blog Website -- a responsive blog built with HTML and CSS",
        ENTRY_LEVEL_JD,
    )
    assert on_topic > off_topic


# ---------------------------------------------------------------------------
# (c) Unrelated project must NOT get an artificial relevance boost
# ---------------------------------------------------------------------------

def test_c_unrelated_project_does_not_boost_relevance_or_strengths():
    data_no_project = _entry_level_candidate()
    data_unrelated_project = _entry_level_candidate(projects=[
        "Personal Blog Website -- a responsive blog built with HTML, CSS, "
        "and a headless CMS for content management",
    ])

    ev_no_project = evaluate_experience(data_no_project, ENTRY_LEVEL_JD)
    ev_unrelated = evaluate_experience(data_unrelated_project, ENTRY_LEVEL_JD)

    # Same role_relevance as having no projects at all -- education-only baseline.
    assert ev_unrelated.role_relevance == ev_no_project.role_relevance == 0.15
    assert not any("Relevant project" in s for s in ev_unrelated.strengths)


def test_c_best_project_returns_none_below_threshold():
    score, project = _best_project(
        ["Personal Blog Website -- HTML and CSS blog with a CMS"],
        ENTRY_LEVEL_JD,
    )
    assert project is None
    assert score == 0.0


# ---------------------------------------------------------------------------
# (d) Projects must not increase professional experience years
# ---------------------------------------------------------------------------

def test_d_projects_do_not_inflate_years_relevant():
    data = _entry_level_candidate(projects=[
        "Flood Severity Prediction -- built a Machine Learning model in "
        "Python using scikit-learn to predict flood severity",
        "Sentiment Analysis Tool -- applied Deep Learning and NLP with "
        "TensorFlow and PyTorch",
    ])
    ev = evaluate_experience(data, ENTRY_LEVEL_JD)
    assert ev.years_relevant == 0.0


def test_d_projects_do_not_change_experience_score_for_nonzero_years_required():
    """Projects only ever move role_relevance/strengths -- experience_score
    must be identical with or without a highly-relevant project when the
    role requires actual years the candidate doesn't have."""
    reqs = ENTRY_LEVEL_JD.model_copy(update={"min_years_experience": 3})
    ev_no_project = evaluate_experience(_entry_level_candidate(), reqs)
    ev_with_project = evaluate_experience(
        _entry_level_candidate(projects=[
            "Flood Severity Prediction -- built a Machine Learning model "
            "in Python using scikit-learn",
        ]),
        reqs,
    )
    assert ev_no_project.experience_score == ev_with_project.experience_score == 0.0
    assert ev_no_project.years_relevant == ev_with_project.years_relevant == 0.0


def test_d_candidates_with_real_work_experience_are_unaffected_by_projects_change():
    """Sanity check: the project-relevance path only applies when
    work_experience is empty -- a candidate with real work history scores
    exactly as before, project list ignored."""
    data = ResumeData(
        work_experience=[
            WorkExperience(
                title="Backend Software Engineer", company="TechCorp",
                start_date="2022-01", end_date="Present",
                technologies=["Python", "Django", "PostgreSQL"],
                responsibilities=["Built REST APIs"],
            )
        ],
        projects=["Flood Severity Prediction -- built an ML model"],
    )
    reqs = JobRequirements(
        title="Backend Software Engineer",
        required_skills=["Python", "Django"],
        min_years_experience=2,
    )
    with_projects = evaluate_experience(data, reqs)
    without_projects = evaluate_experience(data.model_copy(update={"projects": []}), reqs)
    assert with_projects.model_dump() == without_projects.model_dump()
