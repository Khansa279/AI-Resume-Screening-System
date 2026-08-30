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


# ---------------------------------------------------------------------------
# (e) Project-relevance hard-cliff fix (deferred from the original Batch 6
# investigation, addressed once project-boundary parsing was fixed
# separately).
#
# REAL BUG: `_best_project` zeroed out a project's score entirely whenever
# it fell below `_PROJECT_RELEVANCE_THRESHOLD` (0.3), even though
# `evaluate_experience` feeds that score into
# `role_relevance = max(education_baseline, project_score)` -- a formula
# whose whole point is to take whichever signal is strongest. Because
# `project_relevance()`'s skill_hit component is quantized in thirds
# (0.75 * matched/3), a project that genuinely demonstrates exactly ONE
# of the JD's combined required+preferred skills scores exactly 0.25 --
# real, structured evidence -- which sits just under the 0.3 cliff and
# was previously discarded outright, indistinguishable from having no
# project at all, even when 0.25 > a candidate's own 0.15 education
# baseline.
#
# FIX: `_best_project` now returns the raw, continuous score (never
# gated to 0.0). The 0.3 threshold still gates ONLY whether a project is
# named by title in strengths/reasoning (`relevant_project` in
# evaluate_experience) -- a presentation judgment, kept exactly as
# before. These tests assert the STRUCTURAL invariants (continuity,
# max()-based contribution, unaffected messaging threshold) rather than
# any single arbitrary percentage.
# ---------------------------------------------------------------------------

def test_e_project_matching_exactly_one_required_skill_scores_below_threshold():
    """Pin the exact quantization that produces the cliff case: matching
    1 of several combined required+preferred skills, with zero title/
    responsibility overlap, must land at 0.25 -- below the 0.3 threshold
    but clearly non-zero, real evidence."""
    project = (
        "Factory Logistics Simulator -- a desktop app written in Python "
        "to model factory floor logistics"
    )
    score = project_relevance(project, ENTRY_LEVEL_JD)
    assert score == 0.25
    assert score < 0.3


def test_e_best_project_no_longer_zeroes_out_a_below_threshold_score():
    """Core fix: _best_project must return the real score/text, not
    collapse it to (0.0, None), once it clears zero."""
    project = (
        "Factory Logistics Simulator -- a desktop app written in Python "
        "to model factory floor logistics"
    )
    score, text = _best_project([project], ENTRY_LEVEL_JD)
    assert score == 0.25
    assert text == project


def test_e_modest_project_raises_role_relevance_above_baseline_via_max():
    """End-to-end: a project scoring between the education baseline
    (0.15) and the naming threshold (0.3) must still raise role_relevance
    above the baseline -- the exact case the old cliff discarded."""
    project = (
        "Factory Logistics Simulator -- a desktop app written in Python "
        "to model factory floor logistics"
    )
    baseline_ev = evaluate_experience(_entry_level_candidate(), ENTRY_LEVEL_JD)
    boosted_ev = evaluate_experience(_entry_level_candidate(projects=[project]), ENTRY_LEVEL_JD)

    assert baseline_ev.role_relevance == 0.15
    assert boosted_ev.role_relevance == 0.25
    assert boosted_ev.role_relevance > baseline_ev.role_relevance


def test_e_below_threshold_project_still_not_named_in_strengths():
    """The 0.3 threshold is UNCHANGED for the "name it" messaging
    decision: a 0.25-scoring project raises the number but is still not
    confident enough to be called out by title in strengths."""
    project = (
        "Factory Logistics Simulator -- a desktop app written in Python "
        "to model factory floor logistics"
    )
    ev = evaluate_experience(_entry_level_candidate(projects=[project]), ENTRY_LEVEL_JD)
    assert not any("Relevant project" in s for s in ev.strengths)
    assert "Relevant academic background" in ev.strengths


def test_e_project_at_or_above_threshold_is_still_named_unchanged():
    """Regression guard: a project that DOES clear 0.3 must keep being
    named by title exactly as before this fix."""
    project = (
        "Flood Severity Prediction -- built a Machine Learning model using "
        "Python and scikit-learn to predict flood severity from weather data"
    )
    score, _ = _best_project([project], ENTRY_LEVEL_JD)
    assert score >= 0.3
    ev = evaluate_experience(_entry_level_candidate(projects=[project]), ENTRY_LEVEL_JD)
    assert any("Relevant project" in s for s in ev.strengths)


def test_e_unrelated_zero_score_project_still_contributes_nothing():
    """Regression guard: a genuinely unrelated project (score 0.0) must
    still leave role_relevance exactly at the baseline -- the fix only
    stops discarding NON-zero scores, it does not manufacture relevance
    out of nothing."""
    project = (
        "Personal Blog Website -- a responsive blog built with HTML, CSS, "
        "and a headless CMS for content management"
    )
    assert project_relevance(project, ENTRY_LEVEL_JD) == 0.0
    ev = evaluate_experience(_entry_level_candidate(projects=[project]), ENTRY_LEVEL_JD)
    assert ev.role_relevance == 0.15
    assert not any("Relevant project" in s for s in ev.strengths)


def test_e_best_project_score_is_monotonic_with_threshold_boundary():
    """Continuity check across the old cliff boundary: as project
    relevance crosses 0.3, role_relevance must move smoothly (no
    downward jump), unlike the old behavior where crossing back below
    0.3 caused a discontinuous drop to the bare baseline."""
    below = evaluate_experience(
        _entry_level_candidate(projects=[
            "Factory Logistics Simulator -- a desktop app written in Python "
            "to model factory floor logistics"
        ]),
        ENTRY_LEVEL_JD,
    )
    above = evaluate_experience(
        _entry_level_candidate(projects=[
            "Flood Severity Prediction -- built a Machine Learning model using "
            "Python and scikit-learn to predict flood severity from weather data"
        ]),
        ENTRY_LEVEL_JD,
    )
    assert below.role_relevance == 0.25
    assert above.role_relevance == 0.5
    assert below.role_relevance < above.role_relevance
