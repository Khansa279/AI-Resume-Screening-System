"""Regression tests for the missing AI/ML role-family taxonomy entry.

REAL BUG (confirmed via the real job_relevance()/evaluate_experience()
production path, reproduced against a real candidate -- Alex Morgan, an
AI/ML Engineer with 3 years of directly-matching experience -- screened
against a real "AI/ML Engineer" JD): the role-family taxonomy
(_ROLE_FAMILY_KEYWORDS in src/agents/experience_eval.py) had no entry at
all for AI/ML/data-science titles. Both "AI/ML Engineer" (the JD title)
and "Machine Learning Engineer" (the candidate's real, near-identical
title) tokenized to nothing but the generic word "engineer", which is
the ONLY keyword shared with the domain-less "generic_swe" catch-all
bucket. Two titles denoting the SAME specialization therefore collapsed
to the ambiguous "both landed in the generic bucket" case, which is
deliberately capped at 0.5 relatedness (see _role_family_score's
same-family special case) rather than the 1.0 a genuine same-specific-
family match receives everywhere else in this taxonomy (e.g. "Backend
Engineer" vs "Senior Backend Developer").

This under-credited role_relevance (0.48 instead of a value reflecting
the near-exact title match) for an extremely well-matched real candidate,
directly lowering the final match_score below what the underlying
evidence supports.

FIX: a new "ai_ml" specific family, keyed on the distinctive, low-
ambiguity abbreviations "ai"/"ml"/"nlp" (same safety bar already applied
to "ui" for frontend and "qa" for QA) plus a literal-phrase check for
"machine learning"/"artificial intelligence"/"deep learning"/"data
science" (whole multi-word phrases, since the individual words in them
-- "machine", "learning", "data" -- are exactly the kind of ambiguous
common-English words already excluded from this taxonomy for "server"/
"client"/"platform"). Relatedness to "data" and "generic_swe" mirrors
the existing backend<->devops / backend<->generic_swe adjacencies.

This does NOT touch the 0.45/0.40/0.15 job_relevance weights, skill_hit,
resp_score, or any existing family/relatedness entry.
"""

from src.agents.experience_eval import (
    job_relevance,
    _primary_role_family,
    _role_family_score,
    _ROLE_RELATEDNESS,
)
from src.models import JobRequirements, WorkExperience


# ---------------------------------------------------------------------------
# _primary_role_family: AI/ML titles now resolve to a real family
# ---------------------------------------------------------------------------

def test_ai_ml_engineer_title_resolves_to_ai_ml_family():
    assert _primary_role_family("AI/ML Engineer") == "ai_ml"


def test_bare_ai_and_ml_and_nlp_titles_resolve_to_ai_ml_family():
    assert _primary_role_family("AI Engineer") == "ai_ml"
    assert _primary_role_family("ML Engineer") == "ai_ml"
    assert _primary_role_family("NLP Engineer") == "ai_ml"


def test_machine_learning_phrase_titles_resolve_to_ai_ml_family():
    assert _primary_role_family("Machine Learning Engineer") == "ai_ml"
    assert _primary_role_family("Senior Machine Learning Engineer") == "ai_ml"


def test_other_ai_ml_phrase_titles_resolve_to_ai_ml_family():
    assert _primary_role_family("Deep Learning Engineer") == "ai_ml"
    assert _primary_role_family("Artificial Intelligence Engineer") == "ai_ml"
    assert _primary_role_family("Data Science Engineer") == "ai_ml"


# ---------------------------------------------------------------------------
# Ambiguous single words from the AI/ML phrases must NOT become
# standalone false-positive triggers (same safety bar as "server"/
# "client"/"platform" already excluded from this taxonomy).
# ---------------------------------------------------------------------------

def test_machine_operator_is_not_misclassified_as_ai_ml():
    assert _primary_role_family("Machine Operator") != "ai_ml"


def test_learning_and_development_manager_is_not_misclassified_as_ai_ml():
    assert _primary_role_family("Learning & Development Manager") != "ai_ml"


# ---------------------------------------------------------------------------
# Core fix: same-specialization AI/ML titles now score full relatedness,
# not the ambiguous generic_swe 0.5 cap.
# ---------------------------------------------------------------------------

def test_ai_ml_engineer_and_machine_learning_engineer_are_same_family_full_credit():
    assert _role_family_score("Machine Learning Engineer", "AI/ML Engineer") == 1.0


def test_ai_ml_title_score_no_longer_capped_at_generic_bucket_value():
    jd = JobRequirements(title="AI/ML Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Machine Learning Engineer", company="X")
    # Isolated title-only relevance (no technologies/responsibilities):
    # must now be the FULL 0.45 (same-specific-family == 1.0 title_score),
    # not the old 0.45*0.5=0.225 generic_swe-bucket value.
    relevance = job_relevance(exp, jd)
    assert relevance == 0.45


def test_reverse_direction_also_gets_full_credit():
    jd = JobRequirements(title="Machine Learning Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="AI/ML Engineer", company="X")
    assert job_relevance(exp, jd) == 0.45


# ---------------------------------------------------------------------------
# Relatedness to other families: mirrors existing adjacency pattern
# ---------------------------------------------------------------------------

def test_ai_ml_data_relatedness_is_defensible_and_nonzero():
    assert _ROLE_RELATEDNESS[frozenset({"ai_ml", "data"})] == 0.45


def test_ai_ml_generic_swe_relatedness_matches_other_specific_families():
    assert _ROLE_RELATEDNESS[frozenset({"ai_ml", "generic_swe"})] == 0.55


def test_data_scientist_gets_partial_credit_against_ai_ml_jd():
    assert _role_family_score("Data Scientist", "AI/ML Engineer") == 0.45


def test_unrelated_family_still_scores_zero_against_ai_ml_jd():
    assert _role_family_score("Frontend Developer", "AI/ML Engineer") == 0.0


# ---------------------------------------------------------------------------
# Regression guard: existing families/behavior are completely unaffected
# ---------------------------------------------------------------------------

def test_data_analyst_family_unaffected_by_new_ai_ml_phrase_check():
    """'Data Analyst' has no 'data science' substring -- must still
    resolve via the existing 'data' family (token 'data'/'analyst'),
    not be swallowed by the new phrase check."""
    assert _primary_role_family("Data Analyst") == "data"


def test_backend_engineer_family_unaffected():
    assert _primary_role_family("Backend Engineer") == "backend"


def test_generic_software_engineer_still_generic_swe_when_no_ai_ml_signal():
    assert _primary_role_family("Software Engineer") == "generic_swe"


def test_end_to_end_real_candidate_role_relevance_improves():
    """End-to-end reproduction of the real diagnosed case: an AI/ML
    Engineer candidate with directly matching titles/skills against an
    AI/ML Engineer JD must no longer have role_relevance suppressed by
    the missing family entry."""
    from src.agents.experience_eval import evaluate_experience
    from src.models import ResumeData

    jd = JobRequirements(
        title="AI/ML Engineer",
        required_skills=["Python", "PyTorch", "NLP"],
        min_years_experience=2,
    )
    data = ResumeData(
        work_experience=[
            WorkExperience(
                title="Machine Learning Engineer", company="Nova Analytics",
                start_date="2023-01", end_date="Present",
                technologies=["Python", "PyTorch", "Natural Language Processing"],
                responsibilities=["Developed end-to-end machine learning pipelines"],
            ),
        ],
    )
    ev = evaluate_experience(data, jd)
    assert ev.role_relevance >= 0.5, (
        f"role_relevance={ev.role_relevance} still looks suppressed by the "
        f"missing ai_ml family entry for a near-exact title match"
    )