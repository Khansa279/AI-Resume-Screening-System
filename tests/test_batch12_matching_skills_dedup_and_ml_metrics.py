"""Batch 12 regression tests: two confirmed, independent fixes.

1. DUPLICATE `matching_skills` (src/services/screening_service.py)
   Multiple JD requirements can independently resolve to the SAME
   candidate skill (e.g. a "Machine Learning" JD requirement AND a
   separate "ML experience" requirement both match on the candidate's
   single "Machine Learning" skill). Each such match is a legitimate,
   distinct SkillMatch row -- scoring and per-requirement matching are
   completely unaffected -- but `_build_explanation()` previously built
   `matching_skills` by listing `matched_skill` once per matched
   requirement, so the human-facing explanation could read "Machine
   Learning, Machine Learning, Machine Learning" for a single real
   skill. Fixed by deduplicating (case-insensitively, order-preserving)
   in `_build_explanation()` only -- `skill_gaps`, `overall_score`,
   `required_skills_met/total`, and every other field are untouched.

2. COMPOUND ML REQUIREMENT GAPS (src/skill_taxonomy.py)
   Precision, Recall, F1-score, ROC-AUC, MAE, RMSE, and Data
   Preprocessing were entirely absent from the shared skill taxonomy, so
   a compound requirement like "Model evaluation techniques
   (cross-validation, precision, recall, F1-score, ROC-AUC, MAE, RMSE)"
   could never be satisfied even when a resume's own text clearly
   demonstrated these concepts, because none of the sub-concepts inside
   the requirement string were ever recognized by
   extract_canonical_skills() in the first place.

   Fix: seven new taxonomy entries, added with the SAME "don't register
   ambiguous bare words" discipline this file already uses for
   Regression/Classification -- "Precision" and "Recall" are NOT
   registered with their bare single-word form (real collisions:
   precision manufacturing/instruments, product/memory recall), only
   with unambiguous ML-metric phrasing (including the "precision,
   recall" pairing the reported requirement itself uses, which
   normalizes to the "precision recall" alias). F1-score/ROC-AUC/MAE/
   RMSE/Data Preprocessing are unambiguous enough to register more
   directly.
"""

from src.services.screening_service import _build_explanation
from src.skill_taxonomy import extract_canonical_skills, normalize_requirement_skills
from src.models import ExperienceEvaluation, ScreeningOutput, SkillMatch, SkillsMatchResult


# ---------------------------------------------------------------------------
# 1. matching_skills deduplication
# ---------------------------------------------------------------------------

def _explanation_for(matches: list[SkillMatch]) -> dict:
    skills_match = SkillsMatchResult(
        matches=matches,
        required_skills_met=sum(1 for m in matches if m.matched),
        required_skills_total=len(matches),
        overall_score=0.9,
        confidence=0.9,
    )
    exp = ExperienceEvaluation(years_relevant=2.0, role_relevance=0.5)
    output = ScreeningOutput(
        match_score=0.8, recommendation="Proceed", requires_human=False,
        confidence=0.8, reasoning_summary="ok",
    )
    return _build_explanation(skills_match, exp, output)


def test_duplicate_matched_skill_appears_once_in_matching_skills():
    matches = [
        SkillMatch(requirement="Machine Learning", matched=True, matched_skill="Machine Learning",
                   match_quality="exact", confidence=1.0),
        SkillMatch(requirement="ML experience", matched=True, matched_skill="Machine Learning",
                   match_quality="semantic", confidence=0.9),
        SkillMatch(requirement="Applied ML", matched=True, matched_skill="Machine Learning",
                   match_quality="semantic", confidence=0.9),
    ]
    explanation = _explanation_for(matches)
    assert explanation["matching_skills"] == ["Machine Learning"]


def test_dedup_is_case_insensitive_and_keeps_first_seen_casing():
    matches = [
        SkillMatch(requirement="Python", matched=True, matched_skill="Python",
                   match_quality="exact", confidence=1.0),
        SkillMatch(requirement="python programming", matched=True, matched_skill="python",
                   match_quality="partial", confidence=0.5),
    ]
    explanation = _explanation_for(matches)
    assert explanation["matching_skills"] == ["Python"]


def test_dedup_preserves_first_seen_order_for_distinct_skills():
    matches = [
        SkillMatch(requirement="Python", matched=True, matched_skill="Python",
                   match_quality="exact", confidence=1.0),
        SkillMatch(requirement="SQL", matched=True, matched_skill="SQL",
                   match_quality="exact", confidence=1.0),
        SkillMatch(requirement="More Python", matched=True, matched_skill="Python",
                   match_quality="exact", confidence=1.0),
        SkillMatch(requirement="Git", matched=True, matched_skill="Git",
                   match_quality="exact", confidence=1.0),
    ]
    explanation = _explanation_for(matches)
    assert explanation["matching_skills"] == ["Python", "SQL", "Git"]


def test_unmatched_and_empty_matched_skill_entries_are_excluded():
    matches = [
        SkillMatch(requirement="Docker", matched=False, matched_skill="",
                   match_quality="none", confidence=0.9),
        SkillMatch(requirement="Python", matched=True, matched_skill="Python",
                   match_quality="exact", confidence=1.0),
        # Matched but no matched_skill recorded -- should not appear.
        SkillMatch(requirement="Weird", matched=True, matched_skill="",
                   match_quality="partial", confidence=0.5),
    ]
    explanation = _explanation_for(matches)
    assert explanation["matching_skills"] == ["Python"]
    # skill_gaps logic is untouched by this fix.
    assert "Docker" in explanation["skill_gaps"]


def test_no_matches_at_all_gives_empty_matching_skills():
    explanation = _explanation_for([])
    assert explanation["matching_skills"] == []


# ---------------------------------------------------------------------------
# 2. ML evaluation-metric taxonomy entries
# ---------------------------------------------------------------------------

REPORTED_REQUIREMENT = (
    "Model evaluation techniques (cross-validation, precision, recall, "
    "F1-score, ROC-AUC, MAE, RMSE)"
)


def test_reported_compound_requirement_now_recognizes_multiple_metrics():
    """Direct reproduction of the reported requirement string: several
    of its sub-concepts must now be recognized by the taxonomy, where
    previously none of these six were."""
    hits = {name for name, _cat in extract_canonical_skills(REPORTED_REQUIREMENT)}
    assert "Precision" in hits
    assert "F1-score" in hits
    assert "ROC-AUC" in hits
    assert "MAE" in hits
    assert "RMSE" in hits
    assert "Cross-validation" in hits  # pre-existing entry, must still work


def test_normalize_requirement_skills_decomposes_the_reported_requirement():
    keys = normalize_requirement_skills(REPORTED_REQUIREMENT)
    assert "precision" in keys
    assert "f1 score" in keys
    assert "roc auc" in keys
    assert "mae" in keys
    assert "rmse" in keys


def test_f1_score_recognized_in_natural_resume_phrasing():
    hits = {name for name, _cat in extract_canonical_skills(
        "Achieved a strong F1 score and low RMSE on the validation set."
    )}
    assert "F1-score" in hits
    assert "RMSE" in hits


def test_roc_auc_recognized_in_natural_resume_phrasing():
    hits = {name for name, _cat in extract_canonical_skills(
        "Evaluated classifiers using ROC-AUC and cross-validation."
    )}
    assert "ROC-AUC" in hits
    assert "Cross-validation" in hits


def test_mae_and_rmse_recognized_bare_in_regression_context():
    hits = {name for name, _cat in extract_canonical_skills(
        "Built regression models and reported MAE and RMSE on held-out data."
    )}
    assert "MAE" in hits
    assert "RMSE" in hits


def test_data_preprocessing_recognized():
    hits = {name for name, _cat in extract_canonical_skills(
        "Performed data preprocessing and feature engineering before training."
    )}
    assert "Data Preprocessing" in hits


def test_precision_and_recall_paired_phrasing_recognized():
    hits = {name for name, _cat in extract_canonical_skills(
        "Reported precision and recall for each class."
    )}
    assert "Precision" in hits


# ---------------------------------------------------------------------------
# 3. Ambiguous standalone "precision" / "recall" must NOT false-positive
# ---------------------------------------------------------------------------

def test_standalone_precision_in_non_ml_context_is_not_matched():
    hits = {name for name, _cat in extract_canonical_skills(
        "The machinist worked with great precision on the lathe."
    )}
    assert "Precision" not in hits


def test_standalone_recall_product_recall_is_not_matched():
    hits = {name for name, _cat in extract_canonical_skills(
        "Coordinated a product recall after a manufacturing defect was found."
    )}
    assert "Recall" not in hits


def test_standalone_recall_memory_sense_is_not_matched():
    hits = {name for name, _cat in extract_canonical_skills(
        "I recall that the meeting was rescheduled twice."
    )}
    assert "Recall" not in hits


def test_qualified_recall_phrasing_is_still_matched():
    hits = {name for name, _cat in extract_canonical_skills(
        "Reported model recall for the classifier."
    )}
    assert "Recall" in hits


def test_precision_and_recall_pairing_matches_both_directions():
    forward = {name for name, _cat in extract_canonical_skills("precision and recall")}
    reverse = {name for name, _cat in extract_canonical_skills("recall and precision")}
    assert "Precision" in forward
    assert "Recall" in reverse


# ---------------------------------------------------------------------------
# 4. End-to-end: SkillsMatcherAgent now credits the compound requirement
# ---------------------------------------------------------------------------

def test_skills_matcher_credits_compound_requirement_from_metric_evidence():
    import asyncio
    from src.agents.skills_matcher import SkillsMatcherAgent
    from src.models import Skill, JobRequirements

    skills = [
        Skill(name="F1-score", category="technical", source="inferred", confidence=0.7),
        Skill(name="RMSE", category="technical", source="inferred", confidence=0.7),
    ]
    reqs = JobRequirements(required_skills=[REPORTED_REQUIREMENT], preferred_skills=[])

    agent = SkillsMatcherAgent()
    result = asyncio.run(agent.process({
        "extracted_skills": skills, "job_requirements": reqs,
    }))["skills_match"]

    assert result.required_skills_met == 1
    assert result.matches[0].matched is True
