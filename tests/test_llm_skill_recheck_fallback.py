"""Regression tests for SkillsMatcherAgent's LLM fallback/recheck layer.

BUG BEING FIXED: the deterministic taxonomy/alias matcher
(src/skill_taxonomy.py) cannot scale to every semantic relationship
between skill concepts -- e.g. "classification"/"regression" imply
"supervised learning", "clustering" implies "unsupervised learning", and
"cross-validation"/"F1 score"/"ROC-AUC"/"MAE"/"RMSE" all imply "model
evaluation". Hardcoding an alias for every such relationship doesn't
scale.

FIX: SkillsMatcherAgent._llm_recheck_gaps() runs AFTER deterministic
matching (_match_deterministic), and only for requirements that are
STILL genuinely unmatched (not soft-skill "not_mentioned" entries, which
were never gaps). It sends one BATCHED LLM call (not one per skill) with
the JD requirement text + the candidate's extracted-skill evidence, and
promotes a confidently-evidenced MATCH verdict to a match_quality=
"semantic" SkillMatch -- the same quality level deterministic alias
matches already use, scored by the SAME existing CREDIT_BY_QUALITY /
REQUIRED_WEIGHT / PREFERRED_WEIGHT formula (via _aggregate). The LLM
never sets a score directly, never overrides an already-matched
requirement, and any failure (no API key, API error, malformed
response) leaves the deterministic result completely unchanged.

All tests here mock `SkillsMatcherAgent._call_llm_for_json_async`
directly (the same pattern already used elsewhere in this suite, e.g.
tests/test_batch5_workflow_fanin_race.py patching JobAnalyzerAgent) so
no API key or network access is required and results are fully
reproducible.
"""

import asyncio
import json

import pytest

from src.agents.skills_matcher import SkillsMatcherAgent
from src.models import JobRequirements, Skill


def _run(agent: SkillsMatcherAgent, skills: list[Skill], reqs: JobRequirements):
    return asyncio.run(agent.process({"extracted_skills": skills, "job_requirements": reqs}))["skills_match"]


def _mock_llm(monkeypatch, response_text: str, call_log: list | None = None):
    async def fake_call(self, prompt, max_retries=1, max_rate_limit_retries=3):
        if call_log is not None:
            call_log.append(prompt)
        return response_text
    monkeypatch.setattr(SkillsMatcherAgent, "_call_llm_for_json_async", fake_call)


# ---------------------------------------------------------------------------
# Core promotion behavior: the exact semantic relationships from the bug
# report.
# ---------------------------------------------------------------------------

def test_classification_and_regression_satisfy_supervised_learning(monkeypatch):
    _mock_llm(monkeypatch, json.dumps({
        "results": [
            {
                "requirement": "Supervised Learning",
                "match": "MATCH",
                "evidence": "Candidate lists Classification and Regression, both supervised learning techniques.",
                "confidence": 0.9,
            }
        ]
    }))
    skills = [Skill(name="Classification", category="technical"), Skill(name="Regression", category="technical")]
    reqs = JobRequirements(required_skills=["Supervised Learning"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)

    assert result.required_skills_met == 1
    assert result.required_skills_total == 1
    m = result.matches[0]
    assert m.matched is True
    assert m.match_quality == "semantic"
    assert m.confidence == 0.9
    assert "LLM semantic recheck" in m.notes


def test_cross_validation_and_metrics_satisfy_model_evaluation(monkeypatch):
    _mock_llm(monkeypatch, json.dumps({
        "results": [
            {
                "requirement": "Model Evaluation",
                "match": "MATCH",
                "evidence": "Cross-validation and ROC-AUC are model evaluation techniques.",
                "confidence": 0.82,
            }
        ]
    }))
    skills = [Skill(name="Cross-validation", category="technical"), Skill(name="ROC-AUC", category="technical")]
    reqs = JobRequirements(required_skills=["Model Evaluation"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)

    assert result.required_skills_met == 1
    assert result.matches[0].matched is True
    assert result.matches[0].match_quality == "semantic"


def test_clustering_satisfies_unsupervised_learning(monkeypatch):
    _mock_llm(monkeypatch, json.dumps({
        "results": [
            {"requirement": "Unsupervised Learning", "match": "MATCH",
             "evidence": "Candidate lists Clustering.", "confidence": 0.75}
        ]
    }))
    skills = [Skill(name="Clustering", category="technical")]
    reqs = JobRequirements(required_skills=["Unsupervised Learning"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)
    assert result.matches[0].matched is True


# ---------------------------------------------------------------------------
# Deterministic matches are never touched or reconsidered
# ---------------------------------------------------------------------------

def test_deterministic_matches_are_never_sent_to_the_llm_or_overridden(monkeypatch):
    call_log: list = []
    # Even if the LLM (hypothetically) disagreed with an already-matched
    # requirement, it is never even asked about it.
    _mock_llm(monkeypatch, json.dumps({"results": []}), call_log)

    skills = [Skill(name="Python", category="language")]
    reqs = JobRequirements(required_skills=["Python"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)

    assert result.matches[0].matched is True
    assert result.matches[0].match_quality == "exact"
    # No gaps at all -- the LLM must not be called.
    assert call_log == []


def test_llm_only_asked_about_genuinely_unmatched_requirements(monkeypatch):
    call_log: list = []
    _mock_llm(monkeypatch, json.dumps({
        "results": [
            {"requirement": "Supervised Learning", "match": "MATCH",
             "evidence": "Classification.", "confidence": 0.9},
        ]
    }), call_log)

    skills = [Skill(name="Python", category="language"), Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Python", "Supervised Learning"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)

    assert len(call_log) == 1
    prompt = call_log[0]
    assert "Supervised Learning" in prompt
    assert "Python" not in prompt.split("UNMATCHED JOB REQUIREMENTS")[1]

    python_match = next(m for m in result.matches if m.requirement == "Python")
    sl_match = next(m for m in result.matches if m.requirement == "Supervised Learning")
    assert python_match.match_quality == "exact"
    assert sl_match.match_quality == "semantic"
    assert result.required_skills_met == 2
    assert result.required_skills_total == 2


# ---------------------------------------------------------------------------
# Batching: exactly ONE call regardless of how many gaps exist
# ---------------------------------------------------------------------------

def test_single_batched_call_for_multiple_gaps_not_one_per_skill(monkeypatch):
    call_log: list = []
    _mock_llm(monkeypatch, json.dumps({
        "results": [
            {"requirement": "Supervised Learning", "match": "MATCH", "evidence": "Classification.", "confidence": 0.9},
            {"requirement": "Model Evaluation", "match": "MATCH", "evidence": "F1 score.", "confidence": 0.85},
            {"requirement": "Deep Learning", "match": "NO_MATCH", "evidence": "", "confidence": 0.0},
        ]
    }), call_log)

    skills = [Skill(name="Classification", category="technical"), Skill(name="F1 score", category="technical")]
    reqs = JobRequirements(
        required_skills=["Supervised Learning", "Model Evaluation", "Deep Learning"],
        preferred_skills=[],
    )

    result = _run(SkillsMatcherAgent(), skills, reqs)

    assert len(call_log) == 1, "expected exactly one batched LLM call, not one per skill"
    assert result.required_skills_met == 2
    assert result.required_skills_total == 3
    dl_match = next(m for m in result.matches if m.requirement == "Deep Learning")
    assert dl_match.matched is False


# ---------------------------------------------------------------------------
# Graceful fallback on any kind of failure
# ---------------------------------------------------------------------------

def test_falls_back_silently_when_llm_returns_empty(monkeypatch):
    """Empty string is exactly what _call_llm_for_json_async returns on a
    total API failure or an unconfigured provider (see BaseAgent)."""
    _mock_llm(monkeypatch, "")
    skills = [Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Supervised Learning"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)

    assert result.matches[0].matched is False
    assert result.matches[0].match_quality == "none"
    assert result.required_skills_met == 0


def test_falls_back_silently_on_malformed_json(monkeypatch):
    _mock_llm(monkeypatch, "this is not valid json at all")
    skills = [Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Supervised Learning"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)

    assert result.matches[0].matched is False


def test_falls_back_when_results_key_is_missing_or_wrong_shape(monkeypatch):
    _mock_llm(monkeypatch, json.dumps({"unexpected": "shape"}))
    skills = [Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Supervised Learning"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)
    assert result.matches[0].matched is False


def test_falls_back_when_llm_call_raises_unexpectedly(monkeypatch):
    async def raising_call(self, prompt, max_retries=1, max_rate_limit_retries=3):
        raise RuntimeError("boom")
    monkeypatch.setattr(SkillsMatcherAgent, "_call_llm_for_json_async", raising_call)

    skills = [Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Supervised Learning"], preferred_skills=[])

    # Must not raise out of process() -- the deterministic result is
    # still returned.
    result = _run(SkillsMatcherAgent(), skills, reqs)
    assert result.matches[0].matched is False
    assert result.required_skills_total == 1


def test_disabling_the_flag_skips_the_llm_entirely(monkeypatch):
    call_log: list = []
    _mock_llm(monkeypatch, json.dumps({
        "results": [{"requirement": "Supervised Learning", "match": "MATCH", "evidence": "x", "confidence": 0.9}]
    }), call_log)

    agent = SkillsMatcherAgent()
    agent.ENABLE_LLM_RECHECK = False
    skills = [Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Supervised Learning"], preferred_skills=[])

    result = _run(agent, skills, reqs)

    assert call_log == []
    assert result.matches[0].matched is False


# ---------------------------------------------------------------------------
# Verdict quality gates: low confidence / no evidence are rejected
# ---------------------------------------------------------------------------

def test_low_confidence_match_is_not_promoted(monkeypatch):
    _mock_llm(monkeypatch, json.dumps({
        "results": [{"requirement": "Supervised Learning", "match": "MATCH",
                     "evidence": "Maybe related.", "confidence": 0.3}]
    }))
    skills = [Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Supervised Learning"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)
    assert result.matches[0].matched is False


def test_match_with_no_evidence_is_not_promoted(monkeypatch):
    """A bare MATCH with nothing behind it must not be trusted, even at
    high stated confidence -- guards against the LLM asserting a verdict
    with nothing to back it up."""
    _mock_llm(monkeypatch, json.dumps({
        "results": [{"requirement": "Supervised Learning", "match": "MATCH",
                     "evidence": "", "confidence": 0.95}]
    }))
    skills = [Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Supervised Learning"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)
    assert result.matches[0].matched is False


def test_no_match_verdict_leaves_requirement_unmatched(monkeypatch):
    _mock_llm(monkeypatch, json.dumps({
        "results": [{"requirement": "Kubernetes", "match": "NO_MATCH", "evidence": "", "confidence": 0.0}]
    }))
    skills = [Skill(name="Python", category="language")]
    reqs = JobRequirements(required_skills=["Kubernetes"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)
    assert result.matches[0].matched is False


# ---------------------------------------------------------------------------
# Soft-skill "not_mentioned" entries are never sent to the LLM (they were
# never gaps to begin with -- see skill_taxonomy.requirement_is_pure_soft_skill)
# ---------------------------------------------------------------------------

def test_soft_skill_not_mentioned_requirement_is_not_sent_to_llm(monkeypatch):
    call_log: list = []
    _mock_llm(monkeypatch, json.dumps({"results": []}), call_log)

    skills = [Skill(name="Python", category="language")]
    reqs = JobRequirements(required_skills=["Good communication skills"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)

    assert call_log == [], "a soft-skill 'not_mentioned' requirement is not a real gap and must not reach the LLM"
    assert result.matches[0].soft_skill_status == "not_mentioned"
    assert result.required_skills_total == 0  # excluded from scoring, unaffected by this fix


# ---------------------------------------------------------------------------
# Scoring: the LLM verdict feeds the SAME existing formula, never its own
# score. This is checked structurally, not by asserting on a magic number.
# ---------------------------------------------------------------------------

def test_promoted_match_is_scored_by_existing_semantic_credit_not_llm_confidence(monkeypatch):
    """overall_score must come from CREDIT_BY_QUALITY['semantic'] (0.9),
    exactly like a deterministic alias match -- NOT from the LLM's own
    reported confidence (0.99 here), proving the LLM supplies a verdict,
    not a score."""
    _mock_llm(monkeypatch, json.dumps({
        "results": [{"requirement": "Supervised Learning", "match": "MATCH",
                     "evidence": "Classification.", "confidence": 0.99}]
    }))
    agent = SkillsMatcherAgent()
    skills = [Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Supervised Learning"], preferred_skills=[])

    result = _run(agent, skills, reqs)

    assert result.overall_score == agent.CREDIT_BY_QUALITY["semantic"]


def test_mixed_deterministic_and_llm_promoted_matches_score_correctly(monkeypatch):
    """One exact deterministic match (credit 1.0) + one LLM-promoted
    semantic match (credit 0.9) must average to exactly (1.0+0.9)/2,
    using the SAME formula as an all-deterministic result would."""
    _mock_llm(monkeypatch, json.dumps({
        "results": [{"requirement": "Supervised Learning", "match": "MATCH",
                     "evidence": "Classification.", "confidence": 0.8}]
    }))
    agent = SkillsMatcherAgent()
    skills = [Skill(name="Python", category="language"), Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Python", "Supervised Learning"], preferred_skills=[])

    result = _run(agent, skills, reqs)

    expected = round((agent.CREDIT_BY_QUALITY["exact"] + agent.CREDIT_BY_QUALITY["semantic"]) / 2, 4)
    assert result.overall_score == expected


# ---------------------------------------------------------------------------
# matched_skill prefers a real candidate skill name over raw LLM text
# ---------------------------------------------------------------------------

def test_matched_skill_prefers_actual_candidate_skill_name_from_evidence(monkeypatch):
    _mock_llm(monkeypatch, json.dumps({
        "results": [{"requirement": "Supervised Learning", "match": "MATCH",
                     "evidence": "The candidate's Classification project demonstrates this.",
                     "confidence": 0.8}]
    }))
    skills = [Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Supervised Learning"], preferred_skills=[])

    result = _run(SkillsMatcherAgent(), skills, reqs)
    assert result.matches[0].matched_skill == "Classification"


# ---------------------------------------------------------------------------
# Determinism given a fixed (mocked) LLM response
# ---------------------------------------------------------------------------

def test_result_is_stable_across_repeated_calls_with_same_mocked_response(monkeypatch):
    _mock_llm(monkeypatch, json.dumps({
        "results": [{"requirement": "Supervised Learning", "match": "MATCH",
                     "evidence": "Classification.", "confidence": 0.8}]
    }))
    skills = [Skill(name="Classification", category="technical")]
    reqs = JobRequirements(required_skills=["Supervised Learning"], preferred_skills=[])

    agent = SkillsMatcherAgent()
    r1 = _run(agent, skills, reqs)
    r2 = _run(agent, skills, reqs)
    assert r1.model_dump() == r2.model_dump()


# ---------------------------------------------------------------------------
# No job_requirements at all -- unaffected pre-existing early-return path
# ---------------------------------------------------------------------------

def test_no_job_requirements_still_short_circuits_without_calling_llm(monkeypatch):
    call_log: list = []
    _mock_llm(monkeypatch, json.dumps({"results": []}), call_log)

    agent = SkillsMatcherAgent()
    result = asyncio.run(agent.process({"extracted_skills": [], "job_requirements": None}))["skills_match"]

    assert call_log == []
    assert result.confidence == 0.0