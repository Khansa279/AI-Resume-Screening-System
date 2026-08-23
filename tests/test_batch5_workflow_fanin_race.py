"""Batch 5 regression tests.

REAL BUG: a candidate's resume clearly listed skills the JD required
(Python, Machine Learning, Deep Learning, Artificial Intelligence,
Scikit-learn, TensorFlow, PyTorch -- all recognized by SkillExtractorAgent
and confirmed present in extracted_skills), yet the final ScreeningOutput
came back at match_score=0.02 with SkillsMatchResult reporting "0 of 7
required skills met".

Root cause: match_skills has two incoming edges of different depths from
parse_document -- extract_skills (parse_resume -> extract_skills, depth 3)
and analyze_job (depth 2) -- so LangGraph invokes the node once per edge
that resolves rather than once after both are ready. Whichever edge
resolves first triggers an invocation where the OTHER input is still
missing. Before this fix, extracted_skills defaulted to [] (not a
distinguishable "not computed yet" sentinel), so the premature invocation
-- job_requirements present, extracted_skills still genuinely empty --
computed a fully-formed but WRONG SkillsMatchResult ("0 of N required")
and wrote it into shared state.

That alone would have been overwritten by the later, correctly-timed
invocation (LangGraph's last-write-wins), EXCEPT DecisionSynthesizerAgent
gained its own idempotency guard (to stop its LLM call firing twice, for
the identical depth-mismatch reason one hop downstream). That guard fires
on "already ran once", not "ran with genuinely complete inputs" -- so it
locked in whichever skills_match happened to be present the FIRST time
synthesize_decision was invoked, which was the premature, wrong one.

Fix (src/workflow.py):
  - extracted_skills now defaults to None (not []) until SkillExtractorAgent
    has genuinely run once, so "not computed yet" is distinguishable from
    "computed, found nothing".
  - _match_skills_node skips (no-op) unless BOTH extracted_skills is not
    None AND job_requirements is truthy.
  - _synthesize_decision_node skips (no-op, workflow_complete NOT set)
    unless BOTH skills_match and experience_eval are not None, in addition
    to its existing "already completed" guard.

These tests reproduce the exact race by making JobAnalyzerAgent's (LLM)
call artificially slow relative to the fully deterministic extract_skills
path, in both possible orderings, and assert the final result is always
the correct, fully-resolved one -- never a partial/premature one.
"""

import asyncio
import json
from unittest.mock import patch

import pytest

from src.agents.job_analyzer import JobAnalyzerAgent
from src.agents.skills_matcher import SkillsMatcherAgent
from src.agents.decision_synth import DecisionSynthesizerAgent
from src.models import JobRequirements, ResumeData, Skill
from src.workflow import create_screening_workflow


RESUME_TEXT = """John Smith
john@example.com

SKILLS
Python, Machine Learning, Deep Learning, Artificial Intelligence, SQL, Git, Scikit-learn, TensorFlow, PyTorch, Data Structures, Algorithms

EDUCATION
Bachelor of Science in Computer Science, XYZ University, 2024
"""

FAKE_JD_JSON = json.dumps({
    "title": "Associate AI Engineer",
    "summary": "AI engineering role",
    "required_skills": [
        "Python", "Machine Learning", "Deep Learning",
        "Artificial Intelligence", "Scikit-learn", "TensorFlow", "PyTorch",
    ],
    "preferred_skills": ["Data Structures", "Algorithms", "SQL", "Git"],
    "min_years_experience": 0,
    "education_requirements": [],
    "certifications_required": [],
    "responsibilities": [],
    "requirements": [],
    "parsing_confidence": 0.9,
})


def _make_slow_llm(delay: float):
    async def slow_llm(self, prompt):
        await asyncio.sleep(delay)
        return FAKE_JD_JSON
    return slow_llm


async def _run_with_job_analyzer_delay(delay: float):
    slow_llm = _make_slow_llm(delay)
    with patch.object(JobAnalyzerAgent, "_call_llm_for_json_async", slow_llm, create=True), \
         patch.object(JobAnalyzerAgent, "_call_llm_async", slow_llm):
        workflow = create_screening_workflow()
        return await workflow.run(resume_text=RESUME_TEXT, job_description="irrelevant, mocked")


@pytest.mark.parametrize("delay", [0.0, 0.05, 0.3])
def test_workflow_never_reports_premature_zero_match_regardless_of_llm_timing(delay):
    """Whether JobAnalyzerAgent's (mocked) LLM call is instant, slightly
    slower, or much slower than the deterministic extract_skills path,
    the final result must reflect the REAL match (7/7 required here) --
    never the premature "0 of 7" that a fan-in race can produce."""
    output = asyncio.run(_run_with_job_analyzer_delay(delay))
    assert output.match_score > 0.5, (
        f"delay={delay}: got match_score={output.match_score} -- looks like "
        f"the premature/incomplete skills_match got locked in"
    )


def test_match_skills_node_skips_when_extraction_not_yet_run():
    """Direct unit test of the guard: extracted_skills=None (not yet run)
    must produce a no-op, not a computed-but-wrong SkillsMatchResult."""
    workflow = create_screening_workflow()
    state = {
        "extracted_skills": None,
        "job_requirements": JobRequirements(required_skills=["Python"]).model_dump(),
    }
    result = asyncio.run(workflow._match_skills_node(state))
    assert result == {}, "should no-op instead of matching against an unextracted skill list"


def test_match_skills_node_skips_when_job_requirements_not_yet_ready():
    """Symmetric case: extracted_skills is genuinely ready ([] or a real
    list) but job_requirements hasn't arrived yet -- also must no-op."""
    workflow = create_screening_workflow()
    state = {
        "extracted_skills": [Skill(name="Python", category="language").model_dump()],
        "job_requirements": None,
    }
    result = asyncio.run(workflow._match_skills_node(state))
    assert result == {}


def test_match_skills_node_runs_for_real_once_both_inputs_present():
    workflow = create_screening_workflow()
    state = {
        "extracted_skills": [Skill(name="Python", category="language").model_dump()],
        "job_requirements": JobRequirements(required_skills=["Python"]).model_dump(),
    }
    result = asyncio.run(workflow._match_skills_node(state))
    assert result.get("skills_match") is not None
    assert result["skills_match"].required_skills_met == 1


def test_match_skills_node_treats_genuinely_empty_extraction_as_ready():
    """A resume that really has zero recognizable skills ([] after
    extraction genuinely ran) must still be scored (as 0 matches), not
    skipped forever."""
    workflow = create_screening_workflow()
    state = {
        "extracted_skills": [],
        "job_requirements": JobRequirements(required_skills=["Python"]).model_dump(),
    }
    result = asyncio.run(workflow._match_skills_node(state))
    assert result.get("skills_match") is not None
    assert result["skills_match"].required_skills_total == 1
    assert result["skills_match"].required_skills_met == 0


def test_synthesize_decision_node_skips_until_both_inputs_ready():
    workflow = create_screening_workflow()

    # Neither ready yet.
    state = {"skills_match": None, "experience_eval": None, "workflow_complete": False}
    assert asyncio.run(workflow._synthesize_decision_node(state)) == {}

    # Only one ready.
    state = {
        "skills_match": {"overall_score": 0.9, "confidence": 0.9},
        "experience_eval": None,
        "workflow_complete": False,
    }
    assert asyncio.run(workflow._synthesize_decision_node(state)) == {}


def test_decision_synthesizer_llm_call_fires_exactly_once_under_race():
    """The other half of the invariant: fixing the premature-lock-in bug
    must not reintroduce the double-LLM-call bug the guard originally
    existed to prevent."""
    call_count = {"n": 0}
    orig_process = DecisionSynthesizerAgent.process

    async def counting_process(self, state):
        call_count["n"] += 1
        return await orig_process(self, state)

    slow_llm = _make_slow_llm(0.2)
    with patch.object(DecisionSynthesizerAgent, "process", counting_process), \
         patch.object(JobAnalyzerAgent, "_call_llm_for_json_async", slow_llm, create=True), \
         patch.object(JobAnalyzerAgent, "_call_llm_async", slow_llm):
        workflow = create_screening_workflow()
        asyncio.run(workflow.run(resume_text=RESUME_TEXT, job_description="irrelevant, mocked"))

    assert call_count["n"] == 1, f"DecisionSynthesizerAgent.process called {call_count['n']} times, expected 1"
