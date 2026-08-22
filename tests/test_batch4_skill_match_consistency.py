"""Batch 4 regression tests.

REAL BUG (as originally reported against an earlier version of this
pipeline, before the Batch 3 deterministic-taxonomy fix):

    A candidate's "matching skills" list (built from
    SkillsMatchResult.matches) named specific skills as matched, while
    the SAME SkillsMatchResult's required_skills_met/required_skills_total
    counters said "0 of 7" and the free-text reasoning listed some of
    those SAME matched skills as "missing required skills".

Root cause: the two pieces of information -- (a) the per-requirement
`matches` list and (b) the aggregate required/preferred met/total counts
-- were sourced independently (an LLM was asked to report both in one
JSON response), so nothing forced them to agree.

Batch 3 already fixed this for SkillsMatcherAgent itself: matches and
counts are now computed together, in the same deterministic loop
(src/agents/skills_matcher.py::_match_deterministic), so they can no
longer diverge for that agent in isolation.

This file:

  1. Locks that invariant in with explicit exact / alias / grouped-phrase
     cases (the scenarios named in the original bug report), so a future
     change to the matching logic can't reintroduce the divergence.
  2. Fixes and regression-tests a second, narrower instance of the same
     class of bug found downstream in DecisionSynthesizerAgent's
     free-text reasoning: `_build_reasoning_summary` had a fallback that,
     when it couldn't identify which unmatched requirements were
     specifically REQUIRED (vs preferred), silently treated ALL unmatched
     requirements as required-missing -- capable of mislabeling a missing
     PREFERRED skill as "required" in the reasoning text. Fixed in
     src/agents/decision_synth.py.
  3. Verifies the exact invariant asked for in the bug report: a skill
     that satisfies a requirement must simultaneously and consistently
     affect required_skills_met/total, matches, matching_skills, and
     skill_gaps (mirroring src/services/screening_service.py's
     _build_explanation, which is exercised directly here rather than
     going through the DB-backed screen_candidate()).
"""

import asyncio

from src.agents.decision_synth import DecisionSynthesizerAgent
from src.agents.skills_matcher import SkillsMatcherAgent
from src.models import (
    ExperienceEvaluation,
    JobRequirements,
    ResumeData,
    Skill,
    SkillMatch,
    SkillsMatchResult,
)


def _run_matcher(skills: list[Skill], requirements: JobRequirements) -> SkillsMatchResult:
    agent = SkillsMatcherAgent()

    async def run():
        return await agent.process({
            "extracted_skills": skills,
            "job_requirements": requirements,
        })

    return asyncio.run(run())["skills_match"]


def _build_explanation_like_service(skills_match: SkillsMatchResult) -> tuple[list[str], list[str]]:
    """Mirrors src/services/screening_service.py::_build_explanation's
    matching_skills/skill_gaps derivation exactly, without needing a DB
    session -- so this test module can check the SAME invariant the real
    persisted Explanation row is built from."""
    matching_skills = [
        m.matched_skill for m in skills_match.matches
        if m.matched and m.matched_skill
    ]
    skill_gaps = [m.requirement for m in skills_match.matches if not m.matched]
    return matching_skills, skill_gaps


def _assert_fully_consistent(skills_match: SkillsMatchResult, requirements: JobRequirements) -> None:
    """The core invariant from the bug report: a requirement that is
    matched=True in `matches` must be reflected consistently everywhere
    else derived from this SkillsMatchResult, and a requirement that is
    matched=False must never appear as "met"."""
    matching_skills, skill_gaps = _build_explanation_like_service(skills_match)

    required_set = {s.casefold() for s in requirements.required_skills}
    preferred_set = {s.casefold() for s in requirements.preferred_skills}

    matched_required = 0
    matched_preferred = 0
    for m in skills_match.matches:
        req_key = m.requirement.casefold()
        if m.matched:
            # A matched requirement must not simultaneously show up as a gap.
            assert m.requirement not in skill_gaps, (
                f"{m.requirement!r} is matched=True but also listed in skill_gaps"
            )
            # Its matched skill must be represented in matching_skills.
            assert m.matched_skill in matching_skills, (
                f"{m.requirement!r} matched {m.matched_skill!r} but it's missing "
                f"from matching_skills={matching_skills}"
            )
            if req_key in required_set:
                matched_required += 1
            if req_key in preferred_set:
                matched_preferred += 1
        else:
            assert m.requirement in skill_gaps, (
                f"{m.requirement!r} is matched=False but missing from skill_gaps"
            )
            assert m.matched_skill not in matching_skills or not m.matched_skill, (
                f"{m.requirement!r} is unmatched but {m.matched_skill!r} still "
                f"appears in matching_skills"
            )

    # The aggregate counters must agree with what we just counted directly
    # off the `matches` list -- this is the exact "0 of 7 required" vs
    # "8 matching skills" contradiction from the bug report.
    assert skills_match.required_skills_met == matched_required, (
        f"required_skills_met={skills_match.required_skills_met} but counting "
        f"matched=True entries against required_skills gives {matched_required}"
    )
    assert skills_match.preferred_skills_met == matched_preferred, (
        f"preferred_skills_met={skills_match.preferred_skills_met} but counting "
        f"matched=True entries against preferred_skills gives {matched_preferred}"
    )
    assert skills_match.required_skills_total == len(requirements.required_skills)
    assert skills_match.preferred_skills_total == len(requirements.preferred_skills)


# ---------------------------------------------------------------------------
# 1. Exact match: Python -> Python
# ---------------------------------------------------------------------------

def test_exact_match_is_fully_consistent():
    skills = [Skill(name="Python", category="language")]
    reqs = JobRequirements(required_skills=["Python"], preferred_skills=[])

    result = _run_matcher(skills, reqs)

    assert result.required_skills_met == 1
    assert result.required_skills_total == 1
    assert result.matches[0].matched is True
    assert result.matches[0].match_quality == "exact"
    _assert_fully_consistent(result, reqs)


# ---------------------------------------------------------------------------
# 2. Alias: JS -> JavaScript
# ---------------------------------------------------------------------------

def test_alias_js_to_javascript_is_fully_consistent():
    skills = [Skill(name="JS", category="language")]
    reqs = JobRequirements(required_skills=["JavaScript"], preferred_skills=[])

    result = _run_matcher(skills, reqs)

    assert result.required_skills_met == 1
    assert result.matches[0].matched is True
    _assert_fully_consistent(result, reqs)


# ---------------------------------------------------------------------------
# 3. Equivalent / grouped requirement phrasing -- each is a real requirement
#    string style seen in this repo's own sample_data job descriptions.
# ---------------------------------------------------------------------------

def test_grouped_and_verbose_requirement_phrasing_is_fully_consistent():
    skills = [
        Skill(name="Django", category="framework"),
        Skill(name="REST APIs", category="technical"),
        Skill(name="PostgreSQL", category="tool"),
        Skill(name="Docker", category="tool"),
    ]
    reqs = JobRequirements(
        required_skills=[
            "Django or Flask",
            "REST API design and implementation",
            "Relational databases (PostgreSQL)",
        ],
        preferred_skills=[
            "Docker / containerization",
        ],
    )

    result = _run_matcher(skills, reqs)

    assert result.required_skills_met == 3
    assert result.required_skills_total == 3
    assert result.preferred_skills_met == 1
    assert result.preferred_skills_total == 1
    assert all(m.matched for m in result.matches), (
        f"expected every requirement matched, got: {result.matches}"
    )
    _assert_fully_consistent(result, reqs)


# ---------------------------------------------------------------------------
# 4. The exact contradiction from the bug report: a matched required skill
#    must never simultaneously be reported as missing.
# ---------------------------------------------------------------------------

def test_matched_required_skill_never_appears_as_missing():
    skills = [
        Skill(name="Python", category="language"),
        Skill(name="Machine Learning", category="technical"),
        Skill(name="Artificial Intelligence", category="technical"),
        Skill(name="Data Structures", category="technical"),
        Skill(name="Algorithms", category="technical"),
        Skill(name="SQL", category="language"),
        Skill(name="Git", category="tool"),
    ]
    reqs = JobRequirements(
        required_skills=[
            "Python", "Machine Learning", "Deep Learning", "Artificial Intelligence",
            "Data Structures", "Algorithms", "SQL",
        ],
        preferred_skills=["Git"],
    )

    result = _run_matcher(skills, reqs)

    matching_skills, skill_gaps = _build_explanation_like_service(result)

    # 6 of the 7 required skills are present (Deep Learning is not) --
    # required_skills_met must say so, and it must NOT say 0.
    assert result.required_skills_met == 6
    assert result.required_skills_total == 7
    assert result.preferred_skills_met == 1
    assert result.preferred_skills_total == 1

    for satisfied in ("Python", "Machine Learning", "Artificial Intelligence",
                      "Data Structures", "Algorithms", "SQL"):
        assert satisfied in matching_skills, f"{satisfied!r} should be in matching_skills"
        assert satisfied not in skill_gaps, f"{satisfied!r} should NOT be in skill_gaps"

    assert "Deep Learning" in skill_gaps
    _assert_fully_consistent(result, reqs)


# ---------------------------------------------------------------------------
# 5. Second John Doe example from the bug report.
# ---------------------------------------------------------------------------

def test_john_doe_style_backend_match_is_fully_consistent():
    skills = [
        Skill(name="Python", category="language"),
        Skill(name="Django", category="framework"),
        Skill(name="REST APIs", category="technical"),
        Skill(name="PostgreSQL", category="tool"),
        Skill(name="Docker", category="tool"),
        Skill(name="Microservices", category="technical"),
        Skill(name="Kafka", category="tool"),
        Skill(name="AWS", category="tool"),
    ]
    reqs = JobRequirements(
        required_skills=[
            "Python", "Django or Flask", "REST API design and implementation",
            "PostgreSQL", "Docker", "Software testing practices",
        ],
        preferred_skills=["Microservices architecture", "Kafka", "AWS"],
    )

    result = _run_matcher(skills, reqs)

    matching_skills, skill_gaps = _build_explanation_like_service(result)

    assert result.required_skills_met == 5  # everything but "Software testing practices"
    assert result.required_skills_total == 6
    for satisfied in ("Python", "Django", "REST APIs", "PostgreSQL", "Docker"):
        assert satisfied in matching_skills
        assert satisfied not in skill_gaps
    _assert_fully_consistent(result, reqs)


# ---------------------------------------------------------------------------
# 6. DecisionSynthesizerAgent's free-text reasoning must agree with the
#    SkillsMatchResult it was built from -- specifically, it must never
#    name a missing PREFERRED skill in the "Missing required skills"
#    sentence just because it couldn't cleanly separate the two lists.
#    This is the exact bug fixed in src/agents/decision_synth.py.
# ---------------------------------------------------------------------------

def test_decision_reasoning_never_mislabels_preferred_as_required():
    """Direct reproduction of the exact bug fixed in
    src/agents/decision_synth.py::_build_reasoning_summary.

    Scenario: the SkillsMatchResult handed to DecisionSynthesizerAgent
    reports required_skills_met < required_skills_total (a real "some
    required skill is missing" situation), but the job_requirements
    object available at this point doesn't carry a required_skills list
    to cross-check against (e.g. state drift, or a caller that only
    passes job_requirements.preferred_skills). The only unmatched
    requirement is actually a PREFERRED skill ("Docker").

    OLD behavior: `_build_reasoning_summary` fell back to treating EVERY
    unmatched requirement as "required-missing" whenever it couldn't
    positively confirm otherwise, producing the sentence "Missing
    required skills: Docker." -- mislabeling a preferred gap as required.
    This is verified failing against the pre-fix logic in this same PR
    (see the bug report's root-cause writeup); this test pins the fixed
    behavior so it can't regress.

    NEW behavior: with no required_skills list to check against, the
    agent does not guess -- it omits the specific "Missing required
    skills: ..." sentence rather than naming the wrong skill.
    """
    skills_match = SkillsMatchResult(
        matches=[
            SkillMatch(requirement="Docker", matched=False, matched_skill="", match_quality="none"),
        ],
        required_skills_met=0,
        required_skills_total=2,
        preferred_skills_met=0,
        preferred_skills_total=1,
        overall_score=0.1,
        confidence=0.8,
    )
    reqs = JobRequirements(
        title="Backend Engineer",
        required_skills=[],          # <- not available to cross-check against
        preferred_skills=["Docker"],  # Docker is actually PREFERRED, not required
    )
    exp = ExperienceEvaluation(
        years_relevant=1.0, years_required=2, experience_score=0.3,
        role_relevance=0.3, confidence=0.8,
    )

    agent = DecisionSynthesizerAgent()
    state = {
        "resume_data": ResumeData(),
        "job_requirements": reqs,
        "skills_match": skills_match,
        "experience_eval": exp,
        "agent_confidences": {"SkillsMatcherAgent": 0.8, "ExperienceEvaluatorAgent": 0.8},
        "errors": [],
    }
    result = asyncio.run(agent.process(state))
    reasoning = result["final_output"].reasoning_summary

    assert "meet 0 of 2 required skills" in reasoning
    assert "Missing required skills: Docker" not in reasoning, (
        f"Docker is a PREFERRED skill and must never be named in a "
        f"'Missing required skills' sentence: {reasoning!r}"
    )


def test_decision_reasoning_still_correct_when_required_fully_met():
    """Companion happy-path case with the normal (matches-derived)
    SkillsMatcherAgent output: required skills fully met, only a
    preferred skill missing -- must not mention any missing required
    skills at all."""
    skills = [
        Skill(name="Python", category="language"),
        Skill(name="SQL", category="language"),
    ]
    reqs = JobRequirements(
        title="Backend Engineer",
        required_skills=["Python", "SQL"],       # both satisfied
        preferred_skills=["Docker"],              # NOT satisfied
    )
    skills_match = _run_matcher(skills, reqs)

    assert skills_match.required_skills_met == skills_match.required_skills_total == 2
    assert skills_match.preferred_skills_met == 0
    assert skills_match.preferred_skills_total == 1

    exp = ExperienceEvaluation(
        years_relevant=3.0, years_required=2, experience_score=0.9,
        role_relevance=0.8, confidence=0.9,
    )

    agent = DecisionSynthesizerAgent()
    state = {
        "resume_data": ResumeData(),
        "job_requirements": reqs,
        "skills_match": skills_match,
        "experience_eval": exp,
        "agent_confidences": {"SkillsMatcherAgent": 0.9, "ExperienceEvaluatorAgent": 0.9},
        "errors": [],
    }
    result = asyncio.run(agent.process(state))
    reasoning = result["final_output"].reasoning_summary

    assert "meet 2 of 2 required skills" in reasoning
    assert "Missing required skills" not in reasoning, (
        f"required skills are fully met, but reasoning still names some as "
        f"missing: {reasoning!r}"
    )


def test_decision_reasoning_correctly_names_required_gaps_when_present():
    """Companion case: when a required skill genuinely IS missing, the
    reasoning must name it (regression guard against over-correcting the
    fix above into never naming anything)."""
    skills = [Skill(name="Python", category="language")]
    reqs = JobRequirements(
        title="Backend Engineer",
        required_skills=["Python", "PostgreSQL"],
        preferred_skills=[],
    )
    skills_match = _run_matcher(skills, reqs)
    assert skills_match.required_skills_met == 1
    assert skills_match.required_skills_total == 2

    exp = ExperienceEvaluation(
        years_relevant=2.0, years_required=2, experience_score=0.6,
        role_relevance=0.6, confidence=0.9,
    )
    agent = DecisionSynthesizerAgent()
    state = {
        "resume_data": ResumeData(),
        "job_requirements": reqs,
        "skills_match": skills_match,
        "experience_eval": exp,
        "agent_confidences": {"SkillsMatcherAgent": 0.9, "ExperienceEvaluatorAgent": 0.9},
        "errors": [],
    }
    result = asyncio.run(agent.process(state))
    reasoning = result["final_output"].reasoning_summary

    assert "meet 1 of 2 required skills" in reasoning
    assert "Missing required skills: PostgreSQL" in reasoning
