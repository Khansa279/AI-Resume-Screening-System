"""Deterministic pipeline tests (no LLM required)."""

import asyncio
from pathlib import Path

from src.agents.decision_synth import DecisionSynthesizerAgent
from src.agents.experience_eval import evaluate_experience
from src.agents.resume_parser import parse_resume_text
from src.agents.skill_extractor import SkillExtractorAgent, extract_skills_from_resume
from src.agents.skills_matcher import SkillsMatcherAgent
from src.document_parser import normalize_extracted_text, resume_content_hash
from src.models import (
    ExperienceEvaluation,
    JobRequirements,
    ResumeData,
    Skill,
    SkillsMatchResult,
    WorkExperience,
)
from src.skill_taxonomy import canonical_skill_key, extract_canonical_skills


SAMPLE_RESUME = Path("sample_data/resumes/senior_python_dev.txt")
JUNIOR_RESUME = Path("sample_data/resumes/junior_dev.txt")


def test_content_hash_ignores_trivial_whitespace():
    a = "John Doe\n\nPython\n"
    b = "John Doe\n\n\nPython  \n"
    c = "John Doe\r\n\r\nPython\r\n"
    assert normalize_extracted_text(a) == normalize_extracted_text(b)
    assert resume_content_hash(a) == resume_content_hash(b) == resume_content_hash(c)
    assert resume_content_hash(a) != resume_content_hash("Jane Doe\n\nPython\n")


def test_taxonomy_aliases():
    assert canonical_skill_key("JS") == canonical_skill_key("JavaScript")
    assert canonical_skill_key("sklearn") == canonical_skill_key("Scikit-learn")
    assert canonical_skill_key("ML") == canonical_skill_key("Machine Learning")
    assert canonical_skill_key("OOP") == canonical_skill_key("Object Oriented Programming")
    names = {n for n, _c in extract_canonical_skills("Python, scikit-learn, ML, C++ STL, GitHub")}
    assert "Python" in names
    assert "Scikit-learn" in names
    assert "Machine Learning" in names
    assert "C++ STL" in names
    assert "GitHub" in names


# ---------------------------------------------------------------------------
# Batch 3: taxonomy consolidation. SkillExtractorAgent, SkillsMatcherAgent,
# and ExperienceEvaluatorAgent must all resolve the SAME alias to the SAME
# canonical skill -- otherwise a resume that lists "JS" could be extracted
# correctly but then fail to satisfy a JD that asks for "JavaScript" (or
# vice versa), purely because the two agents used different taxonomies.
# ---------------------------------------------------------------------------

def test_skill_agents_use_the_single_canonical_taxonomy_module():
    """Regression guard: both agents must import their taxonomy functions
    from src.skill_taxonomy (the one module also used by
    ExperienceEvaluatorAgent) -- not a second, separately-defined module
    with the same function names. Identity check, not just 'produces the
    same answer', so a future accidental re-fork would fail loudly here."""
    import src.skill_taxonomy as canonical_taxonomy
    from src.agents import skill_extractor as se_mod
    from src.agents import skills_matcher as sm_mod

    assert se_mod.extract_canonical_skills is canonical_taxonomy.extract_canonical_skills
    assert se_mod.skill_category is canonical_taxonomy.skill_category
    assert sm_mod.extract_canonical_skills is canonical_taxonomy.extract_canonical_skills
    assert sm_mod.canonical_skill_key is canonical_taxonomy.canonical_skill_key


def test_extractor_and_matcher_agree_on_common_aliases():
    """End-to-end: for each alias pair, a resume listing the SHORT form
    must satisfy a JD requirement written in the LONG form, and vice
    versa -- proving SkillExtractorAgent and SkillsMatcherAgent now share
    one normalization scheme instead of two that could disagree."""
    alias_pairs = [
        ("JS", "JavaScript"),
        ("ML", "Machine Learning"),
        ("sklearn", "Scikit-learn"),
        ("OOP", "Object Oriented Programming"),
    ]
    extractor = SkillExtractorAgent()
    matcher = SkillsMatcherAgent()

    async def resume_satisfies_requirement(resume_alias: str, jd_alias: str) -> bool:
        resume_data = ResumeData(skills_section=[resume_alias])
        extraction = await extractor.process({"resume_data": resume_data})
        extracted_skills = extraction["extracted_skills"]
        assert extracted_skills, f"{resume_alias!r} was not extracted as a skill at all"

        reqs = JobRequirements(required_skills=[jd_alias])
        match_state = await matcher.process({
            "extracted_skills": extracted_skills,
            "job_requirements": reqs,
        })
        match = match_state["skills_match"]
        return match.required_skills_met == 1

    for resume_alias, jd_alias in alias_pairs:
        assert asyncio.run(resume_satisfies_requirement(resume_alias, jd_alias)), (
            f"resume skill {resume_alias!r} should satisfy JD requirement {jd_alias!r}"
        )
        # And the reverse direction: JD written in the short form, resume
        # lists the long form.
        assert asyncio.run(resume_satisfies_requirement(jd_alias, resume_alias)), (
            f"resume skill {jd_alias!r} should satisfy JD requirement {resume_alias!r}"
        )


def test_migrated_domain_specific_aliases_still_recognized():
    """Batch 3 merged src/skills_taxonomy.py's project-specific aliases
    (seen in this repo's own sample_data) into src/skill_taxonomy.py
    before deleting the duplicate -- confirm they actually survived the
    merge and are still recognized by the extractor."""
    resume_data = ResumeData(
        skills_section=["Heroku", "RabbitMQ", "Jira"],
        certifications=["dbt Fundamentals"],
        work_experience=[
            WorkExperience(
                title="Data Engineer", company="DataFlow Analytics",
                responsibilities=[
                    "Used Airflow and Snowflake for orchestration and warehousing",
                    "Mentored two junior engineers on the team",
                ],
            )
        ],
    )
    skills = extract_skills_from_resume(resume_data)
    names = {s.name for s in skills}
    for expected in ("Heroku", "RabbitMQ", "Jira", "dbt", "Airflow", "Snowflake", "Leadership"):
        assert expected in names, f"expected {expected!r} in extracted skills, got {sorted(names)}"


def test_skill_extraction_is_stable():
    text = SAMPLE_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)
    first = extract_skills_from_resume(data)
    second = extract_skills_from_resume(data)
    assert [s.name for s in first] == [s.name for s in second]
    assert first
    names = {s.name for s in first}
    assert "Python" in names
    assert "Django" in names


def test_skill_extractor_agent_twice_identical():
    text = SAMPLE_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)
    agent = SkillExtractorAgent()

    async def twice():
        a = await agent.process({"resume_data": data})
        b = await agent.process({"resume_data": data})
        return a["extracted_skills"], b["extracted_skills"]

    left, right = asyncio.run(twice())
    assert [s.name for s in left] == [s.name for s in right]


def test_resume_parser_deterministic_on_sample():
    text = SAMPLE_RESUME.read_text(encoding="utf-8")
    a = parse_resume_text(text)
    b = parse_resume_text(text)
    assert a.contact.email == b.contact.email == "john.doe@email.com"
    assert a.contact.name
    assert a.education
    assert a.work_experience
    assert a.skills_section
    assert a.model_dump() == b.model_dump()

    junior = parse_resume_text(JUNIOR_RESUME.read_text(encoding="utf-8"))
    assert junior.contact.email == "sarah.chen@email.com"
    assert junior.work_experience or junior.education


def test_skills_matcher_js_javascript():
    agent = SkillsMatcherAgent()
    skills = [Skill(name="JavaScript", category="language")]
    reqs = JobRequirements(required_skills=["JS"], preferred_skills=[])

    async def run():
        return await agent.process({"extracted_skills": skills, "job_requirements": reqs})

    result = asyncio.run(run())["skills_match"]
    assert result.required_skills_met == 1
    assert result.matches[0].matched


def test_experience_empty_history_rule():
    data = ResumeData(
        education=[{"degree": "BS", "field": "Computer Science", "institution": "X"}]
    )
    reqs = JobRequirements(title="Associate AI Engineer", min_years_experience=0)
    ev = evaluate_experience(data, reqs)
    assert ev.experience_score == 0.0
    assert ev.role_relevance == 0.15
    assert ev.years_relevant == 0.0


def test_experience_and_matcher_stable():
    text = SAMPLE_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)
    reqs = JobRequirements(
        title="Backend Software Engineer",
        required_skills=["Python", "Django", "SQL"],
        preferred_skills=["Docker"],
        min_years_experience=3,
    )
    e1 = evaluate_experience(data, reqs)
    e2 = evaluate_experience(data, reqs)
    assert e1.model_dump() == e2.model_dump()
    assert e1.years_relevant > 0


# ---------------------------------------------------------------------------
# Batch 2: deterministic experience evaluation -- date-handling edge cases.
# evaluate_experience() must never call an LLM and must give byte-identical
# ExperienceEvaluation output for identical input, no matter how messy the
# resume's dates/durations are.
# ---------------------------------------------------------------------------

def test_experience_no_dates_or_duration_is_deterministic():
    """A role with no start/end date and no duration text at all must not
    crash, and must fall back to the same small default every time."""
    exp = WorkExperience(title="Software Engineer", company="Acme Corp")
    data = ResumeData(work_experience=[exp])
    reqs = JobRequirements(title="Software Engineer", min_years_experience=2)

    e1 = evaluate_experience(data, reqs)
    e2 = evaluate_experience(data, reqs)
    assert e1.model_dump() == e2.model_dump()
    assert e1.years_relevant >= 0.0


def test_experience_unparseable_dates_degrade_gracefully():
    """Garbage date strings (not a real date, no year, no month) must not
    raise, and must resolve to the same deterministic fallback every time."""
    exp = WorkExperience(
        title="Data Analyst", company="Beta LLC",
        start_date="whenever", end_date="who knows", duration="",
    )
    data = ResumeData(work_experience=[exp])
    reqs = JobRequirements(title="Data Analyst", min_years_experience=1)

    e1 = evaluate_experience(data, reqs)
    e2 = evaluate_experience(data, reqs)
    assert e1.model_dump() == e2.model_dump()


def test_experience_duration_text_only_is_deterministic():
    """No start/end dates, only a free-text duration -- must be parsed from
    the duration string, and identically so on repeated evaluation."""
    exp = WorkExperience(
        title="Backend Engineer", company="Gamma Inc",
        duration="2 years 3 months",
    )
    data = ResumeData(work_experience=[exp])
    reqs = JobRequirements(title="Backend Engineer", min_years_experience=2)

    e1 = evaluate_experience(data, reqs)
    e2 = evaluate_experience(data, reqs)
    assert e1.model_dump() == e2.model_dump()
    assert e1.years_relevant > 0


def test_experience_partial_date_range_is_deterministic():
    """Only a start date (role presumed ongoing, end date missing/blank) --
    must still produce a stable result across repeated calls in the same run."""
    exp = WorkExperience(title="Product Manager", company="Delta Co", start_date="2021")
    data = ResumeData(work_experience=[exp])
    reqs = JobRequirements(title="Product Manager", min_years_experience=1)

    e1 = evaluate_experience(data, reqs)
    e2 = evaluate_experience(data, reqs)
    assert e1.model_dump() == e2.model_dump()
    assert e1.years_relevant > 0


def test_experience_multiple_roles_mixed_relevance_is_deterministic():
    """Several roles of very different relevance -- the whole evaluation
    (years_relevant, role_relevance, experience_score, gaps, strengths)
    must be byte-identical across repeated calls on the same input."""
    exps = [
        WorkExperience(
            title="Senior Backend Engineer", company="TechCorp",
            start_date="2022-01", end_date="Present",
            responsibilities=["Built REST APIs with Django and PostgreSQL"],
            technologies=["Python", "Django", "PostgreSQL"],
        ),
        WorkExperience(
            title="Barista", company="Coffee Shop",
            start_date="2018-01", end_date="2019-01",
            responsibilities=["Made coffee", "Handled the cash register"],
            technologies=[],
        ),
    ]
    data = ResumeData(work_experience=exps)
    reqs = JobRequirements(
        title="Backend Software Engineer",
        required_skills=["Python", "Django", "PostgreSQL"],
        min_years_experience=2,
    )

    e1 = evaluate_experience(data, reqs)
    e2 = evaluate_experience(data, reqs)
    assert e1.model_dump() == e2.model_dump()
    # The relevant senior role should clearly outweigh the unrelated one.
    assert e1.role_relevance > 0.3


def test_experience_evaluator_agent_makes_no_llm_call():
    """End-to-end guard: ExperienceEvaluatorAgent.process() must never touch
    self.llm. We assert this by simply never providing an API key/LLM and
    confirming the call still succeeds (BaseAgent.llm is lazy -- if the
    agent tried to use it, this would raise trying to build a real client)."""
    from src.agents.experience_eval import ExperienceEvaluatorAgent

    text = SAMPLE_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)
    reqs = JobRequirements(title="Backend Software Engineer", min_years_experience=2)
    agent = ExperienceEvaluatorAgent()
    assert agent._llm is None  # never lazily created

    async def run():
        return await agent.process({"resume_data": data, "job_requirements": reqs})

    result = asyncio.run(run())
    assert isinstance(result["experience_eval"], ExperienceEvaluation)
    assert agent._llm is None  # still never created -- no LLM call was made


def test_final_score_formula_unchanged():
    skills = SkillsMatchResult(overall_score=0.8, confidence=0.9)
    exp = ExperienceEvaluation(experience_score=0.5, role_relevance=0.4, confidence=0.9)
    adjusted = exp.experience_score * 0.7 + exp.role_relevance * 0.3
    expected = round(min(max(skills.overall_score * 0.6 + adjusted * 0.4, 0.0), 1.0), 2)
    got = DecisionSynthesizerAgent()._calculate_match_score(skills, exp)
    assert got == expected
    assert got == 0.67


def test_decision_reasoning_has_no_llm_and_is_stable():
    agent = DecisionSynthesizerAgent()
    text = SAMPLE_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)
    skills = SkillsMatchResult(
        overall_score=0.7,
        required_skills_met=3,
        required_skills_total=4,
        preferred_skills_met=1,
        preferred_skills_total=2,
        matches=[],
        confidence=0.9,
    )
    exp = ExperienceEvaluation(
        years_relevant=4.0,
        years_required=3,
        experience_score=1.0,
        role_relevance=0.8,
        confidence=0.9,
        reasoning="ok",
    )
    reqs = JobRequirements(title="Backend Engineer", required_skills=["Python"])
    state = {
        "resume_data": data,
        "job_requirements": reqs,
        "skills_match": skills,
        "experience_eval": exp,
        "agent_confidences": {"SkillsMatcherAgent": 0.9, "ExperienceEvaluatorAgent": 0.9},
        "errors": [],
    }

    async def twice():
        a = await agent.process(state)
        b = await agent.process(state)
        return a["final_output"], b["final_output"]

    out_a, out_b = asyncio.run(twice())
    assert out_a.match_score == out_b.match_score
    assert out_a.reasoning_summary == out_b.reasoning_summary
    assert out_a.reasoning_summary
