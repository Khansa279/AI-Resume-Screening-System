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
)
from src.skills_taxonomy import canonical_skill_key, extract_canonical_skills


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
