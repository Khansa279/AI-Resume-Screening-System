"""Skill Extractor Agent - Identifies and categorizes candidate skills.

Deterministic: scans resume text against the shared canonical taxonomy.
The same resume text always yields the same skill list (names, categories,
order). No LLM call on the normal path.
"""

from typing import Any

from .base import BaseAgent
from ..models import Skill, ResumeData
from ..skill_taxonomy import extract_canonical_skills


_VALID_CATEGORIES = {"technical", "soft_skill", "tool", "language", "framework", "other"}


def extract_skills_from_resume(resume_data: ResumeData) -> list[Skill]:
    """Pure function used by the agent and by tests."""
    parts: list[str] = []
    if resume_data.skills_section:
        parts.append(" ".join(resume_data.skills_section))
    for exp in resume_data.work_experience:
        if exp.technologies:
            parts.append(" ".join(exp.technologies))
        if exp.responsibilities:
            parts.append(" ".join(exp.responsibilities))
        if exp.title:
            parts.append(exp.title)
    if resume_data.projects:
        parts.append(" ".join(resume_data.projects))
    if resume_data.certifications:
        parts.append(" ".join(resume_data.certifications))
    if resume_data.raw_text:
        parts.append(resume_data.raw_text)

    haystack = "\n".join(parts)
    found = extract_canonical_skills(haystack)

    skills: list[Skill] = []
    for name, category in found:
        if category not in _VALID_CATEGORIES:
            category = "other"
        skills.append(
            Skill(
                name=name,
                category=category,  # type: ignore[arg-type]
                proficiency="intermediate",
                source="explicit",
                confidence=0.95,
            )
        )
    return skills


class SkillExtractorAgent(BaseAgent):
    """Extract and categorize skills without an LLM call."""

    name = "SkillExtractorAgent"
    description = "Extract and categorize technical and soft skills from resume data (deterministic)"
    temperature = 0.0

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        resume_data = state.get("resume_data")

        if not resume_data:
            return {
                "extracted_skills": [],
                "errors": ["SkillExtractorAgent: No resume data available"],
                "agent_confidences": {self.name: 0.0}
            }

        if isinstance(resume_data, dict):
            resume_data = ResumeData.model_validate(resume_data)

        skills = extract_skills_from_resume(resume_data)
        confidence = 0.95 if skills else 0.4

        return {
            "extracted_skills": skills,
            "agent_confidences": {self.name: confidence}
        }
