"""Skills Matcher Agent - Compares candidate skills against job requirements.

This agent is intentionally NOT an LLM call. Matching a requirement string
against an already-extracted list of skill names is a deterministic lookup/
classification problem, not a judgment call -- routing it through an LLM was
the second (and largest) source of run-to-run score variance, stacked on top
of SkillExtractorAgent's own variance. Given the same extracted_skills and
the same job_requirements, this agent now ALWAYS produces the same
SkillsMatchResult.
"""

import difflib
from typing import Any

from .base import BaseAgent
from ..models import Skill, JobRequirements, SkillMatch, SkillsMatchResult
from ..skill_taxonomy import canonical_skill_key


_QUALITY_CREDIT = {"exact": 1.0, "semantic": 0.9, "partial": 0.5, "none": 0.0}
_PARTIAL_MATCH_THRESHOLD = 0.62


class SkillsMatcherAgent(BaseAgent):
    """
    Agent responsible for matching candidate skills to job requirements.

    Deterministic by design: normalizes both sides, checks exact/alias
    matches, then substring containment ("semantic"), then fuzzy string
    similarity ("partial"). No LLM call -- see module docstring.
    """

    name = "SkillsMatcherAgent"
    description = "Compare candidate skills against job requirements and score the match (deterministic, no LLM call)"

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        extracted_skills = state.get("extracted_skills", [])
        job_requirements = state.get("job_requirements")

        if not job_requirements:
            return {
                "skills_match": SkillsMatchResult(confidence=0.0, reasoning="No job requirements to match against"),
                "errors": ["SkillsMatcherAgent: No job requirements available"],
                "agent_confidences": {self.name: 0.0}
            }

        if isinstance(job_requirements, dict):
            job_requirements = JobRequirements.model_validate(job_requirements)

        skills_list: list[Skill] = []
        for skill in extracted_skills:
            skills_list.append(Skill.model_validate(skill) if isinstance(skill, dict) else skill)

        required_matches = [
            self._match_one(req, skills_list) for req in job_requirements.required_skills
        ]
        preferred_matches = [
            self._match_one(req, skills_list) for req in job_requirements.preferred_skills
        ]
        all_matches = required_matches + preferred_matches

        required_skills_met = sum(1 for m in required_matches if m.matched)
        preferred_skills_met = sum(1 for m in preferred_matches if m.matched)

        required_score = self._avg_credit(required_matches)
        preferred_score = self._avg_credit(preferred_matches)
        overall_score = round(required_score * 0.7 + preferred_score * 0.3, 2)

        reasoning = self._build_reasoning(
            required_skills_met, len(required_matches),
            preferred_skills_met, len(preferred_matches),
            overall_score, required_matches,
        )

        result = SkillsMatchResult(
            matches=all_matches,
            required_skills_met=required_skills_met,
            required_skills_total=len(required_matches),
            preferred_skills_met=preferred_skills_met,
            preferred_skills_total=len(preferred_matches),
            overall_score=overall_score,
            confidence=0.9,  # deterministic logic -> fixed confidence, not LLM self-reported
            reasoning=reasoning,
        )

        return {
            "skills_match": result,
            "agent_confidences": {self.name: result.confidence}
        }

    def _match_one(self, requirement: str, skills: list[Skill]) -> SkillMatch:
        """Match a single requirement string against the skills list.
        Same inputs -> same output, always."""
        req_canon = canonical_skill_key(requirement)

        best_skill: Skill | None = None
        best_quality = "none"
        best_score = 0.0

        for skill in skills:
            skill_canon = canonical_skill_key(skill.name)

            if req_canon == skill_canon:
                return SkillMatch(
                    requirement=requirement, matched=True, matched_skill=skill.name,
                    match_quality="exact", confidence=0.97,
                    notes="Exact/alias match",
                )

            if skill_canon and (skill_canon in req_canon or req_canon in skill_canon):
                if 0.85 > best_score:
                    best_skill, best_quality, best_score = skill, "semantic", 0.85
                continue

            ratio = difflib.SequenceMatcher(None, req_canon, skill_canon).ratio()
            if ratio >= _PARTIAL_MATCH_THRESHOLD and ratio > best_score:
                best_skill, best_quality, best_score = skill, "partial", round(ratio, 2)

        if best_skill:
            return SkillMatch(
                requirement=requirement, matched=True, matched_skill=best_skill.name,
                match_quality=best_quality, confidence=best_score,
                notes=f"{best_quality} match",
            )

        return SkillMatch(
            requirement=requirement, matched=False, matched_skill="",
            match_quality="none", confidence=0.9,
            notes="No matching skill found among extracted skills",
        )

    @staticmethod
    def _avg_credit(matches: list[SkillMatch]) -> float:
        if not matches:
            return 0.0
        return sum(_QUALITY_CREDIT[m.match_quality] for m in matches) / len(matches)

    @staticmethod
    def _build_reasoning(
        required_met: int, required_total: int,
        preferred_met: int, preferred_total: int,
        overall_score: float, required_matches: list[SkillMatch],
    ) -> str:
        missing_required = [m.requirement for m in required_matches if not m.matched]
        parts = [
            f"Candidate meets {required_met} of {required_total} required skills "
            f"and {preferred_met} of {preferred_total} preferred skills "
            f"(overall skills score: {overall_score:.0%})."
        ]
        if missing_required:
            parts.append("Missing required skills: " + ", ".join(missing_required) + ".")
        return " ".join(parts)