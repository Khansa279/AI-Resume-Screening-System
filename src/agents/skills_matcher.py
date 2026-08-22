"""Skills Matcher Agent - Compares candidate skills against job requirements.

DETERMINISTIC (no LLM call). This agent still made an LLM call in the
codebase even though the project's LLM-usage budget assumed it didn't --
that's fixed here. It now reuses the exact same taxonomy/alias table as
SkillExtractorAgent and ExperienceEvaluatorAgent (`src/skill_taxonomy.py`
-- Batch 3: previously this agent used a separate, overlapping
`src/skills_taxonomy.py` module, which risked disagreeing with the other
two agents on what a given alias normalizes to) to decide whether a
candidate skill satisfies a JD requirement, instead of asking an LLM to
judge "semantic"/"partial" matches freshly on every run.
"""

from typing import Any

from .base import BaseAgent
from ..models import Skill, JobRequirements, SkillMatch, SkillsMatchResult
from ..skill_taxonomy import canonical_skill_key, extract_canonical_skills


class SkillsMatcherAgent(BaseAgent):
    """
    Agent responsible for matching candidate skills to job requirements.

    This agent:
    - Compares extracted skills against required/preferred skills
    - Handles alias-based "semantic" matching (e.g., "JS" = "JavaScript")
      via the shared taxonomy, plus substring/partial fallback matching
    - Scores each requirement match
    - Calculates overall skills match score

    SCORING (unchanged from the original design):
    - Required skills contribute ~70% of overall_score
    - Preferred skills contribute ~30% of overall_score
    - exact match = full credit, semantic (taxonomy alias) = 90% credit,
      partial (substring) = 50% credit, no match = 0% credit
    """

    name = "SkillsMatcherAgent"
    description = "Compare candidate skills against job requirements and score the match (deterministic)"

    REQUIRED_WEIGHT = 0.7
    PREFERRED_WEIGHT = 0.3

    CREDIT_BY_QUALITY = {
        "exact": 1.0,
        "semantic": 0.9,
        "partial": 0.5,
        "none": 0.0,
    }

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Match candidate skills against job requirements.

        Args:
            state: Current workflow state with extracted_skills and job_requirements

        Returns:
            State updates with skills_match result
        """
        extracted_skills = state.get("extracted_skills", [])
        job_requirements = state.get("job_requirements")

        if not job_requirements:
            return {
                "skills_match": SkillsMatchResult(confidence=0.0, reasoning="No job requirements to match against"),
                "errors": ["SkillsMatcherAgent: No job requirements available"],
                "agent_confidences": {self.name: 0.0}
            }

        # Convert to proper types if needed
        if isinstance(job_requirements, dict):
            job_requirements = JobRequirements.model_validate(job_requirements)

        skills_list: list[Skill] = []
        for skill in extracted_skills:
            if isinstance(skill, dict):
                skills_list.append(Skill.model_validate(skill))
            else:
                skills_list.append(skill)

        match_result = self._match_deterministic(skills_list, job_requirements)

        return {
            "skills_match": match_result,
            "agent_confidences": {self.name: match_result.confidence}
        }

    def _match_deterministic(
        self, skills: list[Skill], requirements: JobRequirements
    ) -> SkillsMatchResult:
        """Score every required/preferred requirement against candidate skills."""
        matches: list[SkillMatch] = []

        required_scores: list[float] = []
        preferred_scores: list[float] = []
        required_met = 0
        preferred_met = 0

        for req in requirements.required_skills or []:
            match = self._score_requirement(req, skills)
            matches.append(match)
            required_scores.append(self.CREDIT_BY_QUALITY[match.match_quality])
            if match.matched:
                required_met += 1

        for req in requirements.preferred_skills or []:
            match = self._score_requirement(req, skills)
            matches.append(match)
            preferred_scores.append(self.CREDIT_BY_QUALITY[match.match_quality])
            if match.matched:
                preferred_met += 1

        required_total = len(requirements.required_skills or [])
        preferred_total = len(requirements.preferred_skills or [])

        required_avg = sum(required_scores) / required_total if required_total else 1.0
        preferred_avg = sum(preferred_scores) / preferred_total if preferred_total else 1.0

        if required_total and preferred_total:
            overall_score = required_avg * self.REQUIRED_WEIGHT + preferred_avg * self.PREFERRED_WEIGHT
        elif required_total:
            overall_score = required_avg
        elif preferred_total:
            overall_score = preferred_avg
        else:
            overall_score = 0.0

        overall_score = round(min(max(overall_score, 0.0), 1.0), 4)

        reasoning = self._build_reasoning(
            required_met, required_total, preferred_met, preferred_total, overall_score
        )

        # Deterministic matching applied identically every time -> stable,
        # high confidence rather than an LLM's self-reported guess.
        confidence = 0.95 if (required_total or preferred_total) else 0.3

        return SkillsMatchResult(
            matches=matches,
            required_skills_met=required_met,
            required_skills_total=required_total,
            preferred_skills_met=preferred_met,
            preferred_skills_total=preferred_total,
            overall_score=overall_score,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _score_requirement(self, requirement: str, skills: list[Skill]) -> SkillMatch:
        """Score a single JD requirement string against the candidate's skill list."""
        req_canonical_key = canonical_skill_key(requirement)
        req_key = requirement.strip().lower()

        # 1. Exact match: requirement string equals a candidate skill name
        #    (case-insensitive), OR both normalize to the same canonical
        #    taxonomy entry (e.g. requirement "JS" vs candidate skill
        #    "JavaScript" both normalize to the same key).
        for skill in skills:
            skill_key = skill.name.strip().lower()
            if skill_key == req_key:
                return SkillMatch(
                    requirement=requirement, matched=True, matched_skill=skill.name,
                    match_quality="exact", confidence=1.0,
                    notes="Exact name match",
                )
            if canonical_skill_key(skill.name) == req_canonical_key:
                return SkillMatch(
                    requirement=requirement, matched=True, matched_skill=skill.name,
                    match_quality="exact", confidence=1.0,
                    notes=f"Both normalize to '{req_canonical_key}'",
                )

        # 2. Semantic (alias) match: the requirement text contains a
        #    taxonomy alias whose canonical form matches a candidate skill,
        #    or vice-versa (candidate skill alias appears in requirement).
        req_skill_hits = {name for name, _category in extract_canonical_skills(requirement)}
        for skill in skills:
            canonical_hits = extract_canonical_skills(skill.name)
            canonical_name = canonical_hits[0][0] if canonical_hits else skill.name
            if canonical_name in req_skill_hits:
                return SkillMatch(
                    requirement=requirement, matched=True, matched_skill=skill.name,
                    match_quality="semantic", confidence=0.9,
                    notes=f"Taxonomy alias match on '{canonical_name}'",
                )

        # 3. Partial match: substring overlap either direction (e.g.
        #    requirement "Python programming experience" vs skill "Python").
        for skill in skills:
            skill_key = skill.name.strip().lower()
            if len(skill_key) >= 3 and (skill_key in req_key or req_key in skill_key):
                return SkillMatch(
                    requirement=requirement, matched=True, matched_skill=skill.name,
                    match_quality="partial", confidence=0.5,
                    notes="Substring overlap",
                )

        # 4. No match
        return SkillMatch(
            requirement=requirement, matched=False, matched_skill="",
            match_quality="none", confidence=0.9,
            notes="No matching candidate skill found",
        )

    def _build_reasoning(
        self, required_met: int, required_total: int,
        preferred_met: int, preferred_total: int, overall_score: float,
    ) -> str:
        parts = [f"Overall skills match: {overall_score:.0%}."]
        if required_total:
            parts.append(f"Met {required_met}/{required_total} required skills.")
        else:
            parts.append("No required skills were specified for this role.")
        if preferred_total:
            parts.append(f"Met {preferred_met}/{preferred_total} preferred skills.")
        return " ".join(parts)
