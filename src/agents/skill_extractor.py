"""Skill Extractor Agent - Identifies and categorizes candidate skills.

DETERMINISTIC (no LLM call). Previously this agent sent the whole resume
context to the LLM and asked it to invent a skill list, which was the
single biggest source of score variance between runs of the *same*
resume: the same text could come back with a different skill set (and
therefore a different match score) every time, and it burned an LLM call
per candidate.

Now it scans resume_data (skills_section, work-experience responsibilities
+ technologies, projects, certifications) against the shared taxonomy in
`src/skill_taxonomy.py` -- the SAME module used by SkillsMatcherAgent and
ExperienceEvaluatorAgent (Batch 3: previously this agent used a separate,
overlapping `src/skills_taxonomy.py` module, which risked the two agents
disagreeing on what a given alias normalizes to) -- and returns a
normalized, de-duplicated, sorted skill list. Same resume text -> same
skills, every time, 0 LLM calls.
"""

from typing import Any

from .base import BaseAgent
from ..models import Skill, ResumeData
from ..skill_taxonomy import extract_canonical_skills, skill_category


def extract_skills_from_resume(resume_data: ResumeData) -> list[Skill]:
    """
    Module-level convenience wrapper around SkillExtractorAgent's
    deterministic extraction, so callers/tests that just want "skills for
    this ResumeData" don't need to construct an agent and go through
    `process(state)`.
    """
    skills, _confidence = SkillExtractorAgent()._extract_deterministic(resume_data)
    return skills


class SkillExtractorAgent(BaseAgent):
    """
    Agent responsible for extracting and categorizing skills from parsed resume.

    This agent:
    - Identifies explicit skills (from the resume's own skills section)
    - Infers skills mentioned in work-experience technologies/responsibilities
      and in project descriptions
    - Categorizes skills using the shared taxonomy (src/skill_taxonomy.py)
    - Deduplicates and sorts the result for a stable, deterministic output
    """

    name = "SkillExtractorAgent"
    description = "Extract and categorize technical and soft skills from resume data (deterministic taxonomy match)"

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Extract skills from the parsed resume.

        Args:
            state: Current workflow state containing resume_data

        Returns:
            State updates with extracted_skills list
        """
        resume_data = state.get("resume_data")

        if not resume_data:
            return {
                "extracted_skills": [],
                "errors": ["SkillExtractorAgent: No resume data available"],
                "agent_confidences": {self.name: 0.0}
            }

        # If resume_data is a dict, convert to ResumeData
        if isinstance(resume_data, dict):
            resume_data = ResumeData.model_validate(resume_data)

        skills, confidence = self._extract_deterministic(resume_data)

        return {
            "extracted_skills": skills,
            "agent_confidences": {self.name: confidence}
        }

    def _extract_deterministic(self, resume_data: ResumeData) -> tuple[list[Skill], float]:
        """Build a normalized, deduplicated, sorted skill list with no LLM call."""
        # canonical_name -> {"source": ..., "hits": int} so we know
        # explicit vs inferred and can bump confidence when a skill is
        # corroborated in multiple places (e.g. listed in skills_section
        # AND used in a project).
        found: dict[str, dict[str, Any]] = {}

        def add(canonical: str, source: str) -> None:
            entry = found.setdefault(canonical, {"source": source, "hits": 0})
            entry["hits"] += 1
            # explicit wins over inferred if seen in both places
            if source == "explicit":
                entry["source"] = "explicit"

        def add_recognized_or_raw(piece: str, source: str) -> None:
            """Scan `piece` for known taxonomy skills; if none are
            recognized, fall back to keeping the raw text as an
            "other"-category skill so nothing the candidate listed is
            lost (same fallback behavior as before Batch 3)."""
            piece = (piece or "").strip()
            if not piece:
                return
            hits = extract_canonical_skills(piece)
            if hits:
                for canonical, _category in hits:
                    add(canonical, source)
            else:
                add(piece, source)

        # 1. Explicit skills section -- exact/alias lookups
        for raw_skill in resume_data.skills_section or []:
            # A skills_section entry may itself be a comma-separated group
            # (e.g. "Languages: Python, JavaScript, SQL") -- split defensively.
            for piece in raw_skill.replace(":", ",").split(","):
                add_recognized_or_raw(piece, "explicit")

        # 2. Work experience: technologies (explicit) + responsibilities (inferred)
        for exp in resume_data.work_experience or []:
            for tech in exp.technologies or []:
                add_recognized_or_raw(tech, "explicit")
            for resp in exp.responsibilities or []:
                for canonical, _category in extract_canonical_skills(resp):
                    add(canonical, "inferred")
            # Title itself can carry a signal, e.g. "Data Engineer"
            for canonical, _category in extract_canonical_skills(exp.title or ""):
                add(canonical, "inferred")

        # 3. Projects (inferred)
        for project in resume_data.projects or []:
            for canonical, _category in extract_canonical_skills(project):
                add(canonical, "inferred")

        # 4. Summary (inferred, light signal)
        if resume_data.summary:
            for canonical, _category in extract_canonical_skills(resume_data.summary):
                add(canonical, "inferred")

        # 5. Certifications imply the certified skill when recognizable
        for cert in resume_data.certifications or []:
            for canonical, _category in extract_canonical_skills(cert):
                add(canonical, "explicit")

        skills: list[Skill] = []
        for canonical, meta in found.items():
            category = skill_category(canonical)
            # Deterministic proficiency heuristic: corroborated in
            # multiple sections -> "advanced", single mention -> "intermediate".
            proficiency = "advanced" if meta["hits"] >= 2 else "intermediate"
            confidence = 0.95 if meta["source"] == "explicit" else 0.7

            skills.append(Skill(
                name=canonical,
                category=category,
                proficiency=proficiency,
                confidence=confidence,
                source=meta["source"],
            ))

        # Stable, deterministic ordering: by name (case-insensitive).
        skills.sort(key=lambda s: s.name.lower())

        # Overall extraction confidence: deterministic taxonomy matching is
        # always applied consistently, so this is high and stable rather
        # than an LLM's self-reported guess. Slightly lower if nothing
        # was found (resume had no recognizable skills content).
        overall_confidence = 0.9 if skills else 0.5

        return skills, overall_confidence
