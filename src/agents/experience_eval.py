"""Experience Evaluator Agent - Assesses work experience relevance.

Deterministic: years from dates/durations, relevance from title/technology
overlap with the job description. Same parsed resume + same requirements
always produce the same ExperienceEvaluation. No LLM on the normal path.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from .base import BaseAgent
from ..models import ResumeData, JobRequirements, ExperienceEvaluation, WorkExperience
from ..skill_taxonomy import canonical_skill_key, normalize_skill_text


_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_PRESENT = re.compile(r"^(present|current|now|ongoing|date)$", re.I)

_STOP = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "at", "to", "for", "on",
    "with", "as", "by", "from", "engineer", "engineering", "developer",
    "software", "associate", "junior", "senior", "lead", "intern",
})

_SENIORITY = [
    ("intern", 0), ("trainee", 0), ("junior", 1), ("associate", 1),
    ("engineer", 2), ("developer", 2), ("analyst", 2),
    ("senior", 3), ("lead", 4), ("principal", 5), ("staff", 5),
    ("manager", 5), ("director", 6),
]


def _parse_one_date(text: str) -> date | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if _PRESENT.match(raw):
        return date.today()

    lowered = raw.lower()
    month = None
    year = None

    y = re.search(r"\b(19|20)\d{2}\b", raw)
    if y:
        year = int(y.group(0))
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}\b", lowered):
            month = num
            break
    mnum = re.search(r"\b(0?[1-9]|1[0-2])[/.-]((?:19|20)\d{2})\b", raw)
    if mnum:
        month = int(mnum.group(1))
        year = int(mnum.group(2))

    if year is None:
        return None
    return date(year, month or 1, 1)


def _parse_duration_years(text: str) -> float | None:
    if not text:
        return None
    years = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text, re.I)
    months = re.search(r"(\d+(?:\.\d+)?)\s*months?", text, re.I)
    total = 0.0
    found = False
    if years:
        total += float(years.group(1))
        found = True
    if months:
        total += float(months.group(1)) / 12.0
        found = True
    return round(total, 2) if found else None


def _split_range(text: str) -> tuple[str, str]:
    parts = re.split(r"\s*(?:-|–|—|to)\s*", text or "", maxsplit=1, flags=re.I)
    if len(parts) == 2:
        return parts[0], parts[1]
    return text, ""


def job_years(exp: WorkExperience) -> float:
    """Best-effort years for one role. Missing dates -> small default if titled."""
    start = _parse_one_date(exp.start_date)
    end = _parse_one_date(exp.end_date)
    if start is None and end is None:
        left, right = _split_range(exp.duration or "")
        start = _parse_one_date(left)
        end = _parse_one_date(right) if right else date.today()
    if start is None and end is None:
        from_duration = _parse_duration_years(exp.duration)
        if from_duration is not None:
            return from_duration
        return 0.5 if (exp.title and exp.company) else 0.0
    if start is None:
        return 0.5
    if end is None:
        end = date.today()
    if end < start:
        start, end = end, start
    days = (end - start).days
    return max(0.0, round(days / 365.25, 2))


def _tokens(text: str) -> set[str]:
    return {
        t for t in normalize_skill_text(text).split()
        if t and t not in _STOP and len(t) > 1
    }


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def job_relevance(exp: WorkExperience, requirements: JobRequirements) -> float:
    jd_title = _tokens(requirements.title)
    jd_skills = {canonical_skill_key(s) for s in (requirements.required_skills + requirements.preferred_skills)}
    jd_skills |= _tokens(" ".join(requirements.required_skills + requirements.preferred_skills))
    resp_text = " ".join(requirements.responsibilities[:8])
    jd_resp = _tokens(resp_text)

    job_title = _tokens(exp.title)
    job_tech = {canonical_skill_key(t) for t in exp.technologies} | _tokens(" ".join(exp.technologies))
    job_resp = _tokens(" ".join(exp.responsibilities[:8]))

    title_score = _overlap(job_title, jd_title | jd_resp | _tokens(" ".join(requirements.required_skills)))
    tech_keys = {canonical_skill_key(t) for t in exp.technologies if t}
    skill_hit = 0.0
    if jd_skills:
        hits = sum(1 for k in tech_keys if k in jd_skills or any(k in s or s in k for s in jd_skills if s))
        skill_hit = min(1.0, hits / max(1, len(jd_skills))) if tech_keys else 0.0
        # Also count responsibility mentions of required skills
        resp_blob = canonical_skill_key(" ".join(exp.responsibilities))
        text_blob = normalize_skill_text(
            " ".join([exp.title, " ".join(exp.technologies), " ".join(exp.responsibilities)])
        )
        mentioned = 0
        for req in requirements.required_skills:
            key = canonical_skill_key(req)
            if key and key in text_blob:
                mentioned += 1
        if requirements.required_skills:
            skill_hit = max(skill_hit, mentioned / len(requirements.required_skills))

    resp_score = _overlap(job_resp, jd_resp | jd_title)
    relevance = 0.45 * title_score + 0.4 * skill_hit + 0.15 * resp_score
    return round(min(1.0, max(0.0, relevance)), 2)


def _seniority(title: str) -> int:
    t = (title or "").lower()
    best = 2
    for word, level in _SENIORITY:
        if word in t:
            best = max(best, level)
    return best


def _career_progression(experiences: list[WorkExperience]) -> str:
    if len(experiences) == 1:
        return "Single listed role."
    levels = [_seniority(e.title) for e in reversed(experiences)]
    if levels[-1] > levels[0]:
        return "Steady growth in seniority across listed roles."
    if levels[-1] == levels[0]:
        return "Lateral moves at a similar seniority level."
    return "Mixed seniority across listed roles."


def evaluate_experience(
    resume_data: ResumeData, requirements: JobRequirements
) -> ExperienceEvaluation:
    years_required = requirements.min_years_experience or 0

    if not resume_data.work_experience:
        has_relevant_education = any(
            any(
                kw in (edu.field or "").lower()
                for kw in [
                    "computer", "data", "software", "engineering", "ai",
                    "machine learning", "information technology",
                ]
            )
            for edu in resume_data.education
        ) if resume_data.education else False
        return ExperienceEvaluation(
            years_relevant=0.0,
            years_required=years_required,
            experience_score=0.0,
            role_relevance=0.15 if has_relevant_education else 0.0,
            career_progression="No professional work history listed.",
            gaps_identified=["No work experience listed"],
            strengths=["Relevant academic background"] if has_relevant_education else [],
            confidence=0.95,
            reasoning=(
                "No work experience was listed on the resume, so experience_score "
                "is set to 0 by rule rather than by LLM judgment. "
                + (
                    "Role relevance reflects a related academic background."
                    if has_relevant_education
                    else "No directly relevant education was found either."
                )
            ),
        )

    per_job: list[tuple[float, float]] = []
    strengths: list[str] = []
    gaps: list[str] = []

    for exp in resume_data.work_experience:
        years = job_years(exp)
        rel = job_relevance(exp, requirements)
        per_job.append((years, rel))
        if rel >= 0.5 and exp.title:
            strengths.append(f"Relevant role: {exp.title} at {exp.company}")
        elif exp.title and rel < 0.2:
            gaps.append(f"Limited relevance: {exp.title} at {exp.company}")

    years_relevant = round(sum(years * (0.35 + 0.65 * rel) for years, rel in per_job), 2)
    if per_job:
        role_relevance = round(
            sum(rel * max(years, 0.25) for years, rel in per_job)
            / sum(max(years, 0.25) for years, _rel in per_job),
            2,
        )
    else:
        role_relevance = 0.0

    if years_required <= 0:
        if years_relevant >= 1:
            experience_score = 1.0
        elif years_relevant > 0:
            experience_score = 0.85
        else:
            experience_score = 0.7
    else:
        experience_score = round(min(1.0, years_relevant / years_required), 2)

    if years_relevant + 0.25 < years_required:
        gaps.append(
            f"Relevant experience ({years_relevant:.1f}y) is below the "
            f"{years_required}y requirement"
        )

    reasoning = (
        f"Estimated {years_relevant:.1f} years of relevant experience "
        f"(required {years_required}). Role relevance {role_relevance:.0%} "
        f"based on title/technology overlap with {requirements.title or 'the role'}."
    )

    return ExperienceEvaluation(
        years_relevant=years_relevant,
        years_required=years_required,
        experience_score=experience_score,
        role_relevance=role_relevance,
        career_progression=_career_progression(resume_data.work_experience),
        gaps_identified=gaps[:6],
        strengths=strengths[:6],
        confidence=0.9,
        reasoning=reasoning,
    )


class ExperienceEvaluatorAgent(BaseAgent):
    name = "ExperienceEvaluatorAgent"
    description = "Evaluate work experience relevance (deterministic, no LLM call)"
    temperature = 0.0

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        resume_data = state.get("resume_data")
        job_requirements = state.get("job_requirements")

        if not resume_data or not job_requirements:
            return {
                "experience_eval": ExperienceEvaluation(
                    confidence=0.0,
                    reasoning="Missing resume or job requirements data"
                ),
                "errors": ["ExperienceEvaluatorAgent: Missing required data"],
                "agent_confidences": {self.name: 0.0}
            }

        if isinstance(resume_data, dict):
            resume_data = ResumeData.model_validate(resume_data)
        if isinstance(job_requirements, dict):
            job_requirements = JobRequirements.model_validate(job_requirements)

        evaluation = evaluate_experience(resume_data, job_requirements)
        return {
            "experience_eval": evaluation,
            "agent_confidences": {self.name: evaluation.confidence}
        }
