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
from ..skill_taxonomy import (
    canonical_skill_key,
    normalize_skill_text,
    extract_canonical_skills,
    normalize_requirement_skills,
)


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

# Title-only stopwords: grammatical filler ONLY. Role/seniority words
# ("engineer", "developer", "software", "senior", ...) are intentionally
# NOT stripped here, unlike _STOP above. Titles are short -- stripping
# those words from a title routinely leaves nothing behind (e.g. "Senior
# Software Engineer" -> {} after the old _STOP), which forced
# title_score to 0 via _overlap()'s empty-set short-circuit for almost
# any conventional job title. Confirmed against real resumes: John Doe's
# "Senior Software Engineer" produced an EMPTY token set under the old
# _STOP list. Role words carry real signal for title comparison (two
# "Engineer" titles overlapping is meaningful), so they stay.
_TITLE_STOP = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "at", "to", "for", "on",
    "with", "as", "by", "from",
})


def _title_tokens(text: str) -> set[str]:
    return {
        t for t in normalize_skill_text(text).split()
        if t and t not in _TITLE_STOP and len(t) > 1
    }

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
    jd_title = _title_tokens(requirements.title)

    # Decompose each requirement (which may be a long descriptive phrase
    # like "REST API development (Django REST Framework or FastAPI)")
    # into the canonical skill keys it actually references, rather than
    # treating the whole phrase as one opaque token. See
    # skill_taxonomy.normalize_requirement_skills for why.
    jd_skills: set[str] = set()
    for s in requirements.required_skills + requirements.preferred_skills:
        jd_skills |= normalize_requirement_skills(s)

    resp_text = " ".join(requirements.responsibilities[:8])
    jd_resp = _tokens(resp_text)

    job_title = _title_tokens(exp.title)
    job_resp = _tokens(" ".join(exp.responsibilities[:8]))

    title_score = _overlap(job_title, jd_title | jd_resp | _tokens(" ".join(requirements.required_skills)))
    tech_keys = {canonical_skill_key(t) for t in exp.technologies if t}
    skill_hit = 0.0
    if jd_skills:
        hits = sum(1 for k in tech_keys if k in jd_skills or any(k in s or s in k for s in jd_skills if s))
        skill_hit = min(1.0, hits / max(1, len(jd_skills))) if tech_keys else 0.0
        # Also count responsibility mentions of required skills -- using
        # the same decomposition, so a candidate whose bullets mention
        # "Django" gets credit against a requirement phrased as "REST API
        # development (Django REST Framework or FastAPI)" instead of
        # needing that whole sentence to appear verbatim.
        text_blob = normalize_skill_text(
            " ".join([exp.title, " ".join(exp.technologies), " ".join(exp.responsibilities)])
        )
        mentioned = 0
        for req in requirements.required_skills:
            req_keys = normalize_requirement_skills(req)
            if any(k and k in text_blob for k in req_keys):
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


# ============================================================================
# Years-required-aware experience_score (Batch 6)
#
# Extracted so BOTH the "has work_experience" path (below) and the "no
# work_experience" path (evaluate_experience's early return) score
# experience_score the same way. Previously, a candidate with zero listed
# work_experience was hard-coded to experience_score=0.0 regardless of
# years_required, which unfairly zeroed out entry-level candidates applying
# to roles that explicitly require 0-1 years -- a candidate who happened to
# have even one (irrelevant) work_experience row would fall through to this
# exact logic and score 0.7, while a candidate with none at all got 0.0.
# ============================================================================

def _experience_score_for_years(years_relevant: float, years_required: int) -> float:
    if years_required <= 0:
        if years_relevant >= 1:
            return 1.0
        elif years_relevant > 0:
            return 0.85
        else:
            return 0.7
    return round(min(1.0, years_relevant / years_required), 2)


# ============================================================================
# Project relevance (Batch 6)
#
# resume_data.projects is a flat list[str] (e.g. "Flood Severity Prediction
# -- built an ML model to predict flood risk using Python and scikit-learn"),
# not a structured object like WorkExperience -- so this reuses the same
# taxonomy-driven skill matching (extract_canonical_skills) and token-overlap
# approach (_tokens/_overlap) as job_relevance() above, adapted for plain
# text instead of a WorkExperience's separate title/technologies/
# responsibilities fields.
#
# IMPORTANT: this only ever feeds role_relevance/strengths. It is
# deliberately never added to years_relevant/years_worked -- projects are
# not professional work history and must not inflate experience_score.
# ============================================================================

_PROJECT_RELEVANCE_THRESHOLD = 0.3  # below this, treat as "not relevant" (no boost, no strength)


def project_relevance(project_text: str, requirements: JobRequirements) -> float:
    """How relevant a single project description is to the target role."""
    if not project_text:
        return 0.0

    jd_skill_keys = {
        canonical_skill_key(s)
        for s in (requirements.required_skills + requirements.preferred_skills)
        if s
    }
    jd_topic_tokens = _tokens(requirements.title) | _tokens(" ".join(requirements.responsibilities[:8]))

    proj_hits = {canonical_skill_key(name) for name, _category in extract_canonical_skills(project_text)}
    matched = len(proj_hits & jd_skill_keys)
    # Cap the denominator at 3 rather than dividing by the FULL required+
    # preferred list (which can be 7-8+ items): a short project blurb that
    # clearly names 2-3 of the JD's core skills is meaningfully relevant on
    # its own merits and shouldn't be diluted just because the JD also has
    # several other, unrelated preferred skills (e.g. "Git", "SQL") the
    # blurb had no occasion to mention.
    skill_hit = min(1.0, matched / 3) if jd_skill_keys else 0.0

    topic_overlap = _overlap(_tokens(project_text), jd_topic_tokens)

    relevance = 0.75 * min(1.0, skill_hit) + 0.25 * topic_overlap
    return round(min(1.0, max(0.0, relevance)), 2)


def _best_project(projects: list[str], requirements: JobRequirements) -> tuple[float, str | None]:
    """Highest-relevance project and its score, or (0.0, None) if none clear the threshold."""
    best_score = 0.0
    best_project: str | None = None
    for project_text in projects or []:
        score = project_relevance(project_text, requirements)
        if score > best_score:
            best_score, best_project = score, project_text
    if best_score < _PROJECT_RELEVANCE_THRESHOLD:
        return 0.0, None
    return best_score, best_project


def _project_title(project_text: str, max_len: int = 60) -> str:
    """Best-effort short label for a project blurb, for use in strengths."""
    for sep in (" -- ", " — ", ": "):
        if sep in project_text:
            return project_text.split(sep, 1)[0].strip()
    text = project_text.strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "\u2026"


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

        # Projects are a secondary relevance signal for candidates with no
        # formal work history -- but they NEVER count as years of
        # professional experience (years_relevant stays 0.0 below).
        project_score, best_project = _best_project(resume_data.projects, requirements)
        relevant_project = best_project is not None

        role_relevance = round(max(
            0.15 if has_relevant_education else 0.0,
            project_score,
        ), 2)

        strengths = []
        if has_relevant_education:
            strengths.append("Relevant academic background")
        if relevant_project:
            strengths.append(f"Relevant project: {_project_title(best_project)}")

        # Reuse the same years-required-aware scoring the "has work
        # experience" path below uses, instead of hard-coding 0.0 -- a
        # candidate with zero years relevant against a 0-1-year-required
        # role should not score worse than one who has some irrelevant
        # experience.
        experience_score = _experience_score_for_years(0.0, years_required)

        gaps = ["No work experience listed"]
        if years_required > 0:
            gaps.append(f"No professional experience found (role requires {years_required}y)")

        reasoning_parts = [
            f"No formal work experience was listed. Against a "
            f"{years_required}-year requirement, experience_score is "
            f"{experience_score:.2f} using the same years-required scale "
            f"applied to candidates who do have work history."
        ]
        if relevant_project:
            reasoning_parts.append(
                f"Role relevance is boosted by a relevant project "
                f"(\"{_project_title(best_project)}\"), which is not "
                f"counted as professional experience."
            )
        elif has_relevant_education:
            reasoning_parts.append("Role relevance reflects a related academic background.")
        else:
            reasoning_parts.append("No directly relevant education or projects were found either.")

        return ExperienceEvaluation(
            years_relevant=0.0,
            years_required=years_required,
            experience_score=experience_score,
            role_relevance=role_relevance,
            career_progression="No professional work history listed.",
            gaps_identified=gaps,
            strengths=strengths,
            confidence=0.9,
            reasoning=" ".join(reasoning_parts),
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

    experience_score = _experience_score_for_years(years_relevant, years_required)

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
