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


# ============================================================================
# Deterministic role-family taxonomy for semantic-ish title relevance
#
# WHY THIS EXISTS: lexical Jaccard overlap (_title_tokens/_overlap) only
# recognizes titles that share literal words -- "Backend Engineer" vs
# "Backend Developer" overlap on "backend"/"engineer"-ish tokens, but
# "Senior Software Engineer" vs "Backend Engineer - Python" share only
# "engineer" against a 5-word union, scoring ~0.09-0.2 even though a
# software engineer is plausibly (not fully) relevant to a backend role,
# and a "Data Analyst" or "Frontend Developer" title shares that same
# "engineer"/"developer" token overlap despite being a genuinely
# different discipline. Lexical overlap alone cannot tell these apart.
#
# WHY NOT AN LLM CALL HERE: job_relevance() runs once per work-experience
# entry, per candidate, per screening -- a batch run over many candidates
# with multiple jobs each would mean many LLM calls on the hot scoring
# path. That reintroduces exactly the problems this pipeline has already
# been fixed to avoid elsewhere (Batch 3/5): Groq rate-limit risk, added
# latency, and nondeterministic output feeding directly into a numeric
# score that this project's own tests require to be reproducible (see
# test_experience_evaluator_agent_makes_no_llm_call and every
# `e1.model_dump() == e2.model_dump()` determinism assertion in this
# file's test suite). A cached/local embedding model was also considered,
# but titles are short, low-vocabulary strings where a small, explicit,
# reviewable mapping is at least as accurate as an embedding similarity
# score, is exactly reproducible, needs no model weights or new heavy
# dependency, and is trivial to unit test and extend by hand.
#
# APPROACH: a small, hand-curated map from title keywords -> a coarse
# "role family" (backend/frontend/fullstack/data/devops/mobile/qa, plus
# a "generic_swe" catch-all for domain-less titles like bare "Software
# Engineer"/"Developer"), plus a symmetric relatedness matrix between
# families. This ONLY ever raises title_score (via max() with the
# existing lexical score below) -- it never lowers a title that already
# scores well lexically, and it deliberately does NOT distinguish
# seniority ("Senior X" and "X" resolve to the same family) since
# seniority isn't a semantic-relevance signal.
# ============================================================================

_ROLE_FAMILY_KEYWORDS: dict[str, frozenset[str]] = {
    "backend": frozenset({"backend", "server"}),
    "frontend": frozenset({"frontend", "ui", "client"}),
    "fullstack": frozenset({"fullstack", "full"}),
    "data": frozenset({"data", "analyst", "analytics", "scientist"}),
    "devops": frozenset({"devops", "sre", "infrastructure", "platform"}),
    "mobile": frozenset({"mobile", "ios", "android"}),
    "qa": frozenset({"qa", "quality", "test", "testing", "sdet"}),
    # Domain-less engineering titles ("Software Engineer", "Developer"
    # alone). Checked LAST -- see _primary_role_family -- so a title that
    # also names a specific domain (e.g. "Backend Developer") is
    # classified by that domain, not diluted into the generic bucket.
    "generic_swe": frozenset({"software", "engineer", "developer", "programmer"}),
}

# Priority order for resolving a title to ONE primary family when its
# tokens match more than one specific-family keyword set (rare, e.g. a
# "Backend/Frontend" combined title) -- first match wins. "generic_swe"
# is intentionally excluded here; it's only used as a last-resort fallback.
_SPECIFIC_FAMILY_ORDER = ("backend", "frontend", "fullstack", "data", "devops", "mobile", "qa")

# Symmetric relatedness between DIFFERENT specific families, and between
# a specific family and the generic-engineering fallback. 0.0-1.0.
#
# IMPORTANT (Batch 10 fix): this matrix ONLY lists pairs with a
# DEFENSIBLE, principled reason to be non-zero. Every other pair --
# including every one NOT listed below -- falls through to 0.0 via
# _ROLE_RELATEDNESS.get(..., 0.0) in _role_family_score. That default
# must stay a genuine "no relationship", not a place to park a vague
# "still kind of engineering-adjacent" guess. An earlier version of this
# matrix included entries like backend<->frontend=0.15 and
# backend<->data=0.15 on exactly that reasoning, which gave Frontend
# Developer and Data Analyst titles a small but real positive semantic
# score against a Backend JD purely because both happen to be software
# roles -- an artificial, unjustified boost. Those entries are removed.
#
# The only relationships kept are ones with an actual reason:
#   - backend<->fullstack, frontend<->fullstack: "fullstack" IS backend
#     + frontend by definition, so this is real overlap, not a guess.
#   - backend<->devops, fullstack<->devops: backend engineers routinely
#     own deployment/CI/infra as part of the role -- a genuine,
#     well-established professional adjacency.
#   - <specific family>: default 0.0 against every OTHER specific family
#     not listed above (frontend<->data, backend<->mobile, data<->qa,
#     etc.) -- these are just different disciplines with no inherent
#     overlap, and get no credit.
#   - <specific family>-generic_swe: a title like "Software Engineer"
#     genuinely doesn't specify a domain, so partial credit here
#     reflects real ambiguity about the candidate's actual focus, not a
#     claim that the domains are related.
#
# Two titles resolving to the SAME specific family score 1.0 (handled as
# a special case in _role_family_score, not listed here); two titles
# both resolving to "generic_swe" score 0.5 (ambiguous on both sides).
_ROLE_RELATEDNESS: dict[frozenset[str], float] = {
    frozenset({"backend", "fullstack"}): 0.75,
    frozenset({"frontend", "fullstack"}): 0.75,
    frozenset({"backend", "devops"}): 0.45,
    frozenset({"fullstack", "devops"}): 0.35,
    frozenset({"backend", "generic_swe"}): 0.55,
    frozenset({"frontend", "generic_swe"}): 0.5,
    frozenset({"fullstack", "generic_swe"}): 0.6,
    frozenset({"data", "generic_swe"}): 0.4,
    frozenset({"devops", "generic_swe"}): 0.45,
    frozenset({"mobile", "generic_swe"}): 0.45,
    frozenset({"qa", "generic_swe"}): 0.4,
}


def _primary_role_family(title: str) -> str | None:
    """The single most-specific role family a title's tokens name, or
    None if nothing in the taxonomy matches at all. Specific domains
    (backend/frontend/...) always win over the generic engineering
    fallback, so "Backend Developer" resolves to "backend", not
    "generic_swe", even though "developer" alone would also match."""
    tokens = set(normalize_skill_text(title).split())
    for family in _SPECIFIC_FAMILY_ORDER:
        if tokens & _ROLE_FAMILY_KEYWORDS[family]:
            return family
    if tokens & _ROLE_FAMILY_KEYWORDS["generic_swe"]:
        return "generic_swe"
    return None


def _role_family_score(candidate_title: str, jd_title: str) -> float | None:
    """Deterministic role-family relatedness between two job titles.

    Returns None when the JD title doesn't name any recognizable role
    family (blank title, or wording entirely outside this taxonomy) --
    callers should fall back to plain lexical overlap in that case, not
    treat "nothing recognized" as "zero relevance".
    """
    jd_family = _primary_role_family(jd_title)
    if jd_family is None:
        return None
    cand_family = _primary_role_family(candidate_title)
    if cand_family is None:
        return 0.0
    if cand_family == jd_family:
        return 0.5 if jd_family == "generic_swe" else 1.0
    return _ROLE_RELATEDNESS.get(frozenset({jd_family, cand_family}), 0.0)


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

    # ISO-style "YYYY-MM" / "YYYY/MM" / "YYYY.MM" (optionally with a
    # trailing "-DD"), e.g. "2023-06" or "2023-06-15". This is checked
    # BEFORE the MM-first regex below because year-first ("2023-06") and
    # month-first ("06-2023") are ambiguous to tell apart by shape alone
    # once separators are stripped -- without this dedicated check,
    # "2023-06" matched neither the month-name loop above (it's numeric)
    # nor the MM/YYYY regex below (that regex requires the 1-2 digit
    # group to come FIRST, but here the 4-digit year comes first), so
    # `month` was silently left None and every such date collapsed to
    # January of that year -- e.g. start="2023-06", end="2023-09" both
    # resolved to 2023-01-01, giving a computed duration of 0 years for
    # a real ~3-month role.
    iso_ym = re.search(r"\b((?:19|20)\d{2})[/.\-](0?[1-9]|1[0-2])(?:[/.\-]\d{1,2})?\b", raw)
    if iso_ym:
        year = int(iso_ym.group(1))
        month = int(iso_ym.group(2))

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

    jd_skills: set[str] = set()
    for s in requirements.required_skills + requirements.preferred_skills:
        jd_skills |= normalize_requirement_skills(s)

    resp_text = " ".join(requirements.responsibilities[:8])
    jd_resp = _tokens(resp_text)

    job_title = _title_tokens(exp.title)
    job_resp = _tokens(" ".join(exp.responsibilities[:8]))

    title_score = _overlap(job_title, jd_title)

    # Deterministic role-family backstop: lexical overlap alone can't see
    # that e.g. "Senior Software Engineer" is plausibly (not fully)
    # relevant to a "Backend Engineer" JD when they share no distinctive
    # words. This only ever RAISES title_score (max()), so a title that
    # already scores well lexically (shared literal words) is unaffected,
    # and it only applies when the JD title names a recognizable family --
    # see _role_family_score. Does not touch skill_hit or resp_score.
    family_score = _role_family_score(exp.title, requirements.title)
    if family_score is not None:
        title_score = max(title_score, family_score)

    tech_keys = {canonical_skill_key(t) for t in exp.technologies if t}
    skill_hit = 0.0
    if jd_skills:
        hits = sum(1 for k in tech_keys if k in jd_skills or any(k in s or s in k for s in jd_skills if s))
        skill_hit = min(1.0, hits / max(1, len(jd_skills))) if tech_keys else 0.0
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


def _experience_score_for_years(years_relevant: float, years_required: int) -> float:
    if years_required <= 0:
        if years_relevant >= 1:
            return 1.0
        elif years_relevant > 0:
            return 0.85
        else:
            return 0.7
    return round(min(1.0, years_relevant / years_required), 2)


# --------------------------------------------------------------------------
# years_relevant per-job credit
#
# CONFIRMED BUG (Priority 6 investigation): the original formula was a flat
#     credit = 0.35 + 0.65 * rel
# applied to EVERY job regardless of how low `rel` (job_relevance()) was.
# There was no lower gate at all -- a job scored at rel=0.01 (e.g. a
# Frontend Developer role against a Backend JD) or rel=0.00 (a resume-
# parsing artifact like a mis-split "EDUCATION & TRAINING" line being
# treated as a job -- see the separate resume-parsing note in
# src/agents/resume_parser.py) still received at least 35% credit toward
# years_relevant, no different from a job that was genuinely, if weakly,
# related. This is what produced the reported anomalies: Ananya Patel's
# 4-year, 1%-relevant Frontend Developer role alone contributed ~1.45
# "relevant" years (35% of 4.07), and Vikram Singh's unrelated Mechanical
# Engineer role plus several parsing-artifact pseudo-jobs (career break,
# education section header, university line) each got a guaranteed floor
# of credit purely for having *some* duration, inflating years_relevant
# well beyond what role_relevance (7%/33% respectively) would suggest is
# defensible.
#
# The 0.35 floor itself was not baseless: job_relevance() is a lexical/
# taxonomy approximation (see the Batch 8/9/10 history in this file) that
# can under-score genuinely related work, so SOME hedge against
# under-scoring is reasonable for a job that shows real, if weak,
# relevance signal. The bug was applying that hedge unconditionally, even
# to jobs with essentially no relevance signal at all.
#
# Fix: _years_relevant_credit() below only applies the 0.35+0.65*rel
# hedge to jobs at/above _YEARS_RELEVANT_FLOOR_THRESHOLD (chosen to match
# the rel < 0.2 cutoff this module already uses, a few lines below in
# evaluate_experience(), to flag a job as "Limited relevance" -- a job
# already flagged that way should not simultaneously receive a guaranteed
# 35% credit toward years_relevant). Below that threshold, credit ramps
# linearly from 0 at rel=0 up to the existing floor-formula's value AT
# the threshold, so the curve is continuous (no cliff at the boundary)
# and a genuinely irrelevant job (rel near 0) now contributes years near
# 0, while the existing above-threshold behavior for real matches is
# completely unchanged.
# --------------------------------------------------------------------------
_YEARS_RELEVANT_FLOOR_THRESHOLD = 0.2
_YEARS_RELEVANT_FLOOR_CREDIT = 0.35


def _years_relevant_credit(rel: float) -> float:
    """Weight applied to a job's duration when accumulating years_relevant.
    See the module-level comment above for the full rationale."""
    threshold_credit = _YEARS_RELEVANT_FLOOR_CREDIT + (1 - _YEARS_RELEVANT_FLOOR_CREDIT) * _YEARS_RELEVANT_FLOOR_THRESHOLD
    if rel < _YEARS_RELEVANT_FLOOR_THRESHOLD:
        return threshold_credit * (rel / _YEARS_RELEVANT_FLOOR_THRESHOLD)
    return _YEARS_RELEVANT_FLOOR_CREDIT + (1 - _YEARS_RELEVANT_FLOOR_CREDIT) * rel


_PROJECT_RELEVANCE_THRESHOLD = 0.3


def project_relevance(project_text: str, requirements: JobRequirements) -> float:
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
    skill_hit = min(1.0, matched / 3) if jd_skill_keys else 0.0
    topic_overlap = _overlap(_tokens(project_text), jd_topic_tokens)
    relevance = 0.75 * min(1.0, skill_hit) + 0.25 * topic_overlap
    return round(min(1.0, max(0.0, relevance)), 2)


def _best_project(projects: list[str], requirements: JobRequirements) -> tuple[float, str | None]:
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
    for sep in (" -- ", " — ", ": "):
        if sep in project_text:
            return project_text.split(sep, 1)[0].strip()
    text = project_text.strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "\u2026"


def evaluate_experience(resume_data: ResumeData, requirements: JobRequirements) -> ExperienceEvaluation:
    years_required = requirements.min_years_experience or 0

    if not resume_data.work_experience:
        has_relevant_education = any(
            any(
                kw in (edu.field or "").lower()
                for kw in ["computer", "data", "software", "engineering", "ai", "machine learning", "information technology"]
            )
            for edu in resume_data.education
        ) if resume_data.education else False

        project_score, best_project = _best_project(resume_data.projects, requirements)
        relevant_project = best_project is not None
        role_relevance = round(max(0.15 if has_relevant_education else 0.0, project_score), 2)

        strengths = []
        if has_relevant_education:
            strengths.append("Relevant academic background")
        if relevant_project:
            strengths.append(f"Relevant project: {_project_title(best_project)}")

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
            years_relevant=0.0, years_required=years_required, experience_score=experience_score,
            role_relevance=role_relevance, career_progression="No professional work history listed.",
            gaps_identified=gaps, strengths=strengths, confidence=0.9,
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

    years_relevant = round(sum(years * _years_relevant_credit(rel) for years, rel in per_job), 2)
    if per_job:
        role_relevance = round(
            sum(rel * max(years, 0.25) for years, rel in per_job) / sum(max(years, 0.25) for years, _rel in per_job), 2,
        )
    else:
        role_relevance = 0.0

    experience_score = _experience_score_for_years(years_relevant, years_required)

    if years_relevant + 0.25 < years_required:
        gaps.append(f"Relevant experience ({years_relevant:.1f}y) is below the {years_required}y requirement")

    reasoning = (
        f"Estimated {years_relevant:.1f} years of relevant experience "
        f"(required {years_required}). Role relevance {role_relevance:.0%} "
        f"based on title/technology overlap with {requirements.title or 'the role'}."
    )

    return ExperienceEvaluation(
        years_relevant=years_relevant, years_required=years_required, experience_score=experience_score,
        role_relevance=role_relevance, career_progression=_career_progression(resume_data.work_experience),
        gaps_identified=gaps[:6], strengths=strengths[:6], confidence=0.9, reasoning=reasoning,
    )


class ExperienceEvaluatorAgent(BaseAgent):
    name = "ExperienceEvaluatorAgent"
    description = "Evaluate work experience relevance (deterministic, no LLM call)"
    temperature = 0.0

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        resume_data = state.get("resume_data")
        job_requirements = state.get("job_requirements")
        if not resume_data or not job_requirements:
            return {"experience_eval": ExperienceEvaluation(confidence=0.0, reasoning="Missing resume or job requirements data"),
                    "errors": ["ExperienceEvaluatorAgent: Missing required data"], "agent_confidences": {self.name: 0.0}}
        if isinstance(resume_data, dict):
            resume_data = ResumeData.model_validate(resume_data)
        if isinstance(job_requirements, dict):
            job_requirements = JobRequirements.model_validate(job_requirements)
        evaluation = evaluate_experience(resume_data, job_requirements)
        return {"experience_eval": evaluation, "agent_confidences": {self.name: evaluation.confidence}}
