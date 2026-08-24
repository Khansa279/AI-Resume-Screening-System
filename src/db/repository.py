"""
Repository (CRUD) layer for the recruitment platform.

Every function here takes a `Session` explicitly rather than opening its
own -- callers (scripts today, API route handlers later) own the session
lifecycle via `database.get_session()` or `database.get_db()`. This keeps
the repository layer testable and framework-agnostic.

Functions are grouped by entity and named consistently:
    create_x / get_x / list_x / update_x / delete_x
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import Session

from . import models


def _now() -> datetime:
    return datetime.now(timezone.utc)


def compute_content_hash(text: str) -> str:
    """SHA-256 of *extracted* resume text, used to recognize 'this is the
    same resume I've already processed' regardless of file name, upload
    timestamp, or which literal file bytes were uploaded (a re-saved PDF
    of the same resume extracts to the same text and should reuse it).

    Whitespace is normalized before hashing so trivial extraction-order
    differences (e.g. an extra trailing newline from one parser vs
    another) don't defeat the dedup.
    """
    normalized = " ".join((text or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ============================================================================
# Organization / Department / Position
# ============================================================================

def create_organization(db: Session, name: str) -> models.Organization:
    org = models.Organization(name=name)
    db.add(org)
    db.flush()  # populate org.id without requiring a full commit
    return org


def get_organization(db: Session, organization_id: int) -> Optional[models.Organization]:
    return db.get(models.Organization, organization_id)

def get_organization_by_name(db: Session, name: str) -> Optional[models.Organization]:
    """Lookup by the column that already carries a UNIQUE constraint, so
    callers can do get-or-create at the call site instead of hitting
    IntegrityError on repeat runs. Constraint itself is untouched."""
    return db.scalar(select(models.Organization).where(models.Organization.name == name))

def list_organizations(db: Session) -> list[models.Organization]:
    return list(db.scalars(select(models.Organization)))


def create_department(db: Session, organization_id: int, name: str) -> models.Department:
    dept = models.Department(organization_id=organization_id, name=name)
    db.add(dept)
    db.flush()
    return dept


def list_departments(db: Session, organization_id: int) -> list[models.Department]:
    stmt = select(models.Department).where(models.Department.organization_id == organization_id)
    return list(db.scalars(stmt))


def create_position(
    db: Session, department_id: int, title: str, status: str = "open"
) -> models.Position:
    position = models.Position(department_id=department_id, title=title, status=status)
    db.add(position)
    db.flush()
    return position


def list_positions(db: Session, department_id: int) -> list[models.Position]:
    stmt = select(models.Position).where(models.Position.department_id == department_id)
    return list(db.scalars(stmt))


def get_position(db: Session, position_id: int) -> Optional[models.Position]:
    return db.get(models.Position, position_id)

def get_department_by_name(db: Session, organization_id: int, name: str) -> Optional[models.Department]:
    """Matches the existing uq_dept_org_name constraint shape
    (organization_id, name) -- same get-or-create purpose."""
    stmt = (
        select(models.Department)
        .where(models.Department.organization_id == organization_id)
        .where(models.Department.name == name)
    )
    return db.scalar(stmt)

def get_position_by_title(db: Session, department_id: int, title: str) -> Optional[models.Position]:
    """Positions have no UNIQUE constraint on (department_id, title) in
    the current schema, so this doesn't enforce dedup at the DB level --
    it just lets callers that want dedup behavior (test/seed scripts)
    opt into it without a schema change."""
    stmt = (
        select(models.Position)
        .where(models.Position.department_id == department_id)
        .where(models.Position.title == title)
    )
    return db.scalar(stmt)
# ============================================================================
# Job Description + Requirements
# ============================================================================

def create_job_description(db: Session, position_id: int, raw_text: str) -> models.JobDescription:
    """Creates a new JD version for a position. If a current JD already
    exists, it is marked no-longer-current and the new one becomes
    version = previous.version + 1. This is how JD edits are versioned
    instead of mutated in place."""
    previous = db.scalar(
        select(models.JobDescription)
        .where(models.JobDescription.position_id == position_id)
        .where(models.JobDescription.is_current.is_(True))
    )
    next_version = 1
    if previous is not None:
        previous.is_current = False
        next_version = previous.version + 1

    jd = models.JobDescription(
        position_id=position_id, raw_text=raw_text, version=next_version, is_current=True
    )
    db.add(jd)
    db.flush()
    return jd


def get_current_job_description(db: Session, position_id: int) -> Optional[models.JobDescription]:
    stmt = (
        select(models.JobDescription)
        .where(models.JobDescription.position_id == position_id)
        .where(models.JobDescription.is_current.is_(True))
    )
    return db.scalar(stmt)


def get_job_description(db: Session, job_description_id: int) -> Optional[models.JobDescription]:
    return db.get(models.JobDescription, job_description_id)


def save_job_requirements(
    db: Session,
    job_description_id: int,
    *,
    title: str = "",
    summary: str = "",
    min_years_experience: int = 0,
    education_requirements: list[str] | None = None,
    certifications_required: list[str] | None = None,
    responsibilities: list[str] | None = None,
    parsing_confidence: float = 0.0,
    required_skills: list[str] | None = None,
    preferred_skills: list[str] | None = None,
    requirement_items: list[dict] | None = None,
) -> models.JobRequirements:
    """Persist the output of JobAnalyzerAgent for one JD version.

    `requirement_items` expects dicts shaped like the Requirement model:
        {"description": ..., "category": ..., "priority": ..., "years_needed": ...}
    """
    jr = models.JobRequirements(
        job_description_id=job_description_id,
        title=title,
        summary=summary,
        min_years_experience=min_years_experience,
        education_requirements=education_requirements or [],
        certifications_required=certifications_required or [],
        responsibilities=responsibilities or [],
        parsing_confidence=parsing_confidence,
    )
    db.add(jr)
    db.flush()

    for skill_name in required_skills or []:
        db.add(models.JobRequirementSkill(
            job_requirements_id=jr.id, skill_name=skill_name, priority="required"
        ))
    for skill_name in preferred_skills or []:
        db.add(models.JobRequirementSkill(
            job_requirements_id=jr.id, skill_name=skill_name, priority="preferred"
        ))
    for item in requirement_items or []:
        db.add(models.JobRequirementItem(
            job_requirements_id=jr.id,
            description=item.get("description", ""),
            category=item.get("category", "other"),
            priority=item.get("priority", "required"),
            years_needed=item.get("years_needed"),
        ))

    db.flush()
    return jr


def get_job_requirements(db: Session, job_description_id: int) -> Optional[models.JobRequirements]:
    stmt = select(models.JobRequirements).where(
        models.JobRequirements.job_description_id == job_description_id
    )
    return db.scalar(stmt)


# ============================================================================
# Candidate + Resume
# ============================================================================

def get_or_create_candidate(
    db: Session,
    *,
    name: str = "",
    email: str = "",
    phone: str = "",
    location: str = "",
    linkedin: str = "",
    github: str = "",
) -> models.Candidate:
    """Dedupe by email when we have one; otherwise always create a new
    candidate record (we can't reliably dedupe on name alone)."""
    if email:
        existing = db.scalar(select(models.Candidate).where(models.Candidate.email == email))
        if existing is not None:
            return existing

    candidate = models.Candidate(
        name=name, email=email, phone=phone, location=location,
        linkedin=linkedin, github=github,
    )
    db.add(candidate)
    db.flush()
    return candidate


def create_resume(
    db: Session,
    *,
    candidate_id: int,
    position_id: int,
    file_path: str = "",
    file_type: str = "unknown",
    raw_text: str = "",
    content_hash: str | None = None,
    summary: str = "",
    skills_section: list[str] | None = None,
    certifications: list[str] | None = None,
    projects: list[str] | None = None,
    parsing_confidence: float = 0.0,
    parsing_notes: list[str] | None = None,
    education: list[dict] | None = None,
    work_experience: list[dict] | None = None,
    skills: list[dict] | None = None,
) -> models.Resume:
    """Persist a fully-parsed resume (i.e. the output of ResumeParserAgent
    + SkillExtractorAgent) in one call.

    `education` items: {"degree", "field", "institution", "graduation_year", "gpa"}
    `work_experience` items: {"title", "company", "duration", "start_date",
                               "end_date", "responsibilities": [...], "technologies": [...]}
    `skills` items: {"name", "category", "proficiency", "confidence", "source"}

    `content_hash` should be `compute_content_hash(extracted_text)` --
    callers (the service layer) are expected to check
    `get_resume_by_content_hash` BEFORE calling this, so this function
    itself does not dedupe; it just persists whatever hash it's given.
    """
    resume = models.Resume(
        candidate_id=candidate_id,
        position_id=position_id,
        file_path=file_path,
        file_type=file_type,
        raw_text=raw_text,
        content_hash=content_hash,
        summary=summary,
        skills_section=skills_section or [],
        certifications=certifications or [],
        projects=projects or [],
        parsing_confidence=parsing_confidence,
        parsing_notes=parsing_notes or [],
    )
    db.add(resume)
    db.flush()

    for edu in education or []:
        db.add(models.ResumeEducation(resume_id=resume.id, **edu))
    for exp in work_experience or []:
        db.add(models.ResumeWorkExperience(resume_id=resume.id, **exp))
    for skill in skills or []:
        db.add(models.CandidateSkill(resume_id=resume.id, **skill))

    db.flush()
    return resume


def list_resumes_for_position(db: Session, position_id: int) -> list[models.Resume]:
    stmt = select(models.Resume).where(models.Resume.position_id == position_id)
    return list(db.scalars(stmt))


def get_resume(db: Session, resume_id: int) -> Optional[models.Resume]:
    return db.get(models.Resume, resume_id)


def get_resume_by_content_hash(
    db: Session, position_id: int, content_hash: str
) -> Optional[models.Resume]:
    """Find a previously-stored resume with identical extracted content
    for this position. This is what lets a re-upload of the same PDF (or
    a byte-different file that extracts to the same text) reuse the
    existing Resume row -- and its already-parsed data/skills -- instead
    of creating a duplicate.

    Scoped to `position_id` because Resume already models "a submission
    for a specific position"; the same person's resume submitted for two
    different positions is still stored as two rows, unchanged from
    today's behavior.
    """
    if not content_hash:
        return None
    stmt = (
        select(models.Resume)
        .where(models.Resume.position_id == position_id)
        .where(models.Resume.content_hash == content_hash)
        .order_by(models.Resume.id.asc())
    )
    return db.scalars(stmt).first()


# ============================================================================
# Screening / Evaluation
# ============================================================================

def create_screening(
    db: Session, resume_id: int, job_description_id: int
) -> models.Screening:
    screening = models.Screening(
        resume_id=resume_id,
        job_description_id=job_description_id,
        status="pending",
    )
    db.add(screening)
    db.flush()
    return screening


def get_or_create_screening(
    db: Session, resume_id: int, job_description_id: int
) -> models.Screening:
    """Like create_screening, but safe to call when a resume is being
    reused (see get_resume_by_content_hash): the (resume_id,
    job_description_id) pair has a UNIQUE constraint
    (uq_screening_resume_jd), so a naive create_screening() would raise
    IntegrityError if a screening for this exact pair already exists --
    e.g. a previous run that started but never completed. The
    already-completed case is handled earlier by the service layer
    (get_completed_screening_for_resume_and_jd short-circuits before we
    get here at all), so reaching this function with an existing row
    means a stale/incomplete screening -- reuse and re-run it rather
    than erroring."""
    existing = get_screening_for_resume_and_jd(db, resume_id, job_description_id)
    if existing is not None:
        return existing
    return create_screening(db, resume_id, job_description_id)


def mark_screening_started(db: Session, screening_id: int) -> None:
    screening = db.get(models.Screening, screening_id)
    if screening:
        screening.status = "running"
        screening.started_at = _now()


def reset_screening_for_rerun(db: Session, screening_id: int) -> None:
    """Clear any partial child rows (skill match, experience eval, result,
    explanation) left over from a previous incomplete attempt at this
    same (resume, JD) screening, so get_or_create_screening's reused row
    can be safely repopulated without hitting the 1:1 unique constraints
    on those child tables. Only ever called on a non-completed screening
    (completed ones are short-circuited earlier and never reach here)."""
    screening = db.get(models.Screening, screening_id)
    if screening is None:
        return
    if screening.skill_match_result is not None:
        db.delete(screening.skill_match_result)
    if screening.experience_evaluation is not None:
        db.delete(screening.experience_evaluation)
    if screening.result is not None:
        db.delete(screening.result)
    if screening.explanation is not None:
        db.delete(screening.explanation)
    db.flush()


def invalidate_completed_screening(db: Session, screening_id: int) -> None:
    """Force ONE completed screening back to a re-runnable state, for
    demo/dev tooling that intentionally wants to re-screen a resume
    against a JD it was already (possibly staleily) screened against --
    e.g. after an agent/scoring change like Batch 6, where
    get_completed_screening_for_resume_and_jd would otherwise keep
    returning the old cached result forever.

    This does NOT touch scoring, matching, or the schema -- it only:
      1. Deletes this screening's own child rows (SkillMatchResult,
         ExperienceEvaluation, ScreeningResult, Explanation) via the
         existing reset_screening_for_rerun cleanup.
      2. Resets status back to "pending" and clears started_at/
         completed_at, so get_completed_screening_for_resume_and_jd no
         longer short-circuits screen_candidate() for this pair.

    Scope is exactly one screening_id -- the Resume row, the
    JobDescription row, every OTHER Screening, and any Ranking are all
    left untouched. Safe to call on a non-completed screening too (it's
    then just a no-op status-wise, same as reset_screening_for_rerun
    alone would be).
    """
    reset_screening_for_rerun(db, screening_id)
    screening = db.get(models.Screening, screening_id)
    if screening is not None:
        screening.status = "pending"
        screening.started_at = None
        screening.completed_at = None
    db.flush()


def save_skill_match_result(
    db: Session,
    screening_id: int,
    *,
    required_skills_met: int,
    required_skills_total: int,
    preferred_skills_met: int,
    preferred_skills_total: int,
    overall_score: float,
    confidence: float,
    reasoning: str,
    matches: list[dict] | None = None,
) -> models.SkillMatchResult:
    """`matches` items: {"requirement", "matched", "matched_skill",
    "match_quality", "confidence", "notes"}"""
    result = models.SkillMatchResult(
        screening_id=screening_id,
        required_skills_met=required_skills_met,
        required_skills_total=required_skills_total,
        preferred_skills_met=preferred_skills_met,
        preferred_skills_total=preferred_skills_total,
        overall_score=overall_score,
        confidence=confidence,
        reasoning=reasoning,
    )
    db.add(result)
    db.flush()

    for match in matches or []:
        db.add(models.SkillMatch(skill_match_result_id=result.id, **match))

    db.flush()
    return result


def save_experience_evaluation(
    db: Session,
    screening_id: int,
    *,
    years_relevant: float,
    years_required: int,
    experience_score: float,
    role_relevance: float,
    career_progression: str,
    gaps_identified: list[str] | None,
    strengths: list[str] | None,
    confidence: float,
    reasoning: str,
) -> models.ExperienceEvaluation:
    evaluation = models.ExperienceEvaluation(
        screening_id=screening_id,
        years_relevant=years_relevant,
        years_required=years_required,
        experience_score=experience_score,
        role_relevance=role_relevance,
        career_progression=career_progression,
        gaps_identified=gaps_identified or [],
        strengths=strengths or [],
        confidence=confidence,
        reasoning=reasoning,
    )
    db.add(evaluation)
    db.flush()
    return evaluation


def save_screening_result(
    db: Session,
    screening_id: int,
    *,
    match_score: float,
    recommendation: str,
    requires_human: bool,
    confidence: float,
    reasoning_summary: str,
    flags: list[str] | None = None,
) -> models.ScreeningResult:
    result = models.ScreeningResult(
        screening_id=screening_id,
        match_score=match_score,
        recommendation=recommendation,
        requires_human=requires_human,
        confidence=confidence,
        reasoning_summary=reasoning_summary,
        flags=flags or [],
    )
    db.add(result)
    db.flush()

    screening = db.get(models.Screening, screening_id)
    if screening:
        screening.status = "completed"
        screening.completed_at = _now()

    return result


def save_explanation(
    db: Session,
    screening_id: int,
    *,
    why_summary: str = "",
    matching_skills: list[str] | None = None,
    evidence_snippets: list[str] | None = None,
    relevant_experience: str = "",
    skill_gaps: list[str] | None = None,
    confidence_label: str = "",
) -> models.Explanation:
    explanation = models.Explanation(
        screening_id=screening_id,
        why_summary=why_summary,
        matching_skills=matching_skills or [],
        evidence_snippets=evidence_snippets or [],
        relevant_experience=relevant_experience,
        skill_gaps=skill_gaps or [],
        confidence_label=confidence_label,
    )
    db.add(explanation)
    db.flush()
    return explanation


def get_screening(db: Session, screening_id: int) -> Optional[models.Screening]:
    return db.get(models.Screening, screening_id)


def list_screenings_for_job_description(
    db: Session, job_description_id: int
) -> list[models.Screening]:
    stmt = select(models.Screening).where(
        models.Screening.job_description_id == job_description_id
    )
    return list(db.scalars(stmt))


def get_screening_for_resume_and_jd(
    db: Session, resume_id: int, job_description_id: int
) -> Optional[models.Screening]:
    """Look up the (at most one, per the uq_screening_resume_jd constraint)
    Screening already run for this exact resume + JD-version pair,
    regardless of status."""
    stmt = (
        select(models.Screening)
        .where(models.Screening.resume_id == resume_id)
        .where(models.Screening.job_description_id == job_description_id)
        .options(
            selectinload(models.Screening.result),
            selectinload(models.Screening.explanation),
            selectinload(models.Screening.skill_match_result),
            selectinload(models.Screening.experience_evaluation),
            selectinload(models.Screening.resume).selectinload(models.Resume.candidate),
        )
    )
    return db.scalar(stmt)


def get_completed_screening_for_resume_and_jd(
    db: Session, resume_id: int, job_description_id: int
) -> Optional[models.Screening]:
    """Same as `get_screening_for_resume_and_jd`, but only returns it if
    the screening actually finished (`status == "completed"`, i.e. it has
    a ScreeningResult). This is the check the service layer uses to
    decide whether a rerun can be answered with 0 LLM calls by just
    returning the stored result."""
    screening = get_screening_for_resume_and_jd(db, resume_id, job_description_id)
    if screening is not None and screening.status == "completed":
        return screening
    return None


# ============================================================================
# Ranking
# ============================================================================

def create_ranking(
    db: Session, job_description_id: int, ranked_screenings: list[tuple[int, float]],
    method: str = "match_score_desc",
) -> models.Ranking:
    """`ranked_screenings` is a list of (screening_id, score) already sorted
    best-first by the caller (the ranking service, not the repository,
    owns the sorting logic -- this function just persists the outcome)."""
    ranking = models.Ranking(job_description_id=job_description_id, method=method)
    db.add(ranking)
    db.flush()

    for position, (screening_id, score) in enumerate(ranked_screenings, start=1):
        db.add(models.RankingEntry(
            ranking_id=ranking.id,
            screening_id=screening_id,
            rank_position=position,
            score=score,
        ))

    db.flush()
    return ranking


def get_latest_ranking(db: Session, job_description_id: int) -> Optional[models.Ranking]:
    stmt = (
        select(models.Ranking)
        .where(models.Ranking.job_description_id == job_description_id)
        .order_by(models.Ranking.generated_at.desc())
    )
    return db.scalars(stmt).first()

def get_ranking_with_details(db: Session, ranking_id: int) -> Optional[models.Ranking]:
    """Like get_latest_ranking, but eager-loads the full chain needed to
    display a ranking (entries -> screening -> resume -> candidate,
    result, explanation) so the returned object is safe to read from
    AFTER the session that fetched it has closed. Use this whenever a
    caller (like screen_position) needs to hand a Ranking back across a
    `with get_session()` boundary."""
    stmt = (
        select(models.Ranking)
        .where(models.Ranking.id == ranking_id)
        .options(
            selectinload(models.Ranking.entries)
            .selectinload(models.RankingEntry.screening)
            .selectinload(models.Screening.resume)
            .selectinload(models.Resume.candidate),
            selectinload(models.Ranking.entries)
            .selectinload(models.RankingEntry.screening)
            .selectinload(models.Screening.result),
            selectinload(models.Ranking.entries)
            .selectinload(models.RankingEntry.screening)
            .selectinload(models.Screening.explanation),
        )
    )
    return db.scalar(stmt)