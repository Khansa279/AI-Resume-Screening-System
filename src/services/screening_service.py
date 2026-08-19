"""
Service layer connecting the LangGraph screening workflow to persistence.

Two entry points:

    screen_candidate()  -- one resume against one position, fully persisted
    screen_position()   -- many resumes against one position, persisted +
                            ranked (this is the real multi-candidate flow
                            the platform is built around)

Design decisions worth knowing before you extend this:

- Job requirements are parsed ONCE per JobDescription version and reused
  for every candidate (see ensure_job_requirements). Without this, ranking
  50 resumes would re-run JobAnalyzerAgent 50 times against identical text.
- We call workflow.run_full(), not workflow.run(), because we need
  skills_match/experience_eval to persist SkillMatchResult/
  ExperienceEvaluation and to build the Explanation -- run() throws that
  detail away and only returns the synthesized ScreeningOutput.
- Uploaded resume files are copied into RESUME_STORAGE_DIR on disk;
  only the path is stored in the DB (Resume.file_path), per the project's
  "resume files stored separately from the database" requirement.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from ..db import get_session, repository as repo
from ..db import models as db_models
from ..document_parser import parse_document
from ..models import JobRequirements as JobRequirementsModel
from ..workflow import create_screening_workflow
from ..agents.job_analyzer import JobAnalyzerAgent


# Where uploaded resumes get copied to, separate from the DB file itself.
RESUME_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / "resumes"


# ============================================================================
# Job requirements: parse once per JD version, reuse for every candidate
# ============================================================================

async def ensure_job_requirements(
    db: Session, job_description_id: int
) -> db_models.JobRequirements:
    """Return persisted JobRequirements for this JD version. Runs
    JobAnalyzerAgent only if they don't already exist -- this is what
    makes batch screening cheap: one LLM call per JD version, not one
    per resume."""
    existing = repo.get_job_requirements(db, job_description_id)
    if existing is not None:
        return existing

    jd = repo.get_job_description(db, job_description_id)
    if jd is None:
        raise ValueError(f"No JobDescription with id={job_description_id}")

    agent = JobAnalyzerAgent()
    result = await agent.process({"job_description": jd.raw_text})
    parsed: JobRequirementsModel = result["job_requirements"]

    return repo.save_job_requirements(
        db, job_description_id,
        title=parsed.title,
        summary=parsed.summary,
        min_years_experience=parsed.min_years_experience,
        education_requirements=parsed.education_requirements,
        certifications_required=parsed.certifications_required,
        responsibilities=parsed.responsibilities,
        parsing_confidence=parsed.parsing_confidence,
        required_skills=parsed.required_skills,
        preferred_skills=parsed.preferred_skills,
        requirement_items=[r.model_dump() for r in parsed.requirements],
    )


def _job_requirements_to_dict(jr: db_models.JobRequirements) -> dict:
    """Rehydrate the DB row into the shape src.models.JobRequirements
    expects, so it can be passed into workflow.run_full(job_requirements=...)
    and model_validate'd there like any other state value."""
    return {
        "title": jr.title,
        "summary": jr.summary,
        "required_skills": [s.skill_name for s in jr.skills if s.priority == "required"],
        "preferred_skills": [s.skill_name for s in jr.skills if s.priority == "preferred"],
        "min_years_experience": jr.min_years_experience,
        "education_requirements": jr.education_requirements,
        "certifications_required": jr.certifications_required,
        "responsibilities": jr.responsibilities,
        "requirements": [
            {
                "description": i.description,
                "category": i.category,
                "priority": i.priority,
                "years_needed": i.years_needed,
            }
            for i in jr.items
        ],
        "parsing_confidence": jr.parsing_confidence,
    }


# ============================================================================
# Resume file storage
# ============================================================================

def _store_resume_file(source_path: str, position_id: int) -> str:
    """Copy an uploaded resume into RESUME_STORAGE_DIR and return the
    stored path. Filename is randomized to avoid collisions between
    candidates who upload files with the same original name."""
    RESUME_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(source_path)
    dest_name = f"pos{position_id}_{uuid.uuid4().hex[:12]}{src.suffix}"
    dest_path = RESUME_STORAGE_DIR / dest_name
    shutil.copy2(src, dest_path)
    return str(dest_path)


# ============================================================================
# Explanation ("5 Ws"): derived from agent output already produced,
# no extra LLM call needed
# ============================================================================

def _build_explanation(skills_match, experience_eval, final_output) -> dict:
    matching_skills = [
        m.matched_skill for m in skills_match.matches
        if m.matched and m.matched_skill
    ]
    skill_gaps = [
        m.requirement for m in skills_match.matches
        if not m.matched
    ]

    confidence_label = (
        "High" if final_output.confidence >= 0.75
        else "Medium" if final_output.confidence >= 0.5
        else "Low"
    )

    return {
        "why_summary": final_output.reasoning_summary,
        "matching_skills": matching_skills,
        "evidence_snippets": [],
        "relevant_experience": (
            f"{experience_eval.years_relevant:.1f} years relevant "
            f"(role relevance {experience_eval.role_relevance:.0%})"
        ),
        "skill_gaps": skill_gaps,
        "confidence_label": confidence_label,
    }


# ============================================================================
# Single-candidate screening
# ============================================================================

async def screen_candidate(
    db: Session,
    *,
    position_id: int,
    resume_file_path: str,
) -> db_models.Screening:
    """
    Run one resume through the real agent workflow against a position's
    current job description, and persist every stage of the result:
    Candidate, Resume, Screening, SkillMatchResult, ExperienceEvaluation,
    ScreeningResult, Explanation.

    Args:
        db: an open Session (caller owns commit/rollback via get_session())
        position_id: which Position this candidate is being screened for
        resume_file_path: path to the uploaded resume file (PDF/DOCX/TXT)

    Returns:
        The persisted Screening row (with .result and .explanation loaded)
    """
    jd = repo.get_current_job_description(db, position_id)
    if jd is None:
        raise ValueError(f"No current job description for position_id={position_id}")

    jr_row = await ensure_job_requirements(db, jd.id)
    job_requirements_dict = _job_requirements_to_dict(jr_row)

    # Parse the document ONCE here; pass raw text into the workflow so its
    # own _parse_document_node has nothing to do (resume_path left empty).
    parsed_doc = parse_document(resume_file_path)
    if not parsed_doc.success:
        raise ValueError(f"Could not parse resume: {parsed_doc.error_message}")

    workflow = create_screening_workflow()
    final_state = await workflow.run_full(
        resume_text=parsed_doc.text,
        job_description=jd.raw_text,
        job_requirements=job_requirements_dict,
    )

    resume_data = final_state.get("resume_data")
    skills_match = final_state.get("skills_match")
    experience_eval = final_state.get("experience_eval")
    final_output = final_state.get("final_output")

    if final_output is None:
        raise RuntimeError(
            f"Workflow produced no final_output for resume={resume_file_path}. "
            f"errors={final_state.get('errors')}"
        )

    # Convert to pydantic models where the graph returned raw dicts.
    from ..models import ResumeData, SkillsMatchResult, ExperienceEvaluation, ScreeningOutput
    if isinstance(resume_data, dict):
        resume_data = ResumeData.model_validate(resume_data)
    if isinstance(skills_match, dict):
        skills_match = SkillsMatchResult.model_validate(skills_match)
    if isinstance(experience_eval, dict):
        experience_eval = ExperienceEvaluation.model_validate(experience_eval)
    if isinstance(final_output, dict):
        final_output = ScreeningOutput.model_validate(final_output)

    # ---- Candidate + Resume -------------------------------------------
    contact = resume_data.contact if resume_data else None
    candidate = repo.get_or_create_candidate(
        db,
        name=contact.name if contact else "",
        email=contact.email if contact else "",
        phone=contact.phone if contact else "",
        location=contact.location if contact else "",
        linkedin=contact.linkedin if contact else "",
        github=contact.github if contact else "",
    )

    stored_path = _store_resume_file(resume_file_path, position_id)

    resume_row = repo.create_resume(
        db,
        candidate_id=candidate.id,
        position_id=position_id,
        file_path=stored_path,
        file_type=parsed_doc.file_type,
        raw_text=resume_data.raw_text if resume_data else parsed_doc.text,
        summary=resume_data.summary if resume_data else "",
        skills_section=resume_data.skills_section if resume_data else [],
        certifications=resume_data.certifications if resume_data else [],
        projects=resume_data.projects if resume_data else [],
        parsing_confidence=resume_data.parsing_confidence if resume_data else 0.0,
        parsing_notes=resume_data.parsing_notes if resume_data else [],
        education=[e.model_dump() for e in resume_data.education] if resume_data else [],
        work_experience=[w.model_dump() for w in resume_data.work_experience] if resume_data else [],
        skills=[
            {
                "name": s.name, "category": s.category, "proficiency": s.proficiency,
                "confidence": s.confidence, "source": s.source,
            }
            for s in (final_state.get("extracted_skills") or [])
        ],
    )

    # ---- Screening + results -------------------------------------------
    screening = repo.create_screening(db, resume_row.id, jd.id)
    repo.mark_screening_started(db, screening.id)

    if skills_match:
        repo.save_skill_match_result(
            db, screening.id,
            required_skills_met=skills_match.required_skills_met,
            required_skills_total=skills_match.required_skills_total,
            preferred_skills_met=skills_match.preferred_skills_met,
            preferred_skills_total=skills_match.preferred_skills_total,
            overall_score=skills_match.overall_score,
            confidence=skills_match.confidence,
            reasoning=skills_match.reasoning,
            matches=[m.model_dump() for m in skills_match.matches],
        )

    if experience_eval:
        repo.save_experience_evaluation(
            db, screening.id,
            years_relevant=experience_eval.years_relevant,
            years_required=experience_eval.years_required,
            experience_score=experience_eval.experience_score,
            role_relevance=experience_eval.role_relevance,
            career_progression=experience_eval.career_progression,
            gaps_identified=experience_eval.gaps_identified,
            strengths=experience_eval.strengths,
            confidence=experience_eval.confidence,
            reasoning=experience_eval.reasoning,
        )

    repo.save_screening_result(
        db, screening.id,
        match_score=final_output.match_score,
        recommendation=final_output.recommendation,
        requires_human=final_output.requires_human,
        confidence=final_output.confidence,
        reasoning_summary=final_output.reasoning_summary,
        flags=final_output.flags,
    )

    if skills_match and experience_eval:
        explanation = _build_explanation(skills_match, experience_eval, final_output)
        repo.save_explanation(db, screening.id, **explanation)

    return screening


# ============================================================================
# Batch screening + ranking (the real multi-candidate flow)
# ============================================================================

async def screen_position(
    position_id: int,
    resume_file_paths: list[str],
) -> db_models.Ranking:
    """
    Screen MULTIPLE resumes against one position and produce a ranking.
    Opens its own session (one commit for the whole batch) since this is
    meant to be called as a single unit of work from a route/script --
    call screen_candidate() directly instead if you need per-resume
    transaction boundaries.

    Args:
        position_id: the Position being screened for
        resume_file_paths: local file paths of uploaded resumes

    Returns:
        The persisted Ranking row (with .entries loaded, best-first)
    """
    with get_session() as db:
        scored: list[tuple[int, float]] = []
        errors: list[str] = []

        for path in resume_file_paths:
            try:
                screening = await screen_candidate(
                    db, position_id=position_id, resume_file_path=path
                )
                # screening.result was just created in this same session/flush
                db.flush()
                scored.append((screening.id, screening.result.match_score))
            except Exception as e:
                errors.append(f"{path}: {e}")

        if not scored:
            raise RuntimeError(f"No resumes screened successfully. Errors: {errors}")

        scored.sort(key=lambda pair: pair[1], reverse=True)

        jd = repo.get_current_job_description(db, position_id)
        ranking = repo.create_ranking(db, jd.id, scored)
        db.flush()

        if errors:
            print(f"screen_position: {len(errors)} resume(s) failed: {errors}")

        # Re-fetch with everything eager-loaded so the object is safe to
        # read from after this `with get_session()` block exits and the
        # session closes (plain `ranking` would raise DetachedInstanceError
        # on any lazy-loaded attribute access, e.g. ranking.entries).
        return repo.get_ranking_with_details(db, ranking.id)