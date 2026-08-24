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
# Resume rehydration: turn a previously-stored Resume row back into the
# dict shapes src.models.ResumeData / src.models.Skill expect, so a resume
# we've already parsed once can be fed straight into
# workflow.run_full(resume_data=..., extracted_skills=...) and skip
# ResumeParserAgent + SkillExtractorAgent entirely on a rerun against a
# different (or new-version) JD.
# ============================================================================

def _resume_row_to_resume_data_dict(resume: db_models.Resume) -> dict:
    candidate = resume.candidate
    return {
        "contact": {
            "name": candidate.name if candidate else "",
            "email": candidate.email if candidate else "",
            "phone": candidate.phone if candidate else "",
            "location": candidate.location if candidate else "",
            "linkedin": candidate.linkedin if candidate else "",
            "github": candidate.github if candidate else "",
        },
        "summary": resume.summary,
        "education": [
            {
                "degree": e.degree, "field": e.field, "institution": e.institution,
                "graduation_year": e.graduation_year, "gpa": e.gpa,
            }
            for e in resume.education
        ],
        "work_experience": [
            {
                "title": w.title, "company": w.company, "duration": w.duration,
                "start_date": w.start_date, "end_date": w.end_date,
                "responsibilities": w.responsibilities or [],
                "technologies": w.technologies or [],
            }
            for w in resume.work_experience
        ],
        "skills_section": resume.skills_section or [],
        "certifications": resume.certifications or [],
        "projects": resume.projects or [],
        "raw_text": resume.raw_text or "",
        "parsing_confidence": resume.parsing_confidence,
        "parsing_notes": resume.parsing_notes or [],
    }


def _resume_row_to_skills_dicts(resume: db_models.Resume) -> list[dict]:
    return [
        {
            "name": s.name, "category": s.category, "proficiency": s.proficiency,
            "confidence": s.confidence, "source": s.source,
        }
        for s in resume.skills
    ]


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
        if not m.matched and m.soft_skill_status != "not_mentioned"
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

    Content-based reuse (this is what keeps repeated runs off the Groq
    free-tier limits):

      1. The resume's TEXT is hashed (SHA-256, see
         repo.compute_content_hash) as soon as it's extracted -- before
         any agent or LLM call runs.
      2. If a Resume with that exact hash already exists for this
         position AND already has a *completed* Screening against this
         exact JobDescription version, that Screening is returned as-is.
         Zero LLM calls, zero new DB rows -- this is the "rerun the same
         resume against the same JD" case from the bug report.
      3. If a Resume with that hash exists but hasn't been screened
         against this JD version yet, the existing Resume row (and its
         already-parsed data/skills) is reused instead of creating a
         duplicate Resume -- ResumeParserAgent and SkillExtractorAgent
         are skipped (see workflow.run_full's resume_data/extracted_skills
         params), and only JobAnalyzer (already cached separately),
         SkillsMatcher, ExperienceEvaluator, and DecisionSynthesizer run.
      4. Only truly new resume content goes through the full pipeline.

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

    # Parse the document ONCE here, up front -- this is also where the
    # content hash comes from, so we know whether ANY further work
    # (including the LLM-backed workflow) is even necessary before doing
    # any of it.
    parsed_doc = parse_document(resume_file_path)
    if not parsed_doc.success:
        raise ValueError(f"Could not parse resume: {parsed_doc.error_message}")

    content_hash = repo.compute_content_hash(parsed_doc.text)
    existing_resume = repo.get_resume_by_content_hash(db, position_id, content_hash)

    if existing_resume is not None:
        existing_screening = repo.get_completed_screening_for_resume_and_jd(
            db, existing_resume.id, jd.id
        )
        if existing_screening is not None:
            print(
                f"screen_candidate: resume content already screened "
                f"(resume_id={existing_resume.id}, content_hash={content_hash[:12]}..., "
                f"screening_id={existing_screening.id}) -- reusing stored result, "
                f"0 LLM calls."
            )
            return existing_screening

    jr_row = await ensure_job_requirements(db, jd.id)
    job_requirements_dict = _job_requirements_to_dict(jr_row)

    # If we've already parsed this exact resume content before (just not
    # against this JD version), reuse that parsed data/skills so the
    # workflow can skip ResumeParserAgent + SkillExtractorAgent.
    reused_resume_data_dict: dict | None = None
    reused_skills_dicts: list[dict] | None = None
    if existing_resume is not None:
        reused_resume_data_dict = _resume_row_to_resume_data_dict(existing_resume)
        reused_skills_dicts = _resume_row_to_skills_dicts(existing_resume)
        print(
            f"screen_candidate: reusing existing resume_id={existing_resume.id} "
            f"(content_hash={content_hash[:12]}...) for a new job description -- "
            f"skipping resume parsing + skill extraction."
        )

    workflow = create_screening_workflow()
    final_state = await workflow.run_full(
        resume_text=parsed_doc.text,
        job_description=jd.raw_text,
        job_requirements=job_requirements_dict,
        resume_data=reused_resume_data_dict,
        extracted_skills=reused_skills_dicts,
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
    if existing_resume is not None:
        # Same resume content we've seen before for this position --
        # reuse the row instead of creating a duplicate. Nothing to
        # insert here; existing_resume/its candidate are already
        # persisted.
        candidate = existing_resume.candidate
        resume_row = existing_resume
    else:
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
            content_hash=content_hash,
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
    # get_or_create_screening (not create_screening) because resume_row
    # may be a REUSED resume: if a previous screening attempt for this
    # exact (resume, JD) pair started but never completed (e.g. crashed
    # mid-run), a plain create would violate the uq_screening_resume_jd
    # unique constraint. The already-completed case was already handled
    # by the early-return at the top of this function, so if we land here
    # with an existing row it's a stale/incomplete one -- clear its
    # partial child rows and let this run repopulate them.
    screening = repo.get_or_create_screening(db, resume_row.id, jd.id)
    if screening.status != "pending":
        repo.reset_screening_for_rerun(db, screening.id)
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

    # `screening` may have been loaded earlier in this same function call
    # (via get_or_create_screening -> get_screening_for_resume_and_jd,
    # which eager-loads .result/.explanation/.experience_evaluation) at a
    # point when those relationships were genuinely empty -- e.g. a
    # reused Screening row that had just been invalidated/reset (see
    # repository.reset_screening_for_rerun / invalidate_completed_screening)
    # and is being repopulated in this very call. The save_* calls above
    # insert new child rows by screening_id directly; they don't update
    # this already-loaded Python object's relationship attributes, so
    # without this, the caller would read back stale (empty/None)
    # relationships despite the real rows already being flushed. Expiring
    # forces the next attribute access to re-read from the DB within the
    # same transaction, seeing the rows just flushed above. This is a
    # session-freshness fix only -- it does not change what gets
    # computed or persisted.
    db.expire(screening)

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