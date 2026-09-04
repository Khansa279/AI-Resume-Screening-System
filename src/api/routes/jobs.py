"""Job description routes.

Reuses src/db/repository.py's existing get-or-create helpers (the same
ones src/scripts/demo_screening.py and src/scripts/test_real_data.py
already use) -- this module adds no new persistence or parsing logic of
its own beyond an optional caller-supplied organization/department name
(previously hardcoded to a single fixed bucket) and a read-only history
listing endpoint, both purely additive and backward compatible: existing
callers that don't pass `organization`/`department` behave exactly as
before.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.db import get_session, repository as repo

from ..schemas import JobCreateResponse, JobHistoryEntry, JobSummary

router = APIRouter(tags=["jobs"])

_API_ORG_NAME = "API Submitted Jobs"
_API_DEPT_NAME = "General"

_MAX_JD_FILE_BYTES = 2 * 1024 * 1024  # 2 MB is generous for a text JD


def _resolve_api_container(db, organization: Optional[str], department: Optional[str]):
    """Get-or-create the organization/department a job is filed under.

    Backward compatible: when the caller doesn't supply an organization
    name, this falls back to the same fixed placeholder bucket the API
    has always used. When a real organization name is supplied (e.g.
    from the "Organization" field on the screening-setup UI), it is
    get-or-created for real, so results and history can show genuine
    context ("CureMD" / "AI/ML Engineer") instead of a placeholder.
    """
    org_name = (organization or "").strip() or _API_ORG_NAME
    dept_name = (department or "").strip() or _API_DEPT_NAME

    org = repo.get_organization_by_name(db, org_name)
    if org is None:
        org = repo.create_organization(db, org_name)
    dept = repo.get_department_by_name(db, org.id, dept_name)
    if dept is None:
        dept = repo.create_department(db, org.id, dept_name)
    return org, dept


async def _read_jd_text(jd_text: Optional[str], jd_file: Optional[UploadFile]) -> str:
    if jd_file is not None:
        if jd_file.filename and not jd_file.filename.lower().endswith(".txt"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported job description file type: {jd_file.filename!r}. Only .txt is accepted.",
            )
        raw = await jd_file.read()
        if len(raw) > _MAX_JD_FILE_BYTES:
            raise HTTPException(status_code=400, detail="Job description file is too large.")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    elif jd_text is not None:
        text = jd_text
    else:
        raise HTTPException(status_code=400, detail="Provide either jd_text or jd_file.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Job description text is empty.")
    return text


@router.post("", response_model=JobCreateResponse)
async def create_job(
    title: str = Form(..., description="Job title, e.g. 'Backend Engineer - Python'"),
    organization: Optional[str] = Form(None, description="Organization this role belongs to"),
    department: Optional[str] = Form(None, description="Department/team within the organization"),
    jd_text: Optional[str] = Form(None, description="Raw job description text"),
    jd_file: Optional[UploadFile] = File(None, description="Job description as a .txt file"),
) -> JobCreateResponse:
    """Create (or version-update) a job description for a position.

    Exactly one of jd_text / jd_file must be provided. If a position with
    this title already exists under the given organization/department,
    a NEW JD version is created for it (see repository.create_job_description)
    -- the position itself is not duplicated.
    """
    text = await _read_jd_text(jd_text, jd_file)

    with get_session() as db:
        org, dept = _resolve_api_container(db, organization, department)
        position = repo.get_position_by_title(db, dept.id, title)
        if position is None:
            position = repo.create_position(db, dept.id, title)

        jd = repo.create_job_description(db, position.id, raw_text=text)

        return JobCreateResponse(
            position_id=position.id,
            organization=org.name,
            department=dept.name,
            title=position.title,
            jd_version=jd.version,
        )


@router.get("", response_model=list[JobHistoryEntry])
def list_jobs() -> list[JobHistoryEntry]:
    """List every position that has ever been created via this API,
    newest job description first, each annotated with its organization/
    department context and (if any screening has ever completed for its
    current JD version) a summary of the most recent ranking.

    Purely a read-side aggregation over existing repository functions --
    it does not add any new persistence, scoring, or matching logic.
    Powers the frontend's "Screening History" view.
    """
    entries: list[JobHistoryEntry] = []
    with get_session() as db:
        for org in repo.list_organizations(db):
            for dept in repo.list_departments(db, org.id):
                for position in repo.list_positions(db, dept.id):
                    jd = repo.get_current_job_description(db, position.id)

                    ranking = None
                    if jd is not None:
                        ranking = repo.get_latest_ranking(db, jd.id)

                    top_name = None
                    top_score = None
                    candidates_screened = 0
                    generated_at = None
                    ranking_id = None

                    if ranking is not None:
                        ranking_id = ranking.id
                        generated_at = ranking.generated_at
                        sorted_entries = sorted(ranking.entries, key=lambda e: e.rank_position)
                        candidates_screened = len(sorted_entries)
                        if sorted_entries:
                            top_entry = sorted_entries[0]
                            top_score = top_entry.score
                            screening = top_entry.screening
                            candidate = (
                                screening.resume.candidate
                                if screening and screening.resume else None
                            )
                            top_name = candidate.name if candidate and candidate.name else None

                    entries.append(JobHistoryEntry(
                        position_id=position.id,
                        title=position.title,
                        organization=org.name,
                        department=dept.name,
                        jd_version=jd.version if jd else None,
                        status="screened" if ranking is not None else "draft",
                        ranking_id=ranking_id,
                        candidates_screened=candidates_screened,
                        generated_at=generated_at,
                        top_candidate_name=top_name,
                        top_candidate_score=top_score,
                    ))

    entries.sort(key=lambda e: (e.generated_at is None, e.generated_at), reverse=False)
    entries.sort(key=lambda e: e.generated_at or e.position_id, reverse=True)
    return entries


@router.get("/{position_id}", response_model=JobSummary)
def get_job(position_id: int) -> JobSummary:
    with get_session() as db:
        position = repo.get_position(db, position_id)
        if position is None:
            raise HTTPException(status_code=404, detail=f"No position with id={position_id}")

        dept = position.department
        org = dept.organization if dept else None
        jd = repo.get_current_job_description(db, position_id)

        return JobSummary(
            position_id=position.id,
            title=position.title,
            organization=org.name if org else "",
            department=dept.name if dept else "",
            current_jd_version=jd.version if jd else None,
        )
