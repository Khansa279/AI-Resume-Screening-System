"""Job description routes.

Reuses src/db/repository.py's existing get-or-create helpers (the same
ones src/scripts/demo_screening.py and src/scripts/test_real_data.py
already use) -- this module adds no new persistence or parsing logic of
its own. Job descriptions submitted through the API are stored under a
dedicated organization/department bucket so they're clearly identifiable
apart from data seeded by the CLI scripts, without needing any schema
change.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.db import get_session, repository as repo

from ..schemas import JobCreateResponse, JobSummary

router = APIRouter(tags=["jobs"])

_API_ORG_NAME = "API Submitted Jobs"
_API_DEPT_NAME = "General"

_MAX_JD_FILE_BYTES = 2 * 1024 * 1024  # 2 MB is generous for a text JD


def _resolve_api_container(db):
    """Get-or-create the organization/department bucket used for jobs
    submitted via this API. Mirrors the placeholder-container pattern
    already established in src/scripts/demo_screening.py."""
    org = repo.get_organization_by_name(db, _API_ORG_NAME)
    if org is None:
        org = repo.create_organization(db, _API_ORG_NAME)
    dept = repo.get_department_by_name(db, org.id, _API_DEPT_NAME)
    if dept is None:
        dept = repo.create_department(db, org.id, _API_DEPT_NAME)
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
    jd_text: Optional[str] = Form(None, description="Raw job description text"),
    jd_file: Optional[UploadFile] = File(None, description="Job description as a .txt file"),
) -> JobCreateResponse:
    """Create (or version-update) a job description for a position.

    Exactly one of jd_text / jd_file must be provided. If a position with
    this title already exists under the API's job bucket, a NEW JD
    version is created for it (see repository.create_job_description) --
    the position itself is not duplicated.
    """
    text = await _read_jd_text(jd_text, jd_file)

    with get_session() as db:
        org, dept = _resolve_api_container(db)
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