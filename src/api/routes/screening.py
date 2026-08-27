"""Screening routes.

This module does NOT implement any parsing, matching, scoring, or
ranking logic. It only:
  - accepts uploaded resume file(s) over HTTP and writes them to a
    temporary directory (screen_candidate/screen_position already copy
    whatever path they're given into permanent storage -- see
    RESUME_STORAGE_DIR in src/services/screening_service.py -- so this
    temp copy only needs to live long enough for that call to run)
  - calls the existing, unmodified src.services.screen_position /
    screen_candidate functions
  - serializes the already-computed ScreeningResult/Ranking rows for
    the HTTP response

Reuses src/db/repository.py for read-only lookups (position, current JD,
latest ranking) the same way src/scripts/demo_screening.py and
src/scripts/test_real_data.py already do.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.db import get_session, repository as repo
from src.db import models as db_models
from src.services import screen_position

from ..schemas import ScreeningCandidateResult, ScreeningResponse

router = APIRouter(tags=["screening"])

_ALLOWED_RESUME_EXTS = {".pdf", ".docx", ".doc", ".txt"}


def _require_position_with_jd(position_id: int) -> None:
    """Read-only validation reusing existing repository lookups -- raises
    the appropriate HTTP error instead of letting screen_position() fail
    deep inside the pipeline with a less helpful message."""
    with get_session() as db:
        position = repo.get_position(db, position_id)
        if position is None:
            raise HTTPException(status_code=404, detail=f"No position with id={position_id}")
        jd = repo.get_current_job_description(db, position_id)
        if jd is None:
            raise HTTPException(
                status_code=400,
                detail=f"Position {position_id} has no job description yet. "
                       f"Create one first via POST /jobs.",
            )


def _ranking_to_response(ranking: db_models.Ranking, position_id: int) -> ScreeningResponse:
    results: list[ScreeningCandidateResult] = []
    for entry in sorted(ranking.entries, key=lambda e: e.rank_position):
        screening = entry.screening
        candidate = screening.resume.candidate if screening.resume else None
        result = screening.result
        results.append(ScreeningCandidateResult(
            screening_id=screening.id,
            resume_id=screening.resume_id,
            candidate_name=(candidate.name if candidate and candidate.name else "Unknown"),
            match_score=result.match_score,
            recommendation=result.recommendation,
            requires_human=result.requires_human,
            confidence=result.confidence,
        ))
    return ScreeningResponse(
        position_id=position_id,
        ranking_id=ranking.id,
        candidates_screened=len(results),
        results=results,
    )


@router.post("/{position_id}/run", response_model=ScreeningResponse)
async def run_screening(position_id: int, resumes: list[UploadFile] = File(...)) -> ScreeningResponse:
    """Screen one or more uploaded resumes against a position's current
    job description and return the ranked results.

    All actual screening work (document parsing, skill extraction, job
    analysis, matching, experience evaluation, decision synthesis,
    persistence, and ranking) is delegated entirely to
    src.services.screen_position -- unmodified.
    """
    _require_position_with_jd(position_id)

    if not resumes:
        raise HTTPException(status_code=400, detail="Provide at least one resume file.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="resume_upload_"))
    try:
        resume_paths: list[str] = []
        for upload in resumes:
            filename = upload.filename or "resume"
            ext = Path(filename).suffix.lower()
            if ext not in _ALLOWED_RESUME_EXTS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported resume file type for {filename!r}: {ext or '(none)'}. "
                           f"Allowed: {sorted(_ALLOWED_RESUME_EXTS)}",
                )
            dest = tmp_dir / filename
            content = await upload.read()
            dest.write_bytes(content)
            resume_paths.append(str(dest))

        try:
            ranking = await screen_position(position_id=position_id, resume_file_paths=resume_paths)
        except RuntimeError as e:
            # screen_position raises RuntimeError only when EVERY resume in
            # the batch failed to screen (see its docstring/implementation) --
            # surface that as a 422 (valid request, unprocessable content)
            # rather than a generic 500.
            raise HTTPException(status_code=422, detail=str(e))

        return _ranking_to_response(ranking, position_id)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/{position_id}/results", response_model=ScreeningResponse)
def get_results(position_id: int) -> ScreeningResponse:
    """Return the most recent ranking already computed for this position,
    without re-running anything."""
    with get_session() as db:
        position = repo.get_position(db, position_id)
        if position is None:
            raise HTTPException(status_code=404, detail=f"No position with id={position_id}")

        jd = repo.get_current_job_description(db, position_id)
        if jd is None:
            raise HTTPException(status_code=404, detail=f"No job description for position_id={position_id}")

        latest = repo.get_latest_ranking(db, jd.id)
        if latest is None:
            raise HTTPException(status_code=404, detail="No screening results yet for this position.")

        ranking = repo.get_ranking_with_details(db, latest.id)
        return _ranking_to_response(ranking, position_id)