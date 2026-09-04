"""Pydantic request/response models for the FastAPI layer.

These are presentation-layer shapes only -- they mirror fields already
produced by src/db/models.py and src/services/screening_service.py, they
do not introduce any new scoring or parsing concepts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ============================================================================
# Health
# ============================================================================

class HealthResponse(BaseModel):
    status: str = "ok"


# ============================================================================
# Job descriptions
# ============================================================================

class JobCreateResponse(BaseModel):
    position_id: int
    organization: str
    department: str
    title: str
    jd_version: int


class JobSummary(BaseModel):
    position_id: int
    title: str
    organization: str
    department: str
    current_jd_version: Optional[int] = None


class JobHistoryEntry(BaseModel):
    """One row in the screening-history list -- one Position, with its
    current organization/department context and (if any screening has
    ever been run for it) a summary of the most recent ranking. Built
    entirely from existing repository reads; introduces no new scoring
    or persistence concepts."""

    position_id: int
    title: str
    organization: str
    department: str
    jd_version: Optional[int] = None
    status: str = "draft"  # draft | screened
    ranking_id: Optional[int] = None
    candidates_screened: int = 0
    generated_at: Optional[datetime] = None
    top_candidate_name: Optional[str] = None
    top_candidate_score: Optional[float] = None


# ============================================================================
# Screening / candidate results
# ============================================================================

class ScreeningCandidateResult(BaseModel):
    screening_id: int
    resume_id: int
    candidate_name: str
    match_score: float
    recommendation: str
    requires_human: bool
    confidence: float
    error: Optional[str] = None
    matching_skills: list[str] = Field(default_factory=list)
    skill_gaps: list[str] = Field(default_factory=list)
    explanation: Optional[str] = None
    candidate_email: Optional[str] = None
    candidate_phone: Optional[str] = None


class ScreeningResponse(BaseModel):
    position_id: int
    ranking_id: Optional[int] = None
    candidates_screened: int = 0
    results: list[ScreeningCandidateResult] = Field(default_factory=list)
    # Organization/position context so the frontend never has to guess
    # or re-derive which role a set of results belongs to (see
    # JobSummary -- same shape, just carried alongside the results here
    # so a single request gives the frontend everything it needs to
    # render both the header and the ranked list).
    title: Optional[str] = None
    organization: Optional[str] = None
    department: Optional[str] = None
    generated_at: Optional[datetime] = None


# ============================================================================
# Errors
# ============================================================================

class ErrorResponse(BaseModel):
    error: str
    detail: str
