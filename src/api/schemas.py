"""Pydantic request/response models for the FastAPI layer.

These are presentation-layer shapes only -- they mirror fields already
produced by src/db/models.py and src/services/screening_service.py, they
do not introduce any new scoring or parsing concepts.
"""

from __future__ import annotations

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
    # Already computed and persisted by screening_service._build_explanation
    # / repository.save_explanation -- previously never exposed over the
    # API, even though frontend/src/components/CandidateCard.jsx already
    # has UI for exactly these fields (matchedSkills, skillGaps,
    # explanation) and silently renders its "unavailable" fallback state
    # for all of them. No new computation here, just exposing existing data.
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


# ============================================================================
# Errors
# ============================================================================

class ErrorResponse(BaseModel):
    error: str
    detail: str