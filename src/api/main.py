"""FastAPI application entrypoint.

Run locally with:
    uvicorn src.api.main:app --reload

This module wires together:
  - CORS (for the future React frontend)
  - basic, consistent JSON error handling
  - the route modules (health, jobs, screening)

It contains NO screening/parsing/scoring logic itself -- see
src/api/routes/*.py, which delegate entirely to the existing,
unmodified src.services / src.db.repository functions.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.db import init_db

from .routes import health, jobs, screening

logger = logging.getLogger(__name__)

# Comma-separated list of allowed origins, e.g.
#   CORS_ALLOWED_ORIGINS="http://localhost:3000,https://myapp.example.com"
# Defaults cover the standard local React dev server ports so the future
# frontend works out of the box in development.
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def _cors_origins() -> list[str]:
    env_value = os.getenv("CORS_ALLOWED_ORIGINS")
    if env_value:
        return [origin.strip() for origin in env_value.split(",") if origin.strip()]
    return _DEFAULT_CORS_ORIGINS


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Resume Screening API",
        description=(
            "Production API layer over the existing agentic resume "
            "screening pipeline (resume parsing, job analysis, skill "
            "matching, experience evaluation, and ranking)."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Error handling: every error response has the same shape,
    # {"error": "<short code>", "detail": "<human message>"}, so the
    # future React frontend can handle failures generically.
    # ------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "http_error", "detail": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "detail": exc.errors()},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        # The existing service layer (e.g. screen_candidate) raises plain
        # ValueError for caller mistakes ("No current job description for
        # position_id=...") -- surface those as 400s instead of a generic 500.
        return JSONResponse(
            status_code=400,
            content={"error": "bad_request", "detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception while processing %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "detail": "An unexpected error occurred."},
        )

    # ------------------------------------------------------------------
    # Startup: ensure the SQLite schema exists (safe/idempotent -- see
    # src/db/database.py::init_db).
    # ------------------------------------------------------------------

    @app.on_event("startup")
    def _on_startup() -> None:
        init_db()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    app.include_router(health.router)
    app.include_router(jobs.router, prefix="/jobs")
    app.include_router(screening.router, prefix="/screening")

    return app


app = create_app()