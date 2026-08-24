#!/usr/bin/env python3
"""
Supervisor-facing demonstration script.

Presents a numbered menu of the job descriptions and resumes already
present under sample_data/, runs them through the EXISTING screening
pipeline (src/services/screening_service.py::screen_candidate --
unmodified), and prints a clean, non-technical report.

This script does NOT implement any screening/matching/scoring logic of
its own. It only:
  - discovers files on disk and derives display labels for them
  - resolves (or creates a bare-bones placeholder) Organization /
    Department / Position container so screen_candidate() has a
    position_id to screen against, reusing repository.py functions
  - calls the real, unmodified screen_candidate()
  - formats already-computed results (ScreeningResult, Explanation,
    ExperienceEvaluation) for a non-technical audience

No IDs, file paths, hashes, or cache/reuse messages are printed here.

Run from the project root:
    python src/scripts/demo_screening.py
"""

import asyncio
import contextlib
import io
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.db import init_db, get_session, repository as repo
from src.services import screen_candidate
from src.agents.resume_parser import parse_resume_text
from src.document_parser import parse_document

# PyPDF2 logs routine, non-fatal recovery notices (e.g. "incorrect
# startxref pointer(1)") at WARNING level via logging.getLogger(__name__)
# in its own submodules (PyPDF2._reader, PyPDF2.generic._data_structures,
# etc.), which are children of the "PyPDF2" logger and inherit its level.
# Raising just this logger to ERROR hides that specific noise without
# touching Python's root logger or any other library's messages -- actual
# parsing failures still raise a real exception (see parse_document's
# success/error_message), so nothing here can hide a genuine error.
logging.getLogger("PyPDF2").setLevel(logging.ERROR)

JD_DIR = Path("sample_data/job_descriptions")
RESUME_DIR = Path("sample_data/resumes")

# Internal-only bucket used when a job description has no known
# organization/department in the database yet. Never shown to the user --
# see _display_org_dept below, which prints "Not specified" instead.
_PLACEHOLDER_ORG = "Demo Screening Session"
_PLACEHOLDER_DEPT = "General"

_SUPPORTED_RESUME_EXTS = {".pdf", ".docx", ".doc", ".txt"}


# ============================================================================
# Discovery + label derivation (no hardcoded names -- everything below is
# derived from the actual file on disk)
# ============================================================================

def _discover_job_descriptions() -> list[Path]:
    if not JD_DIR.exists():
        return []
    return sorted(p for p in JD_DIR.iterdir() if p.suffix == ".txt")


def _discover_resumes() -> list[Path]:
    if not RESUME_DIR.exists():
        return []
    return sorted(p for p in RESUME_DIR.iterdir() if p.suffix.lower() in _SUPPORTED_RESUME_EXTS)


def _extract_job_title(jd_text: str, fallback_filename: str) -> str:
    """Best-effort job title straight from the JD text -- checks the
    common 'POSITION: ...' field format used by this repo's sample JDs,
    then falls back to a short first line, then to the filename."""
    m = re.search(r"^\s*POSITION\s*:\s*(.+)$", jd_text, re.MULTILINE | re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        if title:
            return title

    for line in jd_text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) < 80 and not stripped.startswith("="):
            return stripped

    return Path(fallback_filename).stem.replace("_", " ").title()


def _extract_experience_label(jd_text: str, min_years_experience: int | None) -> str:
    """Best-effort human-readable experience range, preferring the JD's
    own 'EXPERIENCE: ...' field (when present) over the single number
    JobAnalyzerAgent extracted, since the JD's own wording (e.g. '0-1
    years') is more informative than a single min_years integer."""
    m = re.search(r"^\s*EXPERIENCE\s*:\s*(.+)$", jd_text, re.MULTILINE | re.IGNORECASE)
    if m:
        label = m.group(1).strip()
        if label:
            return label
    if min_years_experience is not None:
        return f"{min_years_experience}+ years"
    return "Not specified"


def _display_name(name: str) -> str:
    """Cosmetic-only casing fix for a display label. Resume headers are
    often ALL CAPS ("SARAH CHEN"); this never touches the underlying
    parsed data used by the pipeline itself, only what's printed."""
    name = (name or "").strip()
    return name.title() if name and name.isupper() else name


def _derive_candidate_name(resume_path: Path) -> str:
    """Deterministic name derivation -- reuses the existing rule-based
    resume parser (src/agents/resume_parser.py::parse_resume_text, no LLM
    call) to read the candidate's name straight off the resume. Falls
    back to a prettified filename only if the parser can't find one."""
    parsed = parse_document(str(resume_path))
    if parsed.success and parsed.text:
        resume_data = parse_resume_text(parsed.text)
        if resume_data.contact.name:
            return _display_name(resume_data.contact.name)

    stem = resume_path.stem
    stem = re.sub(r"^resume[_\-]?\d*[_\-]?", "", stem, flags=re.IGNORECASE)
    return stem.replace("_", " ").replace("-", " ").strip().title() or resume_path.stem


# ============================================================================
# Menu / input handling
# ============================================================================

def _print_header(title: str) -> None:
    print("=" * 60)
    print(title.center(60))
    print("=" * 60)


def _prompt_job_description(jd_files: list[Path]) -> Path:
    print()
    print("Select a Job Description:")
    print()
    labels = []
    for i, path in enumerate(jd_files, start=1):
        text = path.read_text(encoding="utf-8")
        title = _extract_job_title(text, path.name)
        labels.append(title)
        print(f"  {i}. {title}")
    print()

    while True:
        choice = input(f"Enter choice [1-{len(jd_files)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(jd_files):
            return jd_files[int(choice) - 1]
        print(f"Invalid choice. Please enter a number between 1 and {len(jd_files)}.")


def _prompt_resumes(resume_files: list[Path]) -> list[Path]:
    print()
    print("Select resumes to screen:")
    print()
    names = [_derive_candidate_name(p) for p in resume_files]
    for i, name in enumerate(names, start=1):
        print(f"  {i}. {name}")
    print()

    while True:
        raw = input("Enter resume numbers (e.g. 5,6) or 'all': ").strip()
        if raw.lower() == "all":
            return resume_files

        indices: list[int] = []
        valid = True
        for piece in raw.split(","):
            piece = piece.strip()
            if not piece.isdigit() or not (1 <= int(piece) <= len(resume_files)):
                valid = False
                break
            indices.append(int(piece))

        if valid and indices:
            # de-dupe while preserving order
            seen: set[int] = set()
            ordered = [i for i in indices if not (i in seen or seen.add(i))]
            return [resume_files[i - 1] for i in ordered]

        print(f"Invalid selection. Enter numbers between 1 and {len(resume_files)} "
              f"separated by commas, or 'all'.")


def _prompt_rescreen_mode() -> bool:
    """Demo/dev option, asked once per run, applied to every selected
    candidate: reuse a saved result if one already exists (default,
    matches screen_candidate()'s normal caching behavior -- 0 LLM calls
    on a rerun), or force a fresh run through the CURRENT pipeline code
    even if a completed screening already exists for that resume+JD
    pair. Returns True for "force re-screen"."""
    print()
    print("Before screening:")
    print()
    print("  1. Use saved results when available (default)")
    print("  2. Re-screen selected candidates using current logic")
    print()

    choice = input("Enter choice [1-2] (press Enter for 1): ").strip()
    while choice not in ("", "1", "2"):
        choice = input("Invalid choice. Enter 1 or 2 (press Enter for 1): ").strip()
    return choice == "2"


# ============================================================================
# Organization / Department / Position resolution
#
# Reuses repository.py lookups only -- no new persistence logic. If this
# exact JD text already belongs to a real, previously-seeded position
# (e.g. one created by scripts/test_real_data.py), that real
# organization/department is reused and displayed. Otherwise a
# placeholder container is used internally so screen_candidate() has a
# position_id, but the report displays "Not specified" rather than
# inventing an organization/department name.
# ============================================================================

def _find_existing_position_for_jd(db, jd_text: str):
    """Search for a Position already tied to this exact JD text. A match
    under a real (non-placeholder) organization always wins; a match
    under our own placeholder bucket is only used as a fallback (to
    avoid creating a second, redundant placeholder position for the same
    JD text on a later run) -- otherwise, if a real org for this JD gets
    seeded later (e.g. by scripts/test_real_data.py), an earlier
    placeholder row could keep shadowing it."""
    placeholder_match = None
    for org in repo.list_organizations(db):
        for dept in repo.list_departments(db, org.id):
            for position in repo.list_positions(db, dept.id):
                current_jd = repo.get_current_job_description(db, position.id)
                if current_jd is not None and current_jd.raw_text == jd_text:
                    if org.name == _PLACEHOLDER_ORG:
                        placeholder_match = (org, dept, position)
                    else:
                        return org, dept, position
    return placeholder_match


def _resolve_position(db, jd_path: Path) -> tuple[int, str, str, str]:
    """Returns (position_id, org_display, dept_display, position_title)."""
    jd_text = jd_path.read_text(encoding="utf-8")
    title = _extract_job_title(jd_text, jd_path.name)

    found = _find_existing_position_for_jd(db, jd_text)
    if found is not None:
        org, dept, position = found
        return position.id, org.name, dept.name, position.title

    # No existing record for this JD text -- use (or create) the
    # placeholder container. Never shown to the user as a real org.
    org = repo.get_organization_by_name(db, _PLACEHOLDER_ORG)
    if org is None:
        org = repo.create_organization(db, _PLACEHOLDER_ORG)
    dept = repo.get_department_by_name(db, org.id, _PLACEHOLDER_DEPT)
    if dept is None:
        dept = repo.create_department(db, org.id, _PLACEHOLDER_DEPT)
    position = repo.get_position_by_title(db, dept.id, title)
    if position is None:
        position = repo.create_position(db, dept.id, title)

    current_jd = repo.get_current_job_description(db, position.id)
    if current_jd is None or current_jd.raw_text != jd_text:
        repo.create_job_description(db, position.id, raw_text=jd_text)

    return position.id, org.name, dept.name, position.title


def _display_org_dept(org_name: str, dept_name: str) -> tuple[str, str]:
    if org_name == _PLACEHOLDER_ORG:
        return "Not specified", "Not specified"
    return org_name, dept_name


# ============================================================================
# Screening + report data collection
#
# All scoring/matching/experience data below comes straight from the
# existing pipeline (screen_candidate -> ScreeningResult / Explanation /
# ExperienceEvaluation). Nothing here recomputes or adjusts a score.
# ============================================================================

def _invalidate_existing_screening(db, position_id: int, job_description_id: int, resume_path: Path) -> None:
    """Only used when the user picked "re-screen with current logic".
    Reuses the exact same content-hash lookup screen_candidate() itself
    uses to find the ONE Screening row for this specific (resume, JD)
    pair, and -- only if it's already completed -- invalidates just that
    row via repository.invalidate_completed_screening(). This never
    touches the Resume, the JobDescription, any other candidate's
    Screening, or any Ranking; if this resume has never been screened
    before, there is nothing to invalidate and this is a no-op.
    """
    parsed = parse_document(str(resume_path))
    if not parsed.success:
        return  # let screen_candidate() raise/report the real parsing error itself

    content_hash = repo.compute_content_hash(parsed.text)
    existing_resume = repo.get_resume_by_content_hash(db, position_id, content_hash)
    if existing_resume is None:
        return

    existing_screening = repo.get_screening_for_resume_and_jd(
        db, existing_resume.id, job_description_id
    )
    if existing_screening is not None and existing_screening.status == "completed":
        repo.invalidate_completed_screening(db, existing_screening.id)


async def _screen_selected(position_id: int, resume_paths: list[Path], force_rescreen: bool) -> list[dict]:
    rows: list[dict] = []
    with get_session() as db:
        current_jd = repo.get_current_job_description(db, position_id)

        for resume_path in resume_paths:
            try:
                if force_rescreen and current_jd is not None:
                    _invalidate_existing_screening(db, position_id, current_jd.id, resume_path)

                # screen_candidate() prints its own internal cache/reuse
                # notices (e.g. "resume content already screened...") --
                # useful for the technical scripts, noise for a
                # supervisor-facing report. Swallow stdout for just this
                # call; a real failure still raises and is caught below.
                with contextlib.redirect_stdout(io.StringIO()):
                    screening = await screen_candidate(
                        db, position_id=position_id, resume_file_path=str(resume_path)
                    )
            except Exception as e:
                rows.append({
                    "name": _derive_candidate_name(resume_path),
                    "error": str(e),
                })
                continue

            db.flush()
            result = screening.result
            explanation = screening.explanation
            experience = screening.experience_evaluation
            candidate = screening.resume.candidate

            strengths = list(explanation.matching_skills) if explanation else []
            if experience:
                for s in experience.strengths:
                    if s not in strengths:
                        strengths.append(s)

            rows.append({
                "name": _display_name(candidate.name) if candidate and candidate.name else _derive_candidate_name(resume_path),
                "match_score": result.match_score,
                "recommendation": result.recommendation,
                "confidence": result.confidence,
                "requires_human": result.requires_human,
                "strengths": strengths,
                "skill_gaps": list(explanation.skill_gaps) if explanation else [],
                "years_relevant": experience.years_relevant if experience else None,
                "years_required": experience.years_required if experience else None,
                "role_relevance": experience.role_relevance if experience else None,
            })
    return rows


# ============================================================================
# Report rendering
# ============================================================================

def _print_screening_report(
    org_display: str, dept_display: str, position_title: str,
    experience_label: str, rows: list[dict],
) -> None:
    print()
    _print_header("SCREENING REPORT")
    print()
    print(f"Organization : {org_display}")
    print(f"Department   : {dept_display}")
    print(f"Position     : {position_title}")
    print(f"Experience   : {experience_label}")
    print()
    print(f"Candidates Screened: {len(rows)}")
    print()

    for i, row in enumerate(rows, start=1):
        print()
        print("-" * 60)
        print(f"{i}. {row['name']}")
        print("-" * 60)
        print()

        if "error" in row:
            print(f"  Could not be screened: {row['error']}")
            continue

        print(f"Match Score       {row['match_score']:.0%}")
        print(f"Recommendation    {row['recommendation']}")
        print(f"Confidence        {row['confidence']:.0%}")
        print(f"Human Review      {'Yes' if row['requires_human'] else 'No'}")
        print()

        if row["strengths"]:
            print("Strengths")
            for s in row["strengths"]:
                print(f"  \u2022 {s}")
            print()

        if row["skill_gaps"]:
            print("Skill Gaps")
            for g in row["skill_gaps"]:
                print(f"  \u2022 {g}")
            print()

        if row["years_relevant"] is not None:
            print("Experience")
            print(f"  Relevant experience : {row['years_relevant']:.1f} years")
            print(f"  Required experience : {row['years_required']} years")
            print(f"  Role relevance      : {row['role_relevance']:.0%}")
            print()

    print()
    _print_header("FINAL CANDIDATE RANKING")
    print()

    ranked = sorted(
        (r for r in rows if "error" not in r),
        key=lambda r: r["match_score"],
        reverse=True,
    )

    if ranked:
        print(f"{'Rank':<6} {'Candidate':<22} {'Score':<8} {'Recommendation'}")
        print("-" * 60)
        for rank, row in enumerate(ranked, start=1):
            print(f"{rank:<6} {row['name']:<22} {row['match_score']:>5.0%}   {row['recommendation']}")
    else:
        print("No candidates were successfully screened.")

    print()
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

async def main() -> None:
    init_db()

    _print_header("AI RESUME SCREENING SYSTEM")

    jd_files = _discover_job_descriptions()
    if not jd_files:
        print(f"\nNo job description files found in {JD_DIR}/")
        return

    resume_files = _discover_resumes()
    if not resume_files:
        print(f"\nNo resume files found in {RESUME_DIR}/")
        return

    jd_path = _prompt_job_description(jd_files)
    selected_resumes = _prompt_resumes(resume_files)
    force_rescreen = _prompt_rescreen_mode()

    jd_text = jd_path.read_text(encoding="utf-8")

    with get_session() as db:
        position_id, org_name, dept_name, position_title = _resolve_position(db, jd_path)

    org_display, dept_display = _display_org_dept(org_name, dept_name)

    print(f"\nScreening {len(selected_resumes)} candidate(s) against: {position_title}\n")
    print("Please wait...")

    rows = await _screen_selected(position_id, selected_resumes, force_rescreen)

    # Read min_years_experience AFTER screening -- screen_candidate() is
    # what actually triggers JobAnalyzerAgent (via ensure_job_requirements)
    # the first time this JD version is screened, so the requirements row
    # is only guaranteed to exist once at least one candidate has run.
    with get_session() as db:
        current_jd = repo.get_current_job_description(db, position_id)
        jr_row = repo.get_job_requirements(db, current_jd.id) if current_jd else None
        min_years = jr_row.min_years_experience if jr_row else None
    experience_label = _extract_experience_label(jd_text, min_years)

    _print_screening_report(org_display, dept_display, position_title, experience_label, rows)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(1)
