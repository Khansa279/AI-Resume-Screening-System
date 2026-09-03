#!/usr/bin/env python3
"""
Maintenance script: re-run the (fixed) deterministic resume parser +
skill extractor against ALREADY-STORED resumes' raw_text, and write the
corrected work_experience/skills back onto their existing Resume rows.

WHY THIS EXISTS: content-hash-based resume reuse (see
services/screening_service.py and repository.get_resume_by_content_hash)
means a Resume's raw_text is extracted from its source file exactly
once -- every later screening against a new/changed job description
reuses the already-stored PARSED representation (work_experience,
skills, ...) instead of re-running ResumeParserAgent. That's the right
performance tradeoff for normal operation, but it also means a bug fix
in src/agents/resume_parser.py (e.g. the wrapped-bullet / inline-header
/ "Selected Projects" heading fixes this script accompanies) does NOT
automatically apply to resumes that were already parsed and stored
under the old, buggy code -- their stored work_experience keeps
reflecting the old (wrong) parse until something re-parses raw_text and
writes the corrected result back.

This script is that "something." It is intentionally general: it
targets every Resume whose CURRENT stored work_experience looks like it
was produced by the fragmentation bug (many low-content entries with no
usable dates), not any specific candidate or job title -- see
_looks_fragmented() below. It only re-runs the DETERMINISTIC parsing
path (parse_resume_text + SkillExtractorAgent's rule-based extraction),
so it makes no LLM calls and needs no API key.

After correcting a Resume's stored data, any of its COMPLETED
Screenings are invalidated (repository.invalidate_completed_screening)
so the next screen_candidate() call for that (resume, job description)
pair re-runs the real, unmodified scoring pipeline against the
corrected data instead of serving a stale cached result -- this is the
same "invalidate one row, let the real pipeline recompute it for real"
pattern already used for CURRENT_ALGORITHM_VERSION staleness.

Usage:
    python src/scripts/reparse_resume_experience.py            # scan + fix all resumes
    python src/scripts/reparse_resume_experience.py --resume-id 99   # just one resume
    python src/scripts/reparse_resume_experience.py --dry-run  # report only, no writes
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.db import init_db, get_session, repository as repo
from src.agents.resume_parser import parse_resume_text
from src.agents.skill_extractor import extract_skills_from_resume
from src.models import ResumeData


def _looks_fragmented(resume) -> bool:
    """Heuristic, evidence-based flag for 'this Resume's stored
    work_experience was very likely produced by the fragmentation bug':
    several entries, most of which have neither a start_date nor an
    end_date. A resume with a couple of genuinely undated entries is
    normal; one where the large majority of ALL entries are undated is
    the signature this bug leaves behind (see the module docstring).
    Never touches resumes that don't match this pattern, so re-running
    this script is safe/idempotent and doesn't churn resumes that were
    already parsed correctly."""
    jobs = resume.work_experience
    if len(jobs) < 4:
        return False
    undated = sum(1 for j in jobs if not j.start_date and not j.end_date)
    return undated / len(jobs) >= 0.5


def reparse_one(db, resume, *, dry_run: bool) -> bool:
    """Returns True if this resume's data was (or, in dry-run mode,
    would be) corrected."""
    resume_data = parse_resume_text(resume.raw_text or "")

    old_count = len(resume.work_experience)
    new_count = len(resume_data.work_experience)

    print(f"Resume id={resume.id} (position_id={resume.position_id}): "
          f"work_experience {old_count} -> {new_count} entr(y/ies)")
    for exp in resume_data.work_experience:
        print(f"    - {exp.title!r} @ {exp.company!r}  "
              f"({exp.start_date or '?'} - {exp.end_date or '?'})")

    if dry_run:
        return True

    skills = extract_skills_from_resume(resume_data)

    repo.replace_resume_parsed_data(
        db, resume.id,
        skills_section=resume_data.skills_section,
        certifications=resume_data.certifications,
        projects=resume_data.projects,
        parsing_confidence=resume_data.parsing_confidence,
        parsing_notes=list(resume_data.parsing_notes) + [
            "Re-parsed by reparse_resume_experience.py after the "
            "wrapped-bullet/inline-header resume_parser.py fix."
        ],
        work_experience=[w.model_dump() for w in resume_data.work_experience],
        skills=[
            {"name": s.name, "category": s.category, "proficiency": s.proficiency,
             "confidence": s.confidence, "source": s.source}
            for s in skills
        ],
    )

    invalidated = 0
    for screening in resume.screenings:
        if screening.status == "completed":
            repo.invalidate_completed_screening(db, screening.id)
            invalidated += 1
    if invalidated:
        print(f"    invalidated {invalidated} completed screening(s) for re-scoring")

    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-id", type=int, help="Only reparse this one resume id")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change, write nothing")
    args = ap.parse_args()

    init_db()

    fixed = 0
    with get_session() as db:
        if args.resume_id is not None:
            candidates = [repo.get_resume(db, args.resume_id)]
            candidates = [r for r in candidates if r is not None]
        else:
            from sqlalchemy import select
            from src.db import models as db_models
            candidates = list(db.scalars(select(db_models.Resume)))

        for resume in candidates:
            if args.resume_id is None and not _looks_fragmented(resume):
                continue
            if reparse_one(db, resume, dry_run=args.dry_run):
                fixed += 1

    verb = "Would fix" if args.dry_run else "Fixed"
    print(f"\n{verb} {fixed} resume(s).")


if __name__ == "__main__":
    main()
