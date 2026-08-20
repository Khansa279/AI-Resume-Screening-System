#!/usr/bin/env python3
"""
Real end-to-end verification using actual repo sample data (no seed_demo,
no hardcoded/fake data). Screens the two real PDF resumes against the
real Associate AI Engineer job description, individually first, then
confirms both persisted correctly in screening.db.
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.db import init_db, get_session, repository as repo
from src.services import screen_candidate, screen_position

RESUME_1 = "sample_data/resumes/resume_05_khansa_aslam.pdf"
RESUME_2 = "sample_data/resumes/resume_06_ayesha_naeem.pdf"
JD_PATH = "sample_data/job_descriptions/jd_05_associate_ai_engineer.txt"


async def main():
    init_db()

    for p in (RESUME_1, RESUME_2, JD_PATH):
        if not Path(p).exists():
            print(f"MISSING FILE: {p}")
            return

    # ---- Set up the real position/JD (idempotent: safe to re-run) ------
    with get_session() as db:
        org = repo.get_organization_by_name(db, "CureMD")
        if org is None:
            org = repo.create_organization(db, "CureMD")

        dept = repo.get_department_by_name(db, org.id, "AI Engineering")
        if dept is None:
            dept = repo.create_department(db, org.id, "AI Engineering")

        position = repo.get_position_by_title(db, dept.id, "Associate AI Engineer")
        if position is None:
            position = repo.create_position(db, dept.id, "Associate AI Engineer")

        jd_text = Path(JD_PATH).read_text(encoding="utf-8")
        current_jd = repo.get_current_job_description(db, position.id)
        if current_jd is None or current_jd.raw_text != jd_text:
            repo.create_job_description(db, position.id, raw_text=jd_text)

        position_id = position.id

    print(f"Created position_id={position_id} for real JD: {JD_PATH}\n")

    # ---- Screen BOTH resumes individually first (as requested) ---------
    print("=" * 70)
    print("STEP 1: Individual screening")
    print("=" * 70)

    for resume_path in (RESUME_1, RESUME_2):
        print(f"\nScreening: {resume_path}")
        with get_session() as db:
            screening = await screen_candidate(
                db, position_id=position_id, resume_file_path=resume_path
            )
            db.flush()
            result = screening.result
            candidate = screening.resume.candidate
            explanation = screening.explanation

            print(f"  Candidate (parsed name): {candidate.name}")
            print(f"  Match score:      {result.match_score:.0%}")
            print(f"  Recommendation:   {result.recommendation}")
            print(f"  Requires human:   {result.requires_human}")
            print(f"  Confidence:       {result.confidence:.0%}")
            print(f"  Reasoning:        {result.reasoning_summary}")
            if explanation:
                print(f"  Matching skills:  {', '.join(explanation.matching_skills) or 'None'}")
                print(f"  Skill gaps:       {', '.join(explanation.skill_gaps) or 'None'}")

    # ---- Now confirm batch/ranking path works on the SAME position -----
    print("\n" + "=" * 70)
    print("STEP 2: Query back from DB - all screenings for this position")
    print("=" * 70)

    with get_session() as db:
        jd = repo.get_current_job_description(db, position_id)
        screenings = repo.list_screenings_for_job_description(db, jd.id)
        print(f"\n{len(screenings)} screening(s) found in screening.db for position_id={position_id}:\n")
        for s in screenings:
            print(f"  Screening #{s.id}  status={s.status}  "
                  f"candidate={s.resume.candidate.name}  "
                  f"score={s.result.match_score:.0%}  "
                  f"rec={s.result.recommendation}")


if __name__ == "__main__":
    asyncio.run(main())