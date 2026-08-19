#!/usr/bin/env python3
"""
Manual smoke test for the new screening service layer.
Screens 2 sample resumes against the seeded backend position and prints
the ranking. Run scripts/seed_demo.py first isn't required -- this
creates its own org/dept/position/JD fresh.
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.db import init_db, get_session, repository as repo
from src.services import screen_position


async def main():
    init_db()

    # Set up a fresh org/dept/position/JD to screen against
    with get_session() as db:
        org = repo.create_organization(db, "TestOrg " + str(id(object())))
        dept = repo.create_department(db, org.id, "Engineering")
        position = repo.create_position(db, dept.id, "Backend Engineer - Python")

        jd_text = Path("sample_data/job_descriptions/backend_engineer.txt").read_text()
        repo.create_job_description(db, position.id, raw_text=jd_text)

        position_id = position.id

    print(f"Screening against position_id={position_id}...")

    ranking = await screen_position(
        position_id=position_id,
        resume_file_paths=[
            "sample_data/resumes/senior_python_dev.txt",
            "sample_data/resumes/junior_dev.txt",
        ],
    )

    print(f"\nRanking #{ranking.id} ({len(ranking.entries)} candidates):\n")
    for entry in sorted(ranking.entries, key=lambda e: e.rank_position):
        screening = entry.screening
        candidate = screening.resume.candidate
        result = screening.result
        explanation = screening.explanation

        print(f"#{entry.rank_position}  {candidate.name}  score={entry.score:.0%}")
        print(f"    Recommendation: {result.recommendation}")
        print(f"    Requires human review: {result.requires_human}")
        if explanation:
            print(f"    Why: {explanation.why_summary}")
            print(f"    Matching skills: {', '.join(explanation.matching_skills)}")
            print(f"    Gaps: {', '.join(explanation.skill_gaps) or 'None'}")
        print()


if __name__ == "__main__":
    asyncio.run(main())