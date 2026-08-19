#!/usr/bin/env python3
"""
Read-side demo: shows the query pattern the future ranking API/frontend
will use -- "give me the ranked candidates for this job description, with
their scores and explanations." Run scripts/seed_demo.py first.

Run from the project root:
    python scripts/query_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_session, repository as repo


def main() -> None:
    with get_session() as db:
        jd = repo.get_current_job_description(db, position_id=1)
        if jd is None:
            print("No job description found for position 1 -- run scripts/seed_demo.py first.")
            return

        print(f"Job Description v{jd.version} for position_id={jd.position_id}")
        print(f"  {jd.raw_text[:80]}...\n")

        ranking = repo.get_latest_ranking(db, jd.id)
        if ranking is None:
            print("No ranking found for this JD version yet.")
            return

        print(f"Ranking #{ranking.id} ({ranking.method}, generated {ranking.generated_at}):\n")
        for entry in sorted(ranking.entries, key=lambda e: e.rank_position):
            screening = entry.screening
            resume = screening.resume
            candidate = resume.candidate
            result = screening.result
            explanation = screening.explanation

            print(f"  #{entry.rank_position}  {candidate.name}  (score={entry.score:.0%})")
            print(f"      Recommendation: {result.recommendation}")
            print(f"      Requires human review: {result.requires_human}")
            if explanation:
                print(f"      Why: {explanation.why_summary}")
                print(f"      Matching skills: {', '.join(explanation.matching_skills)}")
                print(f"      Gaps: {', '.join(explanation.skill_gaps) or 'None noted'}")
            print()


if __name__ == "__main__":
    main()
