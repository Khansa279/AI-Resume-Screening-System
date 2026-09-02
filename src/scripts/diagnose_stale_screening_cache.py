#!/usr/bin/env python3
"""
Read-only diagnostic for the "I bumped CURRENT_ALGORITHM_VERSION but I'm
still seeing the old cached score" symptom.

This does NOT modify any code, any scoring logic, or any database row.
It only READS the real screening.db (or DATABASE_URL, if set) and prints:

  1. What CURRENT_ALGORITHM_VERSION is in the CODE that THIS PROCESS has
     currently loaded (proves whether the process you're running was
     actually restarted after the version-bump edit).
  2. Every Candidate row matching the given name (default: "Alex
     Morgan"), and every Resume/Screening row for them -- including the
     REAL algorithm_version, match_score, status, and completed_at
     stamped on each Screening row in the database right now.
  3. Every Ranking (and RankingEntry) that references one of those
     Screenings, with generated_at, so you can see whether the ranking
     the frontend/API is serving was generated BEFORE or AFTER you made
     the version-bump edit.
  4. A plain-English verdict for each Screening: would screen_candidate()
     reuse it as-is today, or would it invalidate + recompute it?

Run from the project root:
    python src/scripts/diagnose_stale_screening_cache.py
    python src/scripts/diagnose_stale_screening_cache.py --name "Alex Morgan"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.db import get_session, repository as repo
from src.db import models as db_models
from src.db.database import DATABASE_URL
from src.services.screening_service import CURRENT_ALGORITHM_VERSION
from sqlalchemy import select


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Alex Morgan", help="Candidate name to search for (case-insensitive substring)")
    args = ap.parse_args()

    hr("0. WHAT THIS PROCESS HAS LOADED RIGHT NOW")
    print(f"DATABASE_URL in use:            {DATABASE_URL}")
    print(f"CURRENT_ALGORITHM_VERSION (as imported by THIS process): {CURRENT_ALGORITHM_VERSION!r}")
    print(
        "If you edited screening_service.py and the value above still "
        "shows the OLD version string, the process running this script "
        "(and, more importantly, whatever process serves your API/app) "
        "has NOT picked up the edit -- see verdict at the bottom."
    )

    with get_session() as db:
        candidates = db.scalars(
            select(db_models.Candidate).where(db_models.Candidate.name.ilike(f"%{args.name}%"))
        ).all()

        if not candidates:
            hr("RESULT")
            print(f"No Candidate row found matching name like {args.name!r}.")
            print(
                "This itself is useful signal: if the UI is showing a "
                "result for this person but no Candidate row exists with "
                "this name, the number on screen is NOT coming from this "
                "database at all (wrong DATABASE_URL / wrong screening.db "
                "file / frontend showing stale in-memory/mock state)."
            )
            return

        for candidate in candidates:
            hr(f"CANDIDATE id={candidate.id}  name={candidate.name!r}  email={candidate.email!r}")

            resume_rows = db.scalars(
                select(db_models.Resume).where(db_models.Resume.candidate_id == candidate.id)
            ).all()

            if not resume_rows:
                print("  No Resume rows for this candidate.")
                continue

            for resume in resume_rows:
                print(f"\n  RESUME id={resume.id}  position_id={resume.position_id}  "
                      f"content_hash={ (resume.content_hash or '')[:16] }...")

                screenings = db.scalars(
                    select(db_models.Screening).where(db_models.Screening.resume_id == resume.id)
                ).all()

                if not screenings:
                    print("    No Screening rows for this resume.")
                    continue

                for screening in screenings:
                    result = screening.result
                    exp = screening.experience_evaluation
                    print(f"\n    SCREENING id={screening.id}  job_description_id={screening.job_description_id}  "
                          f"status={screening.status}")
                    print(f"      algorithm_version (STORED IN DB) = {screening.algorithm_version!r}")
                    print(f"      completed_at                    = {screening.completed_at}")
                    if result:
                        print(f"      match_score (STORED)            = {result.match_score:.0%}")
                        print(f"      recommendation                  = {result.recommendation}")
                    else:
                        print("      match_score                     = (no ScreeningResult row)")
                    if exp:
                        print(f"      role_relevance (STORED)         = {exp.role_relevance:.0%}")
                        print(f"      years_relevant (STORED)         = {exp.years_relevant}")
                    else:
                        print("      experience_evaluation           = (no ExperienceEvaluation row)")

                    # ---- verdict: would screen_candidate() reuse this row AS-IS today? ----
                    if screening.status != "completed":
                        verdict = "NOT completed -- screen_candidate() cannot reuse this; it will run the full pipeline."
                    elif screening.algorithm_version == CURRENT_ALGORITHM_VERSION:
                        verdict = (
                            "MATCHES current CURRENT_ALGORITHM_VERSION -> screen_candidate() "
                            "WILL reuse this row as-is (0 recomputation) THE NEXT TIME "
                            "screen_candidate() is called for this exact (resume, "
                            "job_description_id) pair."
                        )
                    else:
                        verdict = (
                            f"STORED version {screening.algorithm_version!r} != "
                            f"CURRENT_ALGORITHM_VERSION {CURRENT_ALGORITHM_VERSION!r} -> "
                            f"screen_candidate() WILL invalidate and recompute this row "
                            f"THE NEXT TIME it is called for this exact (resume, "
                            f"job_description_id) pair."
                        )
                    print(f"      VERDICT: {verdict}")

                    # ---- any Ranking referencing this screening ----
                    entries = db.scalars(
                        select(db_models.RankingEntry).where(db_models.RankingEntry.screening_id == screening.id)
                    ).all()
                    if not entries:
                        print("      No RankingEntry references this screening.")
                    for entry in entries:
                        ranking = db.get(db_models.Ranking, entry.ranking_id)
                        print(f"      RANKING id={ranking.id}  generated_at={ranking.generated_at}  "
                              f"rank_position={entry.rank_position}  score(at ranking time)={entry.score:.0%}  "
                              f"job_description_id={ranking.job_description_id}")

                    # ---- is this screening's JD the position's CURRENT JD? ----
                    jd_row = db.get(db_models.JobDescription, screening.job_description_id)
                    if jd_row is not None:
                        current_jd = repo.get_current_job_description(db, jd_row.position_id)
                        is_current = current_jd is not None and current_jd.id == jd_row.id
                        print(f"      This screening's JD version={jd_row.version}  "
                              f"is_current_jd_for_position={is_current}")
                        if not is_current:
                            print(
                                "      NOTE: this screening is NOT against the position's "
                                "CURRENT job description version. GET /screening/{position_id}/results "
                                "only ever looks at the ranking for the CURRENT JD -- a "
                                "screening against an OLDER JD version is invisible to that "
                                "endpoint regardless of algorithm_version."
                            )

    hr("HOW TO READ THIS")
    print(
        "- If a Screening's STORED algorithm_version already equals the "
        "new CURRENT_ALGORITHM_VERSION, but shows the OLD 78%/10%/1.4y "
        "numbers, then it WAS already recomputed under the new version "
        "and produced those SAME numbers again -- i.e. this is not a "
        "cache-staleness issue at all, the scoring pipeline itself is "
        "genuinely returning the same result (a different bug).\n"
        "- If a Screening's STORED algorithm_version is still the OLD "
        "string, no request has actually reached screen_candidate() for "
        "this exact (resume, job_description_id) pair since the bump -- "
        "whatever surface is showing you the 78%/10%/1.4y numbers is "
        "either (a) reading a Ranking/RankingEntry snapshot instead of "
        "calling screen_candidate() at all, or (b) calling "
        "screen_candidate() against a DIFFERENT job_description_id than "
        "the one you think, or (c) a server process that hasn't reloaded "
        "the edited screening_service.py (see section 0 above)."
    )


if __name__ == "__main__":
    main()
