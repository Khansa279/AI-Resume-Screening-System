#!/usr/bin/env python3
"""
Demo / smoke-test script for the new persistence layer.

This does NOT call any agents or LLMs -- it hand-builds the same shape of
data the agents would produce, and persists it, to prove the database
layer works end to end and to show the intended call pattern for the
future service layer that will sit between workflow.py and src/db.

Run from the project root:
    python scripts/seed_demo.py

This script wipes and recreates screening.db each run so it stays a clean,
repeatable demo. Organizations/departments/positions are NOT deduped by
name in the schema itself (real orgs get created once via an admin flow
and referenced by id from then on) -- get_or_create_candidate() is the
only dedup helper, because email is a natural unique key for a person in
a way name is not for an organization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.database import DEFAULT_DB_PATH
from src.db import init_db, get_session, repository as repo


def main() -> None:
    # Demo convenience only: start from a clean file every run. Do NOT do
    # this in the real application -- it exists here purely so this script
    # is repeatable without accumulating duplicate demo orgs/positions.
    if DEFAULT_DB_PATH.exists():
        DEFAULT_DB_PATH.unlink()

    init_db()
    print(f"Database ready at {DEFAULT_DB_PATH}")

    with get_session() as db:
        # ---- Org hierarchy -------------------------------------------------
        org = repo.create_organization(db, "TechVentures Pvt Ltd")
        dept = repo.create_department(db, org.id, "Engineering")
        position = repo.create_position(db, dept.id, "Backend Engineer - Python")

        # ---- Job description + requirements --------------------------------
        # (raw_text shortened here; in real use this is the full JD text,
        # e.g. the contents of sample_data/job_descriptions/jd_01_backend_python_standard.txt)
        jd = repo.create_job_description(
            db, position.id,
            raw_text="Backend Engineer - Python. 2+ years Python, REST APIs, "
                     "PostgreSQL. Nice to have: Django/FastAPI, Docker, AWS.",
        )
        repo.save_job_requirements(
            db, jd.id,
            title="Backend Engineer - Python",
            summary="Platform team role building and maintaining core APIs.",
            min_years_experience=2,
            required_skills=["Python", "REST APIs", "PostgreSQL", "Git"],
            preferred_skills=["Django", "FastAPI", "Docker", "AWS"],
            parsing_confidence=0.9,
            requirement_items=[
                {"description": "2+ years professional Python experience",
                 "category": "experience", "priority": "required", "years_needed": 2},
            ],
        )

        # ---- Candidate + resume ---------------------------------------------
        candidate = repo.get_or_create_candidate(
            db, name="John Doe", email="john.doe@email.com", location="San Francisco, CA",
            linkedin="linkedin.com/in/johndoe", github="github.com/johndoe",
        )
        resume = repo.create_resume(
            db,
            candidate_id=candidate.id,
            position_id=position.id,
            file_type="txt",
            raw_text="(full extracted resume text would go here)",
            summary="Backend engineer with 4+ years Python/Django experience.",
            skills_section=["Python", "Django", "PostgreSQL", "Docker", "AWS"],
            parsing_confidence=0.95,
            work_experience=[
                {"title": "Senior Software Engineer", "company": "TechCorp Inc.",
                 "start_date": "2022-01", "end_date": "Present",
                 "responsibilities": ["Designed RESTful APIs serving 10M+ daily requests"],
                 "technologies": ["Python", "Django", "PostgreSQL"]},
            ],
            skills=[
                {"name": "Python", "category": "language", "proficiency": "expert",
                 "confidence": 0.95, "source": "explicit"},
                {"name": "Django", "category": "framework", "proficiency": "advanced",
                 "confidence": 0.9, "source": "explicit"},
            ],
        )

        # ---- Screening (this is where agent output would normally land) -----
        screening = repo.create_screening(db, resume.id, jd.id)
        repo.mark_screening_started(db, screening.id)

        repo.save_skill_match_result(
            db, screening.id,
            required_skills_met=4, required_skills_total=4,
            preferred_skills_met=3, preferred_skills_total=4,
            overall_score=0.88, confidence=0.9,
            reasoning="Strong match on all required skills; missing only FastAPI exposure.",
            matches=[
                {"requirement": "Python", "matched": True, "matched_skill": "Python",
                 "match_quality": "exact", "confidence": 0.98, "notes": ""},
            ],
        )
        repo.save_experience_evaluation(
            db, screening.id,
            years_relevant=4.0, years_required=2,
            experience_score=0.9, role_relevance=0.85,
            career_progression="Steady growth from junior to senior engineer.",
            gaps_identified=[], strengths=["Led migration to microservices"],
            confidence=0.85,
            reasoning="Experience comfortably exceeds the role's requirements.",
        )
        repo.save_screening_result(
            db, screening.id,
            match_score=0.87, recommendation="Proceed to technical interview",
            requires_human=False, confidence=0.88,
            reasoning_summary="Strong technical match with relevant senior-level experience.",
            flags=[],
        )
        repo.save_explanation(
            db, screening.id,
            why_summary="Candidate exceeds required experience and matches all required skills.",
            matching_skills=["Python", "Django", "PostgreSQL", "Docker"],
            evidence_snippets=["Designed RESTful APIs serving 10M+ daily requests using Python and Django"],
            relevant_experience="4+ years, most recently as Senior Software Engineer",
            skill_gaps=["No direct FastAPI experience mentioned"],
            confidence_label="High",
        )

        # ---- Ranking (normally computed across MANY screenings) -------------
        ranking = repo.create_ranking(
            db, jd.id, ranked_screenings=[(screening.id, 0.87)],
        )

        print(f"Organization:   {org.id} {org.name}")
        print(f"Department:     {dept.id} {dept.name}")
        print(f"Position:       {position.id} {position.title}")
        print(f"Job Description:{jd.id} (v{jd.version})")
        print(f"Candidate:      {candidate.id} {candidate.name}")
        print(f"Resume:         {resume.id}")
        print(f"Screening:      {screening.id} (status will show 'completed' after commit)")
        print(f"Ranking:        {ranking.id} with {len(ranking.entries)} entr(y/ies)")

    print("\nSeed complete. Re-query with scripts/query_demo.py or inspect screening.db directly.")


if __name__ == "__main__":
    main()
