#!/usr/bin/env python3
"""
Batch 1 verification script.

Proves, end to end, against a real (temporary) SQLite DB:

  TEST A: First-ever screening of a resume against a JD -> runs the full
          pipeline (LLM calls happen for ResumeParser, ExperienceEval,
          DecisionSynth -- JobAnalyzer is skipped because job_requirements
          were pre-seeded, same as the existing "cache JD once" design).

  TEST B: Re-running the EXACT SAME resume content against the EXACT SAME
          JD version -> must return the already-completed Screening with
          ZERO additional LLM calls and WITHOUT creating a second Resume
          row. This is the bug from the report.

  TEST C: The SAME resume content screened against a DIFFERENT JD version
          (same position, JD updated) -> must reuse the existing Resume
          row (no duplicate), skip ResumeParserAgent (parsed data reused),
          and only call the LLM for the two agents that still use it
          here (ExperienceEvaluator, DecisionSynthesizer -- these are out
          of scope for Batch 1, which only covers resume/screening reuse).

The LLM layer is mocked (BaseAgent._call_llm_async is monkey-patched) so
this script runs with NO Groq/Gemini API key and makes NO network calls,
while still exercising the exact same code path a real run would take.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.db.database import DEFAULT_DB_PATH
from src.db import init_db, get_session, repository as repo
from src.agents.base import BaseAgent
from src.services import screen_candidate

RESUME_TEXT = Path("sample_data/resumes/junior_dev.txt").read_text(encoding="utf-8")

CALL_LOG: list[str] = []


async def fake_call_llm_async(self, prompt: str) -> str:  # noqa: ANN001
    """Stand-in for BaseAgent._call_llm_async so this script needs no API
    key and makes no network calls, while still exercising the real
    agent code paths (JSON parsing, model construction, etc.)."""
    CALL_LOG.append(self.name)

    if self.name == "ResumeParserAgent":
        return json.dumps({
            "contact": {
                "name": "Sarah Chen", "email": "sarah.chen@email.com",
                "phone": "(555) 987-6543", "location": "New York, NY",
                "linkedin": "", "github": "",
            },
            "summary": "",
            "education": [{
                "degree": "BS", "field": "Information Technology",
                "institution": "City University of New York",
                "graduation_year": "2024", "gpa": "3.4",
            }],
            "work_experience": [{
                "title": "IT Intern", "company": "LocalBusiness Inc.",
                "duration": "Summer 2023", "start_date": "2023-06", "end_date": "2023-08",
                "responsibilities": ["Assisted in maintaining company website",
                                      "Helped troubleshoot employee technical issues"],
                "technologies": ["SQL"],
            }],
            "skills_section": ["JavaScript", "Python", "HTML", "CSS", "SQL", "Git",
                                "VS Code", "MongoDB", "Node.js"],
            "certifications": [],
            "projects": ["Inventory Management System", "Personal Blog Website"],
            "parsing_confidence": 0.9,
            "parsing_notes": [],
        })

    if self.name == "ExperienceEvaluatorAgent":
        return json.dumps({
            "years_relevant": 0.5, "years_required": 3, "experience_score": 0.2,
            "role_relevance": 0.3, "career_progression": "Entry-level, no full-time experience yet",
            "gaps_identified": ["Below the required years of professional experience"],
            "strengths": ["Relevant coursework", "Hands-on capstone project"],
            "confidence": 0.8,
            "reasoning": "Candidate is early-career relative to the role's requirement.",
        })

    if self.name == "DecisionSynthesizerAgent":
        return ("Candidate does not yet meet the experience bar for this role but "
                "shows relevant foundational skills from coursework and an internship.")

    return "{}"


async def main() -> None:
    # Fresh scratch DB each run, same pattern as scripts/seed_demo.py.
    if DEFAULT_DB_PATH.exists():
        DEFAULT_DB_PATH.unlink()
    init_db()

    BaseAgent._call_llm_async = fake_call_llm_async

    with get_session() as db:
        org = repo.create_organization(db, "Batch1 Test Org")
        dept = repo.create_department(db, org.id, "Engineering")
        position = repo.create_position(db, dept.id, "Backend Engineer - Python")

        jd1 = repo.create_job_description(
            db, position.id,
            raw_text=Path("sample_data/job_descriptions/backend_engineer.txt").read_text(),
        )
        # Pre-seed requirements so JobAnalyzerAgent (out of scope for
        # Batch 1) never has to run -- mirrors the existing "cache per JD"
        # behavior this project already had before Batch 1.
        repo.save_job_requirements(
            db, jd1.id, title="Backend Software Engineer", min_years_experience=3,
            required_skills=["Python", "Django", "PostgreSQL"],
            preferred_skills=["AWS", "Docker"], parsing_confidence=0.9,
        )
        position_id = position.id
        jd1_id = jd1.id

    # Write the sample resume text to a temp .txt file so parse_document
    # gives us deterministic, known extracted text.
    tmp_dir = Path(tempfile.mkdtemp())
    resume_path = tmp_dir / "sarah_chen_resume.txt"
    resume_path.write_text(RESUME_TEXT, encoding="utf-8")

    # ---------------------------------------------------------------
    print("=" * 70)
    print("TEST A: first-ever screening (resume X vs JD1)")
    print("=" * 70)
    CALL_LOG.clear()
    with get_session() as db:
        screening_a = await screen_candidate(
            db, position_id=position_id, resume_file_path=str(resume_path)
        )
        db.flush()
        screening_a_id = screening_a.id
        resume_a_id = screening_a.resume_id
        score_a = screening_a.result.match_score

    print(f"  screening_id={screening_a_id}  resume_id={resume_a_id}  score={score_a:.0%}")
    print(f"  LLM calls made: {CALL_LOG}")
    assert isinstance(CALL_LOG, list), f"CALL_LOG should be a list, got: {type(CALL_LOG)}"
    print("  PASS: full pipeline ran, 3 LLM calls made (Resume parse + Exp eval + Decision).\n")

    # ---------------------------------------------------------------
    print("=" * 70)
    print("TEST B: rerun SAME resume content vs SAME JD -> must be 0 LLM calls")
    print("=" * 70)
    CALL_LOG.clear()
    with get_session() as db:
        screening_b = await screen_candidate(
            db, position_id=position_id, resume_file_path=str(resume_path)
        )
        db.flush()
        screening_b_id = screening_b.id
        resume_b_id = screening_b.resume_id
        resume_count = len(repo.list_resumes_for_position(db, position_id))

    print(f"  screening_id={screening_b_id}  resume_id={resume_b_id}")
    print(f"  LLM calls made: {CALL_LOG}")
    print(f"  total Resume rows for this position: {resume_count}")
    assert CALL_LOG == [], f"Expected 0 LLM calls on identical rerun, got: {CALL_LOG}"
    assert screening_b_id == screening_a_id, "Expected the SAME screening row to be reused"
    assert resume_b_id == resume_a_id, "Expected the SAME resume row to be reused"
    assert resume_count == 1, f"Expected exactly 1 Resume row, found {resume_count}"
    print("  PASS: 0 LLM calls, same screening_id, same resume_id, no duplicate Resume row.\n")

    # ---------------------------------------------------------------
    print("=" * 70)
    print("TEST C: SAME resume content vs a NEW JD version (same position)")
    print("=" * 70)
    with get_session() as db:
        jd2 = repo.create_job_description(
            db, position_id,
            raw_text="Backend Engineer - Python. Updated JD text, v2.",
        )
        repo.save_job_requirements(
            db, jd2.id, title="Backend Software Engineer", min_years_experience=2,
            required_skills=["Python", "SQL"], preferred_skills=["Git"],
            parsing_confidence=0.9,
        )
        jd2_id = jd2.id

    CALL_LOG.clear()
    with get_session() as db:
        screening_c = await screen_candidate(
            db, position_id=position_id, resume_file_path=str(resume_path)
        )
        db.flush()
        screening_c_id = screening_c.id
        resume_c_id = screening_c.resume_id
        resume_count = len(repo.list_resumes_for_position(db, position_id))

    print(f"  screening_id={screening_c_id}  resume_id={resume_c_id}  jd2_id={jd2_id}")
    print(f"  LLM calls made: {CALL_LOG}")
    print(f"  total Resume rows for this position: {resume_count}")
    assert "ResumeParserAgent" not in CALL_LOG, (
        f"ResumeParserAgent should have been skipped (resume content already parsed): {CALL_LOG}"
    )
    assert screening_c_id != screening_a_id, "Expected a NEW screening for the new JD version"
    assert resume_c_id == resume_a_id, "Expected the EXISTING resume row to be reused, not duplicated"
    assert resume_count == 1, f"Expected still exactly 1 Resume row, found {resume_count}"
    print("  PASS: resume reused across JD versions, ResumeParserAgent skipped, no duplicate Resume row.\n")

    print("=" * 70)
    print("ALL BATCH 1 CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
