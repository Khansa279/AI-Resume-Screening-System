#!/usr/bin/env python3
"""
Phase 1 diagnostic: traces role_relevance / skills matching end-to-end for
ONE resume against ONE position's current (cached) JobRequirements, using
the REAL, unmodified pipeline code -- no scoring logic is touched.

Drop this into src/scripts/ and run from the project root:

    python src/scripts/diagnose_role_relevance.py --position-id 3 \
        --resume sample_data/resumes/priya_sharma.txt

If you don't know the position_id, run it with --list-positions first.

What it prints, in order, matching what's asked for in Phase 1:
  1. JobAnalyzerAgent's structured output for the position's current JD
     (title / required / preferred / responsibilities / education /
     min_years) -- pulled from the DB cache via ensure_job_requirements,
     the SAME cache screen_candidate() uses (no extra LLM call if it
     already exists).
  2. ResumeParserAgent's structured output (deterministic parse).
  3. SkillExtractorAgent's output, with source=explicit/inferred per skill.
  4. SkillsMatcherAgent's per-requirement verdict (matched/quality/notes) --
     this is WHY each requirement is or isn't a "gap".
  5. ExperienceEvaluatorAgent's per-job job_relevance() breakdown -- title
     tokens after stopword removal, technologies list, and the resulting
     relevance number -- which is exactly where the low role_relevance
     numbers come from.
  6. Which stages made an LLM call (should only ever be JobAnalyzerAgent,
     plus ResumeParserAgent IF its deterministic parse was judged
     insufficient).
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.db import init_db, get_session, repository as repo
from src.document_parser import parse_document
from src.agents.resume_parser import ResumeParserAgent
from src.agents.skill_extractor import SkillExtractorAgent
from src.agents.skills_matcher import SkillsMatcherAgent
from src.agents.experience_eval import ExperienceEvaluatorAgent, job_relevance, _tokens
from src.services.screening_service import ensure_job_requirements, _job_requirements_to_dict
from src.models import ResumeData, JobRequirements


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--position-id", type=int)
    ap.add_argument("--resume", type=str)
    ap.add_argument("--list-positions", action="store_true")
    args = ap.parse_args()

    init_db()

    if args.list_positions or not args.position_id:
        with get_session() as db:
            orgs = repo.list_organizations(db)
            for org in orgs:
                for dept in repo.list_departments(db, org.id):
                    for pos in repo.list_positions(db, dept.id):
                        print(f"position_id={pos.id:<4} {org.name} / {dept.name} / {pos.title}")
        if not args.position_id:
            return

    if not args.resume:
        print("Pass --resume path/to/resume.txt")
        return

    with get_session() as db:
        jd = repo.get_current_job_description(db, args.position_id)
        if jd is None:
            print(f"No current job description for position_id={args.position_id}")
            return
        jr_row = await ensure_job_requirements(db, jd.id)  # LLM call ONLY if not cached yet
        jr_dict = _job_requirements_to_dict(jr_row)

    requirements = JobRequirements.model_validate(jr_dict)

    hr("STAGE 1: JobAnalyzerAgent structured output (from DB cache)")
    print(f"Title:              {requirements.title!r}")
    print(f"Min years:          {requirements.min_years_experience}")
    print(f"Education reqs:     {requirements.education_requirements}")
    print(f"Certifications:     {requirements.certifications_required}")
    print(f"Responsibilities:")
    for r in requirements.responsibilities:
        print(f"  - {r}")
    print(f"REQUIRED skills ({len(requirements.required_skills)}):")
    for s in requirements.required_skills:
        flag = "  <-- long descriptive phrase, not a bare skill token" if len(s.split()) > 3 else ""
        print(f"  - {s!r}{flag}")
    print(f"PREFERRED skills ({len(requirements.preferred_skills)}):")
    for s in requirements.preferred_skills:
        flag = "  <-- long descriptive phrase, not a bare skill token" if len(s.split()) > 3 else ""
        print(f"  - {s!r}{flag}")

    parsed = parse_document(args.resume)
    if not parsed.success:
        print(f"Document parsing failed: {parsed.error_message}")
        return

    hr("STAGE 2: ResumeParserAgent output (deterministic)")
    rp = ResumeParserAgent()
    rp_out = await rp.process({"resume_raw_text": parsed.text})
    resume_data: ResumeData = rp_out["resume_data"]
    print(f"Name: {resume_data.contact.name!r}  parsing_confidence={resume_data.parsing_confidence}")
    print(f"parsing_notes: {resume_data.parsing_notes}")
    print("work_experience:")
    for w in resume_data.work_experience:
        print(f"  - title={w.title!r} company={w.company!r} dates=({w.start_date!r},{w.end_date!r},{w.duration!r})")
        print(f"      technologies={w.technologies!r}   <-- watch this: deterministic parser never fills this in")
        print(f"      responsibilities={w.responsibilities}")

    hr("STAGE 3: SkillExtractorAgent output (deterministic taxonomy match)")
    se = SkillExtractorAgent()
    se_out = await se.process({"resume_data": resume_data})
    extracted_skills = se_out["extracted_skills"]
    for s in extracted_skills:
        print(f"  {s.name:<30} category={s.category:<12} source={s.source:<9} confidence={s.confidence}")

    hr("STAGE 4: SkillsMatcherAgent per-requirement verdicts (deterministic)")
    sm = SkillsMatcherAgent()
    sm_out = await sm.process({"extracted_skills": extracted_skills, "job_requirements": jr_dict})
    skills_match = sm_out["skills_match"]
    for m in skills_match.matches:
        status = "MATCH " if m.matched else "GAP   "
        print(f"  [{status}] quality={m.match_quality:<8} req={m.requirement!r}")
        print(f"           matched_skill={m.matched_skill!r}  notes={m.notes!r}")
    print(f"\noverall_score={skills_match.overall_score}  "
          f"required {skills_match.required_skills_met}/{skills_match.required_skills_total}  "
          f"preferred {skills_match.preferred_skills_met}/{skills_match.preferred_skills_total}")

    hr("STAGE 5: ExperienceEvaluatorAgent / job_relevance() breakdown (deterministic)")
    ee = ExperienceEvaluatorAgent()
    ee_out = await ee.process({"resume_data": resume_data, "job_requirements": jr_dict})
    experience_eval = ee_out["experience_eval"]
    print("Per-job relevance breakdown:")
    for exp in resume_data.work_experience:
        title_tokens = _tokens(exp.title)
        rel = job_relevance(exp, requirements)
        print(f"  title={exp.title!r}")
        print(f"    title tokens AFTER stopword removal: {title_tokens or '(EMPTY -- title_score forced to 0)'}")
        print(f"    technologies used for skill_hit:      {exp.technologies or '(EMPTY)'}")
        print(f"    -> job_relevance() = {rel}")
    print(f"\nFinal role_relevance (weighted avg across jobs): {experience_eval.role_relevance}")
    print(f"years_relevant: {experience_eval.years_relevant}  experience_score: {experience_eval.experience_score}")
    print(f"gaps_identified: {experience_eval.gaps_identified}")
    print(f"strengths: {experience_eval.strengths}")
    print(f"reasoning: {experience_eval.reasoning}")

    hr("STAGE 6: which stages actually called an LLM")
    print("JobAnalyzerAgent: LLM call happened iff this was the FIRST time this JD version was screened")
    print("  (check server logs / call count -- this run reused the DB cache if a JobRequirements row already existed)")
    print(f"ResumeParserAgent: notes={resume_data.parsing_notes} "
          f"-- an LLM fallback only fires if the deterministic parse was judged insufficient")
    print("SkillExtractorAgent, SkillsMatcherAgent, ExperienceEvaluatorAgent: NEVER call an LLM (fully deterministic)")


if __name__ == "__main__":
    asyncio.run(main())
