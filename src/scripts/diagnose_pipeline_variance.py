#!/usr/bin/env python3
"""
Pipeline-variance diagnostic.

Runs ResumeParserAgent -> SkillExtractorAgent -> SkillsMatcherAgent and
ExperienceEvaluatorAgent TWICE against the exact same resume text and the
exact same (cached, DB-persisted) job_requirements dict, and diffs every
stage's output to find the FIRST point where two runs of identical input
produce different data.

Deliberately does NOT touch document_parser.py or the DB schema:
  - The resume's raw text is extracted ONCE up front (via the real,
    unmodified parse_document()) and the identical string object is fed
    into both runs, so any difference seen below cannot be document
    parsing noise.
  - job_requirements is fetched ONCE from the DB cache (via the real,
    unmodified ensure_job_requirements(), same as screen_candidate()
    uses) and the identical dict object is reused for both runs, so any
    difference seen below cannot be job-description re-analysis noise.

That leaves exactly four remaining stages to isolate:
  A. ResumeParserAgent   (temperature = agent class default -> config,
                           currently 0.3)
  B. SkillExtractorAgent (temperature = agent class default -> config,
                           currently 0.3)
  C. SkillsMatcherAgent  (temperature = 0.0, set explicitly)
  D. ExperienceEvaluatorAgent (temperature = 0.0, set explicitly)
  E. The _calculate_match_score() arithmetic itself (deterministic,
     included only to show its exact inputs/outputs per run)

Usage:
    python src/scripts/diagnose_pipeline_variance.py
"""

import sys
import asyncio
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.db import init_db, get_session, repository as repo
from src.document_parser import parse_document
from src.agents.resume_parser import ResumeParserAgent
from src.agents.skill_extractor import SkillExtractorAgent
from src.agents.skills_matcher import SkillsMatcherAgent
from src.agents.experience_eval import ExperienceEvaluatorAgent
from src.models import ResumeData, Skill, SkillsMatchResult, ExperienceEvaluation
from src.services.screening_service import ensure_job_requirements, _job_requirements_to_dict

RESUME_PATH = "sample_data/resumes/resume_05_khansa_aslam.pdf"
ORG_NAME = "CureMD"
DEPT_NAME = "AI Engineering"
POSITION_TITLE = "Associate AI Engineer"


# ============================================================================
# Small printing / comparison helpers
# ============================================================================

def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)


def skill_key(s: Skill) -> tuple:
    """Comparable key for a Skill, ignoring case/whitespace and rounding
    confidence so trivial float noise doesn't register as a difference."""
    return (s.name.strip().casefold(), s.category, s.proficiency, s.source, round(s.confidence, 2))


def match_key(m) -> tuple:
    """Comparable key for a SkillMatch."""
    return (
        m.requirement.strip().casefold(),
        m.matched,
        (m.matched_skill or "").strip().casefold(),
        m.match_quality,
        round(m.confidence, 2),
    )


def diff_sets(label: str, a: list, b: list, key=lambda x: x) -> bool:
    """Prints a set-style diff between two lists. Returns True if any
    difference was found (order-independent -- LLMs are not guaranteed
    to return items in the same order across calls, so order differences
    alone are not treated as meaningful divergence here)."""
    set_a = {key(x) for x in a}
    set_b = {key(x) for x in b}
    only_a = set_a - set_b
    only_b = set_b - set_a
    if not only_a and not only_b:
        print(f"  {label}: IDENTICAL ({len(set_a)} items)")
        return False

    print(f"  {label}: DIFFERENT  (run1={len(set_a)} items, run2={len(set_b)} items)")
    if only_a:
        print("    Only in RUN 1:")
        for item in sorted(only_a, key=str):
            print(f"      - {item}")
    if only_b:
        print("    Only in RUN 2:")
        for item in sorted(only_b, key=str):
            print(f"      - {item}")
    return True


# ============================================================================
# Pipeline execution (real, unmodified agents)
# ============================================================================

async def run_pipeline_once(run_label: str, raw_text: str, job_requirements_dict: dict) -> dict[str, Any]:
    print(f"\n--- {run_label}: ResumeParserAgent "
          f"(temperature={ResumeParserAgent.temperature!r}) ---")
    resume_parser = ResumeParserAgent()
    rp_result = await resume_parser.process({"resume_raw_text": raw_text})
    resume_data: ResumeData = rp_result["resume_data"]
    print(f"    parsed name: {resume_data.contact.name!r}  "
          f"parsing_confidence={resume_data.parsing_confidence}")

    print(f"--- {run_label}: SkillExtractorAgent "
          f"(temperature={SkillExtractorAgent.temperature!r}) ---")
    skill_extractor = SkillExtractorAgent()
    se_result = await skill_extractor.process({"resume_data": resume_data})
    extracted_skills: list[Skill] = se_result["extracted_skills"]
    print(f"    extracted {len(extracted_skills)} skills")

    print(f"--- {run_label}: SkillsMatcherAgent "
          f"(temperature={SkillsMatcherAgent.temperature!r}) ---")
    skills_matcher = SkillsMatcherAgent()
    sm_result = await skills_matcher.process({
        "extracted_skills": extracted_skills,
        "job_requirements": job_requirements_dict,
    })
    skills_match: SkillsMatchResult = sm_result["skills_match"]
    print(f"    overall_score={skills_match.overall_score}  "
          f"required_met={skills_match.required_skills_met}/{skills_match.required_skills_total}")

    print(f"--- {run_label}: ExperienceEvaluatorAgent "
          f"(temperature={ExperienceEvaluatorAgent.temperature!r}) ---")
    experience_evaluator = ExperienceEvaluatorAgent()
    ee_result = await experience_evaluator.process({
        "resume_data": resume_data,
        "job_requirements": job_requirements_dict,
    })
    experience_eval: ExperienceEvaluation = ee_result["experience_eval"]
    print(f"    experience_score={experience_eval.experience_score}  "
          f"role_relevance={experience_eval.role_relevance}")

    return {
        "resume_data": resume_data,
        "extracted_skills": extracted_skills,
        "skills_match": skills_match,
        "experience_eval": experience_eval,
    }


def calculate_match_score(skills_match: SkillsMatchResult, experience_eval: ExperienceEvaluation) -> dict[str, float]:
    """
    Exact mirror of DecisionSynthesizerAgent._calculate_match_score.
    Reimplemented (not imported) so every intermediate number is exposed
    here, not just the final rounded score -- and so this diagnostic
    still works standalone if that method's internals change later.
    """
    SKILLS_WEIGHT = 0.6
    EXPERIENCE_WEIGHT = 0.4

    skills_score = skills_match.overall_score if skills_match else 0.0
    experience_score = experience_eval.experience_score if experience_eval else 0.0
    role_relevance = experience_eval.role_relevance if experience_eval else 0.5

    adjusted_exp_score = (experience_score * 0.7 + role_relevance * 0.3)
    raw_final = (skills_score * SKILLS_WEIGHT) + (adjusted_exp_score * EXPERIENCE_WEIGHT)
    final_score = round(min(max(raw_final, 0.0), 1.0), 2)

    return {
        "skills_score": skills_score,
        "experience_score": experience_score,
        "role_relevance": role_relevance,
        "adjusted_exp_score": adjusted_exp_score,
        "raw_final": raw_final,
        "final_score": final_score,
    }


# ============================================================================
# Main
# ============================================================================

async def main():
    init_db()

    resume_path = Path(RESUME_PATH)
    if not resume_path.exists():
        print(f"MISSING FILE: {RESUME_PATH}")
        return

    # ---- Parse the resume ONCE. Both runs get the identical raw text
    # object, so any divergence below cannot be document-parsing noise. ----
    parsed = parse_document(str(resume_path))
    if not parsed.success:
        print(f"Document parsing failed: {parsed.error_message}")
        return
    raw_text = parsed.text
    print(f"Resume text extracted once ({len(raw_text)} chars) -- "
          f"the identical string is reused for both runs below.")

    # ---- Fetch the SAME cached job_requirements the real app uses
    # (same lookup as test_real_data.py), ONCE. ----
    with get_session() as db:
        org = repo.get_organization_by_name(db, ORG_NAME)
        dept = repo.get_department_by_name(db, org.id, DEPT_NAME) if org else None
        position = repo.get_position_by_title(db, dept.id, POSITION_TITLE) if dept else None
        if position is None:
            print(
                f"No existing position found for {ORG_NAME}/{DEPT_NAME}/{POSITION_TITLE}. "
                f"Run test_real_data.py first so a cached JobRequirements row exists."
            )
            return

        jd = repo.get_current_job_description(db, position.id)
        jr_row = await ensure_job_requirements(db, jd.id)
        job_requirements_dict = _job_requirements_to_dict(jr_row)

    print(f"job_requirements loaded from job_description_id={jd.id} "
          f"(cached JobRequirements row id={jr_row.id}) -- "
          f"the identical dict object is reused for both runs below.")
    print(f"  required_skills:  {job_requirements_dict['required_skills']}")
    print(f"  preferred_skills: {job_requirements_dict['preferred_skills']}")
    print(f"  min_years_experience: {job_requirements_dict['min_years_experience']}")

    # ---- Run the pipeline twice with identical inputs ----
    hr("RUN 1")
    run1 = await run_pipeline_once("RUN 1", raw_text, job_requirements_dict)

    hr("RUN 2")
    run2 = await run_pipeline_once("RUN 2", raw_text, job_requirements_dict)

    # ========================================================================
    # STAGE-BY-STAGE DIFF
    # ========================================================================
    first_divergence = None

    hr("STAGE A: ResumeParserAgent output (resume_data)")
    rd1, rd2 = run1["resume_data"], run2["resume_data"]
    a_diff = False

    ss1 = sorted(s.strip().casefold() for s in rd1.skills_section)
    ss2 = sorted(s.strip().casefold() for s in rd2.skills_section)
    if ss1 != ss2:
        print("  skills_section: DIFFERENT")
        print(f"    RUN 1: {ss1}")
        print(f"    RUN 2: {ss2}")
        a_diff = True
    else:
        print(f"  skills_section: IDENTICAL ({len(ss1)} items)")

    def work_exp_key(w):
        return (
            w.title.strip().casefold(),
            w.company.strip().casefold(),
            tuple(sorted(t.strip().casefold() for t in w.technologies)),
        )

    we1 = [work_exp_key(w) for w in rd1.work_experience]
    we2 = [work_exp_key(w) for w in rd2.work_experience]
    if we1 != we2:
        print("  work_experience (title/company/technologies): DIFFERENT")
        print(f"    RUN 1: {we1}")
        print(f"    RUN 2: {we2}")
        a_diff = True
    else:
        print(f"  work_experience (title/company/technologies): IDENTICAL ({len(we1)} entries)")

    proj1 = sorted(p.strip().casefold() for p in rd1.projects)
    proj2 = sorted(p.strip().casefold() for p in rd2.projects)
    if proj1 != proj2:
        print("  projects: DIFFERENT")
        print(f"    RUN 1: {proj1}")
        print(f"    RUN 2: {proj2}")
        a_diff = True
    else:
        print(f"  projects: IDENTICAL ({len(proj1)} items)")

    if a_diff and first_divergence is None:
        first_divergence = "A: ResumeParserAgent (resume_data)"

    hr("STAGE B: SkillExtractorAgent output (extracted_skills)")
    b_diff = diff_sets(
        "extracted_skills (name/category/proficiency/source/confidence~0.01)",
        run1["extracted_skills"], run2["extracted_skills"], key=skill_key,
    )
    print("\n  Full extracted_skills lists (for manual inspection):")
    print(f"    RUN 1 ({len(run1['extracted_skills'])}): "
          + ", ".join(f"{s.name}[{s.proficiency}]" for s in run1["extracted_skills"]))
    print(f"    RUN 2 ({len(run2['extracted_skills'])}): "
          + ", ".join(f"{s.name}[{s.proficiency}]" for s in run2["extracted_skills"]))
    if b_diff and first_divergence is None:
        first_divergence = "B: SkillExtractorAgent (extracted_skills)"

    hr("STAGE C: SkillsMatcherAgent input/output")
    print("  INPUT extracted_skills: see Stage B above (this is the actual "
          "input SkillsMatcherAgent received in each run)")
    sm1, sm2 = run1["skills_match"], run2["skills_match"]
    c_diff = diff_sets(
        "matches (requirement/matched/matched_skill/quality/confidence~0.01)",
        sm1.matches, sm2.matches, key=match_key,
    )
    print(f"\n  overall_score:        RUN1={sm1.overall_score:.4f}   RUN2={sm2.overall_score:.4f}   "
          f"delta={sm2.overall_score - sm1.overall_score:+.4f}")
    print(f"  required_skills_met:  RUN1={sm1.required_skills_met}/{sm1.required_skills_total}   "
          f"RUN2={sm2.required_skills_met}/{sm2.required_skills_total}")
    print(f"  preferred_skills_met: RUN1={sm1.preferred_skills_met}/{sm1.preferred_skills_total}   "
          f"RUN2={sm2.preferred_skills_met}/{sm2.preferred_skills_total}")
    print(f"  confidence:           RUN1={sm1.confidence:.4f}   RUN2={sm2.confidence:.4f}")
    if (c_diff or abs(sm1.overall_score - sm2.overall_score) > 1e-9) and first_divergence is None:
        first_divergence = "C: SkillsMatcherAgent (skills_match)"

    hr("STAGE D: ExperienceEvaluatorAgent input/output")
    print("  INPUT resume_data: see Stage A above (this is the actual input "
          "ExperienceEvaluatorAgent received in each run)")
    ee1, ee2 = run1["experience_eval"], run2["experience_eval"]
    print(f"  years_relevant:     RUN1={ee1.years_relevant}   RUN2={ee2.years_relevant}")
    print(f"  experience_score:   RUN1={ee1.experience_score:.4f}   RUN2={ee2.experience_score:.4f}   "
          f"delta={ee2.experience_score - ee1.experience_score:+.4f}")
    print(f"  role_relevance:     RUN1={ee1.role_relevance:.4f}   RUN2={ee2.role_relevance:.4f}   "
          f"delta={ee2.role_relevance - ee1.role_relevance:+.4f}")
    print(f"  career_progression: RUN1={ee1.career_progression!r}")
    print(f"                      RUN2={ee2.career_progression!r}")
    d_diff = (
        abs(ee1.experience_score - ee2.experience_score) > 1e-9
        or abs(ee1.role_relevance - ee2.role_relevance) > 1e-9
        or abs(ee1.years_relevant - ee2.years_relevant) > 1e-9
    )
    if d_diff and first_divergence is None:
        first_divergence = "D: ExperienceEvaluatorAgent (experience_eval)"

    hr("STAGE E: _calculate_match_score formula components")
    formula1 = calculate_match_score(sm1, ee1)
    formula2 = calculate_match_score(sm2, ee2)
    print("  Formula:")
    print("    adjusted_exp_score = experience_score*0.7 + role_relevance*0.3")
    print("    raw_final          = skills_score*0.6 + adjusted_exp_score*0.4")
    print("    final_score        = round(clamp(raw_final, 0.0, 1.0), 2)\n")
    for k in ("skills_score", "experience_score", "role_relevance",
              "adjusted_exp_score", "raw_final", "final_score"):
        v1, v2 = formula1[k], formula2[k]
        print(f"  {k:20s}  RUN1={v1:.4f}   RUN2={v2:.4f}   delta={v2 - v1:+.4f}")

    e_diff = abs(formula1["final_score"] - formula2["final_score"]) > 1e-9
    if e_diff and first_divergence is None:
        first_divergence = "E: _calculate_match_score formula"

    hr("SUMMARY")
    if first_divergence:
        print(f"FIRST STAGE OF DIVERGENCE: {first_divergence}")
    else:
        print("No divergence detected at any stage -- both runs produced identical results.")

    print("\nAgent temperatures in effect for this pipeline run "
          "(None = falls through to config.temperature, currently 0.3):")
    print(f"  ResumeParserAgent.temperature        = {ResumeParserAgent.temperature!r}")
    print(f"  SkillExtractorAgent.temperature      = {SkillExtractorAgent.temperature!r}")
    print(f"  SkillsMatcherAgent.temperature       = {SkillsMatcherAgent.temperature!r}")
    print(f"  ExperienceEvaluatorAgent.temperature = {ExperienceEvaluatorAgent.temperature!r}")
    print(
        "\nNote: temperature=0.0 on an agent makes its OWN judgment call more\n"
        "consistent for a FIXED input -- it does not (a) guarantee bit-identical\n"
        "output between two separate API calls even with identical input (most\n"
        "inference backends, including Groq's, are not strictly deterministic at\n"
        "temperature=0 under batched/parallel serving), and it does nothing at all\n"
        "to stabilize a stage that still runs at the default temperature upstream."
    )


if __name__ == "__main__":
    asyncio.run(main())