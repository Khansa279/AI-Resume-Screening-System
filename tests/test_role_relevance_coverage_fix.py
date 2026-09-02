"""Regression tests for the under-scoring fix identified during the
Alex Morgan production investigation (screening_id=172, screening_id=153).

ROOT CAUSE (traced from the actual production code path, not simulated):
_semantic_role_relevance() in src/agents/experience_eval.py measures
SYMMETRIC Jaccard overlap between a candidate's per-job "role signature"
(taxonomy skills + title tokens found in that job's title/technologies/
responsibilities) and the JD's own signature. Jaccard is the wrong tool
here: it punishes a candidate whenever their OWN signature is simply
larger than the JD's -- which is the ORDINARY case for a detailed/senior
resume entry, since real bullets routinely mention process/collaboration
terms the shared skill taxonomy also recognizes (Agile, Code Review,
Leadership, Teamwork, Communication, ...) right alongside the core
technical work. Every extra recognized term on the candidate's side grows
the union denominator and shrinks the ratio, even though it says nothing
about whether the candidate's core work is aligned with the JD.

This is exactly what happened for a strongly AI/ML-aligned candidate
whose detailed job entry (PyTorch/TensorFlow/deep learning/NLP work,
plus the usual mentoring/agile/stakeholder bullets any senior role has)
Jaccard-scored far lower than the real technical overlap warranted --
and because that deterministic signal was the title_score title_score
went into job_relevance() near the floor, which then ALSO made the
optional embedding signal (when unavailable/unconfigured, which is the
default in this environment -- no GEMINI_API_KEY) unable to help at all.

FIX: _semantic_role_coverage() (new) is a DIRECTIONAL measure anchored
to the JD's OWN skill-taxonomy signature size: |A∩B| / |jd_skill_sig|.
It answers "what fraction of what the JD is actually asking for does the
candidate cover", so a candidate's extra, unrelated-but-real terms no
longer penalize the score. It is restricted to genuine skill_taxonomy.py
matches (NOT plain title tokens -- see _skill_signature()) specifically
so it cannot be trivially maxed out by a candidate's title merely being
a superset of the JD title's own words (verified as a real edge case
during this fix -- see test_title_superset_alone_does_not_trigger_full_coverage
below). It is combined into title_score via max() alongside the existing
Jaccard signature and lexical title overlap -- the SAME "only ever
raises, never lowers" pattern already used for the LLM hint and the
embedding signal -- so it can only rescue previously under-scored cases,
never regress an already-correct one.

Nothing about the 0.45/0.40/0.15 weights, the embedding
corroboration/dampening thresholds, _ROLE_FAMILY_KEYWORDS/
_ROLE_RELATEDNESS (removed in an earlier batch and NOT restored), or
SkillsMatcherAgent was touched.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.agents import experience_eval as ee
from src.agents.experience_eval import (
    job_relevance,
    evaluate_experience,
    _semantic_role_relevance,
    _semantic_role_coverage,
)
from src.agents.skills_matcher import SkillsMatcherAgent
from src.db.models import Base
from src.db import repository as repo
from src.models import JobRequirements, ResumeData, Skill, WorkExperience
from src.services.screening_service import screen_candidate, CURRENT_ALGORITHM_VERSION
from src import embeddings as emb_module


@pytest.fixture(autouse=True)
def _reset_embedding_state():
    emb_module.reset_embedding_provider_cache()
    emb_module.clear_embedding_cache()
    yield
    emb_module.reset_embedding_provider_cache()
    emb_module.clear_embedding_cache()


# ---------------------------------------------------------------------------
# Realistic AI/ML Engineer JD + a detailed/senior candidate entry -- the
# exact shape of scenario that reproduced the reported under-scoring.
# Deliberately includes a verbose, realistic set of process/leadership
# bullets on the candidate side (the actual trigger for Jaccard dilution),
# and is NOT an exact title match, so this exercises the same code path
# a real, differently-worded resume/JD pair would.
# ---------------------------------------------------------------------------

AI_ML_JD = JobRequirements(
    title="AI/ML Engineer",
    required_skills=["Python", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow"],
    preferred_skills=["NLP", "Computer Vision", "Docker", "AWS", "MLOps"],
    responsibilities=["Design and deploy machine learning models for production AI systems"],
    min_years_experience=3,
)

STRONGLY_ALIGNED_ML_EXPERIENCE = WorkExperience(
    title="Senior Machine Learning Engineer",
    company="DeepStack AI",
    start_date="2021-01",
    end_date="Present",
    technologies=["Python", "PyTorch", "TensorFlow", "Machine Learning", "Deep Learning", "NLP"],
    responsibilities=[
        "Led design and deployment of deep learning models for production NLP systems",
        "Trained and fine-tuned neural networks using PyTorch and TensorFlow",
        "Mentored junior engineers and led code reviews across the team",
        "Collaborated closely with product managers and cross-functional stakeholders",
        "Facilitated agile ceremonies including sprint planning and retrospectives",
        "Presented research findings to leadership and drove roadmap decisions",
        "Improved documentation and onboarding processes for new hires",
        "Managed stakeholder communication for quarterly OKRs",
    ],
)


# ---------------------------------------------------------------------------
# 1 & 2. A strongly AI/ML-aligned candidate must be treated as strongly
# role-relevant, and must NOT be reduced to near-zero relevance, through
# the real production job_relevance()/evaluate_experience() functions.
# ---------------------------------------------------------------------------

def test_ai_ml_engineer_candidate_is_treated_as_strongly_role_relevant():
    relevance = job_relevance(STRONGLY_ALIGNED_ML_EXPERIENCE, AI_ML_JD)
    assert relevance >= 0.55, (
        f"expected a strongly AI/ML-aligned candidate to score as strongly "
        f"relevant, got {relevance}"
    )


def test_coverage_signal_is_what_rescues_the_previously_under_scored_case():
    """Confirms the FIX specifically (not a coincidence of some other
    component): the symmetric Jaccard score alone under-scores this
    pair (the confirmed root cause), while the new directional coverage
    signal correctly recognizes the real technical overlap."""
    jaccard_only = _semantic_role_relevance(STRONGLY_ALIGNED_ML_EXPERIENCE, AI_ML_JD)
    coverage = _semantic_role_coverage(STRONGLY_ALIGNED_ML_EXPERIENCE, AI_ML_JD)
    assert coverage > jaccard_only, (
        f"expected directional coverage ({coverage}) to exceed the diluted "
        f"symmetric Jaccard score ({jaccard_only}) for a candidate whose "
        f"detailed bullets add process/leadership terms alongside the core "
        f"technical overlap"
    )
    assert coverage >= 0.5, f"expected the candidate to cover most of the JD's own skill signature, got {coverage}"


def test_evaluate_experience_end_to_end_reflects_strong_alignment():
    """The real production evaluate_experience() entry point (not job_relevance
    called directly) must reflect the same fix -- role_relevance should not
    be near-zero for this candidate."""
    resume = ResumeData(work_experience=[STRONGLY_ALIGNED_ML_EXPERIENCE])
    result = evaluate_experience(resume, AI_ML_JD)
    assert result.role_relevance >= 0.5, (
        f"expected strong role_relevance for an AI/ML-aligned candidate "
        f"via evaluate_experience(), got {result.role_relevance}"
    )
    assert result.years_relevant > 1.5, (
        f"expected most of this ~multi-year, strongly relevant role to "
        f"count toward years_relevant, got {result.years_relevant}"
    )


# ---------------------------------------------------------------------------
# 3. Unrelated roles must NOT be inflated by the new coverage signal.
# ---------------------------------------------------------------------------

def test_unrelated_role_is_not_inflated_by_coverage_signal():
    jd = JobRequirements(
        title="Backend Engineer",
        required_skills=["Python", "Django", "PostgreSQL"],
        responsibilities=["Build and maintain REST APIs using Python and Django"],
    )
    exp = WorkExperience(
        title="Restaurant Shift Manager",
        company="Diner Co",
        technologies=[],
        responsibilities=[
            "Managed front-of-house staff and inventory",
            "Handled customer complaints and scheduling",
            "Trained new hires on service standards",
        ],
    )
    coverage = _semantic_role_coverage(exp, jd)
    assert coverage == 0.0, f"expected zero coverage for a genuinely unrelated role, got {coverage}"
    relevance = job_relevance(exp, jd)
    assert relevance < 0.15, f"expected low overall relevance for an unrelated role, got {relevance}"


def test_title_superset_alone_does_not_trigger_full_coverage():
    """Regression guard for the exact edge case found while implementing
    this fix: a candidate title that is a trivial SUPERSET of the JD
    title's own words (e.g. adding a seniority prefix) must NOT alone
    produce full coverage -- coverage is restricted to genuine
    skill-taxonomy overlap, not plain title wording (which the existing
    lexical title_score already handles correctly)."""
    jd = JobRequirements(title="Backend Developer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Junior Backend Developer", company="X")
    coverage = _semantic_role_coverage(exp, jd)
    assert coverage == 0.0, (
        f"expected 0.0 coverage when the only overlap is plain title "
        f"wording (no real skill/technology signature on either side), "
        f"got {coverage}"
    )
    # The existing lexical title-overlap mechanism still correctly
    # credits this pair -- untouched by this fix.
    relevance = job_relevance(exp, jd)
    assert relevance == round(0.45 * (2 / 3), 2)


def test_unseen_novel_role_stays_generalized_not_hardcoded():
    """The fix must generalize to a role/domain this project has never
    seen before -- no AI/ML- or title-specific special-casing."""
    jd = JobRequirements(
        title="Structural Bridge Inspector",
        required_skills=["AutoCAD", "Structural Analysis"],
        responsibilities=["Inspect bridge structures for safety compliance"],
    )
    exp = WorkExperience(
        title="Senior Structural Bridge Inspector",
        company="CivilWorks Inc",
        technologies=["AutoCAD"],
        responsibilities=[
            "Led structural analysis for bridge safety inspections",
            "Mentored junior inspectors and coordinated with municipal engineers",
        ],
    )
    relevance = job_relevance(exp, jd)
    assert relevance > 0.3, f"expected a genuinely aligned but unseen role to score meaningfully, got {relevance}"


# ---------------------------------------------------------------------------
# 4 & 5. Embedding failure/unavailability still uses the deterministic
# fallback, and the existing embedding corroboration/dampening protection
# remains completely intact (untouched by this fix).
# ---------------------------------------------------------------------------

def test_no_embedding_provider_still_produces_a_sensible_deterministic_score(monkeypatch):
    """The real-world default for this environment (no GEMINI_API_KEY) --
    the fix must make the DETERMINISTIC fallback itself behave sensibly,
    not require an API key to score a strong match reasonably."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    emb_module.reset_embedding_provider_cache()

    relevance = job_relevance(STRONGLY_ALIGNED_ML_EXPERIENCE, AI_ML_JD)
    assert relevance >= 0.55, (
        f"expected the deterministic-only fallback (no embedding provider) "
        f"to still score a strong match well, got {relevance}"
    )


def test_embedding_zero_corroboration_dampening_still_applies_when_genuinely_unrelated(monkeypatch):
    """The corroboration/dampening protection this fix was explicitly
    told to preserve: when BOTH deterministic signals (Jaccard AND the
    new coverage signal) find zero overlap, a moderate-but-not-extreme
    embedding score must still be dampened, not trusted at full
    strength -- exactly the pre-existing behavior, untouched."""
    jd = JobRequirements(title="Registered Nurse", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Marketing Manager", company="BrightAd Agency")

    # Confirm both deterministic signals are genuinely zero for this pair
    # first (a precondition for the dampening branch to even apply).
    assert _semantic_role_relevance(exp, jd) == 0.0
    assert _semantic_role_coverage(exp, jd) == 0.0

    # A moderate embedding score (below the existing
    # _EMBEDDING_ZERO_CORROBORATION_TRUST_FLOOR = 0.6) must be dampened,
    # not trusted outright.
    monkeypatch.setattr(ee, "embedding_relevance", lambda a, b: 0.5)
    relevance = job_relevance(exp, jd)
    dampened_title_score = 0.5 * ee._EMBEDDING_ZERO_CORROBORATION_DAMPENING
    assert relevance == round(0.45 * dampened_title_score, 2), (
        f"expected the pre-existing zero-corroboration dampening to still "
        f"apply for a genuinely unrelated pair, got relevance={relevance}"
    )


def test_embedding_full_trust_still_applies_above_the_corroboration_floor(monkeypatch):
    """The other half of the SAME pre-existing protection: an embedding
    score that clears the trust floor is still trusted at full strength
    even with zero deterministic corroboration -- untouched by this fix."""
    jd = JobRequirements(title="Registered Nurse", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Marketing Manager", company="BrightAd Agency")
    assert _semantic_role_relevance(exp, jd) == 0.0
    assert _semantic_role_coverage(exp, jd) == 0.0

    monkeypatch.setattr(ee, "embedding_relevance", lambda a, b: 0.9)
    relevance = job_relevance(exp, jd)
    assert relevance == round(0.45 * 0.9, 2)


# ---------------------------------------------------------------------------
# 6. Identical resume content does not produce inconsistent experience
# calculations unless the JD legitimately changes relevance.
# ---------------------------------------------------------------------------

def test_identical_resume_against_different_jds_differs_only_for_legitimate_jd_reasons():
    """Same WorkExperience content, screened against two DIFFERENT JDs
    (mirroring the real resume_99/resume_103 scenario: same candidate
    content, different positions/JD requirements) -- years_relevant and
    role_relevance are EXPECTED and CORRECT to differ, because relevance
    is inherently relative to what's being screened for. This is not a
    bug; asserting the specific, explainable direction of the difference
    (more relevant for the aligned JD) proves it's driven by real JD
    content, not by any inconsistency in how the same resume content is
    interpreted."""
    resume = ResumeData(work_experience=[STRONGLY_ALIGNED_ML_EXPERIENCE])

    aligned_jd = AI_ML_JD
    unrelated_jd = JobRequirements(
        title="Warehouse Operations Coordinator",
        required_skills=["Forklift Operation", "Inventory Management"],
        responsibilities=["Coordinate warehouse logistics and inventory"],
        min_years_experience=3,
    )

    aligned_result = evaluate_experience(resume, aligned_jd)
    unrelated_result = evaluate_experience(resume, unrelated_jd)

    assert aligned_result.role_relevance > unrelated_result.role_relevance
    assert aligned_result.years_relevant > unrelated_result.years_relevant
    # Same underlying resume content evaluated twice against the SAME JD
    # must still be perfectly consistent/deterministic.
    repeat = evaluate_experience(resume, aligned_jd)
    assert repeat.model_dump() == aligned_result.model_dump()


# ---------------------------------------------------------------------------
# 7. Existing title-score isolation behavior remains intact.
# ---------------------------------------------------------------------------

def test_exact_title_match_still_scores_maximum_lexical_relevance():
    jd = JobRequirements(title="Backend Engineer - Python", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Backend Engineer - Python", company="X")
    assert job_relevance(exp, jd) == 0.45


def test_generic_shared_word_only_title_overlap_behavior_is_unchanged_by_this_fix():
    """The existing 'shared words must be distinctive, not just generic
    seniority/role vocabulary' backstop only zeroes the LEXICAL title_score
    component -- job_relevance() already fell back on the (pre-existing,
    untouched-by-this-fix) Jaccard signature backstop for this exact pair
    before this change (verified: 0.23 both before and after this fix, via
    _semantic_role_relevance's plain-title-token Jaccard overlap, which is
    NOT restricted by _STOP the way the lexical zeroing rule is). This
    test exists to prove the value is IDENTICAL before and after this fix,
    i.e. the new coverage signal contributes nothing extra here (no real
    skills/technologies on either side for it to find)."""
    jd = JobRequirements(title="Senior Backend Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Senior QA Engineer", company="X")
    relevance = job_relevance(exp, jd)
    assert relevance == 0.23, (
        f"expected the pre-existing (unrelated to this fix) Jaccard "
        f"title-token backstop value, got {relevance}"
    )
    # And confirm explicitly that the NEW coverage signal specifically
    # contributes nothing here (no skill-taxonomy overlap at all).
    from src.agents.experience_eval import _semantic_role_coverage
    assert _semantic_role_coverage(exp, jd) == 0.0


# ---------------------------------------------------------------------------
# 8. Existing skill matching behavior remains completely unaffected.
# ---------------------------------------------------------------------------

def test_skills_matcher_agent_unaffected_by_role_relevance_fix():
    agent = SkillsMatcherAgent()
    skills = [Skill(name="Python", category="language"), Skill(name="PyTorch", category="framework")]
    reqs = JobRequirements(required_skills=["Python", "PyTorch", "TensorFlow"], preferred_skills=[])

    async def run():
        return await agent.process({"extracted_skills": skills, "job_requirements": reqs})

    result = asyncio.run(run())["skills_match"]
    assert result.required_skills_met == 2
    assert result.required_skills_total == 3


# ---------------------------------------------------------------------------
# 9. Existing final-score weights (0.45 / 0.40 / 0.15) remain unchanged.
# ---------------------------------------------------------------------------

def test_job_relevance_weights_are_unchanged_by_this_fix(monkeypatch):
    monkeypatch.setattr(ee, "embedding_relevance", lambda a, b: None)
    jd = JobRequirements(
        title="Something With No Recognized Words At All",
        required_skills=["Python", "Django", "PostgreSQL"],
    )
    exp = WorkExperience(
        title="Zzz Unrelated Title Zzz", company="X",
        technologies=["Python", "Django", "PostgreSQL"],
        responsibilities=[],
    )
    relevance = job_relevance(exp, jd)
    expected_title_score = max(
        ee._semantic_role_relevance(exp, jd),
        ee._semantic_role_coverage(exp, jd),
    )
    expected = round(0.45 * expected_title_score + 0.4 * 1.0 + 0.15 * 0.0, 2)
    assert relevance == expected


# ---------------------------------------------------------------------------
# End-to-end: the REAL production path (screen_candidate()), not a copied
# reimplementation -- exercises document parsing, ResumeParserAgent,
# SkillExtractorAgent, SkillsMatcherAgent, ExperienceEvaluatorAgent, and
# DecisionSynthesizerAgent together, with job_requirements pre-seeded so
# no LLM/API key is needed (same pattern as
# tests/test_algorithm_version_cache_invalidation.py).
# ---------------------------------------------------------------------------

RESUME_TEXT = """ALEX MORGAN
Senior Machine Learning Engineer
alex.morgan@email.com | (555) 222-3344 | Remote

PROFESSIONAL SUMMARY
Senior Machine Learning Engineer with 3+ years building and deploying deep
learning models for production NLP and computer vision systems.

WORK EXPERIENCE

Senior Machine Learning Engineer
DeepStack AI | Remote | January 2021 - Present
- Led design and deployment of deep learning models for production NLP systems
- Trained and fine-tuned neural networks using PyTorch and TensorFlow
- Mentored junior engineers and led code reviews across the team
- Collaborated closely with product managers and cross-functional stakeholders
- Facilitated agile ceremonies including sprint planning and retrospectives

EDUCATION

Bachelor of Science in Computer Science
State University | 2020

TECHNICAL SKILLS

Languages: Python
Frameworks: PyTorch, TensorFlow
Concepts: Machine Learning, Deep Learning, NLP, Computer Vision
"""


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = Session()
    try:
        yield session
        session.commit()
    finally:
        session.close()


def test_screen_candidate_end_to_end_scores_ai_ml_aligned_resume_strongly(db_session, tmp_path):
    """Real production path: screen_candidate() -> workflow.run_full() ->
    ResumeParserAgent -> SkillExtractorAgent -> SkillsMatcherAgent ->
    ExperienceEvaluatorAgent -> DecisionSynthesizerAgent, with the current
    algorithm version. Proves the fix holds through the whole pipeline,
    not just at the job_relevance()/evaluate_experience() unit level."""
    org = repo.create_organization(db_session, "Test Org")
    dept = repo.create_department(db_session, org.id, "AI Engineering")
    position = repo.create_position(db_session, dept.id, "AI/ML Engineer")
    jd_text = (
        "AI/ML Engineer. Required: Python, Machine Learning, Deep Learning, "
        "PyTorch, TensorFlow. Preferred: NLP, Computer Vision, Docker, AWS, MLOps."
    )
    jd = repo.create_job_description(db_session, position.id, raw_text=jd_text)
    repo.save_job_requirements(
        db_session, jd.id, title="AI/ML Engineer", min_years_experience=3,
        required_skills=["Python", "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow"],
        preferred_skills=["NLP", "Computer Vision", "Docker", "AWS", "MLOps"],
        parsing_confidence=0.9,
    )
    db_session.flush()

    resume_path = tmp_path / "alex_morgan_resume.txt"
    resume_path.write_text(RESUME_TEXT, encoding="utf-8")

    screening = asyncio.run(
        screen_candidate(db_session, position_id=position.id, resume_file_path=str(resume_path))
    )
    db_session.flush()

    assert screening.algorithm_version == CURRENT_ALGORITHM_VERSION, (
        "result must come from the CURRENT algorithm version, not a cached "
        "old result"
    )
    assert screening.status == "completed"

    exp_eval = screening.experience_evaluation
    result = screening.result

    assert exp_eval.role_relevance >= 0.5, (
        f"expected strong role_relevance for an AI/ML-aligned resume via "
        f"the real production pipeline, got {exp_eval.role_relevance}"
    )
    assert exp_eval.years_relevant > 1.5, (
        f"expected most of the ~multi-year relevant role to count, got "
        f"{exp_eval.years_relevant}"
    )
    assert result.match_score >= 0.5, (
        f"expected a reasonably strong final match_score for a strongly "
        f"aligned candidate, got {result.match_score}"
    )
