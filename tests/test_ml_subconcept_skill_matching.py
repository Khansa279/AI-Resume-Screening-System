"""Regression tests: ML sub-concept skills (Supervised Learning,
Unsupervised Learning, Cross-validation, Classification, Regression) were
entirely absent from src/skill_taxonomy.py's canonical skill entries.

REAL BUG (confirmed via the real, unmodified production pipeline --
SkillExtractorAgent -> SkillsMatcherAgent, no mocking): a resume whose
work-experience bullets literally state "Built supervised classification
models using scikit-learn", "Applied cross-validation to evaluate model
performance", and "Trained regression models to predict housing prices"
produced extracted_skills containing only Machine Learning, Deep
Learning, Scikit-learn, and Python -- none of Supervised Learning,
Unsupervised Learning, Cross-validation, Classification, or Regression
were recognized at all, despite being literally present in the resume
text. Because SkillExtractorAgent never produced a Skill object for
these concepts, SkillsMatcherAgent's existing exact/semantic/partial
matching logic never had anything to compare against, and every one of
these requirements was reported as a complete gap (match_quality="none")
regardless of the real textual evidence in the resume. Root cause: a
taxonomy/extraction gap, not a matching-logic bug -- these terms simply
were not in _SKILL_ENTRIES, so extract_canonical_skills() could never
recognize them in resume text in the first place.

Fix (src/skill_taxonomy.py): five new canonical entries -- Supervised
Learning, Unsupervised Learning, Cross-validation, Classification,
Regression -- added as their OWN distinct skills, matched only against
genuine ML-context phrasing (not blanket keyword equivalence with
"Machine Learning", and not bare "regression"/"classification" alone,
to avoid crediting unrelated resume text like "regression testing" or
generic "classification" outside an ML context). No changes to
SkillsMatcherAgent's scoring logic, CREDIT_BY_QUALITY weights, role-family
logic, or any existing scoring threshold.

These tests exercise the REAL production functions end-to-end
(extract_skills_from_resume -> SkillsMatcherAgent.process), matching this
repo's existing test-suite convention (see
test_batch4_skill_match_consistency.py, test_batch7_role_relevance_and_soft_skills.py).
"""

import asyncio

from src.agents.skill_extractor import extract_skills_from_resume
from src.agents.skills_matcher import SkillsMatcherAgent
from src.models import JobRequirements, ResumeData, Skill, WorkExperience
from src.skill_taxonomy import extract_canonical_skills


AI_ML_JD_REQUIRED_SKILLS = [
    "Machine Learning",
    "Supervised Learning",
    "Unsupervised Learning",
    "Cross-validation",
    "Classification",
    "Regression",
]


def _run_matcher(skills: list[Skill], reqs: JobRequirements):
    agent = SkillsMatcherAgent()
    return asyncio.run(agent.process({
        "extracted_skills": skills, "job_requirements": reqs,
    }))["skills_match"]


# ---------------------------------------------------------------------------
# 1. Core reproduction: literal textual evidence for sub-concepts must no
#    longer be reported as a complete gap.
# ---------------------------------------------------------------------------

def test_literal_subconcept_evidence_is_no_longer_a_false_gap():
    """Direct reproduction of the reported bug: a resume whose bullets
    literally name these ML sub-concepts must have them recognized and
    credited, not reported as completely missing."""
    resume = ResumeData(
        skills_section=["Python", "Machine Learning", "Deep Learning"],
        work_experience=[
            WorkExperience(
                title="ML Engineer", company="X",
                responsibilities=[
                    "Built supervised classification models using scikit-learn",
                    "Applied cross-validation to evaluate model performance",
                    "Trained regression models to predict housing prices",
                ],
            )
        ],
    )
    skills = extract_skills_from_resume(resume)
    names = {s.name for s in skills}
    assert "Classification" in names
    assert "Cross-validation" in names
    assert "Regression" in names

    reqs = JobRequirements(title="AI/ML Engineer", required_skills=AI_ML_JD_REQUIRED_SKILLS)
    result = _run_matcher(skills, reqs)

    matched = {m.requirement for m in result.matches if m.matched}
    assert "Machine Learning" in matched
    assert "Classification" in matched
    assert "Cross-validation" in matched
    assert "Regression" in matched
    # Only what the resume ACTUALLY provides evidence for is credited --
    # "Supervised Learning" and "Unsupervised Learning" (the literal
    # phrases) were never mentioned, only "supervised classification".
    assert "Supervised Learning" not in matched
    assert "Unsupervised Learning" not in matched
    assert result.required_skills_met == 4
    assert result.required_skills_total == 6


def test_exact_phrase_evidence_for_supervised_and_unsupervised_learning_is_credited():
    """When the resume DOES use the exact phrases, they must be credited
    too -- this is not special-cased away by the classification/regression
    guard logic."""
    resume = ResumeData(
        work_experience=[
            WorkExperience(
                title="Data Scientist", company="X",
                responsibilities=[
                    "Built supervised learning models for churn prediction",
                    "Applied unsupervised learning techniques for customer segmentation",
                ],
            )
        ],
    )
    skills = extract_skills_from_resume(resume)
    names = {s.name for s in skills}
    assert "Supervised Learning" in names
    assert "Unsupervised Learning" in names

    reqs = JobRequirements(required_skills=["Supervised Learning", "Unsupervised Learning"])
    result = _run_matcher(skills, reqs)
    assert result.required_skills_met == 2


# ---------------------------------------------------------------------------
# 2. No over-crediting: generic "Machine Learning" evidence alone must NOT
#    satisfy any of the more specific sub-concept requirements. This is
#    the explicit "do not make ML automatically mean X" constraint.
# ---------------------------------------------------------------------------

def test_generic_machine_learning_evidence_does_not_satisfy_subconcepts():
    """A candidate who only demonstrates generic ML (no mention of
    supervised/unsupervised/cross-validation/classification/regression
    anywhere) must still show these as real gaps -- exactly the
    pre-existing, correct behavior for unrelated/unclaimed skills."""
    resume = ResumeData(
        skills_section=["Python", "Machine Learning"],
        projects=[
            "Predictive Model -- built a Machine Learning model in Python "
            "to predict outcomes from tabular data",
        ],
    )
    skills = extract_skills_from_resume(resume)
    names = {s.name for s in skills}
    assert "Machine Learning" in names
    for absent in ("Supervised Learning", "Unsupervised Learning", "Cross-validation",
                   "Classification", "Regression"):
        assert absent not in names, f"{absent!r} should NOT be inferred from generic ML text alone"

    reqs = JobRequirements(required_skills=AI_ML_JD_REQUIRED_SKILLS)
    result = _run_matcher(skills, reqs)
    matched = {m.requirement for m in result.matches if m.matched}
    assert matched == {"Machine Learning"}
    assert result.required_skills_met == 1
    assert result.required_skills_total == 6


def test_pytorch_transformers_and_unsupervised_learning_remain_fully_independent():
    """Explicit guard from the task's own example: ML must NOT imply
    Cross-validation, PyTorch, Transformers, or Unsupervised Learning."""
    resume = ResumeData(skills_section=["Machine Learning"])
    skills = extract_skills_from_resume(resume)
    names = {s.name for s in skills}
    assert names == {"Machine Learning"}

    reqs = JobRequirements(required_skills=[
        "Machine Learning", "PyTorch", "Transformers", "Unsupervised Learning", "Cross-validation",
    ])
    result = _run_matcher(skills, reqs)
    matched = {m.requirement for m in result.matches if m.matched}
    assert matched == {"Machine Learning"}
    assert result.required_skills_met == 1
    assert result.required_skills_total == 5


# ---------------------------------------------------------------------------
# 3. False-positive guard: bare "regression"/"classification" in a
#    NON-ML context must not be credited as the ML skill.
# ---------------------------------------------------------------------------

def test_qa_regression_testing_does_not_falsely_credit_ml_regression_skill():
    resume = ResumeData(
        work_experience=[
            WorkExperience(
                title="QA Engineer", company="X",
                responsibilities=[
                    "Ran regression testing suites before each release",
                    "Automated regression tests with Selenium",
                ],
            )
        ],
    )
    skills = extract_skills_from_resume(resume)
    names = {s.name for s in skills}
    assert "Regression" not in names, (
        "'regression testing' (a QA term) must not be credited as the ML "
        "'Regression' skill"
    )


def test_bare_regression_and_classification_words_are_not_standalone_aliases():
    """Direct taxonomy-level guard: the bare single words must not match
    on their own (only genuine ML-context phrases should)."""
    assert extract_canonical_skills("regression") == []
    assert extract_canonical_skills("classification") == []
    # But ML-context phrasing IS recognized.
    assert any(name == "Regression" for name, _ in extract_canonical_skills("regression model"))
    assert any(name == "Classification" for name, _ in extract_canonical_skills("classification algorithm"))


def test_bare_jd_requirement_string_still_matches_once_genuinely_earned():
    """A JD asking for the bare word "Regression"/"Classification" (as in
    the reported example) must still resolve correctly once the candidate
    has genuine ML-context evidence -- confirms the canonical display name
    (not just the alias) is what the exact-match step compares against."""
    resume = ResumeData(
        work_experience=[
            WorkExperience(
                title="ML Engineer", company="X",
                responsibilities=["Built regression models and classification algorithms for pricing"],
            )
        ],
    )
    skills = extract_skills_from_resume(resume)
    reqs = JobRequirements(required_skills=["Regression", "Classification"])
    result = _run_matcher(skills, reqs)
    assert result.required_skills_met == 2
    for m in result.matches:
        assert m.matched
        assert m.match_quality == "exact"


# ---------------------------------------------------------------------------
# 4. Preserve previously-fixed substring protections (Java != JavaScript,
#    C != C++) -- unaffected by this change.
# ---------------------------------------------------------------------------

def test_java_still_does_not_satisfy_javascript_after_this_change():
    skills = [Skill(name="Java", category="language")]
    reqs = JobRequirements(required_skills=["JavaScript"])
    result = _run_matcher(skills, reqs)
    assert result.required_skills_met == 0


def test_c_still_does_not_satisfy_cplusplus_after_this_change():
    skills = [Skill(name="C", category="language")]
    reqs = JobRequirements(required_skills=["C++"])
    result = _run_matcher(skills, reqs)
    assert result.required_skills_met == 0


# ---------------------------------------------------------------------------
# 5. Determinism (project-wide convention).
# ---------------------------------------------------------------------------

def test_subconcept_extraction_and_matching_is_deterministic():
    resume = ResumeData(
        work_experience=[
            WorkExperience(
                title="ML Engineer", company="X",
                responsibilities=[
                    "Built supervised classification models using scikit-learn",
                    "Applied cross-validation to evaluate model performance",
                    "Trained regression models to predict housing prices",
                ],
            )
        ],
    )
    s1 = extract_skills_from_resume(resume)
    s2 = extract_skills_from_resume(resume)
    assert [s.name for s in s1] == [s.name for s in s2]

    reqs = JobRequirements(required_skills=AI_ML_JD_REQUIRED_SKILLS)
    r1 = _run_matcher(s1, reqs)
    r2 = _run_matcher(s2, reqs)
    assert r1.model_dump() == r2.model_dump()
