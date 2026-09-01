"""Regression tests: "Supervised Learning" / "Unsupervised Learning"
elided-coordination extraction bug.

REAL BUG (confirmed via the real, unmodified production pipeline --
SkillExtractorAgent -> SkillsMatcherAgent, no mocking): the taxonomy
registered "Supervised Learning" and "Unsupervised Learning" only as
their bare, literal phrase ("supervised learning" / "unsupervised
learning" -- see the Batch that added
tests/test_ml_subconcept_skill_matching.py). extract_canonical_skills
does literal contiguous-substring matching, so a resume/JD written the
extremely common way -- coordinating the two terms and eliding the
repeated head noun, e.g. "supervised and unsupervised learning" rather
than "supervised learning and unsupervised learning" -- never contains
the literal substring "supervised learning" at all (the word after
"supervised" is "and", not "learning"). Only the second half's full
phrase ("unsupervised learning") is literally present, so
"Supervised Learning" was silently dropped even though the resume text
plainly claims it. The same elision happens symmetrically when the
coordination is written the other way round ("unsupervised and
supervised learning"), dropping "Unsupervised Learning" instead.

This is an EXTRACTION bug (SkillExtractorAgent/extract_canonical_skills
never produces the Skill object in the first place), not a
normalization or matching-logic bug -- once a Skill object exists,
SkillsMatcherAgent's existing exact-name-match step already credits it
correctly (this file confirms that too, end to end).

Fix (src/skill_taxonomy.py): both canonical entries gained the
coordinated-phrase forms, in both orders and covering the two distinct
strings normalize_skill_text produces from "and"/"&" vs "/"/"," ("X and
Y learning" vs "X Y learning") -- not a new matching mechanism, just the
additional literal phrases the elision produces. The plain bare-phrase
alias ("supervised learning" / "unsupervised learning" said on its own)
is completely unaffected.

This does NOT touch SkillsMatcherAgent, CREDIT_BY_QUALITY weights, any
scoring formula/threshold, or any other taxonomy entry (Cross-validation,
Model Evaluation, Pandas, NumPy, Scikit-learn, etc. -- all independently
verified against the real production pipeline to already match correctly
for standard phrasings and are left untouched here).
"""

import asyncio

from src.agents.skill_extractor import extract_skills_from_resume
from src.agents.skills_matcher import SkillsMatcherAgent
from src.models import JobRequirements, ResumeData, WorkExperience
from src.skill_taxonomy import extract_canonical_skills


def _run_matcher(skills, reqs):
    agent = SkillsMatcherAgent()
    return asyncio.run(agent.process({
        "extracted_skills": skills, "job_requirements": reqs,
    }))["skills_match"]


# ---------------------------------------------------------------------------
# 1. Taxonomy-level: the coordinated/elided forms are recognized, in both
#    orders and punctuation variants, without breaking the plain phrase.
# ---------------------------------------------------------------------------

def test_coordinated_and_form_extracts_both_terms():
    hits = {name for name, _ in extract_canonical_skills("Supervised and Unsupervised Learning")}
    assert hits == {"Supervised Learning", "Unsupervised Learning"}


def test_coordinated_ampersand_form_extracts_both_terms():
    hits = {name for name, _ in extract_canonical_skills("Supervised & Unsupervised Learning")}
    assert hits == {"Supervised Learning", "Unsupervised Learning"}


def test_coordinated_slash_form_extracts_both_terms():
    hits = {name for name, _ in extract_canonical_skills("Supervised/Unsupervised Learning")}
    assert hits == {"Supervised Learning", "Unsupervised Learning"}


def test_coordinated_comma_form_extracts_both_terms():
    hits = {name for name, _ in extract_canonical_skills("Supervised, Unsupervised Learning")}
    assert hits == {"Supervised Learning", "Unsupervised Learning"}


def test_reverse_order_coordination_extracts_both_terms():
    """The symmetric case: coordination written unsupervised-first must
    not drop 'Unsupervised Learning' the same way the forward order used
    to drop 'Supervised Learning'."""
    for text in (
        "Unsupervised and Supervised Learning",
        "Unsupervised/Supervised Learning",
        "Unsupervised, Supervised Learning",
    ):
        hits = {name for name, _ in extract_canonical_skills(text)}
        assert hits == {"Supervised Learning", "Unsupervised Learning"}, text


def test_coordination_inside_a_real_sentence_still_extracts_both():
    hits = {name for name, _ in extract_canonical_skills(
        "Hands-on experience with supervised and unsupervised learning algorithms"
    )}
    assert "Supervised Learning" in hits
    assert "Unsupervised Learning" in hits


def test_bare_phrase_forms_are_unaffected():
    """The pre-existing, already-correct behavior for the plain phrases
    said on their own must be completely unchanged."""
    assert extract_canonical_skills("supervised learning") == [("Supervised Learning", "technical")]
    assert extract_canonical_skills("unsupervised learning") == [("Unsupervised Learning", "technical")]


def test_unrelated_text_does_not_spuriously_match():
    assert extract_canonical_skills("supervised the deployment process") == []
    assert extract_canonical_skills("unsupervised access to the server") == []


# ---------------------------------------------------------------------------
# 2. End-to-end through the REAL production pipeline: extraction ->
#    matching, using the exact JD phrasing style from the bug report.
# ---------------------------------------------------------------------------

AI_ML_JD_REQUIRED_SKILLS = [
    "Machine Learning",
    "Supervised Learning",
    "Unsupervised Learning",
    "Model Evaluation / Cross-validation",
    "Pandas",
    "NumPy",
    "scikit-learn",
]


def test_elided_coordination_resume_no_longer_shows_false_gaps():
    """Direct reproduction of the reported issue: a resume that only ever
    writes the coordinated/elided form (never the bare "supervised
    learning" phrase on its own) must still get full credit for both
    required skills, not show them as gaps."""
    resume = ResumeData(
        skills_section=["Python", "Pandas", "NumPy", "Scikit-learn", "Machine Learning"],
        work_experience=[
            WorkExperience(
                title="Machine Learning Intern", company="Acme AI",
                responsibilities=[
                    "Built supervised and unsupervised learning models for classification "
                    "and clustering tasks",
                    "Performed model evaluation using cross-validation, precision, recall "
                    "and F1-score",
                    "Used pandas and numpy for data preprocessing and feature engineering",
                ],
                technologies=["Python", "Pandas", "NumPy", "Scikit-learn"],
            )
        ],
    )
    skills = extract_skills_from_resume(resume)
    names = {s.name for s in skills}
    assert "Supervised Learning" in names, (
        f"'Supervised Learning' still missing from extraction -- elision fix regressed. "
        f"Extracted: {sorted(names)}"
    )
    assert "Unsupervised Learning" in names

    reqs = JobRequirements(title="AI/ML Engineer", required_skills=AI_ML_JD_REQUIRED_SKILLS)
    result = _run_matcher(skills, reqs)

    matched = {m.requirement for m in result.matches if m.matched}
    for expected in AI_ML_JD_REQUIRED_SKILLS:
        assert expected in matched, (
            f"{expected!r} should be matched given the resume evidence, but matches were: "
            f"{[(m.requirement, m.matched) for m in result.matches]}"
        )
    assert result.required_skills_met == result.required_skills_total == 7


def test_only_one_half_of_coordination_still_leaves_the_other_a_genuine_gap():
    """Guard against over-fixing: a resume that only demonstrates ONE of
    the two (via the plain phrase) must still show the other as a real
    gap -- this fix must not make the two terms imply each other."""
    resume = ResumeData(
        work_experience=[
            WorkExperience(
                title="Data Scientist", company="X",
                responsibilities=["Built supervised learning models for churn prediction"],
            )
        ],
    )
    skills = extract_skills_from_resume(resume)
    names = {s.name for s in skills}
    assert "Supervised Learning" in names
    assert "Unsupervised Learning" not in names

    reqs = JobRequirements(required_skills=["Supervised Learning", "Unsupervised Learning"])
    result = _run_matcher(skills, reqs)
    assert result.required_skills_met == 1
    gaps = {m.requirement for m in result.matches if not m.matched}
    assert gaps == {"Unsupervised Learning"}


# ---------------------------------------------------------------------------
# 3. Determinism (project-wide convention).
# ---------------------------------------------------------------------------

def test_elided_coordination_extraction_is_deterministic():
    resume = ResumeData(
        work_experience=[
            WorkExperience(
                title="ML Engineer", company="X",
                responsibilities=["Built supervised and unsupervised learning models"],
            )
        ],
    )
    s1 = extract_skills_from_resume(resume)
    s2 = extract_skills_from_resume(resume)
    assert [s.name for s in s1] == [s.name for s in s2]
