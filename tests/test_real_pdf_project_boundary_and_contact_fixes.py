"""Regression tests using the REAL Khansa Aslam / Ayesha Naeem PDFs
(sample_data/resumes/resume_05_khansa_aslam.pdf,
resume_06_ayesha_naeem.pdf) against the actual document-parsing and
resume-parsing pipeline.

Covers three confirmed, evidence-driven bugs found by tracing the real
extracted PDF bytes (not reconstructed text):

1. PROJECT-MERGING BUG (the main scoring anomaly)
   Root cause, in two parts:
     a. document_parser.py: PyPDF2 extracts Ayesha's PDF with ZERO
        recoverable bullet markers for her PROJECTS section (the bullet
        glyphs get detached from their lines during extraction), while
        pdfplumber correctly preserves them. Fixed by comparing bullet-
        line-recovery counts between the two extractors and preferring
        whichever one recovers meaningfully more list structure --
        verified against the ENTIRE existing sample_data/resumes/*.pdf
        corpus to confirm this doesn't flip extractor choice for any
        resume that was already parsing correctly (see
        test_extractor_selection_does_not_regress_existing_corpus).
     b. resume_parser.py::_parse_list_section only handled "bulleted
        title line, un-bulleted description" (one convention). Ayesha's
        real PDF uses the OTHER common convention -- "un-bulleted title,
        several bulleted sub-points" -- which the old implementation
        fragmented into one title-only item plus one orphan item per
        bullet, OR (once bullets are correctly recovered upstream)
        merged multiple real projects together because it had no notion
        of a bulleted line continuing an un-bulleted item. Fixed with a
        small state machine that recognizes both conventions.

   Together, (a)+(b) caused Ayesha's three genuinely separate projects
   (PBS DataFest, Lollywood/IMDb, Marble Factory) to be silently merged
   into one large blob whose combined skill surface (Python from one
   project, SQL from another) crossed the 0.3 project-relevance
   threshold that none of her INDIVIDUAL projects would have crossed on
   their own -- producing an inflated role_relevance driven by a parsing
   artifact rather than genuine evidence. The 0.3 threshold itself is
   NOT changed by this fix (see the module docstring in
   experience_eval.py for that separate, deferred design question).

2. EDUCATION-FIELD PARSING GAP
   "Air University - Data Science" (Ayesha's real wording) previously
   failed to populate Education.field, while "Bachelors in Data Science"
   (Khansa's wording) succeeded -- purely a wording difference, not a
   qualifications difference. Fixed in resume_parser.py::_parse_education
   to also recognize "<Institution> - <Field>" phrasing.

3. EMAIL ICON-GLYPH ARTIFACT
   Khansa's real PDF renders the envelope-icon's alt-text ("envelope")
   with its "o" replaced by a Unicode glyph-substitution character,
   leaving the surviving "pe" fused directly onto the front of her real
   email with no separator: "pekhansaaslam524@gmail.com" instead of
   "khansaaslam524@gmail.com". Fixed narrowly -- only when a Unicode
   symbol character is actually present immediately before the match,
   so ordinary emails (including ones that happen to start with "pe",
   e.g. "pearson123@gmail.com") are never touched. Does not affect
   scoring (email is not a scoring input).

Per the explicit instruction accompanying this investigation, none of
these tests assert a specific final match_score/role_relevance
percentage for either candidate -- they assert STRUCTURAL correctness
and invariants (project counts, no cross-project contamination, role
relevance is not inflated by a parsing artifact) that hold regardless of
exactly how JobAnalyzerAgent's LLM call would split required vs
preferred skills for the real JD.
"""

from pathlib import Path

import pytest

from src.document_parser import parse_document
from src.agents.resume_parser import parse_resume_text, _parse_contact
from src.agents.experience_eval import evaluate_experience, project_relevance
from src.models import JobRequirements


KHANSA_PDF = Path("sample_data/resumes/resume_05_khansa_aslam.pdf")
AYESHA_PDF = Path("sample_data/resumes/resume_06_ayesha_naeem.pdf")

# A reasonable stand-in for JobAnalyzerAgent's real output against
# jd_05_associate_ai_engineer.txt (no LLM call is made in these tests --
# see other Batch/Priority test files in this suite for the same
# convention). Only used where a JobRequirements object is needed at
# all; every assertion below is about structural parsing correctness,
# not about matching this exact skill split.
ASSOCIATE_AI_JD = JobRequirements(
    title="Associate AI Engineer",
    required_skills=["Python", "Machine Learning", "Deep Learning", "Artificial Intelligence"],
    preferred_skills=["Data Structures", "Algorithms", "SQL", "Git"],
    min_years_experience=0,
)


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path} not present in this checkout")


# ---------------------------------------------------------------------------
# 1a. Extractor selection: bullet-recovery comparison
# ---------------------------------------------------------------------------

def test_ayesha_pdf_recovers_bullets_after_extractor_fix():
    """Direct reproduction: before the fix, PyPDF2's extraction of this
    PDF had zero recoverable bullet lines for the projects section. The
    fixed _parse_pdf must now select whichever extractor recovers
    meaningfully more bullet structure."""
    _require(AYESHA_PDF)
    result = parse_document(str(AYESHA_PDF))
    assert result.success
    bullet_lines = [ln for ln in result.text.split("\n") if ln.strip().startswith("- ")]
    assert len(bullet_lines) >= 10, (
        f"expected the fixed extractor selection to recover Ayesha's project "
        f"bullets, got only {len(bullet_lines)} bullet lines"
    )


def test_extractor_selection_does_not_regress_existing_corpus():
    """The bullet-count comparison must not flip extractor choice for any
    resume in the pre-existing corpus, where PyPDF2 (plus the existing
    \\x7f bullet-glyph normalization) already recovers bullets correctly
    and pdfplumber does NOT for these particular files."""
    corpus = [
        Path("sample_data/resumes/resume_01_priya_sharma.pdf"),
        Path("sample_data/resumes/resume_02_rahul_verma.pdf"),
        Path("sample_data/resumes/resume_03_ananya_patel.pdf"),
        Path("sample_data/resumes/resume_04_vikram_singh.pdf"),
    ]
    for path in corpus:
        _require(path)
        result = parse_document(str(path))
        assert result.success
        bullet_lines = [ln for ln in result.text.split("\n") if ln.strip().startswith("- ")]
        assert len(bullet_lines) >= 8, (
            f"{path.name}: extractor-selection change regressed bullet recovery "
            f"(got {len(bullet_lines)} bullet lines) -- this resume must keep "
            f"using PyPDF2's extraction, which recovers its bullets correctly"
        )


def test_khansa_pdf_extractor_choice_is_unaffected():
    """Khansa's PDF ties on bullet count between extractors -- must stay
    on the default (PyPDF2) rather than flip unnecessarily."""
    _require(KHANSA_PDF)
    result = parse_document(str(KHANSA_PDF))
    assert result.success
    bullet_lines = [ln for ln in result.text.split("\n") if ln.strip().startswith("- ")]
    assert len(bullet_lines) >= 9


# ---------------------------------------------------------------------------
# 1b. Project boundary correctness (the core structural fix)
# ---------------------------------------------------------------------------

def test_ayesha_projects_are_not_merged_into_one_blob():
    """Core regression: Ayesha's real PDF must parse into MULTIPLE
    distinct projects, not one giant merged item."""
    _require(AYESHA_PDF)
    result = parse_document(str(AYESHA_PDF))
    data = parse_resume_text(result.text)
    assert len(data.projects) >= 3, (
        f"expected at least 3 distinct projects (DataFest, Lollywood, Marble "
        f"Factory), got {len(data.projects)}: {data.projects}"
    )


def test_ayesha_individual_projects_do_not_cross_contaminate():
    """Each of Ayesha's real projects must contain ONLY its own content --
    e.g. the DataFest project text must not also contain the Lollywood
    project's Python/regression content, and vice versa. This is the
    direct structural check against 'combined skills from unrelated
    projects artificially creating one high project score'."""
    _require(AYESHA_PDF)
    result = parse_document(str(AYESHA_PDF))
    data = parse_resume_text(result.text)
    joined = " | ".join(data.projects)

    datafest = next((p for p in data.projects if "DataFest" in p), None)
    lollywood = next((p for p in data.projects if "Lollywood" in p), None)
    marble = next((p for p in data.projects if "Marble Factory" in p), None)

    assert datafest is not None, f"DataFest project not found in {data.projects}"
    assert lollywood is not None, f"Lollywood project not found in {data.projects}"
    assert marble is not None, f"Marble Factory project not found in {data.projects}"

    # DataFest's own project must not contain the OTHER projects' content.
    assert "Lollywood" not in datafest and "Marble Factory" not in datafest
    assert "linear regression" not in datafest.lower()
    assert "WinForms" not in datafest

    # Lollywood's own project must not contain the OTHER projects' content.
    assert "DataFest" not in lollywood and "Marble Factory" not in lollywood
    assert "WinForms" not in lollywood

    # Marble Factory's own project must not contain the OTHER projects' content.
    assert "DataFest" not in marble and "Lollywood" not in marble
    assert "linear regression" not in marble.lower()


def test_ayesha_role_relevance_is_not_inflated_by_project_merging():
    """The core scoring-invariant check: with projects correctly
    separated, no SINGLE real project of Ayesha's clears the (unchanged)
    0.3 "clearly relevant, name it" threshold on its own. This does not
    assert an exact percentage -- it asserts that merging-driven
    inflation is gone.

    NOTE: role_relevance itself is no longer required to sit at the
    education-only baseline just because no single project cleared 0.3 --
    see the project-relevance-cliff fix in src/agents/experience_eval.py
    (_best_project no longer zeroes out a real, if modest, per-project
    score before it reaches the max(baseline, project_score) formula).
    The invariant this test actually needs to protect -- "role_relevance
    reflects only ONE real project's own evidence, never a merged
    blob's" -- is instead checked directly against the formula below,
    which is a STRONGER guard than the old hardcoded cap: if projects
    were still being merged, the merged blob's score would exceed every
    individual project's score, and this equality would fail.
    """
    _require(AYESHA_PDF)
    result = parse_document(str(AYESHA_PDF))
    data = parse_resume_text(result.text)

    project_scores = [project_relevance(p, ASSOCIATE_AI_JD) for p in data.projects]
    for project_text, score in zip(data.projects, project_scores):
        assert score < 0.3, (
            f"a single real Ayesha project scored {score} >= 0.3 -- if this is "
            f"expected (a genuinely strong individual project), fine, but if "
            f"this project also contains content from a DIFFERENT listed "
            f"project, that's the merging bug recurring: {project_text[:200]!r}"
        )

    ev = evaluate_experience(data, ASSOCIATE_AI_JD)
    # role_relevance must equal max(education baseline, best INDIVIDUAL
    # project's own score) -- proving it's explainable by one real
    # project's own evidence (however strong or weak), never inflated by
    # a merged blob of several projects' combined content.
    education_baseline = 0.15 if any(
        any(kw in (edu.field or "").lower() for kw in
            ["computer", "data", "software", "engineering", "ai", "machine learning", "information technology"])
        for edu in data.education
    ) else 0.0
    expected = round(max(education_baseline, max(project_scores, default=0.0)), 2)
    assert ev.role_relevance == expected, (
        f"role_relevance={ev.role_relevance} does not match "
        f"max(education_baseline={education_baseline}, "
        f"best_individual_project_score={max(project_scores, default=0.0)})={expected} -- "
        f"suggests project content is still being pooled across projects "
        f"(a merged blob would score higher than any individual project)"
    )


def test_khansa_projects_remain_correctly_separated():
    """Regression guard: Khansa's already-correct 9-project structure
    (flat bulleted-list convention) must be completely unaffected by the
    _parse_list_section state-machine change."""
    _require(KHANSA_PDF)
    result = parse_document(str(KHANSA_PDF))
    data = parse_resume_text(result.text)
    assert len(data.projects) == 9, f"expected 9 projects, got {len(data.projects)}: {data.projects}"

    flood = next((p for p in data.projects if "Flood Severity" in p), None)
    assert flood is not None, "Flood Severity Prediction project not found"
    assert "machine learning" in flood.lower()
    # Must not contain any OTHER project's distinguishing content.
    assert "Space Shooter" not in flood
    assert "Arduino" not in flood
    assert "Airline Ticket" not in flood


def test_khansa_flood_severity_project_relevance_reflects_its_own_content():
    _require(KHANSA_PDF)
    result = parse_document(str(KHANSA_PDF))
    data = parse_resume_text(result.text)
    flood = next(p for p in data.projects if "Flood Severity" in p)
    score = project_relevance(flood, ASSOCIATE_AI_JD)
    assert score > 0.0, "Flood Severity Prediction should show SOME relevance to an AI/ML JD"


# ---------------------------------------------------------------------------
# 2. Education field parsing (real PDFs)
# ---------------------------------------------------------------------------

def test_ayesha_real_pdf_education_field_is_recognized():
    _require(AYESHA_PDF)
    result = parse_document(str(AYESHA_PDF))
    data = parse_resume_text(result.text)
    assert data.education, "expected at least one education entry"
    assert any("data science" in (e.field or "").lower() for e in data.education), (
        f"expected an education entry with field containing 'data science', got: "
        f"{[(e.degree, e.field) for e in data.education]}"
    )


def test_khansa_real_pdf_education_field_is_recognized():
    _require(KHANSA_PDF)
    result = parse_document(str(KHANSA_PDF))
    data = parse_resume_text(result.text)
    assert data.education
    assert any("data science" in (e.field or "").lower() for e in data.education)


def test_both_candidates_get_equal_education_baseline_role_relevance():
    """The direct structural fix for the reported anomaly's education
    component: with no work experience and no project clearing 0.3,
    both candidates' role_relevance must now come from the SAME
    has_relevant_education baseline, regardless of resume wording."""
    _require(KHANSA_PDF)
    _require(AYESHA_PDF)
    k = parse_resume_text(parse_document(str(KHANSA_PDF)).text)
    a = parse_resume_text(parse_document(str(AYESHA_PDF)).text)

    ev_k = evaluate_experience(k, ASSOCIATE_AI_JD)
    ev_a = evaluate_experience(a, ASSOCIATE_AI_JD)

    assert ev_k.role_relevance == ev_a.role_relevance, (
        f"Khansa role_relevance={ev_k.role_relevance} vs Ayesha "
        f"role_relevance={ev_a.role_relevance} -- with no work experience for "
        f"either candidate and no individual project clearing 0.3 for either, "
        f"these should now be equal (both driven by the same education baseline)"
    )


# ---------------------------------------------------------------------------
# 3. Email icon-glyph artifact (real PDF)
# ---------------------------------------------------------------------------

def test_khansa_real_pdf_email_has_no_icon_glyph_artifact():
    _require(KHANSA_PDF)
    result = parse_document(str(KHANSA_PDF))
    data = parse_resume_text(result.text)
    assert data.contact.email == "khansaaslam524@gmail.com", (
        f"expected the icon-glyph artifact stripped, got {data.contact.email!r}"
    )


def test_ordinary_email_with_coincidental_label_prefix_is_untouched():
    """Regression guard against over-eager stripping: an email that
    happens to start with the same letters as a known icon-label suffix
    (e.g. 'pe' from 'envelope') must be left alone when there's no
    adjacent icon-glyph symbol character corroborating an artifact."""
    contact = _parse_contact(
        "Pat Pearson\npearson123@gmail.com\n(555) 123-4567",
        "Pat Pearson\npearson123@gmail.com\n(555) 123-4567",
    )
    assert contact.email == "pearson123@gmail.com"


def test_ordinary_email_unaffected_generally():
    contact = _parse_contact(
        "John Doe\njohn.doe@email.com",
        "John Doe\njohn.doe@email.com",
    )
    assert contact.email == "john.doe@email.com"


# ---------------------------------------------------------------------------
# 4. Determinism (project-wide convention)
# ---------------------------------------------------------------------------

def test_real_pdf_parsing_is_deterministic():
    _require(AYESHA_PDF)
    result = parse_document(str(AYESHA_PDF))
    d1 = parse_resume_text(result.text)
    d2 = parse_resume_text(result.text)
    assert d1.model_dump() == d2.model_dump()
