"""Regression tests for a REAL production bug traced through actual
database records (screening.db): candidate "Alex Morgan" screened against
the "AI/ML Engineer" position (resume_id=99, job_description_id=50)
stored a screening result of match_score=78%, role_relevance=10%,
years_relevant~1.4y, despite matching 15/15 required and 3/3 preferred
skills.

ROOT CAUSE (traced against the real DB + real code, not a synthetic
reproduction): the bug is entirely upstream of the relevance-scoring
formulas in src/agents/experience_eval.py. src/agents/resume_parser.py's
_parse_experience() fragmented Alex Morgan's ONE real "Machine Learning
Engineer" role (2023-Present, six real bullets) into 12 separate
WorkExperience rows in the database, because:

  1. The job header ("Machine Learning Engineer — Nova Analytics |
     Islamabad, Pakistan | 2023–Present") is on a SINGLE physical line
     using an em-dash + pipe format the parser didn't recognize -- it
     only knew the "bare title line, then a SEPARATE company/dates
     line" convention. So the whole line became an opaque title with no
     dates at all, and job_years() fell back to a flat, wrong 0.5-year
     default instead of computing "2023 to today".
  2. Each of that job's six real bullets word-wrapped across two
     physical lines in the extracted text (bullet marker only on the
     first line, e.g. "- Developed ... for customer" / "and operational
     prediction tasks."). The parser's bullet-collection loop stopped
     collecting the instant it saw the un-marked second line and handed
     it back to the outer loop, which read it as a brand-new job's
     title -- fabricating one fake, near-empty WorkExperience per
     wrapped bullet.
  3. The resume's "SELECTED PROJECTS" heading (as opposed to bare
     "Projects") wasn't recognized by _HEADING_RE, so that whole section
     also leaked into "experience" and became more fake entries.

12 fragments (11 of them near-empty, all defaulting to job_years()=0.5)
then diluted job_relevance()/evaluate_experience()'s duration-weighted
aggregation far below what the candidate's REAL, single, highly relevant
job history warranted -- 10% role_relevance and ~1.4 "relevant" years
for someone who is a working ML Engineer with 3+ years of directly
on-topic experience is a resume-PARSING defect, not a scoring-formula
defect.

FIX (src/agents/resume_parser.py), fully generalized -- no candidate
name, job title, or company is hardcoded anywhere in the fix:
  1. _split_inline_job_header() recognizes the "Title — Company |
     Location | Dates" single-line format (guarded by requiring BOTH a
     date signal and an em/en-dash or pipe separator, so it can't
     misfire on an ordinary hyphenated title with no dates).
  2. The bullet-collection loop in _parse_experience() now folds a
     non-bulleted, non-blank line back onto the PREVIOUS bullet (as a
     wrapped continuation) unless that line itself looks like a real
     job header (_looks_like_job_header) -- the same date+separator
     signal used by fix 1.
  3. _HEADING_RE / _SECTION_ALIASES now also recognize "Selected
     Projects", "Key Projects", "Notable Projects", "Academic Projects",
     and "Technical Projects" as aliases of the existing "projects"
     section, the same pattern already used for "Career Break" /
     "Education & Training" (see test_priority7_fake_work_experience.py).

These tests use the REAL, unmodified parse_resume_text() and the REAL,
unmodified job_relevance()/evaluate_experience() -- no mocking -- exactly
like test_priority7_fake_work_experience.py and
test_real_pdf_project_boundary_and_contact_fixes.py already do elsewhere
in this suite. The Alex Morgan resume text below is the ACTUAL extracted
text stored in screening.db's resumes.raw_text column for resume_id=99
(copied into sample_data/resumes/alex_morgan_ai_ml_engineer.txt), not a
rewritten or simplified stand-in.
"""

from pathlib import Path

from src.agents.resume_parser import parse_resume_text
from src.agents.experience_eval import evaluate_experience, job_relevance, job_years
from src.models import JobRequirements


ALEX_MORGAN_RESUME = Path("sample_data/resumes/alex_morgan_ai_ml_engineer.txt")

# The real, structured JobRequirements this candidate was actually
# screened against (job_description_id=50 / position "AI/ML Engineer"),
# reconstructed field-for-field from the job_requirements /
# job_requirement_skills rows in screening.db -- not invented.
REAL_AI_ML_ENGINEER_REQUIREMENTS = JobRequirements(
    title="AI/ML Engineer",
    summary=(
        "Develop, evaluate, and deploy machine learning solutions, "
        "including supervised/unsupervised models, NLP and deep "
        "learning applications, and production-ready ML pipelines with "
        "deployment via FastAPI and Docker."
    ),
    required_skills=[
        "Python", "scikit-learn", "Pandas", "NumPy",
        "machine learning algorithms", "PyTorch or TensorFlow/Keras",
        "NLP (text classification, embeddings, transformers)",
        "supervised learning", "unsupervised learning",
        "model evaluation and cross-validation", "FastAPI", "Docker",
        "REST APIs", "SQL", "Git/GitHub",
    ],
    preferred_skills=[
        "AWS or other cloud platforms", "end-to-end ML system development",
        "CI/CD and MLOps practices", "strong communication", "problem-solving",
    ],
    min_years_experience=2,
    responsibilities=[
        "Develop and deploy machine learning models using Python",
        "Build supervised and unsupervised learning solutions",
        "Perform data preprocessing, feature engineering, and exploratory data analysis",
        "Develop classification, regression, and clustering models",
        "Build NLP solutions for text/document classification and information extraction",
        "Develop deep learning models using PyTorch or TensorFlow/Keras",
        "Evaluate models using cross-validation, precision, recall, F1-score, ROC-AUC, MAE, and RMSE",
        "Build reproducible ML pipelines",
        "Deploy models through REST APIs using FastAPI and Docker",
        "Work with SQL databases and structured/unstructured datasets",
        "Use Git/GitHub for version control and collaborative development",
        "Participate in model optimization, error analysis, and experimentation",
    ],
)


# ---------------------------------------------------------------------------
# 1. Core reproduction: the real resume no longer fragments into 12 fake
#    work-experience entries.
# ---------------------------------------------------------------------------

def test_real_bug_resume_no_longer_fragments_into_many_fake_jobs():
    text = ALEX_MORGAN_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)

    # Previously: 12. The real resume has exactly two real jobs.
    assert len(data.work_experience) == 2


def test_real_bug_resume_jobs_have_correct_titles_and_companies():
    text = ALEX_MORGAN_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)

    titles = [w.title for w in data.work_experience]
    companies = [w.company for w in data.work_experience]
    assert titles == ["Machine Learning Engineer", "Junior Data Scientist"]
    assert companies == ["Nova Analytics", "Insight Labs"]


def test_real_bug_resume_dates_are_recovered_from_inline_header():
    """Previously start_date/end_date were BOTH empty for this job (the
    whole 'Title — Company | Location | Dates' line was swallowed as an
    opaque title), so job_years() fell back to a flat wrong 0.5y."""
    text = ALEX_MORGAN_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)

    ml_engineer = data.work_experience[0]
    assert ml_engineer.start_date == "2023"
    assert ml_engineer.end_date.lower() == "present"

    junior_ds = data.work_experience[1]
    assert junior_ds.start_date == "2022"
    assert junior_ds.end_date == "2023"


def test_real_bug_resume_wrapped_bullets_are_recovered_as_full_sentences():
    """Each of these bullets word-wrapped across two physical lines with
    no marker on the second line -- e.g. '- Developed end-to-end machine
    learning pipelines ... for customer' / 'and operational prediction
    tasks.' must be recovered as ONE full bullet, not split into a real
    bullet plus an orphaned fake job."""
    text = ALEX_MORGAN_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)

    ml_engineer = data.work_experience[0]
    assert len(ml_engineer.responsibilities) == 6
    assert any(
        r.startswith("Developed end-to-end machine learning pipelines")
        and r.endswith("operational prediction tasks.")
        for r in ml_engineer.responsibilities
    )
    assert any(
        "XGBoost" in r and r.endswith("hyperparameter tuning.")
        for r in ml_engineer.responsibilities
    )


def test_real_bug_resume_technologies_recognized_across_the_full_bullet():
    """Once bullets are no longer truncated mid-sentence, the taxonomy
    scan over title+bullets recognizes technologies that were previously
    split onto the ORPHANED fragment and therefore invisible to the
    (correctly-titled) job's own technologies list."""
    text = ALEX_MORGAN_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)

    ml_engineer = data.work_experience[0]
    for expected in ("Python", "Pandas", "NumPy", "Scikit-learn",
                      "PyTorch", "TensorFlow", "FastAPI", "Docker"):
        assert expected in ml_engineer.technologies, (
            f"expected {expected!r} in {ml_engineer.technologies}"
        )


def test_real_bug_resume_selected_projects_heading_no_longer_leaks_into_experience():
    """'SELECTED PROJECTS' (as opposed to bare 'Projects') was not a
    recognized heading, so its content used to leak into the 'experience'
    bucket and produce more fake WorkExperience entries."""
    text = ALEX_MORGAN_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)

    for exp in data.work_experience:
        assert "SELECTED PROJECTS" not in exp.company
        assert "Customer Churn Prediction" not in exp.title
        assert "NLP Document Classification" not in exp.title


# ---------------------------------------------------------------------------
# 2. The actual downstream numbers: role_relevance / years_relevant, using
#    the REAL, unmodified scoring functions against the REAL requirements
#    this candidate was screened against.
# ---------------------------------------------------------------------------

def test_real_bug_years_relevant_no_longer_collapses_to_the_fragmented_floor():
    """Stored (buggy) result: years_relevant ~= 1.4-1.45. A working ML
    Engineer since 2023 plus a prior 2022-2023 role should credit well
    over 2 years of relevant experience once the real job durations are
    recovered -- this does not pin an exact number (the formula may be
    retuned later), only that the fragmentation floor is gone."""
    text = ALEX_MORGAN_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)

    evaluation = evaluate_experience(data, REAL_AI_ML_ENGINEER_REQUIREMENTS)
    assert evaluation.years_relevant > 2.5, (
        f"years_relevant={evaluation.years_relevant} still looks fragmented"
    )


def test_real_bug_role_relevance_no_longer_collapses_to_ten_percent():
    """Stored (buggy) result: role_relevance = 0.10. All 15/15 required
    and 3/3 preferred skills matched via the resume's skills_section
    (unaffected by this bug), yet the experience side scored the
    candidate as almost entirely irrelevant -- a direct contradiction
    that was the original report's core complaint. This asserts the
    contradiction is resolved (comfortably relevant), not an exact
    target percentage."""
    text = ALEX_MORGAN_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)

    evaluation = evaluate_experience(data, REAL_AI_ML_ENGINEER_REQUIREMENTS)
    assert evaluation.role_relevance >= 0.4, (
        f"role_relevance={evaluation.role_relevance} still looks collapsed"
    )


def test_real_bug_main_job_is_recognized_as_a_relevant_strength():
    """Previously NEITHER fragment reached job_relevance() >= 0.5, so
    'strengths' came back empty for a candidate whose entire real
    history is on-topic for the JD."""
    text = ALEX_MORGAN_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)

    evaluation = evaluate_experience(data, REAL_AI_ML_ENGINEER_REQUIREMENTS)
    assert any("Machine Learning Engineer" in s for s in evaluation.strengths)


def test_real_bug_main_job_relevance_score_is_no_longer_diluted():
    """Isolates job_relevance() itself (not the years-weighted
    aggregate) for the correctly-recovered main job entry -- confirms
    the fix operates at the level the bug actually lived at (parsed
    input), not by changing the relevance FORMULA."""
    text = ALEX_MORGAN_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)

    ml_engineer = data.work_experience[0]
    rel = job_relevance(ml_engineer, REAL_AI_ML_ENGINEER_REQUIREMENTS)
    years = job_years(ml_engineer)
    assert rel >= 0.5, f"job_relevance={rel} for the real ML Engineer role still too low"
    assert years >= 2.0, f"job_years={years} still using the fragmentation-era 0.5 floor"


# ---------------------------------------------------------------------------
# 3. Generalization checks: the fix is a general parsing fix, not special
#    logic for this one resume -- synthetic resumes using the SAME
#    formatting conventions (inline header, wrapped bullets, "Selected
#    Projects" heading) with entirely different titles/companies/people
#    must be fixed identically.
# ---------------------------------------------------------------------------

GENERIC_WRAPPED_BULLET_RESUME = """JORDAN LEE
jordan.lee@email.com

WORK EXPERIENCE

Site Reliability Engineer — Meridian Cloud | Austin, TX | 2021–Present
- Operated Kubernetes clusters serving production traffic for dozens of internal
teams across multiple regions.
- Automated incident response tooling that cut mean-time-to-recovery by half
using Python and Terraform.

SELECTED PROJECTS

Internal Deploy Dashboard
Built a lightweight dashboard summarizing rollout health across services.
"""


def test_generic_resume_with_inline_header_and_wrapped_bullets_parses_correctly():
    """Different candidate, different title, different company -- proves
    the fix generalizes rather than special-casing Alex Morgan's resume."""
    data = parse_resume_text(GENERIC_WRAPPED_BULLET_RESUME)

    assert len(data.work_experience) == 1
    job = data.work_experience[0]
    assert job.title == "Site Reliability Engineer"
    assert job.company == "Meridian Cloud"
    assert job.start_date == "2021"
    assert job.end_date.lower() == "present"
    assert len(job.responsibilities) == 2
    assert job.responsibilities[0].endswith("across multiple regions.")
    assert job.responsibilities[1].endswith("using Python and Terraform.")


def test_generic_resume_selected_projects_heading_does_not_leak():
    data = parse_resume_text(GENERIC_WRAPPED_BULLET_RESUME)
    titles = [w.title for w in data.work_experience]
    assert "Internal Deploy Dashboard" not in titles


# ---------------------------------------------------------------------------
# 4. False-positive guards: a wrapped continuation must NOT be merged when
#    the following line genuinely IS a new job's header, and an ordinary
#    hyphenated title with no dates must NOT be split by the new inline-
#    header parsing.
# ---------------------------------------------------------------------------

TWO_REAL_JOBS_BOTH_INLINE_HEADER = """CASEY KIM
casey.kim@email.com

WORK EXPERIENCE

Backend Engineer — Alpha Systems | Denver, CO | 2020–2022
- Built REST APIs for the billing platform serving thousands of merchants
across the US and Canada.

Platform Engineer — Beta Cloud | Denver, CO | 2022–Present
- Migrated legacy services to Kubernetes and improved deploy frequency.
"""


def test_wrapped_continuation_does_not_swallow_a_genuinely_new_job():
    """A wrapped bullet continuation must fold back into the previous
    bullet, but a line that legitimately starts a NEW job (its own
    inline title/company/date header) must still end bullet collection
    and start a new WorkExperience -- not be merged as if it were more
    of the previous job's last bullet."""
    data = parse_resume_text(TWO_REAL_JOBS_BOTH_INLINE_HEADER)

    assert len(data.work_experience) == 2
    titles = [w.title for w in data.work_experience]
    companies = [w.company for w in data.work_experience]
    assert titles == ["Backend Engineer", "Platform Engineer"]
    assert companies == ["Alpha Systems", "Beta Cloud"]

    first = data.work_experience[0]
    assert len(first.responsibilities) == 1
    assert first.responsibilities[0].endswith("across the US and Canada.")


def test_ordinary_hyphenated_title_with_no_dates_is_not_split_as_inline_header():
    """'Backend Engineer - Python' (a real title convention used
    elsewhere in this project's own sample data) has a bare hyphen and
    no date signal at all -- _split_inline_job_header must return None
    for it, leaving the existing 'next line is company/dates' parsing
    path completely untouched."""
    text = """JAMIE FOX
jamie.fox@email.com

WORK EXPERIENCE

Backend Engineer - Python
TechVentures Pvt Ltd | Jan 2021 - Dec 2022
- Built and maintained internal APIs
"""
    data = parse_resume_text(text)
    assert len(data.work_experience) == 1
    job = data.work_experience[0]
    assert job.title == "Backend Engineer - Python"
    assert job.company == "TechVentures Pvt Ltd"
    assert job.start_date == "Jan 2021"


# ---------------------------------------------------------------------------
# 5. Regression guard: existing fixture resumes (John Doe, Sarah Chen) --
#    which use neither wrapped bullets nor inline headers -- are
#    completely unaffected.
# ---------------------------------------------------------------------------

def test_john_doe_resume_unaffected_by_bullet_wrap_fix():
    text = Path("sample_data/resumes/senior_python_dev.txt").read_text(encoding="utf-8")
    data = parse_resume_text(text)
    titles = [w.title for w in data.work_experience]
    assert titles == ["Senior Software Engineer", "Software Engineer", "Junior Developer"]


def test_sarah_chen_resume_unaffected_by_bullet_wrap_fix():
    text = Path("sample_data/resumes/junior_dev.txt").read_text(encoding="utf-8")
    data = parse_resume_text(text)
    titles = [w.title for w in data.work_experience]
    assert titles == ["IT Intern"]


# ---------------------------------------------------------------------------
# 6. Determinism, matching this project's existing convention for the
#    deterministic parser.
# ---------------------------------------------------------------------------

def test_real_bug_resume_parse_is_deterministic():
    text = ALEX_MORGAN_RESUME.read_text(encoding="utf-8")
    a = parse_resume_text(text)
    b = parse_resume_text(text)
    assert a.model_dump() == b.model_dump()
