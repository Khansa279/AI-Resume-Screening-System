"""Priority 7 regression tests: resume-parser fake work-experience bug.

REAL BUG: resumes using heading variants outside the small fixed list in
_HEADING_RE -- most notably "EDUCATION & TRAINING" (rather than bare
"Education") and "CAREER BREAK" (not recognized at all) -- were never
recognized as section boundaries. _split_sections() then kept routing
that content into whichever section was still open (in practice,
"experience", since it's almost always the section immediately above
these headings on a resume). _parse_experience()'s title+"Company | Year"
look-ahead heuristic then misread degree/institution lines (e.g.
"Bachelor of Technology in Computer Science" / "Pune Institute of
Technology | 2017") as a fake job, and could do the same for a
free-standing "Career Break" note if it happened to have a date-like
line beneath it.

Fix (src/agents/resume_parser.py), two parts:
  1. _HEADING_RE / _SECTION_ALIASES now recognize "Education & Training"
     / "Education and Training" as aliases for the existing "education"
     section, and "Career Break" / "Career Gap" / "Employment Gap" /
     "Sabbatical" as a new "career_break" section -- a bucket nothing
     downstream reads, so that content is safely dropped rather than
     leaking into "experience".
  2. A defense-in-depth guard in _parse_experience()
     (_looks_like_education_or_break_entry) skips creating a
     WorkExperience when a would-be job's title/company look like a
     degree+institution pair or a career-break phrase, even if some
     OTHER, still-unrecognized heading variant leaked them into the
     experience section. This is intentionally narrower than the
     existing _DEGREE_RE (used only for already-correctly-segmented
     education text) so it doesn't misfire on real ambiguous job titles
     like "Associate Software Engineer" or "Scrum Master".

These tests use the REAL, unmodified parse_resume_text() (no mocking),
exactly like test_deterministic_pipeline.py does elsewhere in this suite.
"""

from pathlib import Path

from src.agents.resume_parser import parse_resume_text


VIKRAM_STYLE_RESUME = """VIKRAM SINGH
Backend Developer
vikram.singh@email.com | (555) 222-3344 | Pune, India

WORK EXPERIENCE

Freelance Backend Developer
Self-employed | Jan 2021 - Dec 2021
- Built REST APIs for small business clients using Python and Flask
- Managed PostgreSQL databases for three client projects

Backend Developer
TechNova Systems | Mar 2018 - Dec 2020
- Designed backend services using Django and PostgreSQL
- Deployed services with Docker

CAREER BREAK

Took a career break from Jan 2021 to Mar 2021 to care for a family member.

EDUCATION & TRAINING

Bachelor of Technology in Computer Science
Pune Institute of Technology | 2017

Full Stack Web Development Bootcamp
CodeAcademy Bootcamp | 2018

PERSONAL PROJECTS

Expense Tracker -- built a personal finance tracker using Python and Flask

TECHNICAL SKILLS

Python, Django, Flask, PostgreSQL, Docker, Git
"""


SAMPLE_RESUME = Path("sample_data/resumes/senior_python_dev.txt")  # John Doe
JUNIOR_RESUME = Path("sample_data/resumes/junior_dev.txt")  # Sarah Chen


# ---------------------------------------------------------------------------
# 1. Core reproduction: "EDUCATION & TRAINING" and "CAREER BREAK" no longer
#    produce fake WorkExperience entries, and real jobs are preserved.
# ---------------------------------------------------------------------------

def test_education_and_training_heading_does_not_produce_fake_work_experience():
    data = parse_resume_text(VIKRAM_STYLE_RESUME)

    titles = [w.title for w in data.work_experience]
    assert "Bachelor of Technology in Computer Science" not in titles
    assert "Full Stack Web Development Bootcamp" not in titles
    assert not any("Pune Institute of Technology" in w.company for w in data.work_experience)
    assert not any("CodeAcademy Bootcamp" in w.company for w in data.work_experience)


def test_career_break_heading_does_not_produce_fake_work_experience():
    data = parse_resume_text(VIKRAM_STYLE_RESUME)

    titles = [w.title.lower() for w in data.work_experience]
    assert not any("career break" in t for t in titles)
    assert not any("took a career break" in t for t in titles)


def test_real_jobs_are_still_parsed_correctly_from_vikram_style_resume():
    """The fix must not throw away the genuinely real jobs while
    filtering out the fake ones."""
    data = parse_resume_text(VIKRAM_STYLE_RESUME)

    assert len(data.work_experience) == 2
    titles = {w.title for w in data.work_experience}
    assert titles == {"Freelance Backend Developer", "Backend Developer"}

    freelance = next(w for w in data.work_experience if w.title == "Freelance Backend Developer")
    assert freelance.company == "Self-employed"
    assert freelance.responsibilities

    technova = next(w for w in data.work_experience if w.title == "Backend Developer")
    assert technova.company == "TechNova Systems"
    assert technova.responsibilities


def test_education_and_training_heading_is_recognized_as_the_education_section():
    """Bonus correctness: once the heading is recognized, the degree
    info is no longer silently lost -- it shows up in `education`
    instead of vanishing or becoming a fake job."""
    data = parse_resume_text(VIKRAM_STYLE_RESUME)

    assert len(data.education) >= 1
    degrees = [e.degree for e in data.education]
    assert any("Bachelor of Technology" in d for d in degrees)
    institutions = [e.institution for e in data.education]
    assert any("Pune Institute of Technology" in i for i in institutions)


# ---------------------------------------------------------------------------
# 2. Defense-in-depth guard: even an UNENUMERATED heading variant that
#    still leaks education/career-break text into "experience" must not
#    produce a fake WorkExperience.
# ---------------------------------------------------------------------------

def test_unenumerated_heading_variant_still_guarded_by_title_company_heuristic():
    text = """JANE ROE
jane.roe@email.com

WORK EXPERIENCE

Backend Developer
Acme Corp | Jan 2020 - Dec 2021
- Built APIs

SOME UNRECOGNIZED HEADING WE DID NOT ANTICIPATE

Bachelor of Science in Computer Science
State University | 2019

CAREER BREAK

Took time off from Jan 2022 to Jun 2022 for personal reasons.
"""
    data = parse_resume_text(text)
    assert len(data.work_experience) == 1
    assert data.work_experience[0].title == "Backend Developer"
    assert data.work_experience[0].company == "Acme Corp"


def test_free_standing_career_break_note_with_dates_is_not_a_fake_job():
    """A 'Career Break' note formatted closely enough to look like a job
    (title line immediately followed by a date-shaped line) must still
    be rejected by the explicit non-job-title guard."""
    text = """JANE ROE
jane.roe@email.com

WORK EXPERIENCE

Backend Developer
Acme Corp | Jan 2020 - Dec 2021
- Built APIs

Career Break
Jan 2022 - Jun 2022
"""
    data = parse_resume_text(text)
    titles = [w.title.lower() for w in data.work_experience]
    assert "career break" not in titles


# ---------------------------------------------------------------------------
# 3. False-positive guard: real, ambiguous-but-legitimate job titles must
#    be completely unaffected by the new guard.
# ---------------------------------------------------------------------------

def test_associate_job_title_is_not_mistaken_for_a_degree():
    """'Associate' is part of _DEGREE_RE (used for real education blocks)
    but must NOT be part of the narrower experience-side guard, since it
    routinely appears in real job titles."""
    text = """ALEX KIM
alex.kim@email.com

WORK EXPERIENCE

Associate Software Engineer
TechCorp Inc. | Jan 2020 - Dec 2021
- Built APIs
"""
    data = parse_resume_text(text)
    assert len(data.work_experience) == 1
    assert data.work_experience[0].title == "Associate Software Engineer"
    assert data.work_experience[0].company == "TechCorp Inc."


def test_scrum_master_at_institute_named_company_is_not_mistaken_for_education():
    """'Master' alone is ambiguous (Scrum Master); the guard specifically
    requires 'master of'/'master's', not the bare word, so this real job
    (even at a company whose name contains 'Institute') survives."""
    text = """ALEX KIM
alex.kim@email.com

WORK EXPERIENCE

Scrum Master
Agile Institute of Delivery | Jan 2022 - Present
- Ran sprint ceremonies
"""
    data = parse_resume_text(text)
    assert len(data.work_experience) == 1
    assert data.work_experience[0].title == "Scrum Master"
    assert data.work_experience[0].company == "Agile Institute of Delivery"


# ---------------------------------------------------------------------------
# 4. Regression guard: real fixture resumes (John Doe, Sarah Chen) are
#    completely unaffected by this fix.
# ---------------------------------------------------------------------------

def test_john_doe_work_experience_unaffected():
    text = SAMPLE_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)
    titles = [w.title for w in data.work_experience]
    assert titles == ["Senior Software Engineer", "Software Engineer", "Junior Developer"]
    companies = [w.company for w in data.work_experience]
    assert companies == ["TechCorp Inc.", "StartupXYZ", "WebAgency"]
    assert len(data.education) == 1
    assert data.education[0].institution == "University of California, Berkeley"


def test_sarah_chen_work_experience_unaffected():
    text = JUNIOR_RESUME.read_text(encoding="utf-8")
    data = parse_resume_text(text)
    titles = [w.title for w in data.work_experience]
    assert titles == ["IT Intern"]
    assert data.work_experience[0].company == "LocalBusiness Inc."
    assert len(data.education) == 1
    assert data.education[0].institution == "City University of New York"


# ---------------------------------------------------------------------------
# 5. Determinism (matches the project-wide convention that the
#    deterministic parser gives byte-identical output for identical input)
# ---------------------------------------------------------------------------

def test_vikram_style_parse_is_deterministic():
    a = parse_resume_text(VIKRAM_STYLE_RESUME)
    b = parse_resume_text(VIKRAM_STYLE_RESUME)
    assert a.model_dump() == b.model_dump()
