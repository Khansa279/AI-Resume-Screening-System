"""Regression tests for bullet/list-marker normalization
(src/document_parser.py::normalize_bullet_markers).

REAL BUG: PDF resumes rendering bullet points via an icon font produce a
raw \\x7f control character (confirmed via byte inspection of
resume_01_priya_sharma.pdf, resume_02_rahul_verma.pdf,
resume_03_ananya_patel.pdf, resume_04_vikram_singh.pdf -- all four use
the same glyph). DocumentParser._sanitize_extracted_text() stripped that
character to nothing (via _CONTROL_CHAR_PATTERN), leaving a bare line
with no marker at all. ResumeParserAgent._parse_experience()'s ONLY
signal for "this line is a bullet, attach it as a responsibility" is a
literal `line.startswith(("-", "*", "\u2022"))` check -- so every real
bullet line in these PDFs was silently reinterpreted as the start of a
brand-new job entry (with the next bullet line misread as its
"company"), shredding e.g. Priya Sharma's 3 real jobs into 9 fake
WorkExperience entries with empty technologies/responsibilities. That
cratered role_relevance (a years-weighted average across all entries,
most of which had ~0 relevance) even though the real job was a strong
match.

FIX (src/document_parser.py): normalize_bullet_markers() runs BEFORE the
existing control-char/private-use stripping and converts any
LINE-LEADING bullet glyph -- control characters, Private-Use-Area
icon-font characters, or a curated set of standard Unicode bullet
symbols -- to the single canonical "- " marker ResumeParserAgent already
recognizes. It is deliberately anchored to the start of a line and
requires the marker to be followed by real content, so ordinary
punctuation inside sentences, dates, and ranges (hyphens, en dashes,
arrows) is never touched.
"""

from pathlib import Path

from src.document_parser import normalize_bullet_markers, parse_document
from src.agents.resume_parser import parse_resume_text


# ---------------------------------------------------------------------------
# 1. The confirmed \x7f case
# ---------------------------------------------------------------------------

def test_confirmed_x7f_bullet_at_line_start_is_normalized():
    text = "Senior Backend Developer\n\x7f Architected and built REST APIs using FastAPI"
    result = normalize_bullet_markers(text)
    assert result == "Senior Backend Developer\n- Architected and built REST APIs using FastAPI"


def test_multiple_x7f_bullets_all_normalized():
    text = "\x7f bullet one\n\x7f bullet two\n\x7f bullet three"
    result = normalize_bullet_markers(text)
    assert result == "- bullet one\n- bullet two\n- bullet three"


def test_x7f_with_no_space_before_text_still_normalized():
    """Guards against a PDF extraction that omits the space between the
    glyph and the following word."""
    result = normalize_bullet_markers("\x7fArchitected the system")
    assert result == "- Architected the system"


# ---------------------------------------------------------------------------
# 2. Multiple common Unicode bullet variants
# ---------------------------------------------------------------------------

def test_standard_unicode_bullet_variants_are_normalized():
    variants = {
        "\u2022": "bullet (U+2022)",
        "\u25E6": "white bullet (U+25E6)",
        "\u25AA": "black small square (U+25AA)",
        "\u25B8": "black right-pointing small triangle (U+25B8)",
        "\u2023": "triangular bullet (U+2023)",
        "\u25CF": "black circle (U+25CF)",
        "\u25A0": "black square (U+25A0)",
        "\u25A1": "white square (U+25A1)",
        "\u27A4": "black rightwards arrowhead (U+27A4)",
        "\u27A2": "three-d top-lighted rightwards arrowhead (U+27A2)",
        "\u00B7": "middle dot (U+00B7)",
        "\u2219": "bullet operator (U+2219)",
    }
    for glyph, label in variants.items():
        result = normalize_bullet_markers(f"{glyph} {label}")
        assert result == f"- {label}", f"failed for {glyph!r}: got {result!r}"


def test_private_use_area_icon_font_glyph_is_normalized():
    """Icon-bullet fonts (Wingdings-style) commonly map their bullet
    glyph into the Private Use Area rather than a real Unicode bullet
    code point or a control character."""
    result = normalize_bullet_markers("\uE001 PUA icon-font bullet")
    assert result == "- PUA icon-font bullet"


def test_other_control_characters_besides_x7f_are_also_normalized():
    """The fix generalizes to the whole control-char range, not just the
    one confirmed byte -- a different PDF export/font could plausibly
    emit \\x02 or \\x0e for the same visual bullet."""
    for ctrl in ("\x01", "\x02", "\x0e", "\x1f"):
        result = normalize_bullet_markers(f"{ctrl} some responsibility")
        assert result == "- some responsibility", f"failed for {ctrl!r}: got {result!r}"


# ---------------------------------------------------------------------------
# 3. Existing -, *, and \u2022 markers keep working
# ---------------------------------------------------------------------------

def test_existing_hyphen_marker_is_left_alone_but_still_a_recognized_bullet():
    """"-" is already handled downstream by ResumeParserAgent -- this
    layer must not alter it (and definitely must not break it)."""
    result = normalize_bullet_markers("- already a dash bullet")
    assert result == "- already a dash bullet"


def test_existing_asterisk_marker_normalizes_to_canonical_dash():
    result = normalize_bullet_markers("* asterisk bullet")
    assert result == "- asterisk bullet"


def test_existing_standard_bullet_symbol_normalizes_to_canonical_dash():
    result = normalize_bullet_markers("\u2022 standard bullet")
    assert result == "- standard bullet"


def test_resume_parser_recognizes_all_three_marker_styles_after_normalization():
    """End-to-end: once normalized, ResumeParserAgent._parse_experience
    must attach all three bullet styles as responsibilities on the SAME
    job, not split them into separate fake jobs."""
    text = (
        "WORK EXPERIENCE\n"
        "Backend Engineer\n"
        "Acme Corp | Jan 2022 - Present\n"
        "- Built REST APIs\n"
        "* Wrote unit tests\n"
        "\u2022 Reviewed pull requests\n"
    )
    data = parse_resume_text(text)
    assert len(data.work_experience) == 1
    job = data.work_experience[0]
    assert job.title == "Backend Engineer"
    assert job.responsibilities == [
        "Built REST APIs",
        "Wrote unit tests",
        "Reviewed pull requests",
    ]


# ---------------------------------------------------------------------------
# 4. Normalization only applies at line starts
# ---------------------------------------------------------------------------

def test_bullet_glyph_mid_line_is_not_converted_to_dash():
    """A bullet-like character appearing INSIDE a sentence (not as the
    first token on its line) must be left completely alone by this
    function -- normalization is about list structure, not about
    scrubbing every occurrence of these characters."""
    text = "some text \u2022 not a bullet mid-line"
    assert normalize_bullet_markers(text) == text


def test_x7f_mid_line_is_not_converted_by_this_function():
    """Mid-line control characters are NOT this function's job -- they're
    still handled (stripped) by the existing _CONTROL_CHAR_PATTERN step
    that runs afterward in _sanitize_extracted_text; this function only
    ever touches the first token of a line."""
    text = "some responsibility text\x7fwith an embedded control char"
    assert normalize_bullet_markers(text) == text


def test_leading_whitespace_before_marker_is_still_recognized_as_line_start():
    """A bullet glyph after leading spaces/tabs (common PDF indentation)
    is still the first non-whitespace token of the line, so it must
    still be recognized and normalized, with the indentation preserved."""
    result = normalize_bullet_markers("    \x7f indented bullet")
    assert result == "    - indented bullet"


# ---------------------------------------------------------------------------
# 5. Date ranges and normal inline punctuation are never touched
# ---------------------------------------------------------------------------

def test_date_range_with_hyphen_is_unchanged():
    text = "January 2022 - Present"
    assert normalize_bullet_markers(text) == text


def test_date_range_with_en_dash_is_unchanged():
    text = "Jan 2022 \u2013 Dec 2023"
    assert normalize_bullet_markers(text) == text


def test_date_range_with_em_dash_is_unchanged():
    text = "2020 \u2014 2022"
    assert normalize_bullet_markers(text) == text


def test_hyphenated_word_is_unchanged():
    text = "co-founder of a self-driving startup"
    assert normalize_bullet_markers(text) == text


def test_arrow_in_prose_is_unchanged():
    text = "Improved throughput from 100 -> 500 requests/sec"
    assert normalize_bullet_markers(text) == text


def test_full_resume_style_block_only_bullet_lines_change():
    text = (
        "TechScale Solutions | Mumbai | January 2022 - Present\n"
        "\x7f Reduced response time by 40%\n"
        "Contact: jan-doe@example.com\n"
    )
    result = normalize_bullet_markers(text)
    lines = result.split("\n")
    assert lines[0] == "TechScale Solutions | Mumbai | January 2022 - Present"
    assert lines[1] == "- Reduced response time by 40%"
    assert lines[2] == "Contact: jan-doe@example.com"


# ---------------------------------------------------------------------------
# 6. Real PDFs: Priya Sharma and Vikram Singh parse into real
#    WorkExperience entries with bullets correctly attached
# ---------------------------------------------------------------------------

PRIYA_PDF = Path("sample_data/resumes/resume_01_priya_sharma.pdf")
VIKRAM_PDF = Path("sample_data/resumes/resume_04_vikram_singh.pdf")


def test_priya_sharma_pdf_parses_into_a_small_number_of_real_jobs():
    """Before the fix this was 9 fragmented entries (one real job
    shredded per bullet line). After the fix it must collapse back down
    to a small, realistic number of jobs, each carrying real content."""
    parsed = parse_document(str(PRIYA_PDF))
    assert parsed.success
    data = parse_resume_text(parsed.text)

    assert 2 <= len(data.work_experience) <= 5, (
        f"expected a realistic job count, got {len(data.work_experience)}: "
        f"{[e.title for e in data.work_experience]}"
    )

    senior_role = next(
        (e for e in data.work_experience if "Senior Backend Developer" in e.title), None
    )
    assert senior_role is not None, (
        f"expected to find the real 'Senior Backend Developer' title, got "
        f"titles={[e.title for e in data.work_experience]}"
    )
    assert senior_role.responsibilities, "senior role must have real bullets attached"
    assert senior_role.technologies, "senior role must have technologies recognized"
    for expected_tech in ("FastAPI", "PostgreSQL"):
        assert expected_tech in senior_role.technologies, senior_role.technologies


def test_vikram_singh_pdf_parses_into_a_small_number_of_real_jobs():
    parsed = parse_document(str(VIKRAM_PDF))
    assert parsed.success
    data = parse_resume_text(parsed.text)

    # Before the fix: 14 fragmented entries. Vikram's resume legitimately
    # has more distinct sections (freelance work, a career break, a
    # bootcamp, personal projects) than Priya's, so the bound here is
    # looser, but it must be nowhere near the pre-fix 14.
    assert len(data.work_experience) <= 9, (
        f"expected a realistic job count, got {len(data.work_experience)}: "
        f"{[e.title for e in data.work_experience]}"
    )

    freelance_role = next(
        (e for e in data.work_experience if "Freelance Backend Developer" in e.title), None
    )
    assert freelance_role is not None, (
        f"expected to find the real 'Freelance Backend Developer' title, got "
        f"titles={[e.title for e in data.work_experience]}"
    )
    assert freelance_role.responsibilities, "freelance role must have real bullets attached"
    assert freelance_role.technologies, "freelance role must have technologies recognized"


def test_priya_and_vikram_role_relevance_no_longer_near_zero():
    """End-to-end regression pinning the reported numbers: role_relevance
    against the real Backend Engineer - Python JD must be well above the
    ~2-4% measured before this fix."""
    from src.agents.experience_eval import evaluate_experience
    from src.models import JobRequirements

    job_requirements = JobRequirements(
        title="Backend Engineer - Python",
        required_skills=[
            "2+ years of professional experience with Python",
            "Experience building REST APIs (Django REST Framework or FastAPI preferred)",
            "Strong SQL skills and experience with relational databases (PostgreSQL)",
            "Familiarity with Git version control",
            "Understanding of software development best practices",
            "Good communication skills in English",
        ],
        preferred_skills=[
            "Experience with Django or FastAPI frameworks",
            "Familiarity with Docker and containerization",
            "Exposure to message queues (Redis, RabbitMQ, Celery)",
            "Basic AWS/cloud experience (EC2, S3, RDS)",
            "Experience with CI/CD pipelines",
            "Knowledge of microservices architecture",
        ],
        min_years_experience=2,
        responsibilities=[
            "Design, build, and maintain RESTful APIs using Python",
            "Write clean, well-tested, production-quality code",
            "Work with PostgreSQL databases - schema design, query optimization",
            "Collaborate with frontend team on API contracts",
        ],
    )

    for pdf_path in (PRIYA_PDF, VIKRAM_PDF):
        parsed = parse_document(str(pdf_path))
        data = parse_resume_text(parsed.text)
        ev = evaluate_experience(data, job_requirements)
        assert ev.role_relevance >= 0.08, (
            f"{pdf_path}: role_relevance={ev.role_relevance} is still near the "
            f"pre-fix ~2-4% floor"
        )
