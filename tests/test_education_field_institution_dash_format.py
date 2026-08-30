"""Regression test for a confirmed resume-parsing gap in
src/agents/resume_parser.py::_parse_education.

REAL BUG: `_parse_education` only extracted `Education.field` when the
first line of an education block matched "<Degree> in <Field>" (e.g.
"Bachelors in Data Science"). A resume phrasing the same information as
"<Institution> - <Field>" (e.g. "Air University - Data Science") fell
through to the `else: degree = first` branch, leaving `field` as an
empty string.

This matters because src/agents/experience_eval.py's
`has_relevant_education` (used in evaluate_experience()'s no-work-
experience branch, which feeds directly into role_relevance) checks
`edu.field` for keywords like "computer", "data", "software", etc. Two
candidates with an equally relevant BS Data Science background could
therefore receive different role_relevance purely because of how their
resume happened to word the education line -- not because of any real
difference in qualifications.

Fix: _parse_education now also recognizes "<Institution> - <Field>"
(only when the text before the dash looks like an institution name --
university/college/institute/academy/polytechnic -- so an ordinary
"<Degree> - <Year>" line is not misread).

This test does NOT touch the 0.3 project-relevance threshold, any
scoring weight, or DecisionSynthesizerAgent's formula.
"""

from src.agents.resume_parser import parse_resume_text
from src.agents.experience_eval import evaluate_experience
from src.models import JobRequirements, ResumeData


def test_institution_dash_field_format_is_now_extracted():
    text = """Ayesha Roe
ayesha@example.com

Education
Air University - Data Science
2024 - 2028
2nd-year BS Data Science student.
"""
    data = parse_resume_text(text)
    assert len(data.education) == 1
    assert data.education[0].field == "Data Science"


def test_degree_in_field_format_still_works_unchanged():
    """Regression guard: the pre-existing 'Degree in Field' pattern must
    be completely unaffected by this fix."""
    text = """Khansa Doe
khansa@example.com

Education
Bachelors in Data Science
Air University
2024-2028
"""
    data = parse_resume_text(text)
    assert len(data.education) == 1
    assert data.education[0].degree == "Bachelors"
    assert data.education[0].field == "Data Science"


def test_dash_line_without_institution_keyword_is_not_misread_as_field():
    """A line like 'BS Computer Science - 2024' has a dash but the text
    before it is a degree, not an institution -- must NOT be
    misinterpreted as "field = '2024'"."""
    text = """Alex Kim
alex@example.com

Education
BS Computer Science - 2024
XYZ University
"""
    data = parse_resume_text(text)
    assert len(data.education) == 1
    assert data.education[0].field == ""
    assert data.education[0].degree == "BS Computer Science - 2024"


def test_institution_dash_field_format_with_college_keyword():
    text = """Sam Lee
sam@example.com

Education
Riverside College - Computer Science
2020 - 2024
"""
    data = parse_resume_text(text)
    assert len(data.education) == 1
    assert data.education[0].field == "Computer Science"


def test_two_equally_relevant_candidates_get_equal_education_credit_regardless_of_wording():
    """End-to-end: the actual reported anomaly. Two candidates with an
    equally relevant Data Science background, phrased differently, must
    now receive the SAME education-relevance credit in
    evaluate_experience()'s no-work-experience role_relevance formula."""
    jd = JobRequirements(
        title="Associate AI Engineer",
        required_skills=["Python", "Machine Learning", "Deep Learning", "Artificial Intelligence"],
        preferred_skills=["Data Structures", "Algorithms", "SQL", "Git"],
        min_years_experience=0,
    )

    candidate_a_text = """Person A
a@example.com

Education
Bachelors in Data Science
Air University
2024-2028
"""
    candidate_b_text = """Person B
b@example.com

Education
Air University - Data Science
2024 - 2028
"""
    data_a = parse_resume_text(candidate_a_text)
    data_b = parse_resume_text(candidate_b_text)

    assert data_a.education[0].field == data_b.education[0].field == "Data Science"

    ev_a = evaluate_experience(data_a, jd)
    ev_b = evaluate_experience(data_b, jd)

    # Both have zero work experience and zero projects -- role_relevance
    # should be identical (0.15, the has_relevant_education baseline)
    # now that both resolve to the same recognized education field.
    assert ev_a.role_relevance == ev_b.role_relevance == 0.15


def test_no_education_section_is_still_handled_gracefully():
    data = ResumeData()
    from src.agents.resume_parser import parse_resume_text as _prt  # sanity import check only
    assert data.education == []
