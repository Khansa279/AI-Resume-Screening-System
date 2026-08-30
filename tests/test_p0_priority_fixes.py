"""P0 regression tests: filename traversal, skill substring false
positives, ISO YYYY-MM date parsing, and silent JD-analysis-failure
caching.

Each of these was independently reproduced against the real,
unmodified code before being fixed -- see the one-day audit report for
the exact repro commands. These tests pin the fixed behavior.
"""

import asyncio
from pathlib import Path

import pytest

from src.agents.experience_eval import _parse_one_date, job_years
from src.agents.job_analyzer import JobAnalyzerAgent
from src.agents.skills_matcher import SkillsMatcherAgent
from src.api.routes.screening import _safe_resume_filename
from src.models import JobRequirements, Skill, WorkExperience


# ---------------------------------------------------------------------------
# F-01: an uploaded resume's client-supplied filename was joined directly
# onto a server-side temp directory path, allowing path traversal /
# absolute-path overwrite.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("malicious_name,forbidden_fragment", [
    ("../../etc/cron.d/evil", ".."),
    ("../../../root/.ssh/authorized_keys", ".."),
    ("/etc/passwd", "/etc/passwd"),
    ("..\\..\\windows\\system32\\evil.txt", ".."),
])
def test_malicious_filenames_are_reduced_to_a_safe_basename(malicious_name, forbidden_fragment, tmp_path):
    safe = _safe_resume_filename(malicious_name)
    assert ".." not in safe
    assert "/" not in safe
    assert "\\" not in safe
    # Confirms joining it onto a directory can never leave that directory.
    # `tmp_path` (a pytest fixture) gives a real, platform-native
    # directory -- a WindowsPath on Windows, a PosixPath elsewhere --
    # so this containment check is meaningful on either OS instead of
    # assuming POSIX-style "/tmp/..." paths.
    dest = tmp_path / safe
    assert dest.resolve().is_relative_to(tmp_path.resolve())
    assert dest.resolve().parent == tmp_path.resolve()


def test_absolute_path_filename_does_not_override_the_destination_directory(tmp_path):
    """The specific pathlib footgun: Path(a) / b returns just `b` if `b`
    is absolute. This must never be allowed to reach the final `/` join
    in the route."""
    safe = _safe_resume_filename("/etc/passwd")
    dest = tmp_path / safe
    assert dest.resolve().is_relative_to(tmp_path.resolve())
    assert dest.resolve() != Path("/etc/passwd").resolve()


def test_normal_filenames_are_preserved():
    assert _safe_resume_filename("resume.pdf") == "resume.pdf"
    assert _safe_resume_filename("John Doe Resume.docx") == "John Doe Resume.docx"


def test_empty_or_dot_filenames_fall_back_to_a_safe_default():
    assert _safe_resume_filename("") == "resume"
    assert _safe_resume_filename(".") == "resume"
    assert _safe_resume_filename("..") == "resume"


# ---------------------------------------------------------------------------
# F-03: ISO "YYYY-MM" dates were silently collapsing to January, making a
# real 3-month role compute as 0.0 years of experience.
# ---------------------------------------------------------------------------

def test_iso_year_month_dates_parse_the_correct_month():
    assert _parse_one_date("2023-06") == _parse_one_date("2023-06").replace(month=6)
    d = _parse_one_date("2023-06")
    assert d.year == 2023 and d.month == 6


def test_iso_year_month_day_dates_parse_the_correct_month():
    d = _parse_one_date("2023-06-15")
    assert d.year == 2023 and d.month == 6


def test_three_month_role_no_longer_computes_zero_years():
    """The exact reported reproduction: start=2023-06, end=2023-09 must
    be ~0.25 years, not 0.0 (both previously collapsed to 2023-01-01)."""
    exp = WorkExperience(title="X", company="Y", start_date="2023-06", end_date="2023-09")
    years = job_years(exp)
    assert years == pytest.approx(0.25, abs=0.02)


def test_multi_year_iso_dates_are_not_rounded_to_whole_years():
    exp = WorkExperience(title="X", company="Y", start_date="2020-01", end_date="2023-06")
    years = job_years(exp)
    assert years == pytest.approx(3.41, abs=0.05)
    assert years != 3.0  # pre-fix behavior: both dates lost their month


def test_month_first_mm_yyyy_format_still_works():
    """Regression guard: the existing MM-YYYY / MM/YYYY path must be
    unaffected by adding the YYYY-MM check."""
    d = _parse_one_date("06-2023")
    assert d.year == 2023 and d.month == 6


def test_month_name_format_still_works():
    d = _parse_one_date("June 2023")
    assert d.year == 2023 and d.month == 6


# ---------------------------------------------------------------------------
# F-02: naive substring matching treated "Java" as satisfying
# "JavaScript...", "SQL" as satisfying "NoSQL...", etc.
# ---------------------------------------------------------------------------

def _match(skill_name: str, requirement: str):
    agent = SkillsMatcherAgent()
    skills = [Skill(name=skill_name, category="other")]
    reqs = JobRequirements(required_skills=[requirement], preferred_skills=[])
    result = asyncio.run(agent.process({"extracted_skills": skills, "job_requirements": reqs}))["skills_match"]
    return result.matches[0]


@pytest.mark.parametrize("skill_name,requirement", [
    ("Java", "JavaScript frontend framework experience"),
    ("SQL", "NoSQL database experience"),
    ("Git", "GitHub Actions CI/CD"),
    ("Ruby", "RubyGems package management"),
])
def test_prefix_substring_false_positives_no_longer_match(skill_name, requirement):
    m = _match(skill_name, requirement)
    assert m.matched is False, (
        f"{skill_name!r} should NOT satisfy {requirement!r} -- it's only a "
        f"substring/prefix of a different, unrelated word"
    )


def test_genuine_whole_word_partial_match_still_works():
    """A real substring-as-a-distinct-word case (skill name is not in the
    taxonomy, but appears as a whole word in the requirement) must still
    be credited as a partial match."""
    m = _match("Terraform", "Experience with Terraform for infrastructure as code")
    assert m.matched is True
    assert m.match_quality == "partial"


def test_exact_and_semantic_matches_are_unaffected_by_the_substring_fix():
    m = _match("Python", "Python programming experience")
    assert m.matched is True


# ---------------------------------------------------------------------------
# F-07: a total LLM API failure during job-description analysis was
# silently turned into a normal-looking (low-confidence) JobRequirements
# result with no error surfaced -- and get_job_requirements() would then
# cache it forever via ensure_job_requirements.
# ---------------------------------------------------------------------------

async def _failing_llm(self, prompt, max_retries=1, max_rate_limit_retries=3):
    """Simulates BaseAgent._call_llm_for_json_async's real contract on
    total API failure: it returns "" (see base.py's docstring), not an
    exception and not a JSON string."""
    return ""


def test_job_analyzer_surfaces_an_error_on_total_llm_failure(monkeypatch):
    monkeypatch.setattr(JobAnalyzerAgent, "_call_llm_for_json_async", _failing_llm)
    agent = JobAnalyzerAgent()
    result = asyncio.run(agent.process({"job_description": "Some real JD text"}))

    assert result["job_requirements"].parsing_confidence == 0.0
    assert result.get("errors"), "a total LLM failure must be reported in errors, not silently swallowed"
    assert any("LLM API call failed" in e for e in result["errors"])


def test_job_analyzer_still_degrades_gracefully_on_malformed_but_real_response(monkeypatch):
    """Regression guard: a response that DID come back from the LLM but
    isn't valid JSON is a different case (genuinely-low-confidence parse)
    and must NOT be treated as the total-failure case."""
    async def malformed_llm(self, prompt, max_retries=1, max_rate_limit_retries=3):
        return "this is not json"

    monkeypatch.setattr(JobAnalyzerAgent, "_call_llm_for_json_async", malformed_llm)
    agent = JobAnalyzerAgent()
    result = asyncio.run(agent.process({"job_description": "Some real JD text"}))

    assert result["job_requirements"].parsing_confidence == 0.3
    assert not result.get("errors")


def test_ensure_job_requirements_does_not_cache_a_failed_analysis(monkeypatch, tmp_path):
    """End-to-end: a total JobAnalyzerAgent failure must raise
    JobAnalysisFailedError instead of persisting an empty JobRequirements
    row that would be served as 'the' analysis on every future call."""
    import importlib
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'jd_failure_test.db'}")
    import src.db.database as db_module
    importlib.reload(db_module)
    from src.db import repository as repo
    from src.services import screening_service
    importlib.reload(screening_service)

    db_module.init_db()
    monkeypatch.setattr(JobAnalyzerAgent, "_call_llm_for_json_async", _failing_llm)

    with db_module.get_session() as db:
        org = repo.create_organization(db, "T")
        dept = repo.create_department(db, org.id, "E")
        pos = repo.create_position(db, dept.id, "P")
        jd = repo.create_job_description(db, pos.id, raw_text="Some real JD text")
        jd_id = jd.id

    with pytest.raises(screening_service.JobAnalysisFailedError):
        with db_module.get_session() as db:
            asyncio.run(screening_service.ensure_job_requirements(db, jd_id))

    with db_module.get_session() as db:
        # Must NOT have been persisted -- a later retry should be able to
        # try again for real instead of finding a cached empty result.
        assert repo.get_job_requirements(db, jd_id) is None

    monkeypatch.delenv("DATABASE_URL", raising=False)
    importlib.reload(db_module)
    importlib.reload(screening_service)
