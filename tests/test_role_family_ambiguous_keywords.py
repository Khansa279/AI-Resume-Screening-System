"""Regression tests: _ROLE_FAMILY_KEYWORDS previously included "server",
"client", and "platform" as standalone family-membership signals for
backend/frontend/devops respectively. _primary_role_family() only ever
sees a single already-tokenized word with no surrounding context, so it
could not distinguish a genuine engineering title from an unrelated job
title that happens to share one of those common English words -- e.g. a
restaurant "Server", a "Client Relations Manager", or a "Platform Sales
Manager".

Because _role_family_score() can hand out a FULL 1.0 title_score (via
max() with the lexical overlap score) whenever the candidate and JD
titles resolve to the SAME family, this let those unrelated titles score
job_relevance() = 0.45 (0.45 weight * 1.0 title_score, with zero real
skill or responsibility overlap) against a same-family engineering JD --
a nontrivial contribution to role_relevance and years_relevant credit for
a candidate whose actual job has nothing to do with software.

Confirmed via the real job_relevance()/_role_family_score() production
path (see the audit's reproduction). Fix (src/agents/experience_eval.py):
remove "server", "client", and "platform" from _ROLE_FAMILY_KEYWORDS.
Genuine engineering titles remain covered by "backend"/"frontend"/
"devops" themselves plus "sre"/"infrastructure"/"ui"; a title that only
carried a removed word (e.g. "Platform Engineer") now falls through to
the "generic_swe" bucket via "engineer" and gets a smaller, but still
nonzero, family credit instead of a false-positive full match. This does
NOT touch the 0.45/0.4/0.15 job_relevance weights, the role-family score
matrix (_ROLE_RELATEDNESS), skill_hit, or resp_score.
"""

from src.agents.experience_eval import (
    job_relevance,
    _role_family_score,
    _primary_role_family,
)
from src.models import JobRequirements, WorkExperience


# ---------------------------------------------------------------------------
# _primary_role_family: ambiguous words no longer resolve to a family
# ---------------------------------------------------------------------------

def test_bare_server_title_has_no_role_family():
    assert _primary_role_family("Server") is None


def test_client_relations_title_has_no_role_family():
    assert _primary_role_family("Client Relations Manager") is None


def test_platform_sales_title_has_no_role_family():
    assert _primary_role_family("Platform Sales Manager") is None


# ---------------------------------------------------------------------------
# _role_family_score: no more same-family false positives
# ---------------------------------------------------------------------------

def test_waiter_server_scores_no_family_relatedness_to_backend():
    assert _role_family_score("Server", "Backend Engineer") == 0.0


def test_client_relations_scores_no_family_relatedness_to_frontend():
    assert _role_family_score("Client Relations Manager", "Frontend Engineer") == 0.0


def test_platform_sales_scores_no_family_relatedness_to_devops():
    assert _role_family_score("Platform Sales Manager", "DevOps Engineer") == 0.0


# ---------------------------------------------------------------------------
# End-to-end job_relevance(): the false-positive contribution is gone
# ---------------------------------------------------------------------------

def test_waiter_server_job_is_not_relevant_to_backend_engineer_jd():
    jd = JobRequirements(title="Backend Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(
        title="Server", company="Diner", technologies=[],
        responsibilities=["Took customer orders", "Served food and drinks"],
    )
    assert job_relevance(exp, jd) == 0.0


def test_client_relations_job_is_not_relevant_to_frontend_engineer_jd():
    jd = JobRequirements(title="Frontend Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(
        title="Client Relations Manager", company="X", technologies=[],
        responsibilities=["Managed client accounts", "Handled client escalations"],
    )
    assert job_relevance(exp, jd) == 0.0


def test_platform_sales_job_is_not_relevant_to_devops_engineer_jd():
    jd = JobRequirements(title="DevOps Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(
        title="Platform Sales Manager", company="X", technologies=[],
        responsibilities=["Sold enterprise software platform licenses"],
    )
    assert job_relevance(exp, jd) == 0.0


# ---------------------------------------------------------------------------
# Genuine matches must still work -- this fix must not create false
# negatives for real engineering titles.
# ---------------------------------------------------------------------------

def test_genuine_backend_developer_still_matches_backend_engineer_jd():
    jd = JobRequirements(title="Backend Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Backend Developer", company="X")
    # "backend" is still a shared literal token, so this is a strong
    # lexical match on its own, unaffected by the family-keyword change.
    assert job_relevance(exp, jd) == 0.45


def test_genuine_frontend_developer_still_matches_frontend_engineer_jd():
    jd = JobRequirements(title="Frontend Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Frontend Developer", company="X")
    assert job_relevance(exp, jd) == 0.45


def test_platform_engineer_title_degrades_gracefully_not_to_zero():
    """"Platform Engineer" no longer resolves to the "devops" family
    directly (the ambiguous "platform" keyword is gone), but it still
    contains "engineer" -- it falls through to the generic_swe bucket and
    still gets a real, if reduced, family-relatedness credit against a
    DevOps JD, rather than being incorrectly treated as fully unrelated."""
    assert _primary_role_family("Platform Engineer") == "generic_swe"
    score = _role_family_score("Platform Engineer", "DevOps Engineer")
    assert 0.0 < score < 1.0


def test_infrastructure_keyword_still_recognizes_devops():
    assert _primary_role_family("Infrastructure Engineer") == "devops"
    assert _role_family_score("Infrastructure Engineer", "DevOps Engineer") == 1.0
