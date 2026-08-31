"""Regression tests: job_relevance()'s lexical title_score previously
credited two titles purely because they shared GENERIC seniority/role
words ("senior", "engineer", "developer", "software", "associate",
"junior", "lead", "intern" -- the existing _STOP list), even when the
titles belong to genuinely unrelated disciplines and share no distinctive
domain word at all.

Confirmed via the real job_relevance() production path:
    "Senior QA Engineer"    vs "Senior Backend Engineer" JD -> 0.23
    "Senior Data Scientist" vs "Senior Backend Engineer" JD -> 0.09
Both scores came ENTIRELY from the shared words "senior"/"engineer" --
neither candidate title shares any backend-specific vocabulary, and the
role-family backstop (_role_family_score) independently and correctly
returns 0.0 for both pairs (QA<->backend and data<->backend are not in
_ROLE_RELATEDNESS, i.e. deliberately "no relationship" -- see that
matrix's Batch 10 docstring, which removed an equivalent artificial
"both happen to be software roles" boost at the family layer for exactly
this reason). The lexical side had no equivalent guard, so it could
smuggle the same kind of unjustified credit back in through generic
vocabulary alone.

Fix (src/agents/experience_eval.py::job_relevance): the raw lexical
Jaccard title_score is discarded (set to 0.0) when every token the two
titles share is a member of the existing `_STOP` word list -- i.e. no
DISTINCTIVE word is shared. This does not modify `_title_tokens()` itself
(so Batch 7's "generic words must stay in the token set, or titles like
'Senior Software Engineer' collapse to an empty set and title_score
zeroes out entirely" fix is untouched), does not modify the role-family
backstop, and never lowers title_score for a case where a real
distinctive word is shared.
"""

from src.agents.experience_eval import job_relevance, _role_family_score
from src.models import JobRequirements, WorkExperience


# ---------------------------------------------------------------------------
# False positives: generic-word-only overlap must contribute nothing
# ---------------------------------------------------------------------------

def test_qa_engineer_title_not_relevant_to_backend_engineer_jd():
    jd = JobRequirements(title="Senior Backend Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Senior QA Engineer", company="X", technologies=[], responsibilities=[])
    # Sanity: the family backstop already (correctly) says "no relationship".
    assert _role_family_score(exp.title, jd.title) == 0.0
    assert job_relevance(exp, jd) == 0.0


def test_data_scientist_title_not_relevant_to_backend_engineer_jd():
    jd = JobRequirements(title="Senior Backend Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Senior Data Scientist", company="X", technologies=[], responsibilities=[])
    assert _role_family_score(exp.title, jd.title) == 0.0
    assert job_relevance(exp, jd) == 0.0


def test_mobile_developer_title_not_relevant_to_backend_engineer_jd_via_generic_words_alone():
    jd = JobRequirements(title="Senior Backend Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Senior Mobile Developer", company="X", technologies=[], responsibilities=[])
    # "developer" is shared and generic; "senior" is shared and generic;
    # "mobile" vs "backend" share nothing distinctive.
    assert job_relevance(exp, jd) == 0.0


# ---------------------------------------------------------------------------
# Genuine matches must be unaffected: any shared DISTINCTIVE word still
# drives a real lexical score, and titles with no lexical overlap at all
# still get full credit from the role-family backstop where applicable.
# ---------------------------------------------------------------------------

def test_shared_distinctive_word_still_credited_despite_shared_generic_words_too():
    jd = JobRequirements(title="Backend Developer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Junior Backend Developer", company="X")
    # Shares "backend" (distinctive) AND "developer" (generic) -- must
    # still score highly; family backstop (same family) also gives 1.0.
    assert job_relevance(exp, jd) == 0.45


def test_exact_title_match_still_scores_full_relevance():
    jd = JobRequirements(title="Backend Engineer - Python", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Backend Engineer - Python", company="X")
    assert job_relevance(exp, jd) == 0.45


def test_generic_only_title_still_gets_family_backstop_credit_not_zero():
    """"Senior Software Engineer" shares only generic words with a
    "Backend Engineer" JD, so the lexical component is now 0 -- but the
    role-family backstop (generic_swe <-> backend = 0.55, a deliberate,
    documented relatedness score) still applies, so this must NOT be
    treated as fully irrelevant."""
    jd = JobRequirements(title="Backend Engineer - Python", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Senior Software Engineer", company="X")
    relevance = job_relevance(exp, jd)
    assert relevance > 0.0, "generic engineering title should still get partial family-backstop credit"
    assert relevance == round(0.45 * 0.55, 2)


def test_no_overlap_at_all_still_scores_zero():
    jd = JobRequirements(title="Platform Engineer", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Kubernetes Specialist", company="X", technologies=[], responsibilities=[])
    assert job_relevance(exp, jd) == 0.0
