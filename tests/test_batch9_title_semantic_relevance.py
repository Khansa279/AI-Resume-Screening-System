"""Batch 9/10 regression tests.

BATCH 10 UPDATE: the first version of _ROLE_RELATEDNESS included entries
like backend<->frontend=0.15 and backend<->data=0.15 on the reasoning
"still loosely engineering-adjacent". That gave Frontend Developer and
Data Analyst titles a small but real positive semantic score (~0.07)
against a Backend JD purely because both are software roles -- an
artificial boost with no real justification. Those entries were removed;
unrelated specific-family pairs now correctly fall through to the 0.0
default. Only DEFENSIBLE adjacencies remain (fullstack literally
contains both backend and frontend; backend<->devops reflects a real,
common professional overlap). The tests below were updated to assert
exact 0.0 for genuinely unrelated titles, and new tests were added to
lock in which specific-family pairs are (and are not) allowed to be
non-zero, so a future edit can't silently reintroduce a vague fallback.

ROOT CAUSE: after the Batch 8 title-score denominator/Jaccard fix,
job_relevance()'s title component was still PURELY LEXICAL (token-set
Jaccard overlap). Two titles that share no literal words score ~0 no
matter how semantically related they actually are -- e.g. "Senior
Software Engineer" vs "Backend Engineer - Python" overlap on only
"engineer" against a 5-token union, scoring ~0.09-0.2, identical to (or
even lower than) a genuinely unrelated title like "Data Analyst" would
score by pure chance. Meanwhile "Senior Backend Developer" -- which
*should* score close to the exact title -- ALSO scored ~0.09, because
"backend"/"developer" vs "backend"/"engineer"/"python" barely overlaps
lexically either. Lexical overlap cannot distinguish "clearly the same
discipline, different wording" from "plausibly related" from "different
discipline entirely".

FIX (src/agents/experience_eval.py): a small, deterministic, hand-curated
role-family taxonomy (_primary_role_family / _role_family_score) that
classifies a title into a coarse family (backend/frontend/fullstack/
data/devops/mobile/qa/generic_swe) and looks up a fixed relatedness score
between families. This is combined into title_score via max() with the
existing lexical score, so it only ever helps titles lexical overlap
under-scores -- it never lowers an already-good lexical match.

This intentionally:
  - Does NOT call an LLM (see the module-level comment in
    experience_eval.py for why: batch screening runs job_relevance() once
    per work-experience entry per candidate, and an LLM call on that hot
    path reintroduces Groq rate-limit risk, latency, and nondeterminism
    into a value that feeds directly into a numeric score).
  - Does NOT touch the 0.45/0.40/0.15 job_relevance weights.
  - Does NOT touch skill_hit or resp_score.
  - Does NOT hardcode any real candidate name or resume.
"""

from src.agents.experience_eval import (
    job_relevance,
    _primary_role_family,
    _role_family_score,
    _title_tokens,
)
from src.models import JobRequirements, WorkExperience


BACKEND_JD = JobRequirements(title="Backend Engineer - Python", required_skills=[], preferred_skills=[])


def _relevance_for_title(title: str, jd: JobRequirements = BACKEND_JD) -> float:
    return job_relevance(WorkExperience(title=title, company="X"), jd)


# ---------------------------------------------------------------------------
# Core semantic-ordering expectations from the diagnostic (Stage 3)
# ---------------------------------------------------------------------------

def test_exact_title_scores_highest():
    assert _relevance_for_title("Backend Engineer - Python") == 0.45


def test_semantically_related_backend_titles_score_as_high_as_exact():
    """'Senior Backend Developer' and 'Freelance Backend Developer' share
    almost no literal tokens with 'Backend Engineer - Python' (Jaccard
    ~0.09 before this fix), but are unambiguously the same discipline --
    they must not score anywhere near a lexically-driven near-zero."""
    for title in ("Senior Backend Developer", "Freelance Backend Developer"):
        rel = _relevance_for_title(title)
        assert rel == 0.45, f"{title!r} should match the exact-title score, got {rel}"


def test_senior_software_engineer_is_meaningfully_relevant_but_not_exact():
    """The exact case from the investigation: a generic senior engineering
    title must score noticeably above a clearly-unrelated title, but
    below an exact/backend-specific title match."""
    generic = _relevance_for_title("Senior Software Engineer")
    exact = _relevance_for_title("Backend Engineer - Python")
    unrelated = _relevance_for_title("Frontend Developer")

    assert 0.15 < generic < exact, f"expected meaningfully relevant but below exact, got {generic}"
    assert generic > unrelated, "a generic SWE title must outrank a clearly different discipline"


def test_generic_software_engineer_scores_the_same_as_senior_variant():
    """Seniority words must not change the semantic relevance verdict --
    only the role family matters here."""
    assert _relevance_for_title("Software Engineer") == _relevance_for_title("Senior Software Engineer")


def test_frontend_developer_gets_no_artificial_semantic_boost_against_backend_jd():
    """Batch 10: must be EXACTLY 0.0, not merely 'low'. Frontend and
    Backend are different disciplines with no inherent overlap -- there
    is no principled reason for any positive family_score here, and the
    lexical component is also genuinely 0 (no shared tokens)."""
    assert _role_family_score("Frontend Developer", "Backend Engineer - Python") == 0.0
    assert _relevance_for_title("Frontend Developer") == 0.0


def test_data_analyst_gets_no_artificial_semantic_boost_against_backend_jd():
    """Batch 10: same as above -- Data and Backend are different
    disciplines; no positive score should be manufactured for the pair."""
    assert _role_family_score("Data Analyst", "Backend Engineer - Python") == 0.0
    assert _relevance_for_title("Data Analyst") == 0.0


def test_mobile_and_qa_titles_get_no_family_score_boost_against_backend_jd():
    """Same principle extended to the other specific families that have
    no defensible adjacency to backend: mobile and QA."""
    assert _role_family_score("iOS Developer", "Backend Engineer - Python") == 0.0
    assert _role_family_score("QA Engineer", "Backend Engineer - Python") == 0.0


def test_only_defensible_specific_family_pairs_have_nonzero_relatedness():
    """Locks in the Batch 10 design intent: ONLY fullstack (which
    literally spans backend+frontend) and devops (a well-established
    backend adjacency) may have a non-zero cross-family score against
    backend. Every OTHER specific family pair must default to 0.0 -- this
    directly guards against reintroducing a vague 'still kind of related'
    fallback like the one this batch removed."""
    from src.agents.experience_eval import _ROLE_RELATEDNESS

    allowed_nonzero_pairs = {
        frozenset({"backend", "fullstack"}),
        frozenset({"frontend", "fullstack"}),
        frozenset({"backend", "devops"}),
        frozenset({"fullstack", "devops"}),
    }
    specific_families = {"backend", "frontend", "fullstack", "data", "devops", "mobile", "qa"}

    for a in specific_families:
        for b in specific_families:
            if a == b:
                continue
            key = frozenset({a, b})
            score = _ROLE_RELATEDNESS.get(key, 0.0)
            if key in allowed_nonzero_pairs:
                assert score > 0.0, f"{key} should be a defensible non-zero adjacency"
            else:
                assert score == 0.0, (
                    f"{key} should default to 0.0 (no principled adjacency), "
                    f"but found score={score} -- this looks like a reintroduced "
                    f"'vaguely related' fallback"
                )


def test_backend_adjacent_families_still_get_defensible_credit():
    """The removal of unjustified pairs must not have thrown out the
    genuinely defensible ones. Fullstack and DevOps must still score
    meaningfully above zero (but below an exact/backend-specific title)
    against a Backend JD."""
    fullstack = _relevance_for_title("Fullstack Engineer")
    devops = _relevance_for_title("DevOps Engineer")
    exact = _relevance_for_title("Backend Engineer - Python")

    assert 0.0 < devops < exact
    assert 0.0 < fullstack < exact


def test_frontend_and_data_analyst_do_not_outscore_generic_backend_relevant_titles():
    """A clearly-different discipline must never look MORE relevant to a
    backend JD than a domain-less engineering title does."""
    generic = _relevance_for_title("Senior Software Engineer")
    assert _relevance_for_title("Frontend Developer") < generic
    assert _relevance_for_title("Data Analyst") < generic


# ---------------------------------------------------------------------------
# The fix must not silently boost every engineering title into "relevant"
# ---------------------------------------------------------------------------

def test_unrelated_titles_are_not_artificially_boosted_to_high_relevance():
    for title in ("Frontend Developer", "Data Analyst", "iOS Developer", "QA Engineer"):
        rel = _relevance_for_title(title)
        assert rel < 0.3, f"{title!r} should not be boosted into 'relevant' territory, got {rel}"


def test_role_family_score_returns_none_when_jd_title_unrecognized():
    """A JD title with no recognizable role-family keyword at all (e.g.
    blank, or purely a company codename) must fall back to lexical
    scoring rather than being treated as zero relevance for everyone."""
    assert _role_family_score("Software Engineer", "") is None
    assert _role_family_score("Software Engineer", "Chief Wizard") is None


def test_role_family_score_is_zero_not_none_when_only_candidate_title_unrecognized():
    assert _role_family_score("Chief Wizard", "Backend Engineer") == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_title_semantic_scoring_is_deterministic_across_repeated_calls():
    titles = ["Senior Software Engineer", "Backend Developer", "Data Analyst", "Frontend Developer"]
    first = [_relevance_for_title(t) for t in titles]
    second = [_relevance_for_title(t) for t in titles]
    assert first == second


def test_primary_role_family_is_pure_and_deterministic():
    for _ in range(3):
        assert _primary_role_family("Senior Backend Engineer") == "backend"
        assert _primary_role_family("Senior Software Engineer") == "generic_swe"
        assert _primary_role_family("Frontend Developer") == "frontend"
        assert _primary_role_family("Data Analyst") == "data"


# ---------------------------------------------------------------------------
# skill_hit / resp_score must be completely unaffected by this change
# ---------------------------------------------------------------------------

def test_skill_hit_component_is_unchanged_by_title_family_backstop():
    """Same technologies, only the title changes -- job_relevance must
    move ONLY by the title component's contribution (0.45 weight), never
    by more than that, proving skill_hit/resp_score are untouched."""
    jd = JobRequirements(
        title="Backend Engineer - Python",
        required_skills=["Python", "Django", "PostgreSQL"],
        preferred_skills=[],
    )
    common_kwargs = dict(
        company="X",
        technologies=["Python", "Django", "PostgreSQL"],
        responsibilities=["Built REST APIs"],
    )
    rel_exact_title = job_relevance(WorkExperience(title="Backend Engineer - Python", **common_kwargs), jd)
    rel_generic_title = job_relevance(WorkExperience(title="Senior Software Engineer", **common_kwargs), jd)

    # Both get full skill_hit credit (identical technologies) -- the only
    # difference between the two calls is the title, and title is 45% of
    # the formula, so the delta must be explainable purely by the title
    # component's swing (family_score 1.0 vs 0.55 here == 0.45 * 0.45).
    delta = round(rel_exact_title - rel_generic_title, 2)
    assert delta == round(0.45 * (1.0 - 0.55), 2), (
        f"expected the delta between exact and generic titles to be exactly "
        f"the title-weight * family-score gap, got {delta}"
    )


def test_responsibility_overlap_scoring_path_is_untouched():
    """job_resp / jd_resp overlap (_overlap over responsibility tokens)
    doesn't go through the title-family code path at all -- sanity check
    that responsibility-only relevance signal still works standalone."""
    jd = JobRequirements(
        title="Something With No Recognized Role Words At All",
        required_skills=[], preferred_skills=[],
        responsibilities=["Design and build RESTful APIs using Python and Django"],
    )
    matching_resp = WorkExperience(
        title="Chief Wizard", company="X",
        responsibilities=["Design and build RESTful APIs using Python and Django"],
    )
    unrelated_resp = WorkExperience(
        title="Chief Wizard", company="X",
        responsibilities=["Watered the office plants"],
    )
    assert job_relevance(matching_resp, jd) > job_relevance(unrelated_resp, jd)


# ---------------------------------------------------------------------------
# Existing (pre-Batch-9) title-token behavior must not regress
# ---------------------------------------------------------------------------

def test_title_tokens_helper_unchanged():
    assert _title_tokens("Senior Software Engineer") == {"senior", "software", "engineer"}
    assert _title_tokens("Backend Engineer - Python") == {"backend", "engineer", "python"}
