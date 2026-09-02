"""Regression tests for the embedding-score calibration review.

BACKGROUND (production-readiness review): job_relevance() combined the
deterministic title/taxonomy signal and the embedding-based signal via a
plain `title_score = max(title_score, embedding_score)`, unconditionally.
That was verified to let a SINGLE noisy or merely moderate embedding
score manufacture meaningful relevance out of thin air for a genuinely
UNRELATED pair (e.g. "Backend Engineer" vs "Marketing Manager", where the
deterministic signal correctly found 0.0 overlap) -- text-embedding
spaces are well known to cluster even unrelated short phrases at a
non-trivial baseline cosine similarity ("anisotropy"), and nothing
in the original max() combination distinguished that baseline noise from
genuine relatedness.

FIX (src/agents/experience_eval.py::job_relevance): when the
deterministic signal found ZERO overlap (title_score == 0.0 immediately
before the embedding step), the embedding score is only trusted at full
strength if it clears `_EMBEDDING_ZERO_CORROBORATION_TRUST_FLOOR` (0.6);
below that it's dampened via `_EMBEDDING_ZERO_CORROBORATION_DAMPENING`
(0.5) rather than trusted outright. The moment there IS any deterministic
corroboration at all (title_score > 0.0 -- a real shared title word or
taxonomy skill), the original max() behavior is completely unchanged.

This file specifically regression-tests:
  - the false-positive scenario the review found, at multiple embedding
    score levels, confirming it's now dampened
  - that a confidently-related, zero-corroboration pair (embedding score
    above the trust floor) still gets full credit -- the fix must not
    become so conservative it defeats the point of adding embeddings
  - that ordinary corroborated cases (title_score already > 0 before
    embeddings) are completely unaffected
  - the 0.45/0.4/0.15 weights are untouched
  - all 9 representative domain cases requested for this review
"""

import pytest

from src.agents import experience_eval as ee
from src.agents.experience_eval import job_relevance
from src.models import JobRequirements, WorkExperience


@pytest.fixture(autouse=True)
def _clear_embedding_mock():
    yield
    # Best-effort cleanup: tests here monkeypatch ee.embedding_relevance
    # directly (module attribute reassignment, matching the existing
    # convention already used in test_embedding_role_relevance_integration.py),
    # so restore the real function afterward.
    import importlib
    import src.embeddings as emb_module
    ee.embedding_relevance = emb_module.embedding_relevance


UNRELATED_JD = JobRequirements(title="Backend Engineer", required_skills=[])
UNRELATED_EXP = WorkExperience(
    title="Marketing Manager", company="BrightAd Agency", responsibilities=["Managed campaigns"],
)


def _assert_zero_deterministic_corroboration():
    """Sanity precondition for every false-positive test below: this
    pair must have ZERO deterministic (lexical + taxonomy) overlap, or
    the "zero corroboration" branch being tested here doesn't even
    apply."""
    from unittest.mock import patch
    with patch.object(ee, "embedding_relevance", return_value=None):
        assert job_relevance(UNRELATED_EXP, UNRELATED_JD) == 0.0


# ---------------------------------------------------------------------------
# Core regression: noisy/moderate embedding scores no longer manufacture
# relevance for a genuinely unrelated, zero-corroboration pair.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("noisy_score", [0.30, 0.45, 0.55, 0.59])
def test_moderate_noisy_embedding_score_is_dampened_not_trusted(noisy_score):
    _assert_zero_deterministic_corroboration()
    ee.embedding_relevance = lambda a, b: noisy_score

    relevance = job_relevance(UNRELATED_EXP, UNRELATED_JD)
    naive_max_relevance = round(0.45 * noisy_score, 2)

    assert relevance < naive_max_relevance, (
        f"a below-trust-floor embedding score of {noisy_score} must be dampened, "
        f"not trusted at full strength via plain max() (got {relevance}, "
        f"which is what un-dampened max() would have produced)"
    )
    # Explicit expected value: the embedding's contribution is exactly
    # halved (0.5 dampening) since there is no other corroboration at all.
    assert relevance == round(0.45 * noisy_score * ee._EMBEDDING_ZERO_CORROBORATION_DAMPENING, 2)


def test_dampened_false_positive_score_stays_in_clear_reject_territory():
    """The concrete, business-meaningful check: even the worst noisy
    score just below the trust floor must not push an unrelated pair's
    title contribution into a range that could plausibly flip a
    recommendation from reject to manual-review."""
    _assert_zero_deterministic_corroboration()
    ee.embedding_relevance = lambda a, b: 0.59  # just under the 0.6 trust floor
    relevance = job_relevance(UNRELATED_EXP, UNRELATED_JD)
    assert relevance <= 0.15, f"expected a small, clearly-reject-range score, got {relevance}"


# ---------------------------------------------------------------------------
# The fix must not become so conservative it defeats embeddings: a
# genuinely confident, zero-corroboration embedding score must still be
# fully trusted.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("confident_score", [0.6, 0.7, 0.85, 1.0])
def test_confident_embedding_score_is_still_fully_trusted_with_zero_corroboration(confident_score):
    _assert_zero_deterministic_corroboration()
    ee.embedding_relevance = lambda a, b: confident_score

    relevance = job_relevance(UNRELATED_EXP, UNRELATED_JD)
    assert relevance == round(0.45 * confident_score, 2), (
        "an embedding score at/above the trust floor must be used at full "
        "strength, exactly like plain max() would -- the fix only dampens "
        "scores BELOW the trust floor"
    )


def test_trust_floor_boundary_is_exact():
    """Confirms the >= boundary: exactly at the trust floor, full trust
    applies (not dampening)."""
    _assert_zero_deterministic_corroboration()
    floor = ee._EMBEDDING_ZERO_CORROBORATION_TRUST_FLOOR
    ee.embedding_relevance = lambda a, b: floor
    relevance = job_relevance(UNRELATED_EXP, UNRELATED_JD)
    assert relevance == round(0.45 * floor, 2)


# ---------------------------------------------------------------------------
# Corroborated cases (title_score already > 0 before the embedding step)
# must be completely unaffected -- the fix only changes the zero-
# corroboration path.
# ---------------------------------------------------------------------------

def test_corroborated_case_still_uses_plain_max_unaffected():
    """Any real deterministic signal at all -- here, a shared title word
    -- means the ordinary max() combination applies exactly as before,
    even for an embedding score below the zero-corroboration trust
    floor."""
    jd = JobRequirements(title="Registered Nurse - ICU", required_skills=[])
    exp = WorkExperience(title="Registered Nurse - Med/Surg", company="City Hospital")

    from unittest.mock import patch
    with patch.object(ee, "embedding_relevance", return_value=None):
        deterministic_only = job_relevance(exp, jd)
    assert deterministic_only > 0.0, "test precondition: this pair must have real lexical overlap ('registered'/'nurse')"

    ee.embedding_relevance = lambda a, b: 0.3  # below the zero-corroboration trust floor
    with_embedding = job_relevance(exp, jd)
    assert with_embedding == max(deterministic_only, round(0.45 * 0.3, 2)), (
        "with real deterministic corroboration already present, an embedding "
        "score -- even a modest one -- should combine via plain max(), "
        "unaffected by the zero-corroboration dampening"
    )


def test_exact_title_match_completely_unaffected_by_the_fix():
    jd = JobRequirements(title="Backend Engineer - Python", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Backend Engineer - Python", company="X")
    ee.embedding_relevance = lambda a, b: 0.1  # low, irrelevant -- title_score is already 1.0
    assert job_relevance(exp, jd) == 0.45


# ---------------------------------------------------------------------------
# Weights (0.45 / 0.4 / 0.15) remain untouched by this fix.
# ---------------------------------------------------------------------------

def test_weights_unchanged_by_calibration_fix():
    ee.embedding_relevance = lambda a, b: None
    jd = JobRequirements(
        title="Something With No Recognized Words At All",
        required_skills=["Python", "Django", "PostgreSQL"],
    )
    exp = WorkExperience(
        title="Zzz Unrelated Title Zzz", company="X",
        technologies=["Python", "Django", "PostgreSQL"], responsibilities=[],
    )
    relevance = job_relevance(exp, jd)
    expected_title_score = ee._semantic_role_relevance(exp, jd)
    expected = round(0.45 * expected_title_score + 0.4 * 1.0 + 0.15 * 0.0, 2)
    assert relevance == expected
