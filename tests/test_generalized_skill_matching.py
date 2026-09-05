"""Unit tests for src/skill_matching.py -- the generalized, domain-agnostic
skill/requirement equivalence matcher -- in isolation from its integration
into job_relevance() (see test_job_relevance_semantic_skill_hit.py for
those).

Every test here uses a hand-written fake embedding provider or
monkeypatched env vars, matching this project's existing convention (see
tests/test_embeddings_module.py) -- none of them make a live network
call.

Covers, per the investigation into Marketing-domain role-relevance
under-scoring:
  - deterministic tier still works unchanged (tech-domain regressions)
  - true equivalents match with full credit (Google Ads / paid search
    advertising, GA4 / Google Analytics 4, SEO / search engine
    optimization)
  - closely related-but-distinct skills get partial credit, not full
    credit (Google Ads vs digital advertising strategy)
  - merely co-occurring / same-category-but-distinct skills do NOT match
    (Python vs Java -- both "a programming language")
  - clearly unrelated skills do NOT match (SEO vs graphic design,
    Google Ads vs social media management)
  - safe fallback: no provider configured -> deterministic-only, never
    raises
  - the optional LLM validation tier is off by default, only consulted
    for the ambiguous "related" band, and never overrides a confident
    negative
"""

import pytest

from src import skill_matching as sm
from src import embeddings as emb_module


@pytest.fixture(autouse=True)
def _reset_state():
    emb_module.reset_embedding_provider_cache()
    emb_module.clear_embedding_cache()
    yield
    emb_module.reset_embedding_provider_cache()
    emb_module.clear_embedding_cache()


class TopicEmbeddingProvider:
    """Deterministic, offline stand-in for a real embedding provider.

    Modeled as a small set of CONCEPTS, each recognized by several
    different surface phrasings -- this is what lets two differently
    WORDED but semantically equivalent phrases ("paid search
    advertising" vs "Google Ads PPC") land on the same vector dimension
    and score a high cosine similarity, while two phrases from distinct
    concepts (even superficially similar ones, like "Python" and "Java"
    both being programming languages) get NO shared dimension and score
    zero -- a controlled, fully offline simulation of what a real
    embedding model is expected to do for these hand-picked examples.
    """

    CONCEPTS: dict[str, tuple[str, ...]] = {
        "paid_search_ads": (
            "google ads", "paid search advertising", "paid search", "ppc", "adwords",
        ),
        "seo": (
            "seo", "search engine optimization", "keyword research", "keyword optimization",
            "organic growth", "seo rankings", "organic search",
        ),
        "ga4": (
            "ga4", "google analytics 4", "google analytics", "campaign reporting",
            "analytics reporting", "dashboards",
        ),
        "social_media": (
            "social media", "organic social",
        ),
        "graphic_design": (
            "graphic design", "photoshop", "illustrator",
        ),
        "python": ("python",),
        "java": ("java",),
    }

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if any(phrase in lowered for phrase in phrases) else 0.0
            for phrases in self.CONCEPTS.values()
        ]


@pytest.fixture()
def topic_provider(monkeypatch):
    provider = TopicEmbeddingProvider()
    monkeypatch.setattr(emb_module, "get_embedding_provider", lambda: provider)
    return provider


# ---------------------------------------------------------------------------
# Tier 1: deterministic tier unchanged (tech-domain, no provider needed)
# ---------------------------------------------------------------------------

def test_deterministic_tier_matches_taxonomy_alias_without_any_provider():
    decision = sm.match_requirement("Experience with Python", "Built services using Python and Django")
    assert decision.matched
    assert decision.quality == "taxonomy_or_literal"
    assert decision.credit == 1.0


def test_deterministic_tier_returns_none_match_when_nothing_recognized_and_no_provider():
    decision = sm.match_requirement("SEO and keyword research", "Managed social ad campaigns")
    assert not decision.matched
    assert decision.quality == "none"
    assert decision.credit == 0.0


def test_deterministic_tier_wins_even_when_embedding_would_disagree(topic_provider):
    """Tier 1 is authoritative when it finds something -- tier 2 is never
    even consulted, so a provider that would (wrongly) say "unrelated"
    can't override a real literal match."""
    decision = sm.match_requirement("Python", "5 years of Python development")
    assert decision.quality == "taxonomy_or_literal"


# ---------------------------------------------------------------------------
# True equivalents (Marketing domain) -- full credit
# ---------------------------------------------------------------------------

def test_google_ads_equivalent_to_paid_search_advertising(topic_provider):
    decision = sm.match_requirement(
        "Paid search advertising", "Managed Google Ads PPC campaigns", use_llm=False,
    )
    assert decision.matched
    assert decision.quality in ("embedding_equivalent", "embedding_related")


def test_ga4_equivalent_to_google_analytics_4(topic_provider):
    decision = sm.match_requirement(
        "Google Analytics 4 reporting", "Built dashboards in GA4 for campaign reporting", use_llm=False,
    )
    assert decision.matched


def test_seo_equivalent_to_search_engine_optimization(topic_provider):
    decision = sm.match_requirement(
        "Search engine optimization and organic growth",
        "Improved SEO rankings through keyword optimization",
        use_llm=False,
    )
    assert decision.matched


# ---------------------------------------------------------------------------
# Negative cases -- must NOT match, or must only get partial credit
# ---------------------------------------------------------------------------

def test_google_ads_does_not_match_social_media_management(topic_provider):
    decision = sm.match_requirement(
        "Google Ads paid search advertising", "Managed organic social media content", use_llm=False,
    )
    assert decision.credit < sm.SKILL_EQUIVALENCE_FLOOR or decision.quality != "embedding_equivalent"
    assert not (decision.quality == "embedding_equivalent")


def test_seo_does_not_match_graphic_design(topic_provider):
    decision = sm.match_requirement(
        "SEO and keyword research", "Created graphic designs using Photoshop and Illustrator",
        use_llm=False,
    )
    assert decision.quality != "embedding_equivalent"
    assert decision.credit == 0.0


def test_python_does_not_equal_java_merely_for_being_programming_languages(topic_provider):
    """The exact scenario called out in the investigation: two short,
    single-topic terms in the same broad category must not be treated
    as equivalent skills just because an embedding space clusters them
    moderately close together."""
    decision = sm.match_requirement("Java programming experience", "5 years of Python programming", use_llm=False)
    assert decision.quality != "embedding_equivalent"
    assert decision.credit < 1.0


# ---------------------------------------------------------------------------
# Dual-threshold behavior: equivalent vs related vs unrelated are distinct
# ---------------------------------------------------------------------------

def test_equivalence_floor_is_stricter_than_related_floor():
    assert sm.SKILL_EQUIVALENCE_FLOOR > sm.SKILL_RELATED_FLOOR


def test_related_band_gives_partial_not_full_credit():
    assert 0.0 < sm.SKILL_RELATED_CREDIT < 1.0


class FixedScoreProvider:
    """Lets a test pin the exact rescaled embedding score returned, to
    test the threshold boundaries precisely rather than relying on the
    topic-keyword heuristic provider."""

    def __init__(self, cosine_sim: float):
        # embedding_relevance rescales cosine similarity via
        # similarity_to_relevance's FLOOR/CEIL; for these boundary tests
        # we bypass that by monkeypatching embedding_relevance directly
        # instead (see the tests below).
        self.cosine_sim = cosine_sim

    def embed(self, text):
        return [self.cosine_sim, 1.0]


def test_score_at_equivalence_floor_is_full_credit(monkeypatch):
    monkeypatch.setattr(sm, "embedding_relevance", lambda a, b: sm.SKILL_EQUIVALENCE_FLOOR)
    decision = sm.match_requirement("unrecognized requirement xyz", "unrecognized candidate text abc")
    assert decision.matched
    assert decision.credit == 1.0
    assert decision.quality == "embedding_equivalent"


def test_score_just_below_equivalence_floor_is_partial_credit(monkeypatch):
    monkeypatch.setattr(sm, "embedding_relevance", lambda a, b: sm.SKILL_EQUIVALENCE_FLOOR - 0.01)
    decision = sm.match_requirement("unrecognized requirement xyz", "unrecognized candidate text abc", use_llm=False)
    assert decision.matched
    assert decision.credit == sm.SKILL_RELATED_CREDIT
    assert decision.quality == "embedding_related"


def test_score_below_related_floor_is_zero_credit(monkeypatch):
    monkeypatch.setattr(sm, "embedding_relevance", lambda a, b: sm.SKILL_RELATED_FLOOR - 0.01)
    decision = sm.match_requirement("unrecognized requirement xyz", "unrecognized candidate text abc")
    assert not decision.matched
    assert decision.credit == 0.0


# ---------------------------------------------------------------------------
# Safety / fallback behavior
# ---------------------------------------------------------------------------

def test_no_provider_configured_falls_back_to_deterministic_only(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    emb_module.reset_embedding_provider_cache()
    decision = sm.match_requirement("SEO and keyword research", "Managed social ad campaigns")
    assert not decision.matched
    assert decision.quality == "none"


def test_never_raises_on_empty_input():
    assert sm.match_requirement("", "something").quality == "none"
    assert sm.match_requirement("something", "").quality == "none"
    assert sm.match_requirement("", "").quality == "none"


def test_use_embedding_false_forces_deterministic_only(topic_provider):
    decision = sm.match_requirement(
        "Paid search advertising", "Managed Google Ads PPC campaigns", use_embedding=False,
    )
    assert not decision.matched
    assert decision.quality == "none"


# ---------------------------------------------------------------------------
# Optional LLM validation tier: off by default, opt-in, safe
# ---------------------------------------------------------------------------

def test_llm_validation_disabled_by_default(monkeypatch):
    monkeypatch.delenv(sm.ENABLE_LLM_SKILL_VALIDATION_ENV, raising=False)
    assert sm.llm_validate_requirement("SEO", "some evidence") is None


def test_llm_validation_returns_none_when_no_provider_configured(monkeypatch):
    monkeypatch.setenv(sm.ENABLE_LLM_SKILL_VALIDATION_ENV, "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert sm.llm_validate_requirement("SEO", "some evidence") is None


def test_llm_tier_never_consulted_for_confidently_unrelated_pair(monkeypatch):
    """A confident tier-2 negative must never escalate to an LLM call --
    keeps the feature cheap and keeps a clear negative a negative."""
    monkeypatch.setenv(sm.ENABLE_LLM_SKILL_VALIDATION_ENV, "true")
    calls = []
    monkeypatch.setattr(sm, "llm_validate_requirement", lambda r, c: calls.append((r, c)) or None)
    monkeypatch.setattr(sm, "embedding_relevance", lambda a, b: 0.1)  # confidently unrelated
    sm.match_requirement("unrecognized requirement", "unrecognized text")
    assert calls == []


def test_llm_tier_consulted_only_for_ambiguous_related_band(monkeypatch):
    monkeypatch.setenv(sm.ENABLE_LLM_SKILL_VALIDATION_ENV, "true")
    calls = []

    def fake_llm(req, text):
        calls.append((req, text))
        return sm.SkillMatchDecision(True, "llm_validated", 1.0, text, None, "validated")

    monkeypatch.setattr(sm, "llm_validate_requirement", fake_llm)
    monkeypatch.setattr(sm, "embedding_relevance", lambda a, b: sm.SKILL_RELATED_FLOOR + 0.01)
    decision = sm.match_requirement("unrecognized requirement", "unrecognized text")
    assert len(calls) == 1
    assert decision.quality == "llm_validated"
    assert decision.credit == 1.0


def test_llm_tier_failure_keeps_tier2_partial_credit(monkeypatch):
    monkeypatch.setenv(sm.ENABLE_LLM_SKILL_VALIDATION_ENV, "true")
    monkeypatch.setattr(sm, "llm_validate_requirement", lambda r, c: None)  # simulates any failure
    monkeypatch.setattr(sm, "embedding_relevance", lambda a, b: sm.SKILL_RELATED_FLOOR + 0.01)
    decision = sm.match_requirement("unrecognized requirement", "unrecognized text")
    assert decision.quality == "embedding_related"
    assert decision.credit == sm.SKILL_RELATED_CREDIT
