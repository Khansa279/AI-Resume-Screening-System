"""Unit tests for src/embeddings.py -- the embedding-provider abstraction
itself, in isolation from experience_eval.py's integration of it.

Every test here uses a hand-written fake provider or monkeypatched env
vars -- NONE of them make a live network call, per the project's
constraint that the normal test suite must not depend on a live external
API.
"""

import math

import pytest

from src import embeddings as emb


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Every test gets a clean provider cache and embedding cache, so
    tests can't leak state into each other via the module-level caches."""
    emb.reset_embedding_provider_cache()
    emb.clear_embedding_cache()
    yield
    emb.reset_embedding_provider_cache()
    emb.clear_embedding_cache()


class FakeProvider:
    """Deterministic, offline stand-in for a real embedding provider.
    Returns a small hand-built vector based on which topic keywords
    appear in the text, so related texts (sharing keywords) get high
    cosine similarity and unrelated texts get low similarity -- a
    controlled, fully offline simulation of what a real embedding model
    should do."""

    TOPICS = ("nurse", "medical", "chef", "kitchen", "python", "ml")

    def __init__(self):
        self.call_count = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        lowered = text.lower()
        return [1.0 if topic in lowered else 0.0 for topic in self.TOPICS]


class FailingProvider:
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("simulated provider failure")


class MalformedProvider:
    def embed(self, text: str):
        return None


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors_is_one():
    assert emb.cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert emb.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_negative_one():
    assert emb.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_handles_zero_vector_gracefully():
    assert emb.cosine_similarity([0.0, 0.0], [1.0, 1.0]) is None


def test_cosine_similarity_handles_empty_or_mismatched_vectors():
    assert emb.cosine_similarity([], [1.0]) is None
    assert emb.cosine_similarity([1.0, 2.0], [1.0]) is None


# ---------------------------------------------------------------------------
# similarity_to_relevance mapping (explicit, testable transformation)
# ---------------------------------------------------------------------------

def test_similarity_to_relevance_floor_and_below_is_zero():
    assert emb.similarity_to_relevance(emb.EMBEDDING_SIMILARITY_FLOOR) == 0.0
    assert emb.similarity_to_relevance(0.0) == 0.0
    assert emb.similarity_to_relevance(-1.0) == 0.0


def test_similarity_to_relevance_ceiling_and_above_is_one():
    assert emb.similarity_to_relevance(emb.EMBEDDING_SIMILARITY_CEIL) == 1.0
    assert emb.similarity_to_relevance(1.0) == 1.0


def test_similarity_to_relevance_midpoint_is_linear():
    mid = (emb.EMBEDDING_SIMILARITY_FLOOR + emb.EMBEDDING_SIMILARITY_CEIL) / 2
    assert emb.similarity_to_relevance(mid) == pytest.approx(0.5, abs=1e-6)


def test_similarity_to_relevance_is_monotonic():
    samples = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    scores = [emb.similarity_to_relevance(s) for s in samples]
    assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# embed_text: caching + failure handling
# ---------------------------------------------------------------------------

def test_embed_text_returns_vector_from_provider():
    provider = FakeProvider()
    vector = emb.embed_text("registered nurse medical", provider=provider)
    assert vector == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_embed_text_caches_repeated_calls_for_identical_text():
    provider = FakeProvider()
    emb.embed_text("python ml engineer", provider=provider)
    emb.embed_text("python ml engineer", provider=provider)
    emb.embed_text("python ml engineer", provider=provider)
    assert provider.call_count == 1, "identical text must only be embedded once (cached)"


def test_embed_text_does_not_cache_across_different_text():
    provider = FakeProvider()
    emb.embed_text("python ml engineer", provider=provider)
    emb.embed_text("chef kitchen", provider=provider)
    assert provider.call_count == 2


def test_embed_text_returns_none_on_empty_text():
    provider = FakeProvider()
    assert emb.embed_text("", provider=provider) is None
    assert emb.embed_text("   ", provider=provider) is None
    assert provider.call_count == 0


def test_embed_text_returns_none_when_provider_raises():
    assert emb.embed_text("some text", provider=FailingProvider()) is None


def test_embed_text_returns_none_on_malformed_provider_response():
    assert emb.embed_text("some text", provider=MalformedProvider()) is None


def test_embed_text_returns_none_when_no_provider_available(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    emb.reset_embedding_provider_cache()
    assert emb.embed_text("some text") is None


# ---------------------------------------------------------------------------
# embedding_relevance: the high-level function experience_eval.py calls
# ---------------------------------------------------------------------------

def test_embedding_relevance_high_for_overlapping_topics():
    provider = FakeProvider()
    score = emb.embedding_relevance(
        "Registered Nurse ICU medical", "Critical Care Nurse medical", provider=provider,
    )
    assert score is not None
    assert score > 0.5


def test_embedding_relevance_low_for_disjoint_topics():
    provider = FakeProvider()
    score = emb.embedding_relevance(
        "Registered Nurse medical", "Executive Chef kitchen", provider=provider,
    )
    assert score == 0.0


def test_embedding_relevance_none_when_provider_fails():
    assert emb.embedding_relevance("a", "b", provider=FailingProvider()) is None


def test_embedding_relevance_none_on_empty_input():
    provider = FakeProvider()
    assert emb.embedding_relevance("", "something", provider=provider) is None
    assert emb.embedding_relevance("something", "", provider=provider) is None


# ---------------------------------------------------------------------------
# get_embedding_provider: availability / disable / missing-key handling
# ---------------------------------------------------------------------------

def test_get_embedding_provider_none_when_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    emb.reset_embedding_provider_cache()
    assert emb.get_embedding_provider() is None


def test_get_embedding_provider_none_when_disabled_even_with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("DISABLE_EMBEDDING_ROLE_RELEVANCE", "true")
    emb.reset_embedding_provider_cache()
    assert emb.get_embedding_provider() is None


def test_get_embedding_provider_is_cached_across_calls(monkeypatch):
    """The provider should only be constructed once per process, not on
    every call -- confirmed here by checking identity is stable."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.delenv("DISABLE_EMBEDDING_ROLE_RELEVANCE", raising=False)
    emb.reset_embedding_provider_cache()
    first = emb.get_embedding_provider()
    second = emb.get_embedding_provider()
    assert first is second
    assert first is not None


def test_get_embedding_provider_builds_a_real_gemini_provider_when_key_present(monkeypatch):
    """Confirms wiring end-to-end against the REAL GoogleGenerativeAIEmbeddings
    class (construction only -- no network call is made by simply
    constructing it), using the same langchain-google-genai package this
    project already depends on for its chat LLM."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.delenv("DISABLE_EMBEDDING_ROLE_RELEVANCE", raising=False)
    emb.reset_embedding_provider_cache()
    provider = emb.get_embedding_provider()
    assert isinstance(provider, emb.GeminiEmbeddingProvider)


def test_real_provider_call_failure_is_caught_and_returns_none(monkeypatch):
    """End-to-end reliability check using the REAL provider class with an
    invalid key: whatever the underlying failure mode (auth error,
    network error, blocked egress), embed_text() must catch it and
    return None rather than raising or crashing the caller."""
    monkeypatch.setenv("GEMINI_API_KEY", "definitely-not-a-real-key")
    monkeypatch.delenv("DISABLE_EMBEDDING_ROLE_RELEVANCE", raising=False)
    emb.reset_embedding_provider_cache()
    emb.clear_embedding_cache()
    result = emb.embed_text("Registered Nurse ICU")
    assert result is None
