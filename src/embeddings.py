"""Optional embedding-based semantic similarity utilities.

This module is a small, swappable abstraction around a text-embedding
provider. It exists to give job_relevance() (src/agents/experience_eval.py)
an additional, TRUE semantic-similarity signal -- cosine similarity
between embedding vectors -- on top of the existing deterministic
skill-taxonomy/lexical signature (_semantic_role_relevance) and the
existing optional LLM hint (_llm_role_relevance_hint).

PROVIDER: Google Gemini, via `langchain_google_genai.GoogleGenerativeAIEmbeddings`
-- part of the `langchain-google-genai` package this project ALREADY
depends on (requirements.txt) for its chat LLM (see
src/agents/base.py::create_llm). No new dependency is introduced.
`langchain-groq` (this project's other supported chat provider) has no
embeddings endpoint, so a Groq-only configuration simply runs without an
embedding signal -- see get_embedding_provider()'s docstring.

RELIABILITY: every public function in this module either returns a real
result or `None` -- never raises, and never requires an embedding
provider to be configured. `None` always means "no signal available";
callers are responsible for falling back to their own existing
deterministic behavior, exactly the same contract this project's
existing optional LLM hint already follows.

TESTABILITY: the actual embedding call is isolated behind
`get_embedding_provider()` / `EmbeddingProvider.embed()`, so unit tests
can monkeypatch it with a small, hand-written fake that returns fixed
vectors -- no test in this project's suite makes a live network call.

CACHING: a per-process, in-memory cache keyed by the exact text embedded,
so the SAME job description -- screened against many candidates in one
batch/process -- is embedded once, not once per candidate.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from typing import Protocol

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Minimal interface any embedding backend must satisfy."""

    def embed(self, text: str) -> list[float]:
        """Return a dense vector for `text`. May raise on failure --
        callers in THIS module are responsible for catching exceptions,
        so implementations should not swallow errors themselves."""
        ...


class GeminiEmbeddingProvider:
    """Wraps `langchain_google_genai.GoogleGenerativeAIEmbeddings`.

    Model name is configurable via the `GEMINI_EMBEDDING_MODEL` env var
    so it can be updated without a code change as Google's model lineup
    evolves; defaults to "models/embedding-001", a stable, widely
    available Gemini text-embedding model.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._client = GoogleGenerativeAIEmbeddings(
            model=model or os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001"),
            google_api_key=api_key,
        )

    def embed(self, text: str) -> list[float]:
        return self._client.embed_query(text)


_provider_cache: EmbeddingProvider | None = None
_provider_cache_attempted = False


def _embeddings_enabled() -> bool:
    """Opt-OUT switch: embeddings are attempted by default (per the
    project requirement that "if embeddings are available, use
    embeddings first"), but can be explicitly disabled -- e.g. to save
    cost/latency, or to force the deterministic-only / LLM-hint path for
    testing or debugging. Leaving this enabled by default is safe: an
    embedding call is only ever ATTEMPTED when a usable provider can
    actually be constructed (see get_embedding_provider()), which itself
    requires a configured API key -- so an environment with no key (like
    this project's default test/dev sandbox) never attempts a real call
    regardless of this flag."""
    return os.getenv("DISABLE_EMBEDDING_ROLE_RELEVANCE", "").strip().lower() not in ("1", "true", "yes")


def get_embedding_provider() -> EmbeddingProvider | None:
    """Return a usable embedding provider, or None if one can't be built
    right now (disabled, package missing, no configured API key, or
    construction otherwise failed). Cached for the lifetime of the
    process so repeated calls (job_relevance() calls this once per
    work-experience entry, per candidate) don't re-attempt construction.

    Only ever returns a Gemini-backed provider today -- Groq (this
    project's other supported chat provider) has no embeddings endpoint
    in its LangChain integration (langchain-groq) as of this writing.
    Screening still works completely normally with a Groq-only
    configuration; it just runs entirely on the deterministic signature
    (and, if enabled, the LLM hint) for role relevance, same as before
    this module existed.
    """
    global _provider_cache, _provider_cache_attempted
    if _provider_cache is not None:
        return _provider_cache
    if _provider_cache_attempted:
        return None
    _provider_cache_attempted = True

    if not _embeddings_enabled():
        return None

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        _provider_cache = GeminiEmbeddingProvider(api_key=api_key)
        return _provider_cache
    except Exception:
        logger.debug("Embedding provider unavailable", exc_info=True)
        return None


def reset_embedding_provider_cache() -> None:
    """Test/ops hook: clears the cached provider (and the "already
    tried, gave up" flag) so a later call re-attempts construction --
    e.g. after monkeypatching env vars in a test, or after a genuinely
    transient failure in a long-running process."""
    global _provider_cache, _provider_cache_attempted
    _provider_cache = None
    _provider_cache_attempted = False


_embedding_cache: dict[str, list[float]] = {}


def clear_embedding_cache() -> None:
    """Test hook: clears the text->vector cache."""
    _embedding_cache.clear()


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_text(text: str, provider: EmbeddingProvider | None = None) -> list[float] | None:
    """Embed `text`, or return None on ANY failure (no provider
    available, package/import error, API error, timeout, malformed
    response) -- callers must treat None as "no embedding signal
    available", never as "zero relevance". Results are cached in-process
    by exact text content, so the SAME text -- e.g. one job description's
    title+skills+responsibilities, embedded once and reused across every
    candidate screened against it in this process -- is never sent to
    the provider twice.
    """
    if not text or not text.strip():
        return None

    key = _cache_key(text)
    if key in _embedding_cache:
        return _embedding_cache[key]

    resolved_provider = provider or get_embedding_provider()
    if resolved_provider is None:
        return None

    try:
        vector = resolved_provider.embed(text)
    except Exception:
        logger.debug("Embedding call failed", exc_info=True)
        return None

    if not vector or not isinstance(vector, (list, tuple)):
        return None

    vector = list(vector)
    _embedding_cache[key] = vector
    return vector


def cosine_similarity(a: list[float], b: list[float]) -> float | None:
    """Standard cosine similarity, or None if either vector is empty/
    zero-length, has zero magnitude (can't be meaningfully normalized),
    or the two vectors have mismatched dimensionality (shouldn't happen
    for embeddings from the same model/provider, but guarded
    defensively rather than raising)."""
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Cosine similarity -> [0, 1] relevance rescaling.
#
# Raw cosine similarity between two text embeddings is NOT already a
# 0.0-1.0 "relevance" score in the sense the rest of this pipeline
# expects: most sentence/text embedding spaces cluster even UNRELATED
# short phrases at a fairly high baseline similarity (a well-documented
# property of embedding geometry -- "anisotropy"), so a naive
# `max(0, cos_sim)` would make almost everything look at least
# moderately relevant. This function makes the floor/ceiling explicit,
# named constants, and independently testable, rather than burying a
# magic number inside job_relevance()'s call site.
#
# IMPORTANT: EMBEDDING_SIMILARITY_FLOOR/CEIL below are a reasonable,
# documented STARTING POINT, not a value calibrated against Gemini's
# actual embedding space -- this sandbox has no live network access to
# the embedding API to calibrate against real score distributions. If
# you deploy this with a real API key, spot-check a handful of known
# related/unrelated title pairs from your own domain and adjust these
# two constants if the resulting relevance scores look off; nothing else
# in this module needs to change.
# ---------------------------------------------------------------------------
EMBEDDING_SIMILARITY_FLOOR = 0.35
EMBEDDING_SIMILARITY_CEIL = 0.90


def similarity_to_relevance(cos_sim: float) -> float:
    """Rescale a raw cosine similarity into this project's existing
    0.0-1.0 relevance range, clamping outside [FLOOR, CEIL]."""
    if cos_sim <= EMBEDDING_SIMILARITY_FLOOR:
        return 0.0
    if cos_sim >= EMBEDDING_SIMILARITY_CEIL:
        return 1.0
    return round(
        (cos_sim - EMBEDDING_SIMILARITY_FLOOR) / (EMBEDDING_SIMILARITY_CEIL - EMBEDDING_SIMILARITY_FLOOR),
        4,
    )


def embedding_relevance(
    text_a: str, text_b: str, provider: EmbeddingProvider | None = None
) -> float | None:
    """High-level convenience used by experience_eval.py: embed both
    texts and return the rescaled 0.0-1.0 relevance score, or None if
    embeddings aren't available/usable for any reason. Never raises.
    """
    vec_a = embed_text(text_a, provider=provider)
    if vec_a is None:
        return None
    vec_b = embed_text(text_b, provider=provider)
    if vec_b is None:
        return None
    sim = cosine_similarity(vec_a, vec_b)
    if sim is None:
        return None
    return similarity_to_relevance(sim)
