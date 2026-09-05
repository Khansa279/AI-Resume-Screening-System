"""Generalized, domain-agnostic skill/requirement equivalence matching.

WHY THIS EXISTS (production investigation, Marketing-domain role-relevance
under-scoring): `job_relevance()`'s `skill_hit` component (40% of
job_relevance, itself feeding `role_relevance` in
src/agents/experience_eval.py) and `SkillsMatcherAgent`'s deterministic
pass both lean on `src/skill_taxonomy.py` for anything beyond an exact
string match. That taxonomy is -- by design and by its own module
docstring -- a technology/ML/software catalogue. It has zero entries for
Marketing, Sales, HR, Finance, or any other non-tech domain. For a
requirement like "SEO and keyword research" against a candidate who
wrote "Performed keyword research and on-page SEO optimization", the
taxonomy recognizes nothing in either string, so scoring falls back to a
literal, word-order-sensitive substring check that under-credits real,
clearly-demonstrated skills purely because of phrasing.

This module provides ONE reusable, generic primitive --
`match_requirement()` -- that decides whether a piece of candidate text
(a work-experience's title/technologies/responsibilities, a resume
summary, etc.) satisfies a JD requirement string, layered
cheapest-and-most-reliable first:

  1. DETERMINISTIC (always on, never disabled): reuses
     src/skill_taxonomy.py's existing exact/alias/normalization logic
     UNCHANGED -- this is exactly what already ran before this module
     existed, so behavior for anything already covered by the taxonomy
     (i.e. every technical case) does not change AT ALL.

  2. EMBEDDING SEMANTIC SIMILARITY (src/embeddings.py), only tried when
     tier 1 found nothing. Generic across any domain -- no fixed list of
     skill names to maintain. Uses TWO thresholds, not one, specifically
     so "vaguely related" is never conflated with "equivalent":
       - >= SKILL_EQUIVALENCE_FLOOR  -> true equivalent (full credit)
       - [SKILL_RELATED_FLOOR, SKILL_EQUIVALENCE_FLOOR) -> closely
         related but not the same skill (SKILL_RELATED_CREDIT partial
         credit only, and the one case tier 3 below is allowed to look
         at)
       - < SKILL_RELATED_FLOOR -> not evidence of this skill (covers
         both "merely co-occurring concepts" -- e.g. Python vs Java,
         both "a programming language" -- and genuinely unrelated pairs)
     These floors are deliberately STRICTER than the role-relevance
     embedding thresholds already used elsewhere in this project
     (src/embeddings.py's EMBEDDING_SIMILARITY_FLOOR/CEIL, used for
     "how related are these two ROLES"): calling two SKILLS "equivalent"
     is a much stronger claim than "these two roles are somewhat
     related", so it needs a tighter bar. Like every threshold constant
     in this project's embedding integration, these are a documented,
     reasonable STARTING POINT -- this sandbox has no live network
     access to a real embedding provider to calibrate against, so if you
     deploy this with a real API key, spot-check a handful of known
     equivalent/related/unrelated skill pairs from your own domain and
     adjust the two constants below if scores look off. Nothing else in
     this module needs to change.

  3. OPTIONAL, OPT-IN LLM VALIDATION, only invoked for a pair that
     landed in tier 2's ambiguous "related, not confidently equivalent"
     band -- never for a confident tier-2 negative, never in place of
     tiers 1/2. Disabled unless ENABLE_LLM_SKILL_VALIDATION is set.
     Uses the SAME provider-agnostic `create_llm()` this project already
     uses for its chat LLM (Gemini OR Groq, whichever is configured) --
     no new provider-specific dependency, and the feature is a complete
     no-op (falls straight back to tier 2's result) when no provider is
     configured, the flag is off, or the call fails for any reason.

RELIABILITY CONTRACT (matches src/embeddings.py's existing contract):
every public function here either returns a real decision or the safe
"no match" default -- never raises, and the whole module is a complete
no-op with IDENTICAL behavior to "just the deterministic tier" when no
embedding provider and no LLM are configured. This is what keeps
deterministic matching working when embeddings/LLM are unavailable
(sandboxes, CI, Groq-only deployments, etc).

EXPLAINABILITY: every decision carries `quality` (which tier decided
it), `credit` (0.0-1.0 contribution), `evidence` (what on the candidate
side justified it), and `notes` (human-readable reasoning) -- so a
result can always be traced back to why it was made, the same
expectation this project's existing SkillMatch/matched_skill/notes
fields already set.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from .embeddings import embedding_relevance
from .skill_taxonomy import normalize_requirement_skills, normalize_skill_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable thresholds -- see module docstring, tier 2, for the full
# rationale. Intentionally stricter than the role-relevance embedding
# thresholds in src/embeddings.py.
# ---------------------------------------------------------------------------
SKILL_EQUIVALENCE_FLOOR = 0.83
SKILL_RELATED_FLOOR = 0.55
SKILL_RELATED_CREDIT = 0.6

ENABLE_LLM_SKILL_VALIDATION_ENV = "ENABLE_LLM_SKILL_VALIDATION"

_LLM_VERDICT_RE = re.compile(r"\b(YES|NO)\b", re.I)


@dataclass(frozen=True)
class SkillMatchDecision:
    """Result of matching one requirement against one piece of candidate
    text. `credit` is what callers should actually add into their score
    (0.0-1.0); `matched` is a convenience boolean (`credit > 0`)."""

    matched: bool
    quality: str  # "taxonomy_or_literal" | "embedding_equivalent" | "embedding_related" | "llm_validated" | "none"
    credit: float
    evidence: str
    score: float | None  # raw embedding-rescaled similarity, when available; None otherwise
    notes: str

    @staticmethod
    def none(notes: str = "No matching evidence found") -> "SkillMatchDecision":
        return SkillMatchDecision(False, "none", 0.0, "", None, notes)


# ---------------------------------------------------------------------------
# Tier 1: deterministic, reusing skill_taxonomy.py unchanged
# ---------------------------------------------------------------------------

def _deterministic_requirement_evidence(
    requirement: str, candidate_text: str
) -> SkillMatchDecision | None:
    """Exact/alias/literal-phrase evidence for `requirement` inside
    `candidate_text`, using the EXACT SAME normalization
    (`normalize_requirement_skills`, which prefers taxonomy-recognized
    sub-terms and falls back to the whole normalized phrase) already
    used throughout this project. Returns None -- not a negative -- when
    there's nothing to find here, so callers know to try the next tier
    rather than treating an empty taxonomy lookup as a confident "no"."""
    if not requirement or not candidate_text:
        return None
    candidate_norm = normalize_skill_text(candidate_text)
    if not candidate_norm:
        return None
    for key in normalize_requirement_skills(requirement):
        if key and re.search(rf"(?<!\S){re.escape(key)}(?!\S)", candidate_norm):
            return SkillMatchDecision(
                True, "taxonomy_or_literal", 1.0, key,
                1.0, f"Deterministic taxonomy/literal match on '{key}'",
            )
    return None


# ---------------------------------------------------------------------------
# Tier 2: embedding semantic similarity, dual-thresholded
# ---------------------------------------------------------------------------

def _embedding_requirement_evidence(
    requirement: str, candidate_text: str
) -> SkillMatchDecision | None:
    """Returns None (no signal) whenever no embedding provider is
    available/usable -- callers MUST treat that as "try the next tier or
    give up", never as a negative. Returns a real decision (matched or
    not) whenever a score WAS computed."""
    score = embedding_relevance(requirement, candidate_text)
    if score is None:
        return None
    if score >= SKILL_EQUIVALENCE_FLOOR:
        return SkillMatchDecision(
            True, "embedding_equivalent", 1.0, candidate_text[:160], score,
            f"Embedding similarity {score:.2f} >= equivalence floor {SKILL_EQUIVALENCE_FLOOR} "
            f"-- treated as a true equivalent skill",
        )
    if score >= SKILL_RELATED_FLOOR:
        return SkillMatchDecision(
            True, "embedding_related", SKILL_RELATED_CREDIT, candidate_text[:160], score,
            f"Embedding similarity {score:.2f} in the related-but-not-equivalent band "
            f"[{SKILL_RELATED_FLOOR}, {SKILL_EQUIVALENCE_FLOOR}) -- partial credit only",
        )
    return SkillMatchDecision(
        False, "none", 0.0, "", score,
        f"Embedding similarity {score:.2f} below the related floor {SKILL_RELATED_FLOOR} "
        f"-- not treated as evidence of this skill",
    )


# ---------------------------------------------------------------------------
# Tier 3: optional, opt-in LLM validation for the embedding-ambiguous band
# ---------------------------------------------------------------------------

def _llm_validation_enabled() -> bool:
    return os.getenv(ENABLE_LLM_SKILL_VALIDATION_ENV, "").strip().lower() in ("1", "true", "yes")


def llm_validate_requirement(requirement: str, candidate_text: str) -> SkillMatchDecision | None:
    """OPT-IN LLM validation for a requirement/candidate-text pair whose
    embedding score already landed in the ambiguous "related" band.
    Never called for a confidently-unrelated pair, never a substitute
    for tiers 1/2. Provider-agnostic (reuses BaseAgent.create_llm, so
    Gemini OR Groq -- whichever this deployment already has configured
    -- works identically; no new provider-specific dependency).

    Returns None -- never raises -- whenever the feature is disabled, no
    LLM provider is configured, or the call fails/returns something
    unusable for any reason. Callers MUST treat None as "no verdict",
    keeping whatever tier-2 result they already had.
    """
    if not _llm_validation_enabled():
        return None
    try:
        from .config import get_config
        from .agents.base import create_llm

        get_config()  # raises ValueError if no provider/API key configured
        llm = create_llm(temperature=0.0)
        prompt = (
            "You are validating whether a candidate's demonstrated work "
            "genuinely satisfies a specific job requirement. Two concepts "
            "may be TRUE EQUIVALENTS (e.g. 'GA4' and 'Google Analytics 4', "
            "or 'paid search advertising' and 'Google Ads'), CLOSELY "
            "RELATED but distinct (e.g. 'SEO' and 'content marketing'), "
            "MERELY CO-OCCURRING in the same field without being evidence "
            "of each other (e.g. 'Python' and 'Java' -- both programming "
            "languages, but knowing one is not evidence of the other), or "
            "UNRELATED (e.g. 'SEO' and 'graphic design').\n\n"
            f"Job requirement: {requirement!r}\n"
            f"Candidate's demonstrated skills/work: {candidate_text[:600]!r}\n\n"
            "Does the candidate's demonstrated work provide genuine, "
            "specific evidence that satisfies this exact requirement -- "
            "not merely a loosely related or co-occurring concept? "
            "Respond with ONLY the single word YES or NO."
        )
        response = llm.invoke(prompt)
        text = getattr(response, "content", str(response)).strip()
        match = _LLM_VERDICT_RE.search(text)
        if not match:
            return None
        if match.group(1).upper() == "YES":
            return SkillMatchDecision(
                True, "llm_validated", 1.0, candidate_text[:160], None,
                "LLM validated genuine equivalence for an embedding-ambiguous pair",
            )
        return SkillMatchDecision(
            False, "none", 0.0, "", None,
            "LLM declined to validate an embedding-ambiguous pair as a genuine match",
        )
    except Exception:
        logger.debug("LLM skill validation failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def match_requirement(
    requirement: str,
    candidate_text: str,
    *,
    use_embedding: bool = True,
    use_llm: bool | None = None,
) -> SkillMatchDecision:
    """Decide whether `candidate_text` satisfies `requirement`, trying
    each tier in order and stopping at the first one that finds
    something (a later tier is only consulted when the previous one
    found NOTHING -- not when it found a confident negative). See the
    module docstring for the full rationale of each tier.

    Args:
        requirement: a JD requirement string, any domain.
        candidate_text: free text describing what the candidate has
            actually done (e.g. a work experience's title + technologies
            + responsibilities, joined).
        use_embedding: set False to force deterministic-only behavior
            (e.g. for a caller that wants to control cost/latency
            explicitly rather than relying on "no provider configured").
        use_llm: override for whether tier 3 may run; defaults to
            `ENABLE_LLM_SKILL_VALIDATION` when None.

    Returns:
        A SkillMatchDecision. Never raises. Worst case (nothing in the
        taxonomy, no embedding provider, LLM disabled/unavailable)
        returns SkillMatchDecision.none() -- identical to how an
        unrecognized requirement already behaved before this module
        existed.
    """
    tier1 = _deterministic_requirement_evidence(requirement, candidate_text)
    if tier1 is not None:
        return tier1

    if not use_embedding:
        return SkillMatchDecision.none(
            "No deterministic evidence found; embedding tier disabled for this call"
        )

    tier2 = _embedding_requirement_evidence(requirement, candidate_text)
    if tier2 is None:
        return SkillMatchDecision.none(
            "No deterministic evidence found and no embedding provider is available"
        )

    should_try_llm = use_llm if use_llm is not None else _llm_validation_enabled()
    if tier2.quality == "embedding_related" and should_try_llm:
        tier3 = llm_validate_requirement(requirement, candidate_text)
        if tier3 is not None and tier3.matched:
            return tier3
        # LLM unavailable/disabled/declined -- tier 2's partial-credit
        # result still stands; never discard real (if partial) evidence
        # just because the optional upgrade step didn't pan out.

    return tier2


def match_requirement_against_snippets(
    requirement: str,
    candidate_snippets: list[str],
    **kwargs,
) -> SkillMatchDecision:
    """Like `match_requirement`, but checks `requirement` against each of
    several individual candidate evidence snippets (e.g. a job title, an
    individual responsibility bullet, or a single extracted skill name)
    and returns the SINGLE BEST decision across them.

    This matters specifically for the embedding tier: cosine similarity
    between a short, single-topic requirement phrase and an ENTIRE
    concatenated resume/work-history blob is systematically diluted --
    the blob's embedding averages over every topic it touches, so even a
    perfect one-sentence match gets drowned out by everything else in
    the blob and can score BELOW the related floor even though a real
    equivalence exists. Comparing against each snippet individually (the
    same granularity a human reviewer would actually check evidence at)
    avoids that dilution and is what preserves explainability too -- the
    returned decision's `evidence` field names the ONE specific snippet
    that justified it, not "somewhere in this whole resume".

    Deterministic matching (tier 1) is still tried against the full
    joined text first inside each per-snippet call, so nothing about
    the taxonomy/alias/literal-phrase behavior changes -- this only
    affects how the embedding tier is scoped.
    """
    best: SkillMatchDecision = SkillMatchDecision.none("No candidate evidence snippets provided")
    for snippet in candidate_snippets:
        if not snippet or not snippet.strip():
            continue
        decision = match_requirement(requirement, snippet, **kwargs)
        if decision.credit > best.credit:
            best = decision
        if best.credit >= 1.0:
            break
    return best
