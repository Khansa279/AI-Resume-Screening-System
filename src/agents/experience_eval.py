"""Experience Evaluator Agent - Assesses work experience relevance.

Deterministic: years from dates/durations, relevance from title/technology
overlap with the job description. Same parsed resume + same requirements
always produce the same ExperienceEvaluation. No LLM/embedding call on the
normal path by default:
  - A small, OPT-IN, fallback-safe LLM hint exists for ambiguous title/
    role matches -- see "generalized semantic role relevance" below --
    disabled unless ENABLE_LLM_ROLE_RELEVANCE is set.
  - A real, embedding-based semantic-similarity signal (src/embeddings.py)
    is used automatically WHEN a provider is configured (a Gemini API
    key present), and is a complete no-op otherwise -- see
    "_embedding_role_relevance" below.
Neither can ever lower a score, run the show by itself, or make the
pipeline fail when unavailable.
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any

from .base import BaseAgent
from ..config import get_config
from ..embeddings import embedding_relevance
from ..models import ResumeData, JobRequirements, ExperienceEvaluation, WorkExperience
from ..skill_matching import match_requirement_against_snippets as match_skill_requirement_against_snippets
from ..skill_taxonomy import (
    canonical_skill_key,
    normalize_skill_text,
    extract_canonical_skills,
    normalize_requirement_skills,
)


_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_PRESENT = re.compile(r"^(present|current|now|ongoing|date)$", re.I)

_STOP = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "at", "to", "for", "on",
    "with", "as", "by", "from", "engineer", "engineering", "developer",
    "software", "associate", "junior", "senior", "lead", "intern",
})

_TITLE_STOP = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "at", "to", "for", "on",
    "with", "as", "by", "from",
})


def _title_tokens(text: str) -> set[str]:
    return {
        t for t in normalize_skill_text(text).split()
        if t and t not in _TITLE_STOP and len(t) > 1
    }

_SENIORITY = [
    ("intern", 0), ("trainee", 0), ("junior", 1), ("associate", 1),
    ("engineer", 2), ("developer", 2), ("analyst", 2),
    ("senior", 3), ("lead", 4), ("principal", 5), ("staff", 5),
    ("manager", 5), ("director", 6),
]


# ============================================================================
# Generalized (non-hardcoded) semantic role relevance
#
# WHY THIS REPLACED THE OLD APPROACH: this module used to carry a small,
# hand-curated map from title keywords ("backend", "frontend", "ai_ml", ...)
# to a coarse "role family", plus a hand-curated relatedness matrix between
# families. That is a finite, hardcoded catalogue of job-role concepts --
# every unseen role either fell through to a vague "generic_swe" bucket or
# scored 0.0 relatedness purely because nobody had added it to the map yet
# (the "ai_ml" family above is itself evidence of this: someone had to keep
# coming back to hand-add new roles one at a time). That's not
# production-ready: it requires a person to edit this file every time a new
# kind of role shows up, and it's fundamentally biased toward common
# software-engineering titles.
#
# GENERALIZED APPROACH: instead of classifying a title into a fixed set of
# named families, this measures how much a candidate's ACTUAL demonstrated
# skills and responsibilities overlap with what the JD actually asks for --
# using the same shared skill taxonomy (src/skill_taxonomy.py) the rest of
# this pipeline already relies on for skill matching. That taxonomy is a
# general-purpose catalogue of technologies/tools/concepts (Python, PyTorch,
# Kubernetes, NLP, ...), not a list of job titles, so it needs no new entry
# to handle a brand-new title. A "Machine Learning Engineer" JD and a "Data
# Scientist" candidate who used PyTorch/NLP overlap heavily on real skill
# signal even though neither title appears anywhere in this file and the
# titles themselves share no words -- see _semantic_role_relevance below.
#
# WHY NOT A FULL EMBEDDING MODEL: this environment can't reliably download
# a hosted embedding model at request time, and job_relevance() runs once
# per work-experience entry per candidate -- a hot scoring path where this
# project's own test suite requires deterministic, reproducible output (see
# every `e1.model_dump() == e2.model_dump()` assertion in this file's
# tests). A signature built from the SAME taxonomy already used for skill
# matching is deterministic, needs no extra dependency or network access,
# and grows automatically as skill_taxonomy.py grows -- no role-specific
# edits required.
#
# OPTIONAL LLM ASSIST: for genuinely ambiguous cases (the deterministic
# signature overlap lands in a middle band -- neither confidently related
# nor confidently unrelated), an OPT-IN LLM hint can nudge title relevance
# further. It is disabled unless ENABLE_LLM_ROLE_RELEVANCE is set, it can
# only ever RAISE the score (never lower it, same as the old family
# backstop it replaces), it never determines the final candidate score by
# itself (it only feeds into title_score, which is one of three weighted
# components inside job_relevance(), itself only one input to the overall
# match score), and any failure/unavailability (no API key, network error,
# bad response) is caught and silently ignored -- the deterministic
# signature score is always what's used when the LLM isn't available or
# doesn't return something usable.
# ============================================================================


def _role_signature(title: str, technologies: list[str] | None, responsibilities: list[str] | None) -> set[str]:
    """A generic, non-hardcoded fingerprint of "what this role actually
    involves": canonical taxonomy skills mentioned anywhere in the title,
    technologies, or responsibility text, plus plain title tokens. Built
    entirely from src/skill_taxonomy.py (a general skills/tools catalogue)
    and simple tokenization -- no fixed list of role/job-title names is
    involved, so this works identically for a title this project has
    never seen before."""
    text_blob = " ".join([title or "", " ".join(technologies or []), " ".join(responsibilities or [])])
    skill_keys = {canonical_skill_key(name) for name, _category in extract_canonical_skills(text_blob)}
    return skill_keys | _title_tokens(title)


def _semantic_role_relevance(exp: WorkExperience, requirements: JobRequirements) -> float:
    """Generalized (non-hardcoded) replacement for the old role-family
    taxonomy backstop. Scores how much a candidate's demonstrated
    skills/responsibilities overlap with what the JD's title, required/
    preferred skills, and responsibilities actually call for -- so
    relevance tracks real overlapping work, not literal title matching.
    Always 0.0-1.0; 0.0 when there's no recognizable overlap. Unlike the
    old family lookup, there is no "unrecognized title" special case to
    fall back from, since nothing here depends on a fixed title
    vocabulary -- an entirely unseen title degrades gracefully to
    whatever real skill/responsibility signal is actually present."""
    candidate_sig = _role_signature(exp.title, exp.technologies, exp.responsibilities)
    jd_sig = _role_signature(
        requirements.title,
        (requirements.required_skills or []) + (requirements.preferred_skills or []),
        requirements.responsibilities,
    )
    return _overlap(candidate_sig, jd_sig)


# ----------------------------------------------------------------------------
# CONFIRMED BUG (production regression investigation, embedding-integration
# follow-up): _semantic_role_relevance() above measures SYMMETRIC (Jaccard)
# overlap -- |A∩B| / |A∪B|. That is the right tool when comparing two
# comparably-sized sets, but it systematically UNDER-SCORES a candidate
# whose own per-job signature is simply larger than the JD's, which is the
# ORDINARY case for a detailed/senior resume entry: real-world job bullets
# routinely mention process/collaboration terms the shared skill taxonomy
# also recognizes (Agile, Code Review, Leadership, Teamwork, Communication,
# ...) alongside the core technical work. Every one of those EXTRA terms
# grows the union denominator and shrinks the ratio, even though none of
# them contradict the candidate's core alignment with the JD -- a detailed
# "Senior Machine Learning Engineer" entry (PyTorch, TensorFlow, deep
# learning, NLP -- plus the usual mentoring/agile/stakeholder bullets any
# senior role has) can end up scoring a LOWER Jaccard overlap against an
# "AI/ML Engineer" JD than a terser, less-detailed entry would, purely
# because the candidate said more about their day-to-day work. Verified:
# a JD requiring Python/ML/DL/PyTorch/TensorFlow/NLP/CV/Docker/AWS/MLOps
# against a candidate who demonstrably covers 7 of the JD's own 14
# signature terms (all the technical ones) still only Jaccard-scores 0.32,
# because the candidate's own signature also picked up 8 process/soft-skill
# terms the JD text never mentions.
#
# FIX: _semantic_role_coverage() below is a DIRECTIONAL (recall) measure,
# anchored to the JD's OWN signature size rather than the candidate's:
# |A∩B| / |jd_sig|. This answers "what fraction of what the JD is actually
# asking for does the candidate's signature cover" -- extra, unrelated
# terms on the CANDIDATE's side (verbose bullets, a broader skillset) no
# longer penalize the score, since they never appear in the denominator.
#
# WHY THIS IS SAFE (does not just inflate everything): the numerator is the
# exact same intersection _semantic_role_relevance() already uses -- a
# genuinely unrelated pair (zero real overlap) still scores 0.0 here,
# identically. And unlike a "coverage relative to the SMALLER of the two
# sets" (Szymkiewicz-Simpson) formula -- which is NOT used here and would
# be unsafe, since a candidate with a nearly-empty signature could then
# trivially "cover" 100% of it with a single shared word -- anchoring
# specifically to the JD's side means the denominator is always the JD's
# own (typically multi-term, for any real job posting) signature size, not
# whichever set happens to be smaller.
#
# This is combined into title_score via max() alongside the existing
# Jaccard-based semantic_score (see job_relevance() below) -- the same
# "only ever raises, never lowers" pattern this module already uses for
# the LLM hint and the embedding signal, so it cannot regress any existing
# case that already scored well; it can only rescue cases the symmetric
# measure was undervaluing. It uses the same general-purpose
# skill_taxonomy.py signature as _semantic_role_relevance() -- no new
# fixed role/title vocabulary, so it generalizes to any unseen role
# exactly like the function it sits alongside.
# ----------------------------------------------------------------------------

def _skill_signature(title: str, technologies: list[str] | None, responsibilities: list[str] | None) -> set[str]:
    """Like _role_signature(), but WITHOUT the plain title-token union --
    only genuine skill_taxonomy.py-recognized canonical skill keys.

    WHY THIS SEPARATE, NARROWER SIGNATURE EXISTS: _semantic_role_coverage()
    below must only credit a candidate for REAL, taxonomy-recognized
    skill/technology overlap -- not for the candidate's title merely
    containing all of the JD title's own words (e.g. "Junior Backend
    Developer" trivially contains every word in "Backend Developer").
    That kind of pure title-wording superset relationship is already
    correctly handled by the existing lexical title_score (_overlap on
    job_title/jd_title) and by _semantic_role_relevance()'s own Jaccard
    backstop -- a directional coverage measure anchored to a JD
    signature made of NOTHING BUT plain title tokens would trivially
    hit 1.0 for any title superset, which is not evidence of skill
    coverage at all. Restricting coverage to genuine taxonomy skill
    keys keeps the new signal scoped to what it was built to fix:
    technology/skill overlap the symmetric Jaccard measure was diluting."""
    text_blob = " ".join([title or "", " ".join(technologies or []), " ".join(responsibilities or [])])
    return {canonical_skill_key(name) for name, _category in extract_canonical_skills(text_blob)}


def _semantic_role_coverage(exp: WorkExperience, requirements: JobRequirements) -> float:
    """Directional counterpart to _semantic_role_relevance(): the fraction
    of the JD's OWN skill/technology signature (genuine taxonomy-
    recognized terms only -- see _skill_signature(), NOT plain title
    words) that the candidate's signature covers, |A∩B| / |jd_skill_sig|,
    rather than the symmetric Jaccard ratio. See the module-level
    comment above _skill_signature for why this is restricted to real
    skill-taxonomy overlap rather than _role_signature()'s full set
    (which also includes plain title tokens). Returns 0.0 when the JD
    has no recognizable skill/technology signature at all (nothing to
    cover) -- never divides by zero, never raises, and never credits a
    candidate purely for their title containing the JD title's words."""
    candidate_sig = _skill_signature(exp.title, exp.technologies, exp.responsibilities)
    jd_sig = _skill_signature(
        requirements.title,
        (requirements.required_skills or []) + (requirements.preferred_skills or []),
        requirements.responsibilities,
    )
    if not jd_sig:
        return 0.0
    return round(len(candidate_sig & jd_sig) / len(jd_sig), 4)


# ============================================================================
# Real embedding-based semantic relevance (src/embeddings.py)
#
# This is an ADDITIONAL signal on top of _semantic_role_relevance() above,
# not a replacement for it. _semantic_role_relevance() is deterministic,
# free, and always available, but it can only recognize overlap through
# literal title words or entries already in skill_taxonomy.py -- a
# "Registered Nurse" vs "Critical Care Nurse" or "Machine Learning
# Engineer" vs "AI Research Scientist" pair may share little of either
# and still be genuinely, highly related work. A text-embedding model
# captures that kind of relatedness directly, without needing every
# possible synonym pair hand-added to the taxonomy.
#
# This is entirely optional: see src/embeddings.py's module docstring for
# the full reliability contract. In short, `embedding_relevance()` never
# raises and returns None whenever no provider is configured/available or
# a call fails for any reason -- the caller (job_relevance(), below) then
# simply doesn't apply it, identical to how the LLM hint's None is
# already handled.
# ============================================================================


def _embedding_context_text(
    title: str, extra_skills: list[str] | None, responsibilities: list[str] | None,
) -> str:
    """Plain-text blob to embed for role/work relevance -- title +
    skills/technologies + responsibilities, the same ingredients
    _role_signature() above uses, just as free text for a text-embedding
    model rather than a taxonomy-driven signature."""
    parts = [title or "", *(extra_skills or []), *(responsibilities or [])]
    return ". ".join(p for p in parts if p).strip()


def _embedding_role_relevance(exp: WorkExperience, requirements: JobRequirements) -> float | None:
    """Real, embedding-based semantic similarity for role/title
    relevance. Returns None whenever an embedding provider isn't
    available or usable for any reason -- callers MUST treat None as "no
    signal", never as zero relevance.

    The JD side of this text is identical for every candidate screened
    against the same JobRequirements, so embed_text()'s cache (keyed by
    exact text content) means the JD is only ever sent to the provider
    once per process, no matter how many candidates are screened against
    it -- see src/embeddings.py.
    """
    jd_text = _embedding_context_text(
        requirements.title,
        (requirements.required_skills or []) + (requirements.preferred_skills or []),
        requirements.responsibilities,
    )
    candidate_text = _embedding_context_text(exp.title, exp.technologies, exp.responsibilities)
    return embedding_relevance(jd_text, candidate_text)


# Band within which the deterministic signature score is genuinely
# ambiguous -- not confidently related, not confidently unrelated -- and
# where an optional LLM hint (see below) is allowed to help, if enabled.
_AMBIGUOUS_RELEVANCE_LOW = 0.05
_AMBIGUOUS_RELEVANCE_HIGH = 0.4

# See the calibration note at the embedding integration point inside
# job_relevance() for the full rationale. In short: when the
# deterministic signal found ZERO overlap, an embedding score must clear
# this stricter bar to be trusted at full strength; below it, the score
# is dampened rather than trusted outright, so a single noisy/moderate
# embedding similarity can't manufacture relevance with no corroborating
# evidence at all.
_EMBEDDING_ZERO_CORROBORATION_TRUST_FLOOR = 0.6
_EMBEDDING_ZERO_CORROBORATION_DAMPENING = 0.5

_LLM_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _llm_role_relevance_hint(
    candidate_title: str, candidate_context: str, jd_title: str, jd_context: str,
) -> float | None:
    """Optional, OPT-IN LLM-assisted relevance hint for ambiguous title/
    role comparisons that the deterministic signature overlap couldn't
    confidently resolve either way.

    Returns a 0.0-1.0 relevance score, or None if the feature is disabled,
    no LLM is configured, or the call fails for any reason -- callers
    MUST treat None as "no hint available" and keep using the
    deterministic score, never as "0 relevance". This function never
    raises; every failure path is caught and degrades to None so the rest
    of the (fully deterministic) pipeline is completely unaffected when
    no LLM is available.
    """
    if os.getenv("ENABLE_LLM_ROLE_RELEVANCE", "").strip().lower() not in ("1", "true", "yes"):
        return None
    try:
        get_config()  # raises ValueError if no provider/API key is configured
        from .base import create_llm

        llm = create_llm(temperature=0.0)
        prompt = (
            "You are helping score how relevant a candidate's past role is to "
            "an open job. Respond with ONLY a single number between 0.0 and "
            "1.0 (no words, no explanation).\n\n"
            f"Open job title: {jd_title!r}\n"
            f"Open job skills/responsibilities: {jd_context[:500]!r}\n\n"
            f"Candidate's past role title: {candidate_title!r}\n"
            f"Candidate's past role skills/responsibilities: {candidate_context[:500]!r}\n\n"
            "How relevant is the candidate's past role to the open job, "
            "based on the ACTUAL WORK described (not just whether the titles "
            "match)? Respond with ONLY the number."
        )
        response = llm.invoke(prompt)
        text = getattr(response, "content", str(response)).strip()
        match = _LLM_SCORE_RE.search(text)
        if not match:
            return None
        return min(1.0, max(0.0, float(match.group(1))))
    except Exception:
        return None


def _parse_one_date(text: str) -> date | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if _PRESENT.match(raw):
        return date.today()
    lowered = raw.lower()
    month = None
    year = None
    y = re.search(r"\b(19|20)\d{2}\b", raw)
    if y:
        year = int(y.group(0))
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}\b", lowered):
            month = num
            break
    mnum = re.search(r"\b(0?[1-9]|1[0-2])[/.-]((?:19|20)\d{2})\b", raw)
    if mnum:
        month = int(mnum.group(1))
        year = int(mnum.group(2))
    else:
        # ISO-style "YYYY-MM" (year first, month second) -- e.g. "2022-01",
        # "2023-06". The MM-YYYY pattern above never matches this order,
        # so previously any YYYY-MM date silently fell through with
        # month=None and defaulted to January regardless of the real
        # month -- correct only by coincidence when the real month
        # happened to BE January. This is the same numeric date format
        # WorkExperience.start_date/end_date use throughout this
        # project's own sample data and tests (e.g. "2022-01", "2023-06").
        ym = re.search(r"\b((?:19|20)\d{2})[/.-](0?[1-9]|1[0-2])\b", raw)
        if ym:
            year = int(ym.group(1))
            month = int(ym.group(2))
    if year is None:
        return None
    return date(year, month or 1, 1)


def _parse_duration_years(text: str) -> float | None:
    if not text:
        return None
    years = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", text, re.I)
    months = re.search(r"(\d+(?:\.\d+)?)\s*months?", text, re.I)
    total = 0.0
    found = False
    if years:
        total += float(years.group(1))
        found = True
    if months:
        total += float(months.group(1)) / 12.0
        found = True
    return round(total, 2) if found else None


def _split_range(text: str) -> tuple[str, str]:
    parts = re.split(r"\s*(?:-|–|—|to)\s*", text or "", maxsplit=1, flags=re.I)
    if len(parts) == 2:
        return parts[0], parts[1]
    return text, ""


def job_years(exp: WorkExperience) -> float:
    start = _parse_one_date(exp.start_date)
    end = _parse_one_date(exp.end_date)
    if start is None and end is None:
        left, right = _split_range(exp.duration or "")
        start = _parse_one_date(left)
        end = _parse_one_date(right) if right else date.today()
    if start is None and end is None:
        from_duration = _parse_duration_years(exp.duration)
        if from_duration is not None:
            return from_duration
        return 0.5 if (exp.title and exp.company) else 0.0
    if start is None:
        return 0.5
    if end is None:
        end = date.today()
    if end < start:
        start, end = end, start
    days = (end - start).days
    return max(0.0, round(days / 365.25, 2))


def _tokens(text: str) -> set[str]:
    return {
        t for t in normalize_skill_text(text).split()
        if t and t not in _STOP and len(t) > 1
    }


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# job_relevance()'s skill_hit component previously used a plain,
# unbounded substring test (`k in s or s in k`, and separately `k in
# text_blob`) to decide whether a candidate's technology/skill key
# "relates to" a JD skill key. Both keys come from the same
# skill_taxonomy normalization (canonical_skill_key /
# normalize_requirement_skills / normalize_skill_text), which strips
# punctuation down to single space-separated tokens -- so a plain
# substring test has no way to tell "java" is a WHOLE word inside
# "javascript" (false positive) apart from "algorithms" being a whole
# word inside "data structures and algorithms" (a legitimate
# multi-word/parent-child match the taxonomy already recognizes as
# related). This produced exactly the same class of false-positive
# bug already fixed in SkillsMatcherAgent (Java credited for
# "JavaScript", C for "C++", Git for "GitHub", etc.) -- see
# tests/test_job_relevance_skill_substring.py for reproductions.
#
# Fix: require the shorter key to appear at a WORD boundary within the
# longer one (bounded by whitespace/start/end, since both sides are
# already normalized to space-separated tokens with no other
# punctuation) rather than anywhere as a raw substring. Exact-string
# equality (k == s) is unaffected and still the primary path; this only
# changes what counts as a valid PARTIAL/multi-word match.
def _bounded_skill_match(a: str, b: str) -> bool:
    """True if normalized skill keys `a` and `b` are equal, or one
    appears as a whole space-bounded run of token(s) inside the other."""
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return re.search(rf"(?<!\S){re.escape(shorter)}(?!\S)", longer) is not None


def job_relevance(exp: WorkExperience, requirements: JobRequirements) -> float:
    jd_title = _title_tokens(requirements.title)

    jd_skills: set[str] = set()
    for s in requirements.required_skills + requirements.preferred_skills:
        jd_skills |= normalize_requirement_skills(s)

    resp_text = " ".join(requirements.responsibilities[:8])
    jd_resp = _tokens(resp_text)

    job_title = _title_tokens(exp.title)
    job_resp = _tokens(" ".join(exp.responsibilities[:8]))

    title_score = _overlap(job_title, jd_title)

    # CONFIRMED BUG: _title_tokens() deliberately keeps generic
    # seniority/role words ("senior", "engineer", "developer", "software",
    # "associate", "junior", "lead", "intern" -- see _STOP, and Batch 7's
    # rationale for why _title_tokens must NOT strip them: doing so
    # collapsed titles like "Senior Software Engineer" to an empty token
    # set and zeroed title_score entirely). That's correct for titles
    # that ALSO share a distinctive/domain word, but when the ONLY shared
    # tokens between two titles are generic role words, the raw Jaccard
    # overlap still credits it -- e.g. "Senior QA Engineer" vs "Senior
    # Backend Engineer" share only {"senior", "engineer"} and score a
    # lexical title_score of 0.5 (job_relevance()=0.23 overall) purely
    # from words that appear in nearly every engineering title regardless
    # of domain. This directly contradicts the role-family matrix's own
    # documented intent a few lines below (the old _ROLE_RELATEDNESS's 0.0
    # default is described as "a genuine 'no relationship'", and the
    # Batch 10 comment explicitly removed backend<->frontend/data credit
    # for the exact same reason: an "artificial, unjustified boost...
    # purely because both happen to be software roles"). The family
    # backstop already declines to credit backend<->qa, backend<->data,
    # backend<->mobile, etc. (not in the matrix, so 0.0) -- but the
    # lexical side had no equivalent guard and could still smuggle that
    # credit back in through shared seniority/role vocabulary alone.
    #
    # Fix: if every token the two titles share is a generic word (per the
    # existing _STOP list -- no new word list invented), the lexical
    # score contributes nothing; only a shared DISTINCTIVE word (e.g.
    # "backend", "python", "qa") counts as real lexical evidence. This
    # does not touch _title_tokens itself (so the Batch 7 empty-set
    # collapse fix is untouched and titles still tokenize the same way),
    # does not touch the role-family backstop below, and does not lower
    # title_score for any case where a real distinctive word is shared.
    if title_score > 0 and not ((job_title & jd_title) - _STOP):
        title_score = 0.0

    # Generalized semantic backstop (replaces the old hardcoded role-family
    # taxonomy): lexical title overlap alone can't see that e.g. a "Data
    # Scientist" candidate with NLP/PyTorch experience is genuinely
    # relevant to a "Machine Learning Engineer" JD when the titles share
    # no words. _semantic_role_relevance() measures overlap of actual
    # demonstrated skills/responsibilities (via the shared skill taxonomy)
    # instead of relying on any fixed list of job-title names, so it works
    # identically for titles this project has never seen before. This
    # only ever RAISES title_score (max()), so a title that already
    # scores well lexically is unaffected, and it never touches skill_hit
    # or resp_score.
    semantic_score = _semantic_role_relevance(exp, requirements)
    title_score = max(title_score, semantic_score)

    # Directional coverage counterpart to the symmetric Jaccard score
    # above -- see _semantic_role_coverage()'s module-level comment for
    # the full rationale (fixes systematic under-scoring of detailed/
    # senior resume entries whose own signature is simply larger than
    # the JD's). Only ever raises title_score, same max()-combination
    # pattern as semantic_score itself.
    coverage_score = _semantic_role_coverage(exp, requirements)
    title_score = max(title_score, coverage_score)

    # Real embedding-based semantic similarity (src/embeddings.py), tried
    # BEFORE the LLM hint below -- it's cheaper and more deterministic
    # per call than a chat-model completion, and doesn't need a chat
    # model configured at all (Gemini's embeddings endpoint is enough).
    #
    # CALIBRATION NOTE (production-readiness review): a plain
    # `title_score = max(title_score, embedding_score)` -- unconditional,
    # regardless of what the deterministic signal (lexical + taxonomy
    # signature) found -- was verified to let a SINGLE noisy or merely
    # moderate embedding score manufacture meaningful relevance out of
    # thin air for a genuinely unrelated pair (e.g. "Backend Engineer" vs
    # "Marketing Manager", where lexical/taxonomy overlap is correctly
    # 0.0). Sentence/text embedding spaces are well known to cluster even
    # UNRELATED short phrases at a non-trivial baseline cosine similarity
    # ("anisotropy") -- EMBEDDING_SIMILARITY_FLOOR already exists to
    # filter the LOW end of that noise, but nothing filtered the MID
    # range: a moderate-but-not-extreme embedding score (e.g. rescaled
    # ~0.45-0.65) could still single-handedly raise title_score with
    # zero other evidence backing it up, since max() has no way to tell
    # "confidently related" apart from "embedding noise that happens to
    # be in the middle of the range."
    #
    # Fix: when the deterministic signal found ZERO overlap (no shared
    # title words, no shared taxonomy skills/responsibilities -- i.e.
    # title_score is still exactly 0.0 at this point), the embedding
    # score is trusted at full strength ONLY if it clears
    # _EMBEDDING_ZERO_CORROBORATION_TRUST_FLOOR -- a stricter bar than
    # the ordinary FLOOR/CEIL rescaling already applies. Below that bar,
    # its contribution is dampened (not zeroed -- it may still be a real,
    # if weaker, signal) rather than trusted outright. Once there IS any
    # deterministic corroboration at all (title_score > 0.0 -- some real
    # shared word or taxonomy skill), the existing max() behavior is
    # unchanged, since a second, independent signal agreeing with an
    # already-present one is exactly when trusting it is safe.
    #
    # _EMBEDDING_ZERO_CORROBORATION_TRUST_FLOOR = 0.6 is not an arbitrary
    # pick: it reuses this module's OWN existing ambiguity boundary
    # (_AMBIGUOUS_RELEVANCE_HIGH = 0.4, the same value already used to
    # decide when the LLM hint may fire) with a symmetric margin above
    # the scale's midpoint (0.5), i.e. "confidently past the ambiguous
    # zone in the SAME way _AMBIGUOUS_RELEVANCE_HIGH already defines
    # 'still ambiguous' below it." In raw-cosine terms (before the
    # existing FLOOR=0.35/CEIL=0.90 rescale), 0.6 rescaled corresponds to
    # a raw cosine of ~0.68 -- comfortably above the ~0.5-0.65 baseline
    # noise range the FLOOR constant's own comment already documents.
    # _EMBEDDING_ZERO_CORROBORATION_DAMPENING = 0.5 is a conservative,
    # simple halving (not a tuned value -- no live embedding calibration
    # data is available in this environment) chosen so an unconfirmed
    # embedding score still contributes SOMETHING (it may be a real, if
    # weaker, signal) but can never alone account for more than half of
    # what full trust would have given it.
    embedding_score = _embedding_role_relevance(exp, requirements)
    if embedding_score is not None:
        if title_score > 0.0 or embedding_score >= _EMBEDDING_ZERO_CORROBORATION_TRUST_FLOOR:
            title_score = max(title_score, embedding_score)
        else:
            title_score = max(title_score, embedding_score * _EMBEDDING_ZERO_CORROBORATION_DAMPENING)

    # Optional, OPT-IN LLM assist for genuinely ambiguous cases only (see
    # module-level comment above _llm_role_relevance_hint). Disabled by
    # default; any failure/unavailability falls straight back to the
    # deterministic (+ embedding, if it fired) score above with no
    # behavior change.
    if _AMBIGUOUS_RELEVANCE_LOW <= title_score <= _AMBIGUOUS_RELEVANCE_HIGH:
        llm_hint = _llm_role_relevance_hint(
            exp.title,
            " ".join((exp.technologies or []) + (exp.responsibilities or [])),
            requirements.title,
            " ".join(
                (requirements.required_skills or [])
                + (requirements.preferred_skills or [])
                + (requirements.responsibilities or [])
            ),
        )
        if llm_hint is not None:
            title_score = max(title_score, llm_hint)

    tech_keys = {canonical_skill_key(t) for t in exp.technologies if t}
    skill_hit = 0.0
    if jd_skills:
        hits = sum(1 for k in tech_keys if any(_bounded_skill_match(k, s) for s in jd_skills if s))
        skill_hit = min(1.0, hits / max(1, len(jd_skills))) if tech_keys else 0.0
        text_blob = normalize_skill_text(
            " ".join([exp.title, " ".join(exp.technologies), " ".join(exp.responsibilities)])
        )
        # Individual candidate evidence snippets (NOT one joined blob)
        # for the generalized semantic fallback below. Each requirement
        # is checked against each snippet separately and the single BEST
        # result is kept -- see match_requirement_against_snippets'
        # docstring for why comparing against the whole concatenated
        # text would dilute embedding similarity and silently under-score
        # a genuine one-sentence match.
        candidate_evidence_snippets = (
            [exp.title] if exp.title else []
        ) + list(exp.technologies or []) + list(exp.responsibilities or [])
        mentioned = 0.0
        for req in requirements.required_skills:
            req_keys = normalize_requirement_skills(req)
            if any(k and re.search(rf"(?<!\S){re.escape(k)}(?!\S)", text_blob) for k in req_keys):
                mentioned += 1.0
                continue
            # GENERALIZED SEMANTIC FALLBACK (src/skill_matching.py): only
            # consulted when the deterministic taxonomy/literal-phrase
            # check just above found NOTHING for this specific
            # requirement. This is what lets a non-technical requirement
            # (e.g. "SEO and keyword research") get credit from a
            # candidate's differently-worded evidence ("Performed
            # keyword research and on-page SEO optimization") without
            # needing skill_taxonomy.py to grow a hardcoded entry for
            # every domain, and without needing the exact phrase to
            # appear verbatim in the same word order. It only ever ADDS
            # credit here -- when no embedding provider is configured
            # (this project's default), match_requirement() returns a
            # 0.0-credit decision and `mentioned` is completely
            # unaffected, so a tech-domain candidate whose taxonomy
            # coverage was already complete sees zero behavior change.
            decision = match_skill_requirement_against_snippets(req, candidate_evidence_snippets)
            mentioned += decision.credit
        if requirements.required_skills:
            skill_hit = max(skill_hit, mentioned / len(requirements.required_skills))

    resp_score = _overlap(job_resp, jd_resp | jd_title)
    relevance = 0.45 * title_score + 0.4 * skill_hit + 0.15 * resp_score
    return round(min(1.0, max(0.0, relevance)), 2)


def _seniority(title: str) -> int:
    t = (title or "").lower()
    best = 2
    for word, level in _SENIORITY:
        if word in t:
            best = max(best, level)
    return best


def _career_progression(experiences: list[WorkExperience]) -> str:
    if len(experiences) == 1:
        return "Single listed role."
    levels = [_seniority(e.title) for e in reversed(experiences)]
    if levels[-1] > levels[0]:
        return "Steady growth in seniority across listed roles."
    if levels[-1] == levels[0]:
        return "Lateral moves at a similar seniority level."
    return "Mixed seniority across listed roles."


def _experience_score_for_years(years_relevant: float, years_required: int) -> float:
    if years_required <= 0:
        if years_relevant >= 1:
            return 1.0
        elif years_relevant > 0:
            return 0.85
        else:
            return 0.7
    return round(min(1.0, years_relevant / years_required), 2)


# --------------------------------------------------------------------------
# years_relevant per-job credit
#
# CONFIRMED BUG (Priority 6 investigation): the original formula was a flat
#     credit = 0.35 + 0.65 * rel
# applied to EVERY job regardless of how low `rel` (job_relevance()) was.
# There was no lower gate at all -- a job scored at rel=0.01 (e.g. a
# Frontend Developer role against a Backend JD) or rel=0.00 (a resume-
# parsing artifact like a mis-split "EDUCATION & TRAINING" line being
# treated as a job -- see the separate resume-parsing note in
# src/agents/resume_parser.py) still received at least 35% credit toward
# years_relevant, no different from a job that was genuinely, if weakly,
# related. This is what produced the reported anomalies: Ananya Patel's
# 4-year, 1%-relevant Frontend Developer role alone contributed ~1.45
# "relevant" years (35% of 4.07), and Vikram Singh's unrelated Mechanical
# Engineer role plus several parsing-artifact pseudo-jobs (career break,
# education section header, university line) each got a guaranteed floor
# of credit purely for having *some* duration, inflating years_relevant
# well beyond what role_relevance (7%/33% respectively) would suggest is
# defensible.
#
# The 0.35 floor itself was not baseless: job_relevance() is a lexical/
# taxonomy approximation (see the Batch 8/9/10 history in this file) that
# can under-score genuinely related work, so SOME hedge against
# under-scoring is reasonable for a job that shows real, if weak,
# relevance signal. The bug was applying that hedge unconditionally, even
# to jobs with essentially no relevance signal at all.
#
# Fix: _years_relevant_credit() below only applies the 0.35+0.65*rel
# hedge to jobs at/above _YEARS_RELEVANT_FLOOR_THRESHOLD (chosen to match
# the rel < 0.2 cutoff this module already uses, a few lines below in
# evaluate_experience(), to flag a job as "Limited relevance" -- a job
# already flagged that way should not simultaneously receive a guaranteed
# 35% credit toward years_relevant). Below that threshold, credit ramps
# linearly from 0 at rel=0 up to the existing floor-formula's value AT
# the threshold, so the curve is continuous (no cliff at the boundary)
# and a genuinely irrelevant job (rel near 0) now contributes years near
# 0, while the existing above-threshold behavior for real matches is
# completely unchanged.
# --------------------------------------------------------------------------
_YEARS_RELEVANT_FLOOR_THRESHOLD = 0.2
_YEARS_RELEVANT_FLOOR_CREDIT = 0.35


def _years_relevant_credit(rel: float) -> float:
    """Weight applied to a job's duration when accumulating years_relevant.
    See the module-level comment above for the full rationale."""
    threshold_credit = _YEARS_RELEVANT_FLOOR_CREDIT + (1 - _YEARS_RELEVANT_FLOOR_CREDIT) * _YEARS_RELEVANT_FLOOR_THRESHOLD
    if rel < _YEARS_RELEVANT_FLOOR_THRESHOLD:
        return threshold_credit * (rel / _YEARS_RELEVANT_FLOOR_THRESHOLD)
    return _YEARS_RELEVANT_FLOOR_CREDIT + (1 - _YEARS_RELEVANT_FLOOR_CREDIT) * rel


_PROJECT_RELEVANCE_THRESHOLD = 0.3


def project_relevance(project_text: str, requirements: JobRequirements) -> float:
    if not project_text:
        return 0.0
    jd_skill_keys = {
        canonical_skill_key(s)
        for s in (requirements.required_skills + requirements.preferred_skills)
        if s
    }
    jd_topic_tokens = _tokens(requirements.title) | _tokens(" ".join(requirements.responsibilities[:8]))
    proj_hits = {canonical_skill_key(name) for name, _category in extract_canonical_skills(project_text)}
    matched = len(proj_hits & jd_skill_keys)
    skill_hit = min(1.0, matched / 3) if jd_skill_keys else 0.0
    topic_overlap = _overlap(_tokens(project_text), jd_topic_tokens)
    relevance = 0.75 * min(1.0, skill_hit) + 0.25 * topic_overlap
    return round(min(1.0, max(0.0, relevance)), 2)


def _best_project(projects: list[str], requirements: JobRequirements) -> tuple[float, str | None]:
    """Return the (score, text) of whichever listed project is most
    relevant to the job requirements, on the full continuous 0-1
    project_relevance() scale.

    Deliberately NOT gated by _PROJECT_RELEVANCE_THRESHOLD here.

    CONFIRMED BUG (this function previously zeroed out any result below
    the threshold): project_relevance()'s skill_hit component is
    quantized in thirds (0.75 * matched/3, for matched in {0, 1, 2,
    3+}), so a project that genuinely demonstrates exactly ONE of the
    JD's combined required+preferred skills -- real, structured
    evidence, not noise -- scores exactly 0.25 with no other overlap.
    That lands just under the 0.3 threshold. evaluate_experience()
    feeds this score into `role_relevance = max(education_baseline,
    project_score)`, a formula whose whole design is "take whichever
    signal is strongest" -- but the old gate silently collapsed that
    0.25 down to 0.0 first, so a project scoring HIGHER than a
    candidate's own education baseline (0.15) could still contribute
    nothing, indistinguishable from having listed no project at all.
    That is a discontinuity in the scoring, not a deliberate design
    choice: nothing about the max()-based role_relevance formula calls
    for discarding real, if modest, evidence.

    The threshold itself is not being removed -- it still governs
    whether a project is clearly strong enough to be *named* in
    strengths/reasoning (see `relevant_project` in evaluate_experience,
    a presentation decision distinct from the underlying score). Only
    the scoring contribution is now continuous, matching how every
    other signal in this module (job_relevance, years_relevant credit)
    already behaves.
    """
    best_score = 0.0
    best_project: str | None = None
    for project_text in projects or []:
        score = project_relevance(project_text, requirements)
        if score > best_score:
            best_score, best_project = score, project_text
    return best_score, best_project


def _project_title(project_text: str, max_len: int = 60) -> str:
    for sep in (" -- ", " — ", ": "):
        if sep in project_text:
            return project_text.split(sep, 1)[0].strip()
    text = project_text.strip()
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "\u2026"


def evaluate_experience(resume_data: ResumeData, requirements: JobRequirements) -> ExperienceEvaluation:
    years_required = requirements.min_years_experience or 0

    if not resume_data.work_experience:
        has_relevant_education = any(
            any(
                kw in (edu.field or "").lower()
                for kw in ["computer", "data", "software", "engineering", "ai", "machine learning", "information technology"]
            )
            for edu in resume_data.education
        ) if resume_data.education else False

        project_score, best_project = _best_project(resume_data.projects, requirements)
        # `relevant_project` gates NAMING a project in strengths/reasoning --
        # a presentation judgment ("is this clearly strong enough to call
        # out"), kept at the existing 0.3 bar. It is intentionally separate
        # from role_relevance below, which uses the full continuous
        # project_score so a project with real-but-modest evidence (e.g.
        # matching exactly one required skill, scoring 0.25) still raises
        # role_relevance via max() when it exceeds the education baseline,
        # instead of being discarded outright. See _best_project's docstring.
        relevant_project = best_project is not None and project_score >= _PROJECT_RELEVANCE_THRESHOLD
        role_relevance = round(max(0.15 if has_relevant_education else 0.0, project_score), 2)

        strengths = []
        if has_relevant_education:
            strengths.append("Relevant academic background")
        if relevant_project:
            strengths.append(f"Relevant project: {_project_title(best_project)}")

        experience_score = _experience_score_for_years(0.0, years_required)

        gaps = ["No work experience listed"]
        if years_required > 0:
            gaps.append(f"No professional experience found (role requires {years_required}y)")

        reasoning_parts = [
            f"No formal work experience was listed. Against a "
            f"{years_required}-year requirement, experience_score is "
            f"{experience_score:.2f} using the same years-required scale "
            f"applied to candidates who do have work history."
        ]
        baseline_only = 0.15 if has_relevant_education else 0.0
        if relevant_project:
            reasoning_parts.append(
                f"Role relevance is boosted by a relevant project "
                f"(\"{_project_title(best_project)}\"), which is not "
                f"counted as professional experience."
            )
        elif best_project is not None and project_score > baseline_only:
            # Project score is real (non-zero, and above whatever baseline
            # already applies) but below the "clearly relevant, name it"
            # bar -- still worth a factual note rather than silence, since
            # role_relevance above did factor it in.
            reasoning_parts.append(
                "Role relevance includes a modest contribution from a listed "
                "project showing partial, though not clearly strong, relevance."
            )
        elif has_relevant_education:
            reasoning_parts.append("Role relevance reflects a related academic background.")
        else:
            reasoning_parts.append("No directly relevant education or projects were found either.")

        return ExperienceEvaluation(
            years_relevant=0.0, years_required=years_required, experience_score=experience_score,
            role_relevance=role_relevance, career_progression="No professional work history listed.",
            gaps_identified=gaps, strengths=strengths, confidence=0.9,
            reasoning=" ".join(reasoning_parts),
        )

    per_job: list[tuple[float, float]] = []
    strengths: list[str] = []
    gaps: list[str] = []

    for exp in resume_data.work_experience:
        years = job_years(exp)
        rel = job_relevance(exp, requirements)
        per_job.append((years, rel))
        if rel >= 0.5 and exp.title:
            strengths.append(f"Relevant role: {exp.title} at {exp.company}")
        elif exp.title and rel < 0.2:
            gaps.append(f"Limited relevance: {exp.title} at {exp.company}")

    years_relevant = round(sum(years * _years_relevant_credit(rel) for years, rel in per_job), 2)
    if per_job:
        role_relevance = round(
            sum(rel * max(years, 0.25) for years, rel in per_job) / sum(max(years, 0.25) for years, _rel in per_job), 2,
        )
    else:
        role_relevance = 0.0

    experience_score = _experience_score_for_years(years_relevant, years_required)

    if years_relevant + 0.25 < years_required:
        gaps.append(f"Relevant experience ({years_relevant:.1f}y) is below the {years_required}y requirement")

    reasoning = (
        f"Estimated {years_relevant:.1f} years of relevant experience "
        f"(required {years_required}). Role relevance {role_relevance:.0%} "
        f"based on title/technology overlap with {requirements.title or 'the role'}."
    )

    return ExperienceEvaluation(
        years_relevant=years_relevant, years_required=years_required, experience_score=experience_score,
        role_relevance=role_relevance, career_progression=_career_progression(resume_data.work_experience),
        gaps_identified=gaps[:6], strengths=strengths[:6], confidence=0.9, reasoning=reasoning,
    )


class ExperienceEvaluatorAgent(BaseAgent):
    name = "ExperienceEvaluatorAgent"
    description = "Evaluate work experience relevance (deterministic, no LLM call)"
    temperature = 0.0

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        resume_data = state.get("resume_data")
        job_requirements = state.get("job_requirements")
        if not resume_data or not job_requirements:
            return {"experience_eval": ExperienceEvaluation(confidence=0.0, reasoning="Missing resume or job requirements data"),
                    "errors": ["ExperienceEvaluatorAgent: Missing required data"], "agent_confidences": {self.name: 0.0}}
        if isinstance(resume_data, dict):
            resume_data = ResumeData.model_validate(resume_data)
        if isinstance(job_requirements, dict):
            job_requirements = JobRequirements.model_validate(job_requirements)
        evaluation = evaluate_experience(resume_data, job_requirements)
        return {"experience_eval": evaluation, "agent_confidences": {self.name: evaluation.confidence}}