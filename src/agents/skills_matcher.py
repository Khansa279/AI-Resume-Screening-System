"""Skills Matcher Agent - Compares candidate skills against job requirements.

PRIMARILY DETERMINISTIC. This agent reuses the exact same taxonomy/alias
table as SkillExtractorAgent and ExperienceEvaluatorAgent
(`src/skill_taxonomy.py` -- Batch 3: previously this agent used a
separate, overlapping `src/skills_taxonomy.py` module, which risked
disagreeing with the other two agents on what a given alias normalizes
to) to decide whether a candidate skill satisfies a JD requirement.

LLM FALLBACK/RECHECK (added after Batch 10): the hardcoded taxonomy
cannot scale to every semantic relationship between ML/AI concepts --
e.g. "classification"/"regression" both imply "supervised learning",
"clustering" implies "unsupervised learning", and "cross-validation",
"F1 score", "ROC-AUC", "MAE", "RMSE" all imply "model evaluation". Rather
than keep hardcoding aliases for every such relationship, requirements
that are STILL unmatched after the deterministic pass (real gaps only --
not soft-skill "not_mentioned" entries, which were never gaps to begin
with) are re-checked in a single batched LLM call: the JD requirement
text plus the candidate's extracted-skill evidence is sent, and the LLM
returns a structured MATCH/NO_MATCH verdict with cited evidence and a
confidence score.

This is strictly a RECHECK layer, not a replacement:
  - The deterministic pass always runs first and is authoritative --
    anything it already matched is never touched or reconsidered here.
  - The LLM never sets a score directly. A promoted requirement is
    assigned match_quality="semantic" (the same quality deterministic
    alias matches already use) and the EXISTING CREDIT_BY_QUALITY /
    REQUIRED_WEIGHT / PREFERRED_WEIGHT aggregation formula (see
    `_aggregate`) computes the final overall_score exactly as before --
    the LLM supplies a verdict, not a number that flows into scoring.
  - A verdict is only accepted with a real confidence >= a fixed
    threshold AND cited evidence naming a specific skill; a bare
    "MATCH" with nothing behind it is discarded.
  - Any failure -- API error, rate limit, malformed/unparseable
    response, or even an unconfigured LLM provider (no API key) -- is
    caught and the ORIGINAL deterministic result is returned unchanged.
    See `_llm_recheck_gaps`.
"""

import logging
import re
from typing import Any

from .base import BaseAgent
from ..models import Skill, JobRequirements, SkillMatch, SkillsMatchResult
from ..skill_taxonomy import canonical_skill_key, extract_canonical_skills, requirement_is_pure_soft_skill

logger = logging.getLogger(__name__)


class SkillsMatcherAgent(BaseAgent):
    """
    Agent responsible for matching candidate skills to job requirements.

    This agent:
    - Compares extracted skills against required/preferred skills
    - Handles alias-based "semantic" matching (e.g., "JS" = "JavaScript")
      via the shared taxonomy, plus substring/partial fallback matching
    - Scores each requirement match
    - Calculates overall skills match score

    SCORING (unchanged from the original design):
    - Required skills contribute ~70% of overall_score
    - Preferred skills contribute ~30% of overall_score
    - exact match = full credit, semantic (taxonomy alias) = 90% credit,
      partial (substring) = 50% credit, no match = 0% credit
    """

    name = "SkillsMatcherAgent"
    description = "Compare candidate skills against job requirements and score the match (deterministic, with an LLM recheck fallback for unmatched requirements)"

    # Deterministic-judgment agent: when the LLM recheck fires, its
    # verdicts should be as consistent as possible run-to-run for the
    # same (requirement, evidence) input, same rationale as
    # ExperienceEvaluatorAgent/JobAnalyzerAgent/ResumeParserAgent.
    temperature = 0.0

    REQUIRED_WEIGHT = 0.7
    PREFERRED_WEIGHT = 0.3

    CREDIT_BY_QUALITY = {
        "exact": 1.0,
        "semantic": 0.9,
        "partial": 0.5,
        "none": 0.0,
    }

    # Toggle for the LLM recheck layer -- exposed as a class attribute so
    # it can be disabled (e.g. in a context with no LLM access at all)
    # without touching call sites. Deterministic matching is completely
    # unaffected either way.
    ENABLE_LLM_RECHECK: bool = True

    # A promoted verdict must clear this confidence bar. Deliberately not
    # very low -- this is a recheck for real semantic relationships
    # (e.g. "classification" -> "supervised learning"), not a place to
    # let a weakly-confident guess become a scored match.
    LLM_RECHECK_MIN_CONFIDENCE: float = 0.6

    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Match candidate skills against job requirements.

        Args:
            state: Current workflow state with extracted_skills and job_requirements

        Returns:
            State updates with skills_match result
        """
        extracted_skills = state.get("extracted_skills", [])
        job_requirements = state.get("job_requirements")

        if not job_requirements:
            return {
                "skills_match": SkillsMatchResult(confidence=0.0, reasoning="No job requirements to match against"),
                "errors": ["SkillsMatcherAgent: No job requirements available"],
                "agent_confidences": {self.name: 0.0}
            }

        # Convert to proper types if needed
        if isinstance(job_requirements, dict):
            job_requirements = JobRequirements.model_validate(job_requirements)

        skills_list: list[Skill] = []
        for skill in extracted_skills:
            if isinstance(skill, dict):
                skills_list.append(Skill.model_validate(skill))
            else:
                skills_list.append(skill)

        match_result = self._match_deterministic(skills_list, job_requirements)

        if self.ENABLE_LLM_RECHECK:
            try:
                match_result = await self._llm_recheck_gaps(match_result, skills_list, job_requirements)
            except Exception as e:  # noqa: BLE001 -- see module docstring: any
                # failure here (API error, unconfigured provider, malformed
                # response, or anything unforeseen) must never take down the
                # whole screening -- the deterministic result already
                # computed above is a perfectly valid result on its own.
                logger.warning(
                    f"[{self.name}] LLM recheck failed unexpectedly; keeping "
                    f"the deterministic result unchanged: {e}"
                )

        return {
            "skills_match": match_result,
            "agent_confidences": {self.name: match_result.confidence}
        }

    def _match_deterministic(
        self, skills: list[Skill], requirements: JobRequirements
    ) -> SkillsMatchResult:
        """Score every required/preferred requirement against candidate skills."""
        matches: list[SkillMatch] = [
            self._score_requirement(req, skills) for req in (requirements.required_skills or [])
        ] + [
            self._score_requirement(req, skills) for req in (requirements.preferred_skills or [])
        ]
        return self._aggregate(matches, requirements)

    def _aggregate(
        self, matches: list[SkillMatch], requirements: JobRequirements
    ) -> SkillsMatchResult:
        """Compute required/preferred met-counts, overall_score, and
        reasoning from a `matches` list, in the SAME order/shape
        `_match_deterministic` builds it in (all required-skill matches
        first, in requirements.required_skills order, then all
        preferred-skill matches). This is the single source of truth for
        the scoring formula -- called both right after the deterministic
        pass and again (on an updated `matches` list) after the LLM
        recheck, so a promoted match is scored by the EXACT SAME
        CREDIT_BY_QUALITY / REQUIRED_WEIGHT / PREFERRED_WEIGHT formula,
        never by anything the LLM itself proposes.
        """
        n_required = len(requirements.required_skills or [])
        required_matches = matches[:n_required]
        preferred_matches = matches[n_required:]

        required_scores: list[float] = []
        preferred_scores: list[float] = []
        required_met = 0
        preferred_met = 0
        required_excluded = 0
        preferred_excluded = 0

        for match in required_matches:
            # A pure soft-skill requirement with no resume evidence either
            # way is NOT a gap -- absence of a keyword like "communication"
            # doesn't mean the candidate lacks it, so it must not reduce
            # the score or count against required_skills_total. It still
            # appears in `matches` (soft_skill_status="not_mentioned") for
            # transparency; it's just excluded from scoring.
            if match.soft_skill_status == "not_mentioned":
                required_excluded += 1
                continue
            required_scores.append(self.CREDIT_BY_QUALITY[match.match_quality])
            if match.matched:
                required_met += 1

        for match in preferred_matches:
            if match.soft_skill_status == "not_mentioned":
                preferred_excluded += 1
                continue
            preferred_scores.append(self.CREDIT_BY_QUALITY[match.match_quality])
            if match.matched:
                preferred_met += 1

        required_total = n_required - required_excluded
        preferred_total = len(requirements.preferred_skills or []) - preferred_excluded

        required_avg = sum(required_scores) / required_total if required_total else 1.0
        preferred_avg = sum(preferred_scores) / preferred_total if preferred_total else 1.0

        if required_total and preferred_total:
            overall_score = required_avg * self.REQUIRED_WEIGHT + preferred_avg * self.PREFERRED_WEIGHT
        elif required_total:
            overall_score = required_avg
        elif preferred_total:
            overall_score = preferred_avg
        else:
            overall_score = 0.0

        overall_score = round(min(max(overall_score, 0.0), 1.0), 4)

        reasoning = self._build_reasoning(
            required_met, required_total, preferred_met, preferred_total, overall_score,
            required_excluded, preferred_excluded,
        )

        # Deterministic matching applied identically every time -> stable,
        # high confidence rather than an LLM's self-reported guess. This
        # confidence formula is unchanged by the LLM recheck -- it is not
        # based on the LLM's own reported confidence for any individual
        # verdict (see LLM_RECHECK_MIN_CONFIDENCE, which only gates
        # WHETHER a verdict is accepted at all, not this value).
        confidence = 0.95 if (required_total or preferred_total) else 0.3

        return SkillsMatchResult(
            matches=matches,
            required_skills_met=required_met,
            required_skills_total=required_total,
            preferred_skills_met=preferred_met,
            preferred_skills_total=preferred_total,
            overall_score=overall_score,
            confidence=confidence,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # LLM fallback/recheck for requirements still unmatched after the
    # deterministic pass.
    # ------------------------------------------------------------------

    def _real_gap_indices(self, matches: list[SkillMatch]) -> list[int]:
        """Indices of matches that are genuine gaps -- unmatched AND not a
        soft-skill 'not_mentioned' entry (those were never gaps; see
        `_aggregate`). Only these are worth spending an LLM call on."""
        return [
            i for i, m in enumerate(matches)
            if not m.matched and m.soft_skill_status != "not_mentioned"
        ]

    async def _llm_recheck_gaps(
        self,
        match_result: SkillsMatchResult,
        skills: list[Skill],
        requirements: JobRequirements,
    ) -> SkillsMatchResult:
        """Re-check genuinely unmatched requirements with a single batched
        LLM call, promoting any confidently-evidenced MATCH to a
        match_quality="semantic" match. Deterministic matches are never
        reconsidered. On ANY failure (no API key configured, network/API
        error, malformed response), returns `match_result` unchanged.
        """
        gap_indices = self._real_gap_indices(match_result.matches)
        if not gap_indices:
            return match_result

        gap_requirements = [match_result.matches[i].requirement for i in gap_indices]
        prompt = self._build_recheck_prompt(gap_requirements, skills)

        # _call_llm_for_json_async already swallows API/rate-limit
        # failures (including "no provider configured") into "" -- see
        # BaseAgent's docstring -- so this never raises for that class of
        # failure. The broad except in `process()` is a second safety net
        # for anything else (e.g. a bug in prompt construction).
        response = await self._call_llm_for_json_async(prompt)
        if not response:
            return match_result

        data = self._extract_json_from_response(response)
        if not data or not isinstance(data.get("results"), list):
            return match_result

        verdicts: dict[str, dict] = {}
        for item in data["results"]:
            if isinstance(item, dict) and item.get("requirement"):
                verdicts[str(item["requirement"]).strip().casefold()] = item

        matches = list(match_result.matches)
        changed = False

        for i in gap_indices:
            original = matches[i]
            verdict = verdicts.get(original.requirement.strip().casefold())
            if not verdict:
                continue
            if str(verdict.get("match", "")).strip().upper() != "MATCH":
                continue

            evidence = str(verdict.get("evidence") or "").strip()
            if not evidence:
                # A MATCH with no cited evidence isn't trustworthy enough
                # to promote -- don't guess.
                continue

            try:
                confidence = float(verdict.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = min(max(confidence, 0.0), 1.0)
            if confidence < self.LLM_RECHECK_MIN_CONFIDENCE:
                continue

            matched_skill_name = self._best_evidence_skill_name(evidence, skills) or evidence[:80]

            matches[i] = SkillMatch(
                requirement=original.requirement,
                matched=True,
                matched_skill=matched_skill_name,
                match_quality="semantic",
                confidence=round(confidence, 2),
                notes=f"LLM semantic recheck: {evidence}",
                soft_skill_status=original.soft_skill_status,
            )
            changed = True

        if not changed:
            return match_result

        return self._aggregate(matches, requirements)

    @staticmethod
    def _best_evidence_skill_name(evidence: str, skills: list[Skill]) -> str | None:
        """If the LLM's cited evidence names one of the candidate's actual
        extracted skills, prefer that exact skill name as `matched_skill`
        (rather than raw LLM text) -- keeps `matched_skill` consistent
        with the candidate's real skill list, the same way deterministic
        matches already report an actual Skill.name."""
        evidence_lower = evidence.lower()
        for skill in skills:
            if skill.name and skill.name.lower() in evidence_lower:
                return skill.name
        return None

    def _build_recheck_prompt(self, gap_requirements: list[str], skills: list[Skill]) -> str:
        skill_lines = "\n".join(f"- {s.name} ({s.category})" for s in skills) or "(no extracted skills)"
        req_lines = "\n".join(f"{i + 1}. {req}" for i, req in enumerate(gap_requirements))
        return f"""{self._build_system_prompt()}

TASK: The requirements below did NOT match any candidate skill through
exact, alias, or substring comparison. Decide, for each one, whether the
candidate's skill evidence semantically satisfies it anyway -- e.g. a
candidate who lists "classification" and "regression" satisfies a
requirement for "supervised learning"; a candidate who lists
"cross-validation", "F1 score", "ROC-AUC", "MAE", or "RMSE" satisfies a
requirement for "model evaluation"; "clustering" satisfies "unsupervised
learning". Only answer MATCH when the evidence specifically and
genuinely supports the requirement -- do not guess from a loosely
related but distinct concept.

CANDIDATE SKILL EVIDENCE (already extracted from the resume):
{skill_lines}

UNMATCHED JOB REQUIREMENTS TO RE-CHECK:
{req_lines}

Return ONLY a JSON object of this exact shape, with exactly one entry
per requirement above, in the same order:
{{
    "results": [
        {{
            "requirement": "<exact requirement text as given above>",
            "match": "MATCH" or "NO_MATCH",
            "evidence": "<the specific candidate skill(s) that justify a MATCH; empty string for NO_MATCH>",
            "confidence": 0.0 to 1.0
        }}
    ]
}}

Respond with ONLY valid JSON, no markdown code fences, no additional text."""

    def _score_requirement(self, requirement: str, skills: list[Skill]) -> SkillMatch:
        """Score a single JD requirement string against the candidate's skill list."""
        req_canonical_key = canonical_skill_key(requirement)
        req_key = requirement.strip().lower()
        is_soft = requirement_is_pure_soft_skill(requirement)

        def _matched(match_quality: str, matched_skill: str, confidence: float, notes: str) -> SkillMatch:
            return SkillMatch(
                requirement=requirement, matched=True, matched_skill=matched_skill,
                match_quality=match_quality, confidence=confidence, notes=notes,
                soft_skill_status="demonstrated" if is_soft else None,
            )

        # 1. Exact match: requirement string equals a candidate skill name
        #    (case-insensitive), OR both normalize to the same canonical
        #    taxonomy entry (e.g. requirement "JS" vs candidate skill
        #    "JavaScript" both normalize to the same key).
        for skill in skills:
            skill_key = skill.name.strip().lower()
            if skill_key == req_key:
                return _matched("exact", skill.name, 1.0, "Exact name match")
            if canonical_skill_key(skill.name) == req_canonical_key:
                return _matched("exact", skill.name, 1.0, f"Both normalize to '{req_canonical_key}'")

        # 2. Semantic (alias) match: the requirement text contains a
        #    taxonomy alias whose canonical form matches a candidate skill,
        #    or vice-versa (candidate skill alias appears in requirement).
        #    For soft skills, this is also where an INFERRED candidate
        #    skill (e.g. "Leadership" inferred from "mentored junior devs")
        #    counts as demonstrated evidence -- SkillExtractorAgent already
        #    gives inferred skills lower confidence (0.7) than explicit
        #    ones (0.95); we surface that provenance in the note here
        #    rather than silently treating inferred and explicit the same.
        req_skill_hits = {name for name, _category in extract_canonical_skills(requirement)}
        for skill in skills:
            canonical_hits = extract_canonical_skills(skill.name)
            canonical_name = canonical_hits[0][0] if canonical_hits else skill.name
            if canonical_name in req_skill_hits:
                note = f"Taxonomy alias match on '{canonical_name}'"
                if is_soft and skill.source == "inferred":
                    note += " (inferred from resume experience, not explicitly listed)"
                return _matched("semantic", skill.name, 0.9, note)

        # 3. Partial match: WORD-BOUNDARY-aware substring overlap either
        #    direction (e.g. requirement "Python programming experience"
        #    vs skill "Python" -- "python" appears in the requirement as
        #    a whole word). Plain (non-word-boundary) substring
        #    containment previously matched unrelated skills that merely
        #    share a prefix -- e.g. candidate skill "Java" incorrectly
        #    satisfied requirement "JavaScript" (since "java" in
        #    "javascript"), and "C++" incorrectly satisfied requirement
        #    "C" (since "c" in "c++"), even though this project's own
        #    taxonomy (skill_taxonomy.py) treats Java/JavaScript and C/C++
        #    as distinct canonical skills. A word-boundary requirement on
        #    whichever side is doing the "containing" fixes this while
        #    leaving genuine substring matches (a whole word appearing
        #    inside a longer phrase) unaffected. The boundary character
        #    class additionally includes +/#/. -- the same punctuation
        #    skill_taxonomy.py's own _PROTECT list already treats as part
        #    of a single compound token (c++, c#, .net) -- so "c" doesn't
        #    read as a standalone word at the start of "c++" either.
        _BOUNDARY_CHARS = r"[a-z0-9+#.]"
        for skill in skills:
            skill_key = skill.name.strip().lower()
            if len(skill_key) < 3:
                continue
            if re.search(rf"(?<!{_BOUNDARY_CHARS})" + re.escape(skill_key) + rf"(?!{_BOUNDARY_CHARS})", req_key):
                return _matched("partial", skill.name, 0.5, "Substring overlap")
            if re.search(rf"(?<!{_BOUNDARY_CHARS})" + re.escape(req_key) + rf"(?!{_BOUNDARY_CHARS})", skill_key):
                return _matched("partial", skill.name, 0.5, "Substring overlap")

        # 4. No match found in the candidate's extracted skills.
        #
        #    For a PURE soft-skill requirement, this means "not mentioned",
        #    NOT "candidate lacks this trait" -- resumes routinely
        #    demonstrate teamwork/communication/leadership without ever
        #    using those exact words, and there is no reliable way to
        #    prove a true negative from resume text. So this is flagged
        #    "not_mentioned" (excluded from scoring/gaps by the caller)
        #    rather than treated as an ordinary unmet requirement.
        #
        #    For a technical requirement, "no match" IS a real gap, same
        #    as before -- that behavior is unchanged.
        return SkillMatch(
            requirement=requirement, matched=False, matched_skill="",
            match_quality="none", confidence=0.9,
            notes=(
                "No explicit or inferred evidence of this soft skill in the resume"
                if is_soft else "No matching candidate skill found"
            ),
            soft_skill_status="not_mentioned" if is_soft else None,
        )

    def _build_reasoning(
        self, required_met: int, required_total: int,
        preferred_met: int, preferred_total: int, overall_score: float,
        required_excluded: int = 0, preferred_excluded: int = 0,
    ) -> str:
        parts = [f"Overall skills match: {overall_score:.0%}."]
        if required_total:
            parts.append(f"Met {required_met}/{required_total} required skills.")
        else:
            parts.append("No required skills were specified for this role.")
        if preferred_total:
            parts.append(f"Met {preferred_met}/{preferred_total} preferred skills.")
        excluded = required_excluded + preferred_excluded
        if excluded:
            parts.append(
                f"{excluded} soft-skill requirement(s) (e.g. communication, teamwork) had no "
                f"explicit or inferred evidence in the resume; not counted as gaps or scored "
                f"against, since absence of a keyword doesn't confirm absence of the trait."
            )
        return " ".join(parts)