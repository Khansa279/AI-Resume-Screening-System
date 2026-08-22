"""Base agent class with common LLM interaction logic."""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel

from ..config import get_config


T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)

_RETRY_AFTER_RE = re.compile(r"try again in\s+([\d.]+)\s*s", re.IGNORECASE)

# Safety cap so a provider-suggested wait (or our own backoff) never
# stalls a batch run indefinitely.
_MAX_RATE_LIMIT_WAIT_SECONDS = 90.0


class LLMCallError(Exception):
    """
    Raised when the underlying LLM API call itself fails -- network
    error, auth failure, rate limit, timeout, etc. -- as distinct from
    the call SUCCEEDING but returning content that isn't valid JSON.

    This distinction matters: previously, a failed API call and a
    malformed-but-real LLM response were both represented as the same
    plain string ("Error calling LLM: ...") flowing into
    _extract_json_from_response, which meant a 429 rate limit was
    treated identically to the LLM writing bad JSON -- and retried with
    a "please write valid JSON" corrective prompt, which obviously does
    nothing for a rate limit and just burns another request.
    """

    def __init__(self, message: str, *, is_rate_limit: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.is_rate_limit = is_rate_limit
        self.retry_after = retry_after


def _classify_llm_error(e: Exception) -> tuple[bool, float | None]:
    """
    Inspect an exception raised by the LLM client and determine:
      - is_rate_limit: does this look like a 429 / rate-limit error?
      - retry_after: how long did the provider say to wait, if it told us?

    Tries structured attributes first (some SDK exceptions expose
    response headers), then falls back to parsing the provider's message
    text, e.g. Groq's "Please try again in 29.6025s".
    """
    message = str(e)
    lowered = message.lower()
    is_rate_limit = (
        "429" in message
        or "rate_limit" in lowered
        or "rate limit" in lowered
        or getattr(e, "status_code", None) == 429
    )

    retry_after: float | None = None

    response = getattr(e, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers:
            value = headers.get("retry-after") or headers.get("Retry-After")
            if value:
                try:
                    retry_after = float(value)
                except (TypeError, ValueError):
                    retry_after = None

    if retry_after is None:
        match = _RETRY_AFTER_RE.search(message)
        if match:
            try:
                retry_after = float(match.group(1))
            except ValueError:
                retry_after = None

    return is_rate_limit, retry_after


def create_llm(temperature: float | None = None):
    """Create LLM instance based on config."""
    config = get_config()
    resolved_temperature = config.temperature if temperature is None else temperature

    if config.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.groq_model,
            api_key=config.groq_api_key,
            temperature=resolved_temperature,
            max_tokens=config.max_tokens,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            google_api_key=config.gemini_api_key,
            temperature=resolved_temperature,
            max_output_tokens=config.max_tokens,
        )


class BaseAgent(ABC):
    """
    Base class for all screening agents.

    Each agent has a clear responsibility and processes a specific part
    of the screening workflow. Agents communicate through structured data
    (Pydantic models) that flows through the shared state.
    """

    name: str = "BaseAgent"
    description: str = "Base agent class"

    # Optional per-agent-class override for LLM temperature. Leave as
    # None to use the global config.temperature default (appropriate for
    # agents like DecisionSynthesizerAgent whose LLM call only produces
    # free-text reasoning prose, where some phrasing variation is
    # harmless). Agents whose LLM output feeds numeric/categorical
    # judgments directly into match_score -- SkillsMatcherAgent's
    # overall_score and match_quality calls, ExperienceEvaluatorAgent's
    # experience_score/role_relevance -- set this to a low value (e.g.
    # 0.0) so identical input scores consistently across runs instead of
    # swinging based on LLM sampling noise.
    temperature: float | None = None

    def __init__(self, llm=None):
        """
        Initialize the agent with an optional LLM.

        LLM clients are created lazily on first use so deterministic agents
        (and skipped workflow nodes) do not require an API key or spend tokens.
        """
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = create_llm(temperature=self.temperature)
        return self._llm

    @abstractmethod
    async def process(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Process the current state and return updates.

        Args:
            state: Current workflow state

        Returns:
            Dictionary of state updates
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Raw LLM call -- raises LLMCallError on failure instead of ever
    # returning an error disguised as content. Everything else builds on
    # top of this.
    # ------------------------------------------------------------------

    async def _call_llm_raw_async(self, prompt: str) -> str:
        """
        Call the LLM and return its text content.

        Raises:
            LLMCallError: if the API call itself fails (network, auth,
                rate limit, etc). Callers that need to distinguish "the
                call failed" from "the call succeeded but the content is
                bad" should catch this explicitly rather than calling
                _call_llm_async (which swallows this into a string, kept
                only for backward compatibility -- see below).
        """
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            is_rate_limit, retry_after = _classify_llm_error(e)
            logger.error(
                f"[{self.name}] LLM call failed"
                f"{' (rate limit)' if is_rate_limit else ''}: {e}"
            )
            raise LLMCallError(str(e), is_rate_limit=is_rate_limit, retry_after=retry_after) from e

    def _call_llm(self, prompt: str) -> str:
        """
        Make a synchronous LLM call.

        Kept for backward compatibility with any caller that doesn't
        need JSON-vs-error handling: returns an "Error calling LLM: ..."
        string on failure rather than raising, same as before.
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"[{self.name}] LLM call failed: {e}")
            return f"Error calling LLM: {str(e)}"

    async def _call_llm_async(self, prompt: str) -> str:
        """
        Make an asynchronous LLM call.

        Kept for backward compatibility (e.g. DecisionSynthesizerAgent's
        free-text reasoning call, which isn't JSON and doesn't need
        rate-limit-aware retry). Returns an "Error calling LLM: ..."
        string on failure rather than raising -- callers that DO need to
        tell an API failure apart from bad-but-real content should use
        _call_llm_raw_async or _call_llm_for_json_async instead.
        """
        try:
            return await self._call_llm_raw_async(prompt)
        except LLMCallError as e:
            return f"Error calling LLM: {str(e)}"

    # ------------------------------------------------------------------
    # Rate-limit-aware call, used internally by _call_llm_for_json_async
    # ------------------------------------------------------------------

    async def _call_llm_with_rate_limit_retry(
        self, prompt: str, max_rate_limit_retries: int
    ) -> str | None:
        """
        Call the LLM, retrying specifically on rate-limit (429) errors by
        waiting out the provider's suggested cooldown (falling back to
        exponential backoff if none was given). Any other API failure
        (auth, network, etc.) is logged and NOT retried here.

        Returns:
            The response text on success, or None if the call never
            succeeded. None means "we got no LLM output at all" -- a
            fundamentally different situation from "the LLM responded
            with something that wasn't valid JSON".
        """
        backoff = 1.0
        for attempt in range(1, max_rate_limit_retries + 2):  # +1 for the initial attempt
            try:
                return await self._call_llm_raw_async(prompt)
            except LLMCallError as e:
                if e.is_rate_limit and attempt <= max_rate_limit_retries:
                    wait = e.retry_after if e.retry_after is not None else backoff
                    wait = min(wait + 0.5, _MAX_RATE_LIMIT_WAIT_SECONDS)  # small buffer + cap
                    logger.warning(
                        f"[{self.name}] Rate limit hit (retry {attempt}/"
                        f"{max_rate_limit_retries}); waiting {wait:.1f}s "
                        f"before retrying the same request."
                    )
                    await asyncio.sleep(wait)
                    backoff = min(backoff * 2, _MAX_RATE_LIMIT_WAIT_SECONDS)
                    continue

                # Not a rate limit, or retries exhausted -- give up here.
                # The rate-limit case specifically is still logged as an
                # error (not silently retried forever) so it's visible
                # in a batch run.
                if e.is_rate_limit:
                    logger.error(
                        f"[{self.name}] Rate limit persisted after "
                        f"{max_rate_limit_retries} retries; giving up on "
                        f"this call."
                    )
                return None
        return None

    # ------------------------------------------------------------------
    # JSON-expecting call: rate-limit/API-failure retry and
    # JSON-corrective retry are two separate, non-overlapping paths.
    # ------------------------------------------------------------------

    async def _call_llm_for_json_async(
        self,
        prompt: str,
        max_retries: int = 1,
        max_rate_limit_retries: int = 3,
    ) -> str:
        """
        Call the LLM expecting a JSON response.

        - If the API call itself fails (network, auth, rate limit), that
          is handled ENTIRELY by _call_llm_with_rate_limit_retry above --
          rate limits get retried with an actual wait; other failures do
          not consume a JSON-corrective retry attempt.
        - Only once we have an actual LLM-generated response do we check
          whether it parses as JSON, and only then do we retry with a
          corrective "please write valid JSON" prompt.

        Returns the raw response text (same contract as _call_llm_async)
        so existing _parse_response() implementations in each agent
        don't need to change. Returns "" if the API call never
        succeeded at all (distinct from a non-empty-but-unparseable
        response) -- _extract_json_from_response handles empty strings
        explicitly and callers' existing "failed to parse" fallback
        paths still apply.
        """
        response = await self._call_llm_with_rate_limit_retry(prompt, max_rate_limit_retries)
        if response is None:
            logger.error(
                f"[{self.name}] Giving up: LLM API call did not succeed "
                f"(see rate-limit/error logs above). No JSON-corrective "
                f"retry was attempted, since there was no LLM content to "
                f"correct."
            )
            return ""

        if self._extract_json_from_response(response) is not None:
            return response

        for attempt in range(1, max_retries + 1):
            logger.warning(
                f"[{self.name}] LLM response was not valid JSON "
                f"(corrective attempt {attempt}/{max_retries}); retrying "
                f"with a corrective prompt."
            )
            retry_prompt = (
                f"{prompt}\n\n"
                "IMPORTANT: Your previous response could not be parsed as "
                "JSON. Respond with ONLY the complete, valid JSON object -- "
                "no markdown code fences, no explanation before or after "
                "it, and make sure every bracket and brace is closed. If "
                "the object is long, keep individual field values concise "
                "so the full response fits."
            )
            response = await self._call_llm_with_rate_limit_retry(retry_prompt, max_rate_limit_retries)
            if response is None:
                logger.error(
                    f"[{self.name}] Giving up: LLM API call failed during "
                    f"JSON-corrective retry."
                )
                return ""
            if self._extract_json_from_response(response) is not None:
                return response

        logger.error(
            f"[{self.name}] LLM response still not valid JSON after "
            f"{max_retries} corrective retry attempt(s); caller will fall "
            f"back to its default result. Last raw response (first 1000 "
            f"chars): {response[:1000]!r}"
        )
        return response

    # ------------------------------------------------------------------
    # JSON extraction / repair (unchanged from the previous fix)
    # ------------------------------------------------------------------

    def _parse_json_response(self, response: str, model_class: type[T]) -> T | None:
        """Parse a JSON response from the LLM into a Pydantic model."""
        data = self._extract_json_from_response(response)
        if data is None:
            return None
        try:
            return model_class.model_validate(data)
        except ValueError as e:
            logger.warning(f"[{self.name}] Failed to validate parsed JSON into {model_class.__name__}: {e}")
            return None

    def _extract_json_from_response(self, response: str) -> dict | None:
        """
        Extract a JSON dictionary from an LLM response.

        Handles, in order:
          1. Markdown code fences (```json ... ``` or ``` ... ```)
          2. Stray prose before/after the JSON object (isolates the
             outermost {...} span)
          3. JSON truncated mid-response because the LLM hit its token
             limit -- attempts a structural repair by trimming back to
             the last complete element and closing whatever
             brackets/braces were left open.

        Every failure path logs why it failed instead of silently
        returning None.
        """
        if not response or not response.strip():
            logger.warning(f"[{self.name}] Empty LLM response -- cannot extract JSON")
            return None

        candidate = self._strip_code_fences(response)

        data = self._try_parse_json(candidate)
        if data is not None:
            return data

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            span = candidate[start:end + 1]

            data = self._try_parse_json(span)
            if data is not None:
                return data

            repaired = self._repair_truncated_json(candidate[start:])
            data = self._try_parse_json(repaired)
            if data is not None:
                logger.warning(
                    f"[{self.name}] LLM response appears to have been "
                    f"truncated mid-JSON; recovered via structural repair "
                    f"(raw response length={len(response)} chars)."
                )
                return data

        logger.error(
            f"[{self.name}] Failed to parse JSON response after all "
            f"fallbacks. Raw response (first 1000 chars): "
            f"{response[:1000]!r}"
        )
        return None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` wrapping if present."""
        if "```json" in text:
            start = text.find("```json") + len("```json")
            end = text.find("```", start)
            return text[start:end if end != -1 else None].strip()
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end if end != -1 else None].strip()
        return text.strip()

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """
        Best-effort repair for JSON truncated because the LLM hit its
        token limit mid-response.

        Pass 1: walk the string tracking bracket depth (skipping over
        string literal contents) to find the index right after the LAST
        fully-closed '}' or ']' -- the last known-good structural point.
        Truncate there.

        Pass 2: on that truncated prefix, replay the same tracking to
        find which brackets are still open, and append matching closers.
        """
        def track(s: str):
            stack: list[str] = []
            in_string = False
            escape = False
            for idx, ch in enumerate(s):
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    yield idx, ch, in_string, stack
                    continue
                if ch == '"':
                    in_string = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if stack:
                        stack.pop()
                yield idx, ch, in_string, stack

        last_safe_end = 0
        for idx, ch, in_string, _stack in track(text):
            if ch in "}]" and not in_string:
                last_safe_end = idx + 1

        truncated = text[:last_safe_end] if last_safe_end else text

        reopen: list[str] = []
        for _idx, _ch, _in_string, stack in track(truncated):
            reopen = stack  # final stack state after walking the (safe) prefix

        trimmed = truncated.rstrip()
        if trimmed.endswith(","):
            trimmed = trimmed[:-1]

        closers = {"{": "}", "[": "]"}
        suffix = "".join(closers[b] for b in reversed(reopen))
        return trimmed + suffix

    def _build_system_prompt(self) -> str:
        """Build the system prompt for this agent."""
        return f"""You are {self.name}, a specialized AI agent in a resume screening system.
Your role: {self.description}

IMPORTANT GUIDELINES:
1. Always respond with valid JSON as specified in the prompt
2. Be thorough but concise in your analysis
3. When uncertain, indicate low confidence rather than guessing
4. Focus only on your specific task - other agents handle other aspects
5. Provide reasoning for your conclusions

Remember: Your output will be used by other agents in the pipeline, so accuracy is crucial."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"