"""Regression tests: job_relevance()'s skill_hit component previously used
an unbounded substring test to decide whether a candidate's technology key
"relates to" a JD skill key -- the same class of bug already fixed in
SkillsMatcherAgent (see skills_matcher.py's word-boundary fix). Because
both candidate/JD keys are produced by the same skill_taxonomy
normalization (single space-separated tokens, no other punctuation), a
plain `k in s or s in k` (and, separately, `k in text_blob`) check could
not distinguish "java" being a whole word inside "javascript" (false
positive -- unrelated skills) from "algorithms" being a whole word inside
"data structures and algorithms" (a legitimate, taxonomy-recognized
multi-word/parent-child match).

Confirmed false-positive reproductions (BEFORE the fix, via the real
job_relevance() path, all using isolated single-technology/single-
required-skill WorkExperience/JobRequirements so the skill_hit
contribution is unambiguous):
    Java tech      vs "JavaScript" required -> credited as if exact match
    C tech         vs "C++" required        -> credited as if exact match
    Git tech       vs "GitHub" required     -> credited as if exact match
    "JavaScript Engineer" title (NO technologies at all) vs "Java"
        required -> credited via the separate text_blob substring check

Fix (src/agents/experience_eval.py): `_bounded_skill_match(a, b)` requires
exact equality OR the shorter normalized key to appear at a whitespace/
start/end-bounded position inside the longer one, and both the tech-key
comparison and the text_blob comparison were switched to use it. This
does NOT change the 0.45/0.40/0.15 job_relevance weights, role-family
logic, or project-relevance logic.
"""

from src.agents.experience_eval import job_relevance, _bounded_skill_match
from src.models import JobRequirements, WorkExperience


def _relevance(tech: str | None, title: str, req_title: str, req_skill: str) -> float:
    exp = WorkExperience(title=title, company="X", technologies=[tech] if tech else [])
    jd = JobRequirements(title=req_title, required_skills=[req_skill], preferred_skills=[])
    return job_relevance(exp, jd)


# ---------------------------------------------------------------------------
# _bounded_skill_match unit tests -- the core fix, in isolation
# ---------------------------------------------------------------------------

def test_bounded_match_rejects_java_javascript():
    assert _bounded_skill_match("java", "javascript") is False
    assert _bounded_skill_match("javascript", "java") is False


def test_bounded_match_rejects_c_cplusplus():
    assert _bounded_skill_match("c", "cplusplus") is False
    assert _bounded_skill_match("cplusplus", "c") is False


def test_bounded_match_rejects_git_github():
    assert _bounded_skill_match("git", "github") is False
    assert _bounded_skill_match("github", "git") is False


def test_bounded_match_accepts_exact_equal_keys():
    for key in ("java", "python", "cplusplus", "git", "github", "javascript"):
        assert _bounded_skill_match(key, key) is True


def test_bounded_match_accepts_legitimate_multiword_containment():
    # "algorithms" is a whole, space-bounded token inside the taxonomy's
    # multi-word canonical key for "Data Structures & Algorithms".
    assert _bounded_skill_match("algorithms", "data structures and algorithms") is True
    assert _bounded_skill_match("data structures and algorithms", "algorithms") is True


def test_bounded_match_rejects_empty_keys():
    assert _bounded_skill_match("", "java") is False
    assert _bounded_skill_match("java", "") is False


# ---------------------------------------------------------------------------
# End-to-end job_relevance(): false positives must be gone
# ---------------------------------------------------------------------------

def test_java_tech_does_not_get_javascript_credit():
    with_java = _relevance("Java", "Frontend Developer", "Frontend Engineer", "JavaScript")
    # Baseline: a technology with NO taxonomy relationship at all to the
    # requirement, same titles -- if Java got real JavaScript credit,
    # `with_java` would be strictly greater than this baseline.
    baseline = _relevance("Ruby", "Frontend Developer", "Frontend Engineer", "JavaScript")
    assert with_java == baseline, (
        f"Java should get the same (zero) skill credit as an unrelated "
        f"tech against a JavaScript requirement, got {with_java} vs baseline {baseline}"
    )


def test_c_tech_does_not_get_cplusplus_credit():
    with_c = _relevance("C", "Systems Developer", "Systems Engineer", "C++")
    baseline = _relevance("Ruby", "Systems Developer", "Systems Engineer", "C++")
    assert with_c == baseline, (
        f"C should get the same (zero) skill credit as an unrelated tech "
        f"against a C++ requirement, got {with_c} vs baseline {baseline}"
    )


def test_git_tech_does_not_get_github_credit():
    with_git = _relevance("Git", "Developer", "Developer", "GitHub")
    baseline = _relevance("Ruby", "Developer", "Developer", "GitHub")
    assert with_git == baseline, (
        f"Git should get the same (zero) skill credit as an unrelated tech "
        f"against a GitHub requirement, got {with_git} vs baseline {baseline}"
    )


def test_title_text_blob_does_not_leak_java_credit_from_javascript_title():
    """Isolates the SECOND substring check (text_blob), which has no
    `technologies` involved at all -- only the job title text feeds it."""
    exp = WorkExperience(title="JavaScript Engineer", company="X", technologies=[])
    jd_java = JobRequirements(title="Java Developer", required_skills=["Java"], preferred_skills=[])
    jd_unrelated = JobRequirements(title="Java Developer", required_skills=["Ruby"], preferred_skills=[])
    with_java_req = job_relevance(exp, jd_java)
    with_unrelated_req = job_relevance(exp, jd_unrelated)
    assert with_java_req == with_unrelated_req, (
        f"A 'JavaScript Engineer' title must not get skill credit toward a "
        f"bare 'Java' requirement, got {with_java_req} vs unrelated-requirement "
        f"baseline {with_unrelated_req}"
    )


# ---------------------------------------------------------------------------
# Legitimate matches must still work at full strength
# ---------------------------------------------------------------------------

def test_exact_java_still_matches_java():
    exact = _relevance("Java", "Developer", "Developer", "Java")
    unrelated = _relevance("Ruby", "Developer", "Developer", "Java")
    assert exact > unrelated


def test_exact_python_still_matches_python():
    exact = _relevance("Python", "Developer", "Developer", "Python")
    unrelated = _relevance("Ruby", "Developer", "Developer", "Python")
    assert exact > unrelated


def test_exact_cplusplus_still_matches_cplusplus():
    exact = _relevance("C++", "Developer", "Developer", "C++")
    unrelated = _relevance("Ruby", "Developer", "Developer", "C++")
    assert exact > unrelated


def test_alias_javascript_still_matches_js_requirement():
    """'JS' is a recognized taxonomy alias for JavaScript -- both
    normalize to the same canonical key, so this is an exact-key match,
    not a substring one, and must be unaffected by the fix."""
    exact = _relevance("JavaScript", "Developer", "Developer", "JS")
    unrelated = _relevance("Ruby", "Developer", "Developer", "JS")
    assert exact > unrelated


def test_legitimate_multiword_parent_child_skill_still_credited():
    """A candidate technology that IS a recognized multi-word taxonomy
    skill ('Data Structures & Algorithms') should still credit a
    requirement for one of its component words ('Algorithms') -- this is
    the legitimate containment case the bounded match must preserve."""
    with_dsa = _relevance(
        "Data Structures & Algorithms", "Developer", "Developer", "Algorithms"
    )
    without = _relevance("Ruby", "Developer", "Developer", "Algorithms")
    assert with_dsa > without
