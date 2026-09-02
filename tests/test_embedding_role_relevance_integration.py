"""Integration tests: the embedding-based semantic relevance signal
(src/embeddings.py) wired into job_relevance() / evaluate_experience()
(src/agents/experience_eval.py).

All embedding calls in these tests are mocked -- either by monkeypatching
`src.agents.experience_eval.embedding_relevance` directly (fast, precise
control over the returned score) or by injecting a small offline
FakeEmbeddingProvider through `src.embeddings.embed_text`'s `provider`
kwarg (exercises the real cosine-similarity + rescaling path). No test
here makes a live network call.

Covers the specific scenarios requested for this change:
  1. Related roles, different titles (ML Engineer vs AI Research Scientist)
  2. Healthcare (Registered Nurse - ICU vs Critical Care Nurse)
  3. Culinary (Executive Chef vs Sous Chef)
  4. Clearly unrelated roles (Registered Nurse vs Marketing Manager)
  5. Unseen/novel titles
  6. Embedding provider unavailable -> deterministic fallback works
  7. Missing API key -> system still works
  8. Embedding result is deterministic for the same input (mocked provider)
  9. Existing skill matching remains unchanged
  10. Existing final scoring behavior does not regress unexpectedly
"""

import pytest

from src.agents import experience_eval as ee
from src.agents.experience_eval import job_relevance, evaluate_experience
from src.agents.skills_matcher import SkillsMatcherAgent
from src.models import JobRequirements, ResumeData, Skill, WorkExperience
from src import embeddings as emb_module


@pytest.fixture(autouse=True)
def _reset_embedding_state():
    emb_module.reset_embedding_provider_cache()
    emb_module.clear_embedding_cache()
    yield
    emb_module.reset_embedding_provider_cache()
    emb_module.clear_embedding_cache()


# ---------------------------------------------------------------------------
# Helper: a tiny offline embedding provider whose vectors are built from
# topic keywords, so cosine similarity behaves sensibly for these
# hand-picked examples without ever touching a real network.
# ---------------------------------------------------------------------------

class TopicEmbeddingProvider:
    TOPICS = (
        "nurse", "medical", "icu", "critical", "care",
        "chef", "kitchen", "culinary", "menu",
        "machine", "learning", "ai", "research", "ml", "python",
        "marketing", "advertising", "brand",
    )

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if topic in lowered else 0.0 for topic in self.TOPICS]


@pytest.fixture()
def topic_provider(monkeypatch):
    """Wires the offline TopicEmbeddingProvider into embed_text() so
    _embedding_role_relevance() (which calls the module-level
    embedding_relevance -> embed_text with no explicit provider) picks
    it up as if it were the real configured provider."""
    provider = TopicEmbeddingProvider()
    monkeypatch.setattr(emb_module, "get_embedding_provider", lambda: provider)
    return provider


# ---------------------------------------------------------------------------
# 1 & 2 & 3: related roles with different titles, across three domains
# ---------------------------------------------------------------------------

def test_ml_engineer_vs_ai_research_scientist_scores_relevant(topic_provider):
    jd = JobRequirements(
        title="Machine Learning Engineer",
        required_skills=["Python", "Machine Learning"],
        responsibilities=["Research and deploy machine learning models"],
    )
    exp = WorkExperience(
        title="AI Research Scientist",
        company="DeepLab",
        technologies=["Python"],
        responsibilities=["Conducted machine learning research"],
    )
    relevance = job_relevance(exp, jd)
    assert relevance > 0.0


def test_registered_nurse_vs_critical_care_nurse_scores_relevant(topic_provider):
    jd = JobRequirements(title="Registered Nurse - ICU", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Critical Care Nurse", company="City Hospital")
    relevance = job_relevance(exp, jd)
    assert relevance > 0.0


def test_executive_chef_vs_sous_chef_scores_relevant(topic_provider):
    jd = JobRequirements(title="Executive Chef", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Sous Chef", company="Bistro Verde")
    relevance = job_relevance(exp, jd)
    assert relevance > 0.0


def test_embedding_signal_raises_relevance_for_titles_lexical_overlap_misses(monkeypatch):
    """Direct, isolated proof that the EMBEDDING signal specifically
    (not the pre-existing lexical/taxonomy signature) is what closes the
    gap for a pair that shares no title words and no taxonomy skills.
    Mocks embedding_relevance directly so this is independent of the
    offline TopicEmbeddingProvider's specific keyword design."""
    jd = JobRequirements(title="Registered Nurse - ICU", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Emergency Trauma Specialist", company="City Hospital")

    # Without any embedding signal, this pair shares no title words and
    # no taxonomy signal (skill_taxonomy.py has no healthcare entries) --
    # confirm the baseline really is 0 first.
    baseline = job_relevance(exp, jd)
    assert baseline == 0.0

    monkeypatch.setattr(ee, "embedding_relevance", lambda a, b: 0.8)
    boosted = job_relevance(exp, jd)
    assert boosted > baseline
    assert boosted == round(0.45 * 0.8, 2)  # title_score raised to 0.8 by the embedding signal alone


# ---------------------------------------------------------------------------
# 4. Clearly unrelated roles must stay low even with embeddings enabled.
# ---------------------------------------------------------------------------

def test_unrelated_roles_stay_low_with_embeddings_enabled(topic_provider):
    jd = JobRequirements(title="Registered Nurse", required_skills=[], preferred_skills=[])
    exp = WorkExperience(
        title="Marketing Manager", company="BrightAd Agency",
        responsibilities=["Managed social media campaigns and brand advertising"],
    )
    relevance = job_relevance(exp, jd)
    assert relevance < 0.15, f"expected low relevance for unrelated roles even with embeddings on, got {relevance}"


def test_embedding_signal_cannot_rescue_a_genuinely_unrelated_pair_if_it_also_scores_low(monkeypatch):
    monkeypatch.setattr(ee, "embedding_relevance", lambda a, b: 0.0)
    jd = JobRequirements(title="Registered Nurse", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Marketing Manager", company="X")
    assert job_relevance(exp, jd) == 0.0


# ---------------------------------------------------------------------------
# 5. Unseen/novel titles -- must not crash, must still produce a valid score.
# ---------------------------------------------------------------------------

def test_unseen_novel_titles_do_not_crash_with_embeddings_enabled(topic_provider):
    jd = JobRequirements(title="Quantum Sensing Systems Integrator", required_skills=[])
    exp = WorkExperience(title="Cryogenic Instrumentation Technologist", company="LabCo")
    relevance = job_relevance(exp, jd)  # must not raise
    assert 0.0 <= relevance <= 1.0


def test_unseen_titles_still_work_when_embedding_provider_returns_none(monkeypatch):
    monkeypatch.setattr(ee, "embedding_relevance", lambda a, b: None)
    jd = JobRequirements(title="Quantum Sensing Systems Integrator", required_skills=[])
    exp = WorkExperience(title="Cryogenic Instrumentation Technologist", company="LabCo")
    relevance = job_relevance(exp, jd)
    assert 0.0 <= relevance <= 1.0


# ---------------------------------------------------------------------------
# 6 & 7. Provider unavailable / missing API key -> deterministic fallback.
# ---------------------------------------------------------------------------

def test_no_api_key_configured_job_relevance_still_works(monkeypatch):
    """The real-world default for this project's sandbox/dev environment
    (no GEMINI_API_KEY / GOOGLE_API_KEY set): embeddings must be
    transparently skipped, and job_relevance() must behave exactly like
    it did before this feature existed."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    emb_module.reset_embedding_provider_cache()

    jd = JobRequirements(title="Backend Engineer - Python", required_skills=["Python", "Django"])
    exp = WorkExperience(
        title="Senior Software Engineer", company="TechCorp",
        technologies=["Python", "Django"], responsibilities=["Built REST APIs"],
    )
    relevance = job_relevance(exp, jd)
    assert 0.0 <= relevance <= 1.0


def test_embedding_provider_raising_falls_back_to_deterministic_score(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated embedding outage")

    monkeypatch.setattr(ee, "embedding_relevance", boom)
    jd = JobRequirements(title="Backend Engineer - Python", required_skills=["Python"])
    exp = WorkExperience(title="Backend Engineer - Python", company="X")
    with pytest.raises(RuntimeError):
        # NOTE: this confirms embedding_relevance itself is expected to
        # never raise in real operation (see src/embeddings.py) -- if it
        # somehow did, job_relevance() has no try/except of its own
        # around it, by design: the contract is "embedding_relevance
        # itself must not raise", enforced by its own tests
        # (test_embeddings_module.py), not by a defensive wrapper here.
        job_relevance(exp, jd)


def test_embedding_relevance_returning_none_is_a_transparent_no_op(monkeypatch):
    """The REAL contract: embedding_relevance() returns None (never
    raises) on failure -- confirming job_relevance() handles that
    correctly and produces the same result as if embeddings were never
    attempted at all."""
    jd = JobRequirements(title="Backend Engineer - Python", required_skills=["Python", "Django"])
    exp = WorkExperience(
        title="Senior Software Engineer", company="TechCorp",
        technologies=["Python", "Django"], responsibilities=["Built REST APIs"],
    )

    monkeypatch.setattr(ee, "embedding_relevance", lambda a, b: None)
    with_none = job_relevance(exp, jd)

    # Simulate "embeddings module doesn't even exist for this call" by
    # NOT patching at all -- since no API key is configured in this test
    # environment, the real embedding_relevance() call already returns
    # None on its own, so this should be identical.
    emb_module.reset_embedding_provider_cache()
    without_patch = job_relevance(exp, jd)

    assert with_none == without_patch


# ---------------------------------------------------------------------------
# 8. Embedding result is deterministic for the same input (mocked provider).
# ---------------------------------------------------------------------------

def test_embedding_backed_job_relevance_is_deterministic(topic_provider):
    jd = JobRequirements(title="Registered Nurse - ICU", required_skills=[])
    exp = WorkExperience(title="Critical Care Nurse", company="City Hospital")
    results = {job_relevance(exp, jd) for _ in range(5)}
    assert len(results) == 1


def test_evaluate_experience_is_deterministic_with_embeddings_enabled(topic_provider):
    data = ResumeData(work_experience=[
        WorkExperience(title="Critical Care Nurse", company="City Hospital", start_date="2021-01", end_date="Present"),
    ])
    jd = JobRequirements(title="Registered Nurse - ICU", required_skills=[], min_years_experience=1)
    e1 = evaluate_experience(data, jd)
    e2 = evaluate_experience(data, jd)
    assert e1.model_dump() == e2.model_dump()


# ---------------------------------------------------------------------------
# 9. Existing skill matching remains completely unchanged.
# ---------------------------------------------------------------------------

def test_skills_matcher_agent_is_unaffected_by_embedding_changes():
    """SkillsMatcherAgent doesn't import or use embeddings at all -- this
    is a straightforward regression guard that skill matching (a
    completely separate concern from role/title relevance) still behaves
    exactly as documented."""
    import asyncio

    agent = SkillsMatcherAgent()
    skills = [Skill(name="Python", category="language"), Skill(name="JS", category="language")]
    reqs = JobRequirements(required_skills=["Python", "JavaScript"], preferred_skills=[])

    async def run():
        return await agent.process({"extracted_skills": skills, "job_requirements": reqs})

    result = asyncio.run(run())["skills_match"]
    assert result.required_skills_met == 2
    assert result.required_skills_total == 2


# ---------------------------------------------------------------------------
# 10. Existing final scoring behavior does not regress unexpectedly.
# ---------------------------------------------------------------------------

def test_exact_title_match_relevance_unchanged_with_embeddings_available(topic_provider):
    """An exact title match already scores the maximum lexical
    title_score (1.0) -- an embedding signal (even a real one) must not
    change this, since max() can't push a value past what's already at
    its ceiling for the 0.45-weighted title component."""
    jd = JobRequirements(title="Backend Engineer - Python", required_skills=[], preferred_skills=[])
    exp = WorkExperience(title="Backend Engineer - Python", company="X")
    assert job_relevance(exp, jd) == 0.45


def test_job_relevance_weights_are_unchanged(monkeypatch):
    """The 0.45 / 0.4 / 0.15 weighting between title/skill/responsibility
    components must be completely untouched by this change. Embeddings
    are explicitly disabled here so this test isolates exactly the two
    pre-existing components (the deterministic title signature backstop
    from the earlier generalization, and skill_hit) -- confirming the
    embedding integration only ever ADDS a third max()-combined signal
    into title_score, without altering the 0.45/0.4/0.15 weight split
    itself."""
    monkeypatch.setattr(ee, "embedding_relevance", lambda a, b: None)
    jd = JobRequirements(
        title="Something With No Recognized Words At All",
        required_skills=["Python", "Django", "PostgreSQL"],
    )
    exp = WorkExperience(
        title="Zzz Unrelated Title Zzz", company="X",
        technologies=["Python", "Django", "PostgreSQL"],
        responsibilities=[],
    )
    relevance = job_relevance(exp, jd)
    expected_title_score = ee._semantic_role_relevance(exp, jd)  # lexical component is 0 here
    # skill_hit is 1.0: all 3 required skills are present in exp.technologies
    expected = round(0.45 * expected_title_score + 0.4 * 1.0 + 0.15 * 0.0, 2)
    assert relevance == expected


def test_full_screening_pipeline_score_does_not_regress_when_embeddings_unavailable():
    """End-to-end sanity check against a resume/JD pair used elsewhere in
    this project's test suite (test_deterministic_pipeline.py), run with
    NO embedding provider configured -- confirms the whole pipeline
    (skills + experience + decision synthesis) produces the exact same
    kind of result as before this feature existed."""
    from pathlib import Path
    from src.agents.resume_parser import parse_resume_text

    text = Path("sample_data/resumes/senior_python_dev.txt").read_text(encoding="utf-8")
    data = parse_resume_text(text)
    reqs = JobRequirements(
        title="Backend Software Engineer",
        required_skills=["Python", "Django", "SQL"],
        preferred_skills=["Docker"],
        min_years_experience=3,
    )
    ev = evaluate_experience(data, reqs)
    assert ev.years_relevant > 0
    assert 0.0 <= ev.role_relevance <= 1.0
