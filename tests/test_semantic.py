"""Tests for semantic matching.

Every test degrades gracefully: the encoder is an optional dependency, so the
suite must pass whether or not sentence-transformers is installed.
"""
import pytest

import semantic
from requirements import score_requirements
from skills import analyse_skills


@pytest.fixture(autouse=True)
def _enable_encoder(monkeypatch):
    """Semantic matching is opt-in; these tests opt in for themselves."""
    monkeypatch.setenv(semantic.ENV_FLAG, "1")

# Gate on whether the LIBRARY is installed, not on `semantic.is_available()` —
# that now also consults the opt-in env flag, which the autouse fixture below
# sets at test-setup time, after skipif conditions are evaluated. Checking
# availability here skipped every test even with the encoder present.
requires_encoder = pytest.mark.skipif(
    "not semantic._encoder_installed()", reason="sentence-transformers not installed")


def test_module_never_raises_without_the_dependency():
    """The whole point of the optional design: callers get None, not an error."""
    assert semantic.similarity("a", "b") is None or 0.0 <= semantic.similarity("a", "b") <= 1.0
    assert semantic.best_match("x", []) is None


def test_rescale_maps_the_useful_band_onto_0_1():
    assert semantic.rescale(0.20) == 0.0     # below floor -> nothing
    assert semantic.rescale(0.80) == 1.0     # above ceiling -> full
    assert 0.4 < semantic.rescale(0.50) < 0.6
    assert semantic.rescale(0.5, floor=0.5, ceiling=0.5) == 0.5   # degenerate


@requires_encoder
def test_paraphrases_score_higher_than_unrelated_text():
    """The capability lexical matching cannot provide."""
    a = "experience training machine learning models"
    paraphrase = "trained LightGBM models and evaluated them with AUC-ROC"
    unrelated = "managed office catering and travel booking"
    assert semantic.similarity(a, paraphrase) > semantic.similarity(a, unrelated)


@requires_encoder
def test_low_token_overlap_paraphrase_is_still_matched():
    req = "Comfortable owning services in production, including on-call"
    bullets = ["Handled L3 production incidents, debugging live issues and on-call rotations",
               "Designed a React dashboard for internal users"]
    idx, score = semantic.best_match(req, bullets)
    assert idx == 0
    assert score > 0.3


def test_non_concrete_requirements_are_filtered_before_matching():
    """Documents a real tension between two fixes in this project.

    `extract_requirements` keeps only lines naming a skill, a year count, or a
    degree — added because scoring every line produced "4 of 25 addressed" on a
    real posting, where the misses were job duties and corporate boilerplate.

    But the semantic matcher exists to score requirements that name NO skill.
    The filter removes exactly its input, so the "meaning" branch in
    requirements.py is currently reachable only for degree-style lines.

    Encoder-level paraphrase matching still works and is covered by
    test_low_token_overlap_paraphrase_is_still_matched; it is the
    requirement-level plumbing that is mostly unreachable. Resolving this means
    choosing which failure to accept — noise from unscoreable duties, or
    blindness to real non-skill requirements — and that needs the golden set to
    decide, not a guess.
    """
    from requirements import extract_requirements

    jd = ("Requirements:\n"
          "- Able to explain tradeoffs to non-technical partners\n"
          "- Strong Python\n"
          "- 5+ years of experience\n")
    kept = extract_requirements(jd)
    # "Strong Python" is two words; it must survive because it names a skill.
    assert any("Python" in r for r in kept), f"short skill requirement dropped: {kept}"
    assert any("years" in r for r in kept)
    assert not any("tradeoffs" in r for r in kept), (
        "non-concrete requirement unexpectedly kept — if this now passes, the "
        "filter was relaxed and the semantic branch is live again")


@requires_encoder
def test_encoder_results_are_deterministic():
    a = semantic.similarity("Python engineer", "software developer using Python")
    b = semantic.similarity("Python engineer", "software developer using Python")
    assert a == b


def test_llm_availability_requires_actual_credentials(monkeypatch):
    """Regression: `anthropic.Anthropic()` constructs fine with no credentials
    and only fails at request time, so an availability check based on
    construction alone reported True on a machine that could not make a call."""
    import llm_matcher

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(llm_matcher, "_client", lambda: object())
    monkeypatch.setattr(llm_matcher, "_ant_profile_active", lambda: False)
    assert llm_matcher.is_available() is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert llm_matcher.is_available() is True


def test_llm_matcher_degrades_without_credentials(monkeypatch):
    import llm_matcher

    monkeypatch.setattr(llm_matcher, "_client", lambda: None)
    assert llm_matcher.match("resume text", "job text") is None
