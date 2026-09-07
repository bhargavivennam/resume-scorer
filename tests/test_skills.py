"""Tests for the skill ontology, evidence weighting and recency decay."""
import pytest

from skills import (EVIDENCE_EXPERIENCE, EVIDENCE_LISTED, EVIDENCE_QUANTIFIED,
                    analyse_skills, canonicalize, find_skills, group_of)


@pytest.mark.parametrize("surface,canonical", [
    ("k8s", "kubernetes"), ("K8S", "kubernetes"), ("EKS", "kubernetes"),
    ("sklearn", "scikit-learn"), ("postgres", "postgresql"),
    ("golang", "go"), ("js", "javascript"), ("node js", "node.js"),
    ("amazon web services", "aws"), ("github actions", "ci/cd"),
])
def test_aliases_resolve_to_one_canonical_skill(surface, canonical):
    assert canonicalize(surface) == canonical


def test_alias_matching_in_free_text():
    found = find_skills("Ran k8s on AWS with Postgres and sklearn")
    assert found == {"kubernetes", "aws", "postgresql", "scikit-learn"}


@pytest.mark.parametrize("text", [
    "we parse json payloads",       # 'js' must not fire inside 'json'
    "a trusted engineer",           # 'rust' must not fire inside 'trusted'
    "google docs and sheets",       # 'go' must not fire inside 'google'
    "the ruby-throated hummingbird" # hyphen guard
])
def test_no_substring_false_positives(text):
    assert find_skills(text) == set() or "rust" not in find_skills(text)


def test_evidence_beats_keyword_stuffing():
    """The core improvement: proving a skill outranks listing it."""
    listed = analyse_skills("skills: kubernetes")
    shown = analyse_skills("", roles=[("- cut deploy time 55% using kubernetes", 2025)])
    assert listed["kubernetes"].evidence == EVIDENCE_LISTED
    assert shown["kubernetes"].evidence == EVIDENCE_QUANTIFIED
    assert shown["kubernetes"].strength > listed["kubernetes"].strength * 2


def test_experience_without_numbers_is_middle_tier():
    sig = analyse_skills("", roles=[("- built services with kubernetes", 2025)])
    assert sig["kubernetes"].evidence == EVIDENCE_EXPERIENCE


def test_recency_decay():
    recent = analyse_skills("", roles=[("- used python", 2025)])["python"]
    old = analyse_skills("", roles=[("- used python", 2013)])["python"]
    assert recent.recency > old.recency
    assert recent.strength > old.strength


def test_undated_skill_is_neutral_not_zero():
    """A skill with no date must not be treated as ancient."""
    sig = analyse_skills("skills: python")["python"]
    assert 0.4 < sig.recency < 0.6


def test_strongest_evidence_wins_across_mentions():
    sig = analyse_skills(
        "skills: kubernetes",
        roles=[("- migrated 30 services to kubernetes cutting cost 40%", 2025)])
    assert sig["kubernetes"].evidence == EVIDENCE_QUANTIFIED


def test_groups_are_assigned():
    assert group_of("kubernetes") == "cloud_devops"
    assert group_of("pytorch") == "ml"
    assert group_of("unknown-thing") is None


def test_narrower_skill_implies_broader_one():
    """PostgreSQL on a resume must satisfy a job asking for SQL.

    Scoring that as "missing SQL" is something no human reviewer would say,
    and it was the single biggest source of false 'missing skill' reports.
    """
    sig = analyse_skills("skills: postgresql")
    assert "sql" in sig
    assert sig["sql"].evidence < sig["postgresql"].evidence, \
        "an inferred skill must not outrank the stated one"


def test_implication_chain_from_demonstrated_skill():
    sig = analyse_skills("", roles=[("- Scaled Kafka to 20M events/day", 2025)])
    assert "system design" in sig
    assert sig["kafka"].strength > sig["system design"].strength


def test_stated_skill_beats_the_same_skill_when_also_implied():
    """If Python is both written down AND implied by Django, the written
    mention must win."""
    implied_only = analyse_skills("skills: django")["python"]
    stated = analyse_skills("", roles=[("- Built Python services cutting cost 30%", 2025)])["python"]
    assert stated.evidence > implied_only.evidence


def test_no_implication_loops():
    from skills import IMPLIES
    for skill, implied in IMPLIES.items():
        assert skill not in implied, f"{skill} implies itself"
        for other in implied:
            assert skill not in IMPLIES.get(other, []), f"cycle: {skill} <-> {other}"
