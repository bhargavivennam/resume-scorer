"""Tests for requirement-level matching and the evaluation harness."""
import json

import pytest

from evaluate import load_golden, pairwise_agreement
from features import parse_roles
from requirements import extract_requirements, score_requirements
from skills import analyse_skills

JD = """Senior Backend Engineer

About us
We are a fast growing company with great benefits and health insurance.

Requirements:
- 6+ years of backend engineering experience
- Strong Python and Go
- Experience mentoring junior engineers and leading code reviews
- Track record designing scalable, reliable APIs
- Comfortable owning services in production, including on-call
- Kubernetes and AWS

Benefits:
- Competitive salary and 401k
- Paid time off and dental
"""

RESUME = """Alex Chen
EXPERIENCE
Staff Engineer - Northwind
Jan 2016 - Present
- Led migration of 30 microservices to Kubernetes on AWS, cutting deploy 55%
- Mentored 6 junior engineers and ran weekly code reviews
SKILLS
Python, Go, Kubernetes, AWS"""


def _signals(text):
    roles = parse_roles(text)
    return analyse_skills(text.lower(), roles=[(r.lower(), y) for r, y in roles])


def test_extracts_only_real_requirements():
    reqs = extract_requirements(JD)
    joined = " ".join(reqs).lower()
    assert any("mentoring" in r.lower() for r in reqs)
    # Boilerplate must not be scored as a requirement.
    assert "401k" not in joined and "dental" not in joined
    assert "health insurance" not in joined


def test_non_skill_requirements_are_scored_not_ignored():
    """The core gap this module closes: most requirements name no skill."""
    r = score_requirements(RESUME, JD, _signals(RESUME), resume_years=9.0)
    mentoring = next(x for x in r["requirements"] if "mentoring" in x["requirement"].lower())
    assert mentoring["verdict"] != "unaddressed", \
        "resume explicitly mentions mentoring engineers and code reviews"


def test_years_requirement_uses_the_timeline_not_word_overlap():
    """Regression: '6+ years of backend engineering' scored 0.15 for a
    candidate with 8 years, because it went through lexical matching."""
    r = score_requirements(RESUME, JD, _signals(RESUME), resume_years=9.0)
    yrs = next(x for x in r["requirements"] if "years" in x["requirement"].lower())
    assert yrs["basis"] == "tenure"
    assert yrs["score"] == 1.0

    short = score_requirements(RESUME, JD, _signals(RESUME), resume_years=2.0)
    yrs_short = next(x for x in short["requirements"] if "years" in x["requirement"].lower())
    assert yrs_short["score"] < 0.5


def test_unaddressed_requirements_are_reported():
    r = score_requirements(RESUME, JD, _signals(RESUME), resume_years=9.0)
    assert any("on-call" in u.lower() or "production" in u.lower()
               for u in r["unaddressed"]) or r["n_addressed"] >= 4


def test_requirements_sorted_worst_first():
    r = score_requirements(RESUME, JD, _signals(RESUME), resume_years=9.0)
    scores = [x["score"] for x in r["requirements"]]
    assert scores == sorted(scores)


def test_unstructured_posting_falls_back_to_sentences():
    prose = ("We need someone who can build scalable APIs in Python. "
             "You should have experience with Kubernetes and cloud platforms. "
             "Mentoring junior developers is a big part of this role.")
    assert len(extract_requirements(prose)) >= 2


def test_empty_posting_returns_no_coverage():
    r = score_requirements(RESUME, "", _signals(RESUME))
    assert r["coverage"] is None and r["requirements"] == []


# ---------------------- evaluation harness -------------------------------

def test_pairwise_agreement_detects_a_correct_scorer():
    pairs = [("good", "bad"), ("great", "awful")]
    good = pairwise_agreement(pairs, lambda t: len(t))       # longer == better here
    assert good["agreement"] == 1.0
    inverted = pairwise_agreement(pairs, lambda t: -len(t))
    assert inverted["agreement"] == 0.0


def test_pairwise_agreement_counts_ties_separately():
    r = pairwise_agreement([("a", "b")], lambda t: 1.0)
    assert r["ties"] == 1 and r["agreement"] is None


def test_golden_loader_expands_grades_into_pairs(tmp_path):
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"better": "A", "worse": "B"},
        {"text": "C", "grade": 4},
        {"text": "D", "grade": 1},
    ]))
    pairs, graded = load_golden(p)
    assert ("A", "B") in pairs
    assert ("C", "D") in pairs      # derived from the grades
    assert len(graded) == 2


def test_golden_loader_handles_missing_file(tmp_path):
    pairs, graded = load_golden(tmp_path / "nope.jsonl")
    assert pairs == [] and graded == []
