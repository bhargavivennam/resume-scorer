"""Tests for the rule-based quality rubric.

The rubric replaced the ML model as the headline quality score because real
resumes fall outside the model's training distribution. These tests pin the
properties that make it trustworthy: it must work on long real-world resumes,
reward content over structure, and always hand back an actionable fix.
"""
import pytest

from rubric import score_rubric

STRONG = """Priya Sharma
priya.sharma@example.com | +1 (555) 442-1180 | linkedin.com/in/priyasharma

SUMMARY
Senior Backend Engineer with 8 years designing distributed systems.

EXPERIENCE
Senior Software Engineer - Vertex Analytics
Jun 2021 - Present
- Led migration of 24 microservices to Kubernetes, cutting deploy time 55%
- Designed a Kafka pipeline processing 18M events/day at 99.95% uptime
- Reduced PostgreSQL p95 latency from 740ms to 85ms by rewriting query plans
- Automated CI/CD with Terraform, cutting release cycle 60%

Software Engineer - Hexbyte
Aug 2017 - May 2021
- Built Spring Boot REST APIs serving 4M requests/day across 3 regions
- Migrated batch ETL to Spark, cutting nightly runtime from 6h to 40m

SKILLS
Java, Python, Go, SQL, PostgreSQL, Kafka, Spark, AWS, Docker, Kubernetes

EDUCATION
M.S. Computer Science, State University, 2015 - 2017"""

WEAK = """Sam Doyle

EXPERIENCE
Developer - Acme Corp
2021 - 2022
- Responsible for writing code and fixing bugs
- Worked on various projects using different technologies

SKILLS
MS Office, Email, Teamwork"""


def test_strong_beats_weak_by_a_wide_margin():
    assert score_rubric(STRONG)["quality_score"] - score_rubric(WEAK)["quality_score"] > 40


def test_score_is_bounded_and_verdict_matches():
    for text in (STRONG, WEAK, "x", ""):
        r = score_rubric(text)
        assert 0 <= r["quality_score"] <= 100
        assert r["verdict"] in {"Strong", "Solid", "Needs work", "Weak"}


def test_works_on_a_long_real_world_resume():
    """The exact case the ML model could not handle — 0% of its training data
    reached this length, so it had no basis for a prediction."""
    long_resume = STRONG + "\n" + "\n".join(
        f"- Delivered internal platform project {i}, cutting processing time by "
        f"{20 + i}% and saving roughly ${i + 5}k per quarter across teams"
        for i in range(45))
    assert len(long_resume.split()) > 600, "the point of this test is a LONG resume"
    r = score_rubric(long_resume)
    assert r["quality_score"] >= 60


def test_quantification_is_the_heaviest_lever():
    stripped = (STRONG.replace("55%", "significantly").replace("18M", "many")
                .replace("from 740ms to 85ms", "considerably").replace("60%", "a lot")
                .replace("4M", "many").replace("from 6h to 40m", "less time")
                .replace("99.95%", "high").replace("24 microservices", "microservices")
                .replace("3 regions", "several regions"))
    assert score_rubric(STRONG)["quality_score"] - score_rubric(stripped)["quality_score"] >= 15


def test_adding_a_section_matters_less_than_adding_results():
    """Content must outweigh structure — the failure the ML model had."""
    with_section = STRONG + "\n\nCERTIFICATIONS\n- AWS Certified Solutions Architect"
    section_gain = score_rubric(with_section)["quality_score"] - score_rubric(STRONG)["quality_score"]
    unquantified = STRONG.replace("55%", "a lot").replace("18M", "many").replace("60%", "a lot")
    content_loss = score_rubric(STRONG)["quality_score"] - score_rubric(unquantified)["quality_score"]
    assert content_loss > section_gain


def test_every_dimension_is_actionable_and_weighted():
    r = score_rubric(STRONG)
    assert abs(sum(d["weight"] for d in r["dimensions"]) - 1.0) < 1e-6
    for d in r["dimensions"]:
        assert d["fix"].strip() and d["detail"].strip()
        assert 0 <= d["score"] <= 100


def test_dimensions_sorted_worst_first():
    scores = [d["score"] for d in score_rubric(STRONG)["dimensions"]]
    assert scores == sorted(scores)


def test_biggest_win_is_returned():
    assert score_rubric(WEAK)["biggest_win"].strip()


def test_filler_language_is_penalised():
    filler = STRONG.replace("Led migration of", "Was responsible for the migration of")
    assert score_rubric(filler)["quality_score"] < score_rubric(STRONG)["quality_score"]


def test_empty_input_does_not_crash():
    assert score_rubric("")["quality_score"] >= 0
