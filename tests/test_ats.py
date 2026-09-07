"""Tests for the deterministic ATS parse-readiness checks."""
from ats import BLOCKER, check_ats_readiness

WELL_FORMED = """Alex Chen
alex.chen@example.com | +1 (555) 201-3344

EXPERIENCE
Staff Engineer - Northwind Systems
Mar 2019 - Present
- Led migration of 30 microservices to Kubernetes, cutting deploy time by 55%
- Designed a Kafka pipeline processing 20M events/day at 99.9% uptime
- Optimized PostgreSQL query plans, reducing p95 latency from 800ms to 90ms
- Automated CI/CD with Terraform and Jenkins, saving $40k annually
- Mentored five engineers and owned on-call for eight production services

SKILLS
Python, Go, PostgreSQL, AWS, Docker, Kubernetes, Terraform, Kafka, React

EDUCATION
M.S. Computer Science, State University, 2013 - 2015
""" + "- Additional relevant detail about delivery and ownership.\n" * 25


def test_well_formed_resume_passes():
    r = check_ats_readiness(WELL_FORMED)
    assert r["n_blockers"] == 0
    assert r["ats_score"] >= 85


def test_missing_contact_and_sections_are_blockers():
    r = check_ats_readiness("My Journey\nI did things at a company\nWhat I Love\nPython")
    issues = " ".join(f["issue"] for f in r["findings"])
    assert "email" in issues.lower()
    assert "section heading" in issues.lower()
    assert r["n_blockers"] >= 2


def test_missing_dates_flagged():
    text = WELL_FORMED.replace("Mar 2019 - Present", "recently")
    issues = " ".join(f["issue"] for f in check_ats_readiness(text)["findings"])
    assert "date" in issues.lower()


def test_multi_column_layout_detected():
    columned = "EXPERIENCE\nEDUCATION\nSKILLS\na@b.com\nJan 2020 - Jan 2024\n" + \
               "\n".join("Engineer     Acme Corp     2020 - 2024" for _ in range(15))
    r = check_ats_readiness(columned)
    assert any("column" in f["issue"].lower() or "table" in f["issue"].lower()
               for f in r["findings"])


def test_findings_are_sorted_by_severity():
    r = check_ats_readiness("nothing useful here at all")
    severities = [f["severity"] for f in r["findings"]]
    assert severities == sorted(severities, key=lambda s: {"blocker": 0, "warning": 1, "tip": 2}[s])


def test_every_finding_carries_a_fix():
    for text in (WELL_FORMED, "junk", "My Journey\nStuff"):
        for f in check_ats_readiness(text)["findings"]:
            assert f["fix"].strip(), "a finding without a fix is not actionable"


def test_score_is_bounded():
    assert 0 <= check_ats_readiness("x")["ats_score"] <= 100
    assert check_ats_readiness("x")["n_blockers"] > 0
