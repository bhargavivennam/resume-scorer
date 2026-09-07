"""Tests for job-description matching and document extraction."""
import pytest

from documents import ExtractionError, extract_text_from_upload, _assert_public_url
from matching import (_classify_requirements, find_skills, match_resume_to_job,
                      required_years, seniority_level)

JD = """Senior Backend Engineer

Requirements:
- 6+ years of professional software engineering experience
- Strong Python and Go
- Experience with Kubernetes, Docker and AWS
- PostgreSQL and Kafka

Preferred qualifications:
- Terraform
- Rust
"""

RESUME = """Alex Chen
EXPERIENCE
Staff Engineer - Northwind
Mar 2016 - Present
- Built Python and Go services on AWS with Kubernetes and Docker
- Owned PostgreSQL and Kafka infrastructure
SKILLS
Python, Go, AWS, Kubernetes, Docker, PostgreSQL, Kafka
"""


def test_required_years_takes_the_minimum_bar():
    assert required_years("5+ years backend, 2+ years Kubernetes") == 2
    assert required_years("at least 8 years of experience") == 8
    assert required_years("no numbers here") is None


def test_required_vs_preferred_split():
    required, preferred = _classify_requirements(JD)
    assert {"python", "go", "kubernetes", "docker", "aws"} <= required
    assert "terraform" in preferred and "rust" in preferred
    assert not (required & preferred)  # a skill can't be in both


def test_skill_named_in_both_sections_counts_as_required():
    jd = "Requirements:\n- Python\nPreferred:\n- Python\n- Rust"
    required, preferred = _classify_requirements(jd)
    assert "python" in required and "python" not in preferred


def test_strong_match_scores_high_with_nothing_missing():
    r = match_resume_to_job(RESUME, JD)
    assert r["match_score"] >= 60
    assert r["missing_required"] == []
    assert r["skill_coverage"] == 1.0


def test_evidence_ordering_listed_below_used_below_quantified():
    """The property that separates this from keyword matching.

    All three resumes contain 100% of the required keywords, so an ATS would
    rank them identically. Ordering them by evidence is the entire point.
    """
    listed = ("Sam\nEXPERIENCE\nEngineer - Acme\nJan 2019 - Present\n"
              "- Worked on various projects\n"
              "SKILLS\nPython, Go, k8s, Docker, AWS, Postgres, Kafka")
    used = ("Alex\nEXPERIENCE\nEngineer - N\nJan 2019 - Present\n"
            "- Built Python and Go services on AWS with Kubernetes and Docker\n"
            "- Owned PostgreSQL and Kafka infrastructure\n"
            "SKILLS\nPython, Go, Kubernetes, Docker, AWS, PostgreSQL, Kafka")
    quantified = ("Alex\nEXPERIENCE\nEngineer - N\nJan 2019 - Present\n"
                  "- Cut deploy time 55% migrating 30 services to k8s on AWS with Docker\n"
                  "- Scaled Kafka to 20M events/day and tuned PostgreSQL p95 by 40%\n"
                  "- Built Go and Python services\n"
                  "SKILLS\nPython, Go, Kubernetes, Docker, AWS, PostgreSQL, Kafka")

    scores = [match_resume_to_job(t, JD) for t in (listed, used, quantified)]
    # Identical keyword coverage...
    assert all(r["skill_coverage"] == 1.0 for r in scores)
    # ...but strictly increasing evidence-weighted scores.
    assert scores[0]["match_score"] < scores[1]["match_score"] < scores[2]["match_score"]
    assert scores[0]["weakly_evidenced"] and not scores[2]["weakly_evidenced"]


def test_missing_skills_are_reported():
    thin = "EXPERIENCE\nDev - Acme\nJan 2023 - Jan 2024\n- Wrote Python scripts\nSKILLS\nPython"
    r = match_resume_to_job(thin, JD)
    assert "kubernetes" in r["missing_required"]
    assert r["match_score"] < 60
    assert any("Missing" in n for n in r["notes"])


def test_match_is_job_specific_not_just_quality():
    """The same strong resume must score lower against an unrelated job."""
    frontend = ("Requirements:\n- 3+ years experience\n- Expert React, Angular, Vue\n"
                "- Strong HTML and CSS")
    assert match_resume_to_job(RESUME, frontend)["match_score"] < \
           match_resume_to_job(RESUME, JD)["match_score"]


def test_seniority_detection():
    assert seniority_level("Senior Backend Engineer") > seniority_level("Junior Developer")


def test_empty_job_text_rejected():
    with pytest.raises(ValueError):
        match_resume_to_job(RESUME, "   ")


def test_find_skills_matches_the_feature_lexicon():
    assert find_skills("We use Node.js and C++") >= {"node.js", "c++"}


# ---------------------- document extraction ------------------------------

def test_txt_upload():
    text = extract_text_from_upload("cv.txt", b"EXPERIENCE\nEngineer at Acme, 2020 - 2024")
    assert "Acme" in text


def test_unsupported_and_legacy_types_rejected():
    with pytest.raises(ExtractionError, match="Unsupported"):
        extract_text_from_upload("cv.pages", b"x" * 100)
    with pytest.raises(ExtractionError, match="save as .docx"):
        extract_text_from_upload("cv.doc", b"x" * 100)


def test_oversized_upload_rejected():
    with pytest.raises(ExtractionError, match="larger than"):
        extract_text_from_upload("cv.pdf", b"x" * (11 * 1024 * 1024))


def test_line_structure_is_preserved():
    """Section headers and bullets are detected with ^-anchored regexes, so
    extraction must not collapse the document onto one line."""
    text = extract_text_from_upload("cv.txt", b"EXPERIENCE\n\n\n\n- did a thing\n- did another")
    assert text.count("\n") >= 2


@pytest.mark.parametrize("url", [
    "http://localhost:8000/admin",
    "http://127.0.0.1/",
    "http://169.254.169.254/latest/meta-data/",
    "file:///etc/passwd",
])
def test_ssrf_guard_blocks_internal_targets(url):
    with pytest.raises(ExtractionError):
        _assert_public_url(url)


def test_html_escaped_board_content_is_unescaped():
    """Greenhouse returns the description HTML-escaped ('&lt;p&gt;').

    Without unescaping first, the tags survive into the text as literal
    '<p>' characters and pollute both the skill matching and TF-IDF.
    """
    from documents import _html_to_text
    text = _html_to_text("&lt;p&gt;Strong &lt;b&gt;Python&lt;/b&gt; and Go&lt;/p&gt;")
    assert "<p>" not in text and "&lt;" not in text
    assert "Python" in text and "Go" in text


def test_list_items_keep_their_line_breaks():
    from documents import _html_to_text
    text = _html_to_text("<ul><li>Python</li><li>Kubernetes</li></ul>")
    assert text.count("\n") >= 1


def test_jsonld_jobposting_is_extracted():
    from bs4 import BeautifulSoup

    from documents import _try_jsonld
    html = """<html><head><script type="application/ld+json">
    {"@type": "JobPosting", "title": "Backend Engineer",
     "hiringOrganization": {"name": "Acme"},
     "description": "<p>We need 5+ years of Python, Kubernetes and AWS experience. """ + ("Details. " * 30) + """</p>"}
    </script></head><body></body></html>"""
    text = _try_jsonld(BeautifulSoup(html, "html.parser"))
    assert text and "Backend Engineer" in text and "Acme" in text
    assert "Kubernetes" in text and "<p>" not in text


def test_missing_similarity_redistributes_weight_instead_of_scoring_zero():
    """A dimension we couldn't MEASURE must not be scored as zero.

    Without a loaded model the TF-IDF similarity is unavailable. Treating that
    as 0.0 silently dragged the headline down and was indistinguishable from a
    genuinely irrelevant resume.
    """
    r = match_resume_to_job(RESUME, JD, bundle=None)
    assert r["components"]["relevance"] is None
    assert r["text_similarity"] is None
    # Skills and experience alone still produce a sensible score.
    assert r["match_score"] > 50


def test_breakdown_is_sorted_worst_first_and_covers_every_dimension():
    from inference import load_bundle, score_resume
    from matching import combined_report
    bundle = load_bundle()
    q = score_resume(RESUME, bundle=bundle)
    m = match_resume_to_job(RESUME, JD, bundle)
    c = combined_report(q, m)

    values = [r["value"] for r in c["breakdown"]]
    assert values == sorted(values), "worst dimension must be listed first"
    labels = {r["label"] for r in c["breakdown"]}
    # Dimensions mirror what Jobscan / Teal actually report.
    assert {"Hard skills", "Experience level", "Resume quality"} <= labels
    assert all(r["detail"] for r in c["breakdown"])
    assert 0 <= c["overall_score"] <= 100


def test_pdf_ligatures_are_normalised():
    """PDF fonts emit "fi" as U+FB01, so "classification" extracts as
    "classiﬁcation" and stops matching the keyword. Any resume exported from
    Word or a browser hits this."""
    from documents import _clean
    text = _clean("Skills: classiﬁcation, workﬂow, eﬃcient staﬀing")
    assert "ﬁ" not in text
    assert "classification" in text and "workflow" in text
    assert "efficient" in text and "staffing" in text


def test_smart_quotes_normalised():
    from documents import _clean
    assert "don't" in _clean("don’t").lower()
