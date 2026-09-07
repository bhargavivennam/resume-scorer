"""Behavioural tests for the scorer and the API."""
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from api import app
from data import generate_dataset
from inference import ModelNotTrained, load_bundle, score_resume

GOOD = """Alex Chen
alex.chen@example.com | +1 (555) 201-3344 | linkedin.com/in/alexchen | github.com/alexchen

SUMMARY
Senior Software Engineer with 10 years building distributed systems.

EXPERIENCE
Staff Engineer - Northwind Systems
Mar 2019 - Present
- Led migration of 30 microservices to Kubernetes, cutting deploy time by 55%
- Designed a Kafka pipeline processing 20M events/day at 99.9% uptime
- Optimized PostgreSQL query plans, reducing p95 latency from 800ms to 90ms

Software Engineer - Bluepeak Labs
Jun 2015 - Feb 2019
- Built a PyTorch ranking model that lifted click-through rate by 22%
- Automated CI/CD with Terraform and Jenkins, saving $40k annually

SKILLS
Python, Java, Go, TypeScript, PostgreSQL, Redis, AWS, Docker, Kubernetes,
Terraform, Spark, Kafka, Airflow, PyTorch, scikit-learn, React, FastAPI

EDUCATION
M.S. Computer Science, State University, 2013 - 2015
B.Tech Computer Science, City College, 2009 - 2013

CERTIFICATIONS
- AWS Certified Solutions Architect - Associate (2023)
- Certified Kubernetes Administrator (CKA), 2022
"""

BAD = """Sam Doyle

EXPERIENCE
Developer - Acme Corp
2021 - 2022
- Responsible for writing code and fixing bugs
- Worked on various projects using different technologies

SKILLS
MS Office, Email, Teamwork
"""


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _require_model():
    try:
        score_resume(GOOD)
    except ModelNotTrained:
        pytest.skip("Model not trained yet — run `python train.py`.")


def test_score_shape_and_ranges():
    _require_model()
    r = score_resume(GOOD)
    assert 0 <= r["score"] <= 100
    assert 0.0 <= r["confidence"] <= 1.0
    assert r["verdict"] in {"good", "needs work"}
    assert r["top_features"], "explanation must not be empty"


def test_strong_resume_outscores_weak_one():
    """The single most important behavioural guarantee of the whole project."""
    _require_model()
    assert score_resume(GOOD)["score"] > score_resume(BAD)["score"]


def test_extracted_facts_are_correct():
    _require_model()
    ex = score_resume(GOOD)["extracted"]
    assert ex["years_experience"] > 8
    assert ex["n_degrees"] >= 2
    assert ex["n_certifications"] >= 1
    assert ex["n_quantified_results"] >= 4
    assert ex["n_employment_gaps"] == 0


def test_scoring_is_deterministic():
    _require_model()
    assert score_resume(GOOD)["score"] == score_resume(GOOD)["score"]


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        score_resume("   ")


def test_ranking_beats_chance_on_holdout():
    """End-to-end check: the model should rank held-out good resumes above bad."""
    _require_model()
    texts, labels = generate_dataset(60, seed=1234)  # unseen seed
    scores = [score_resume(t)["score"] for t in texts]
    good = [s for s, y in zip(scores, labels) if y == 1]
    bad = [s for s, y in zip(scores, labels) if y == 0]
    if not good or not bad:
        pytest.skip("degenerate sample")
    assert sum(good) / len(good) > sum(bad) / len(bad)


# ---------------------------- API tests ----------------------------------

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] in {"ok", "degraded"}


def test_score_endpoint(client):
    _require_model()
    resp = client.post("/score", json={"resume_text": GOOD})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["score"] <= 100
    assert body["top_features"]


def test_score_endpoint_validates_short_input(client):
    assert client.post("/score", json={"resume_text": "hi"}).status_code == 422


def test_compare_endpoint(client):
    _require_model()
    resp = client.post("/compare", json={"resume_a": GOOD, "resume_b": BAD})
    assert resp.status_code == 200
    body = resp.json()
    assert body["winner"] == "a"
    assert body["score_gap"] > 0
    assert body["differences"]


JD_BACKEND = """Senior Backend Engineer
Requirements:
- 6+ years of professional experience
- Strong Python and Go
- Kubernetes, Docker, AWS
- PostgreSQL and Kafka
Preferred qualifications:
- Terraform
"""


def test_score_endpoint_with_job_description(client):
    _require_model()
    resp = client.post("/score", json={"resume_text": GOOD, "job_description": JD_BACKEND})
    assert resp.status_code == 200
    body = resp.json()
    assert body["match"] is not None
    assert 0 <= body["match"]["match_score"] <= 100
    assert body["combined"]["overall_score"] >= 0
    assert body["match"]["required_years"] == 6


def test_score_endpoint_without_job_description_omits_match(client):
    _require_model()
    body = client.post("/score", json={"resume_text": GOOD}).json()
    assert body["match"] is None and body["combined"] is None


def test_extract_endpoint(client):
    files = {"file": ("cv.txt", b"EXPERIENCE\nEngineer at Acme, Jan 2020 - Jan 2024", "text/plain")}
    body = client.post("/extract", files=files).json()
    assert "Acme" in body["text"] and body["n_chars"] > 0


def test_extract_endpoint_rejects_bad_type(client):
    files = {"file": ("cv.pages", b"x" * 100, "application/octet-stream")}
    assert client.post("/extract", files=files).status_code == 422


def test_fetch_job_blocks_internal_urls(client):
    resp = client.post("/fetch-job", json={"url": "http://127.0.0.1:8000/health"})
    assert resp.status_code == 422
    assert "private" in resp.json()["detail"].lower()


def test_compare_uses_job_fit_when_jd_supplied(client):
    _require_model()
    body = client.post("/compare", json={
        "resume_a": GOOD, "resume_b": BAD, "job_description": JD_BACKEND}).json()
    assert body["a"]["match"]["match_score"] > body["b"]["match"]["match_score"]
    assert body["winner"] == "a"


def test_score_endpoint_accepts_job_url(client, monkeypatch):
    """Regression: the UI can pass a bare URL and get job-fit scoring.

    The original bug was on the client side — the URL was collected but never
    sent — so this pins the server contract the UI depends on.
    """
    _require_model()
    import api
    monkeypatch.setattr(api, "fetch_job_description", lambda url, **kw: JD_BACKEND)

    body = client.post("/score", json={
        "resume_text": GOOD,
        "job_url": "https://job-boards.greenhouse.io/example/jobs/1",
    }).json()
    assert body["match"] is not None
    assert body["match"]["match_score"] > 0
    assert body["combined"]["overall_score"] > 0


def test_job_description_wins_over_job_url(client, monkeypatch):
    """If both are supplied, the pasted text is authoritative — no wasted fetch."""
    _require_model()
    import api

    def _boom(url, **kw):
        raise AssertionError("should not fetch when job_description is present")

    monkeypatch.setattr(api, "fetch_job_description", _boom)
    body = client.post("/score", json={
        "resume_text": GOOD, "job_description": JD_BACKEND,
        "job_url": "https://example.com/job",
    }).json()
    assert body["match"]["required_years"] == 6


def test_blocked_job_url_still_returns_a_quality_score(client, monkeypatch):
    """A blocked posting (Indeed, LinkedIn) must degrade, not fail.

    Regression: this used to return 422 and discard the quality score, which we
    could compute perfectly well without the job posting.
    """
    _require_model()
    import api
    from documents import ExtractionError

    def _blocked(url, **kw):
        raise ExtractionError("www.indeed.com blocks automated fetching")

    monkeypatch.setattr(api, "fetch_job_description", _blocked)
    resp = client.post("/score", json={
        "resume_text": GOOD, "job_url": "https://www.indeed.com/viewjob?jk=123"})

    assert resp.status_code == 200          # not 422
    body = resp.json()
    assert body["score"] >= 0               # quality score survived
    assert body["match"] is None
    assert "indeed.com" in body["job_fetch_error"]


# A resume with no certifications section — the common real-world case.
PLAIN_STRONG = """Priya Sharma
priya.sharma@example.com | +1 (555) 442-1180 | linkedin.com/in/priyasharma

SUMMARY
Senior Backend Engineer with 8 years designing distributed systems.

EXPERIENCE
Senior Software Engineer - Vertex Analytics
Jun 2021 - Present
- Led migration of 24 microservices to Kubernetes, cutting deploy time 55%
- Designed a Kafka pipeline processing 18M events/day at 99.95% uptime
- Reduced PostgreSQL p95 latency from 740ms to 85ms by rewriting query plans

Software Engineer - Hexbyte
Aug 2017 - May 2021
- Built Spring Boot REST APIs serving 4M requests/day across 3 regions
- Migrated batch ETL to Spark, cutting nightly runtime from 6h to 40m

SKILLS
Java, Python, Go, SQL, PostgreSQL, Kafka, Spark, AWS, Docker, Kubernetes

EDUCATION
M.S. Computer Science, State University, 2015 - 2017"""

CERT_BLOCK = "\n\nCERTIFICATIONS\n- AWS Certified Solutions Architect - Associate (2023)"


def test_certifications_section_is_not_a_shortcut():
    """Regression: adding one certifications line must not dominate the score.

    The generator used to gate whole sections on hard quality thresholds
    (`if q > 0.55: certifications`), which made section PRESENCE a near-perfect
    proxy for quality. The model learned that shortcut, and a one-line certs
    block moved a real resume from 27% to 56% — so every real resume without
    certifications was stuck around 23%.
    """
    _require_model()
    without = score_resume(PLAIN_STRONG)["score"]
    with_certs = score_resume(PLAIN_STRONG + CERT_BLOCK)["score"]
    assert abs(with_certs - without) <= 12, (
        f"certifications swung the score {without} -> {with_certs}; "
        "the model is keying on section presence rather than content")


def test_strong_resume_without_certifications_scores_well():
    """The concrete symptom the user reported: quality stuck around 23%."""
    _require_model()
    assert score_resume(PLAIN_STRONG)["score"] >= 55


def test_content_moves_the_score_more_than_sections_do():
    """Removing the measured results should cost more than removing a section."""
    _require_model()
    full = score_resume(PLAIN_STRONG)["score"]
    unquantified = score_resume(
        PLAIN_STRONG.replace("55%", "significantly").replace("18M", "many")
        .replace("from 740ms to 85ms", "considerably").replace("4M", "many")
        .replace("from 6h to 40m", "a lot").replace("99.95%", "high"))["score"]
    section_only = score_resume(PLAIN_STRONG + CERT_BLOCK)["score"]
    assert full - unquantified > abs(section_only - full)


def test_stale_model_raises_a_clear_error(monkeypatch):
    """Changing the feature schema must produce an actionable message.

    Adding one skill group silently widened the feature matrix and inference
    died with a sklearn shape error four frames deep. The fix should tell you
    to retrain.
    """
    import inference
    from inference import ModelOutOfDate

    class StaleVectorizer:
        def transform(self, _):
            raise ValueError("X has 35 features, but StandardScaler is expecting 34 features")

    bundle = dict(load_bundle())
    bundle["vectorizer"] = StaleVectorizer()
    with pytest.raises(ModelOutOfDate, match="retrain"):
        score_resume(GOOD, bundle=bundle)
