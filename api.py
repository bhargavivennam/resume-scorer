"""
FastAPI service for the resume scorer.

Run:   uvicorn api:app --reload --port 8000
Docs:  http://localhost:8000/docs

Endpoints
  GET  /health   - is the model loaded? which one? what were its test metrics?
  POST /score    - score one resume
  POST /compare  - score two resumes side by side and diff them
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from documents import (SUPPORTED_EXTENSIONS, ExtractionError,
                       extract_text_from_upload, fetch_job_description)
from inference import (ModelNotTrained, ModelOutOfDate, load_bundle,
                       score_resume)
from matching import combined_report, match_resume_to_job

MAX_CHARS = 50_000

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model at boot so the first real request isn't the slow one."""
    try:
        load_bundle()
        print("Model loaded.")
    except ModelNotTrained as exc:
        print(f"WARNING: {exc}")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Resume Scorer",
    version="1.0.0",
    description="Supervised ML scoring of resume quality, 0-100, with explanations.",
)

# The React dev server runs on a different origin, so the browser needs CORS.
# Tighten allow_origins before putting this anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas — Pydantic validates and documents the contract in one place.
# --------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    resume_text: str = Field(..., min_length=20, description="Raw resume text")
    top_k: int = Field(8, ge=1, le=25, description="How many explaining features to return")
    # Optional: when supplied, the response also carries job-fit analysis.
    job_description: Optional[str] = Field(
        None, description="Job description text to match against")
    job_url: Optional[str] = Field(
        None, description="Job posting URL; fetched server-side if job_description is absent")

    @field_validator("resume_text")
    @classmethod
    def not_too_long(cls, v: str) -> str:
        if len(v) > MAX_CHARS:
            raise ValueError(f"resume_text exceeds {MAX_CHARS} characters")
        return v


class FeatureImpact(BaseModel):
    feature: str
    kind: str
    impact: float
    direction: str
    value: Optional[float] = None


class RequirementResult(BaseModel):
    requirement: str
    score: float
    verdict: str
    basis: str
    evidence: Optional[str] = None


class MatchResult(BaseModel):
    match_score: int
    requirement_coverage: Optional[float] = None
    requirements: List[RequirementResult] = []
    unaddressed_requirements: List[str] = []
    n_requirements: int = 0
    n_requirements_addressed: int = 0
    skill_coverage: float
    evidence_weighted_coverage: float
    weakly_evidenced: List[str]
    stale_skills: List[str]
    required_skills: List[str]
    matched_required: List[str]
    missing_required: List[str]
    matched_preferred: List[str]
    missing_preferred: List[str]
    required_years: Optional[int] = None
    resume_years: float
    years_component: float
    text_similarity: Optional[float] = None
    components: Dict[str, Optional[int]]
    hard_skills_required: List[str] = []
    soft_skills_required: List[str] = []
    keyword_frequency: Dict[str, int] = {}
    notes: List[str]


class BreakdownRow(BaseModel):
    label: str
    value: int
    detail: str


class CombinedResult(BaseModel):
    overall_score: int
    quality_score: int
    match_score: int
    recommendation: str
    breakdown: List[BreakdownRow]


class AtsFinding(BaseModel):
    severity: str
    issue: str
    fix: str


class AtsReport(BaseModel):
    ats_score: int
    n_blockers: int
    n_warnings: int
    verdict: str
    findings: List[AtsFinding]


class QualityDimension(BaseModel):
    name: str
    weight: float
    score: int
    detail: str
    fix: str


class ScoreResponse(BaseModel):
    score: int
    verdict_quality: str
    quality_dimensions: List[QualityDimension]
    biggest_win: str
    scoring_method: str
    ml_percentile: float
    percentile: float
    predicted_grade: int
    grade_label: str
    probability_good: float
    confidence: float
    verdict: str
    decision_threshold: float
    model: str
    top_features: List[FeatureImpact]
    notes: List[str]
    extracted: Dict[str, float]
    ats: AtsReport
    match: Optional[MatchResult] = None
    combined: Optional[CombinedResult] = None
    # Set when a job_url was supplied but couldn't be fetched; the quality
    # score in this same response is still valid.
    job_fetch_error: Optional[str] = None


class ExtractResponse(BaseModel):
    filename: str
    text: str
    n_chars: int


class FetchJobRequest(BaseModel):
    url: str = Field(..., min_length=8)


class FetchJobResponse(BaseModel):
    url: str
    text: str
    n_chars: int


class CompareRequest(BaseModel):
    resume_a: str = Field(..., min_length=20)
    resume_b: str = Field(..., min_length=20)
    top_k: int = Field(8, ge=1, le=25)
    job_description: Optional[str] = None


class CompareResponse(BaseModel):
    a: ScoreResponse
    b: ScoreResponse
    winner: str
    score_gap: int
    differences: List[Dict[str, Any]]


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        bundle = load_bundle()
    except ModelNotTrained as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}
    return {
        "status": "ok",
        "model_loaded": True,
        "model": bundle["model_name"],
        "decision_threshold": bundle["threshold"],
        "n_features": len(bundle["feature_names"]),
        "test_metrics": bundle["metrics"],
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> Dict[str, Any]:
    try:
        bundle = load_bundle()
        result = score_resume(req.resume_text, top_k=req.top_k, bundle=bundle)
    except (ModelNotTrained, ModelOutOfDate) as exc:
        # 503, not 500: the service is fine, it just has no usable model yet.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Job matching is optional — /score stays useful with no posting at all.
    job_text = req.job_description
    if not job_text and req.job_url:
        try:
            job_text = fetch_job_description(req.job_url)
        except ExtractionError as exc:
            # Degrade, don't fail. Previously a blocked URL (Indeed, LinkedIn)
            # returned 422 and the caller lost the quality score too — the one
            # part we could still compute perfectly well.
            result["job_fetch_error"] = str(exc)

    if job_text and job_text.strip():
        match = match_resume_to_job(req.resume_text, job_text, bundle)
        result["match"] = match
        result["combined"] = combined_report(result, match)
    return result


@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Turn an uploaded PDF / DOCX / TXT into plain text.

    Kept separate from /score so the UI can show the parsed text back to the
    user before scoring it — PDF extraction is lossy often enough that silently
    scoring whatever fell out would be dishonest.
    """
    data = await file.read()
    try:
        text = extract_text_from_upload(file.filename or "", data)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"filename": file.filename or "upload", "text": text, "n_chars": len(text)}


@app.post("/fetch-job", response_model=FetchJobResponse)
def fetch_job(req: FetchJobRequest) -> Dict[str, Any]:
    """Best-effort fetch of a job posting URL. Many boards block this by design."""
    try:
        text = fetch_job_description(req.url)
    except ExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"url": req.url, "text": text, "n_chars": len(text)}


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest) -> Dict[str, Any]:
    try:
        bundle = load_bundle()
        a = score_resume(req.resume_a, top_k=req.top_k, bundle=bundle)
        b = score_resume(req.resume_b, top_k=req.top_k, bundle=bundle)
    except ModelNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if req.job_description and req.job_description.strip():
        for result, text in ((a, req.resume_a), (b, req.resume_b)):
            m = match_resume_to_job(text, req.job_description, bundle)
            result["match"] = m
            result["combined"] = combined_report(result, m)

    # Diff the structured features — this is the part a recruiter can act on.
    differences = [
        {"feature": k, "a": a["extracted"][k], "b": b["extracted"][k],
         "delta": round(b["extracted"][k] - a["extracted"][k], 2)}
        for k in a["extracted"]
        if a["extracted"][k] != b["extracted"][k]
    ]
    differences.sort(key=lambda d: abs(d["delta"]), reverse=True)

    # With a job description in play, "better" means better *for that job*.
    key = "overall_score" if a.get("combined") else None
    sa = a["combined"][key] if key else a["score"]
    sb = b["combined"][key] if key else b["score"]

    return {
        "a": a,
        "b": b,
        "winner": "a" if sa > sb else "b" if sb > sa else "tie",
        "score_gap": abs(sa - sb),
        "differences": differences,
    }
