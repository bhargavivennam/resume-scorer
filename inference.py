"""
Score a resume with the trained model.

    from inference import score_resume
    score_resume(text)  ->  {"score": 78, "confidence": 0.62, "top_features": [...]}

The three outputs answer three different questions:
  score       "how good is this resume?"      (calibrated probability x 100)
  confidence  "how sure is the model?"        (distance from the 50/50 line)
  top_features "why did it say that?"         (per-feature contributions)

A score without an explanation is unusable for hiring — nobody can act on
"73" and nobody can audit it for bias. The explanation is not decoration.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import scipy.sparse as sp

from ats import check_ats_readiness
from features import extract_features
from rubric import score_rubric

MODEL_PATH = Path(__file__).parent / "models" / "resume_scorer.joblib"

# Human-readable labels for the handcrafted features.
PRETTY: Dict[str, str] = {
    "years_experience": "Years of experience",
    "n_roles": "Number of roles listed",
    "n_employment_gaps": "Employment gaps (6+ months)",
    "longest_gap_months": "Longest employment gap (months)",
    "has_current_role": "Currently employed",
    "avg_tenure_years": "Average tenure per role",
    "n_skills": "Distinct technical skills",
    "n_skill_groups": "Breadth of skill areas",
    "skill_strength_mean": "Average skill evidence strength",
    "skill_strength_total": "Total skill evidence",
    "n_demonstrated_skills": "Skills backed by measured results",
    "n_recent_skills": "Skills used recently",
    "avg_skill_recency": "How recent the skills are",
    "skills_practices": "Engineering practices (agile, testing, mentoring)",
    "skills_languages": "Programming languages",
    "skills_data": "Data engineering skills",
    "skills_ml": "ML / AI skills",
    "skills_cloud_devops": "Cloud & DevOps skills",
    "skills_databases": "Database skills",
    "skills_web": "Web / framework skills",
    "n_degrees": "Degrees",
    "n_certifications": "Certifications",
    "has_advanced_degree": "Advanced degree",
    "n_action_verbs": "Strong action verbs",
    "n_weak_phrases": "Weak filler phrases",
    "n_quantified_results": "Quantified achievements",
    "quantified_per_bullet": "Quantified results per bullet",
    "n_sections": "Resume sections present",
    "n_bullets": "Bullet points",
    "word_count": "Word count",
    "avg_bullet_words": "Average bullet length",
    "has_email": "Email address present",
    "has_phone": "Phone number present",
    "has_links": "LinkedIn / GitHub links",
}


class ModelNotTrained(RuntimeError):
    pass


class ModelOutOfDate(RuntimeError):
    """The saved model was trained on a different feature schema."""


@functools.lru_cache(maxsize=1)
def load_bundle(path: str | Path = MODEL_PATH) -> Dict[str, Any]:
    """Load once and cache — deserialising per request would dominate latency."""
    path = Path(path)
    if not path.exists():
        raise ModelNotTrained(f"No model at {path}. Run `python train.py` first.")
    return joblib.load(path)


def _pretty_name(raw: str) -> tuple[str, str]:
    """FeatureUnion names look like 'numeric__n_skills' / 'tfidf__machine learning'."""
    if raw.startswith("numeric__"):
        key = raw.split("__", 1)[1]
        return PRETTY.get(key, key.replace("_", " ").capitalize()), "structured"
    if raw.startswith("tfidf__"):
        return f'Mentions "{raw.split("__", 1)[1]}"', "text"
    return raw, "other"


def _contributions(bundle: Dict[str, Any], x: sp.csr_matrix) -> np.ndarray:
    """Per-feature push toward 'good' (positive) or 'bad' (negative), in log-odds.

    Trees:  LightGBM's `pred_contrib=True` returns exact SHAP values.
    Linear: the log-odds ARE a sum of coef_i * x_i, so the decomposition is exact
            with no approximation needed.
    """
    model = bundle["model"]
    n_features = x.shape[1]

    if hasattr(model, "booster_"):  # LightGBM
        contrib = model.predict(x, pred_contrib=True)
        contrib = np.asarray(contrib.todense() if sp.issparse(contrib) else contrib)
        return contrib[0][:n_features]  # last column is the base/expected value

    if hasattr(model, "coef_"):  # LogisticRegression
        return np.asarray(model.coef_[0]) * np.asarray(x.todense())[0]

    # Fallback (HistGradientBoosting): no built-in attribution, so approximate by
    # how far each structured feature sits from the training mean.
    return np.zeros(n_features)


def _rule_based_notes(feats: Dict[str, float]) -> List[str]:
    """Model-independent sanity checks — always shown, always true of the text."""
    notes: List[str] = []
    if feats["years_experience"] >= 8:
        notes.append(f"{feats['years_experience']:.1f} years of experience — senior range.")
    elif feats["years_experience"] < 2:
        notes.append(f"Only {feats['years_experience']:.1f} years of detectable experience.")
    if feats["n_employment_gaps"]:
        notes.append(
            f"{int(feats['n_employment_gaps'])} employment gap(s), longest "
            f"{int(feats['longest_gap_months'])} months."
        )
    if feats["n_quantified_results"] == 0:
        notes.append("No quantified achievements (no %, $, or scale figures).")
    if feats["n_weak_phrases"] >= 2:
        notes.append(f"{int(feats['n_weak_phrases'])} filler phrases like 'responsible for'.")
    if feats["n_skill_groups"] >= 4:
        notes.append(f"Broad skill coverage across {int(feats['n_skill_groups'])} areas.")
    if feats["n_demonstrated_skills"] == 0 and feats["n_skills"] >= 5:
        notes.append("Skills are listed but never demonstrated with a measured result.")
    if feats["avg_skill_recency"] < 0.4 and feats["n_skills"] >= 3:
        notes.append("Most listed skills were last used several years ago.")
    if not feats["has_email"] and not feats["has_phone"]:
        notes.append("No contact details found.")
    return notes


def score_resume(resume_text: str, top_k: int = 8, bundle: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Score one resume. Returns score 0-100, confidence 0-1, and explanations."""
    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text is empty")

    bundle = bundle or load_bundle()
    try:
        x = bundle["vectorizer"].transform([resume_text])
    except ValueError as exc:
        # Adding or removing a handcrafted feature changes the matrix width and
        # the saved model no longer matches. Surface that plainly instead of
        # letting a sklearn shape error surface from four frames deep.
        if "features" in str(exc):
            raise ModelOutOfDate(
                f"The saved model was trained on a different feature set ({exc}). "
                "Run `python train.py` to retrain."
            ) from exc
        raise
    x = sp.csr_matrix(x)

    # --- score: percentile of the graded ranker's prediction --------------
    # Ranking, not probability. "Better than 78% of the pool" survives a resume
    # that doesn't look like the training data; "P(good)=0.02" does not.
    raw_grade = float(bundle["ranker"].predict(x)[0])
    reference = bundle["percentile_reference"]
    percentile = float(np.searchsorted(reference, raw_grade) / len(reference) * 100)

    # --- confidence: calibrated P(this is a grade-3+ resume) --------------
    # Deliberately from the separate binary classifier: it answers a genuine
    # probability question, which a regression output cannot.
    raw_p = float(bundle["classifier"].predict_proba(x)[0, 1])
    p = float(np.clip(bundle["calibrator"].predict([raw_p])[0], 0.0, 1.0))
    grade = int(np.clip(round(raw_grade), 0, 4))

    contrib = _contributions(bundle, x)
    names = bundle["feature_names"]
    nz = x.nonzero()[1]  # only features this resume actually triggered
    ranked = sorted(
        ((float(contrib[i]), names[i]) for i in nz if abs(contrib[i]) > 1e-9),
        key=lambda t: abs(t[0]),
        reverse=True,
    )[:top_k]

    feats = extract_features(resume_text)
    top_features = []
    for weight, raw_name in ranked:
        label, kind = _pretty_name(raw_name)
        key = raw_name.split("__", 1)[1] if raw_name.startswith("numeric__") else None
        top_features.append({
            "feature": label,
            "kind": kind,
            "impact": round(weight, 4),
            "direction": "positive" if weight > 0 else "negative",
            "value": round(feats[key], 2) if key in feats else None,
        })

    # The headline quality number is the RUBRIC, not the model.
    #
    # The learned ranker is trained on synthetic resumes and real ones fall
    # outside its distribution entirely (0% of training resumes reach 662
    # words, an ordinary real length). Trees cannot extrapolate, so its
    # absolute output is not meaningful on real input. It still ranks
    # sensibly, so it's reported alongside as `ml_percentile` — but it does
    # not drive the score until it's trained on real labelled data.
    rubric = score_rubric(resume_text)

    return {
        "score": rubric["quality_score"],
        "verdict_quality": rubric["verdict"],
        "quality_dimensions": rubric["dimensions"],
        "biggest_win": rubric["biggest_win"],
        "scoring_method": "rubric",
        # Secondary, for comparison only — see the note above.
        "ml_percentile": round(percentile, 1),
        "predicted_grade": grade,
        "grade_label": bundle["grade_names"][grade],
        "percentile": round(percentile, 1),
        "probability_good": round(p, 4),
        "confidence": round(abs(p - 0.5) * 2, 4),  # 0 at a coin flip, 1 at certainty
        "verdict": "good" if p >= bundle["threshold"] else "needs work",
        "decision_threshold": round(float(bundle["threshold"]), 4),
        "model": bundle["model_name"],
        "top_features": top_features,
        "notes": _rule_based_notes(feats),
        # Deterministic, model-free, and the one check that can flag a resume
        # nobody will ever read regardless of how good it is.
        "ats": check_ats_readiness(resume_text),
        "extracted": {
            "years_experience": feats["years_experience"],
            "n_skills": int(feats["n_skills"]),
            "n_skill_groups": int(feats["n_skill_groups"]),
            "n_degrees": int(feats["n_degrees"]),
            "n_certifications": int(feats["n_certifications"]),
            "n_employment_gaps": int(feats["n_employment_gaps"]),
            "longest_gap_months": int(feats["longest_gap_months"]),
            "n_quantified_results": int(feats["n_quantified_results"]),
            "n_demonstrated_skills": int(feats["n_demonstrated_skills"]),
            "n_recent_skills": int(feats["n_recent_skills"]),
            "word_count": int(feats["word_count"]),
        },
    }


def score_many(texts: List[str], top_k: int = 8) -> List[Dict[str, Any]]:
    bundle = load_bundle()
    return [score_resume(t, top_k=top_k, bundle=bundle) for t in texts]


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        from data import generate_dataset
        text = generate_dataset(1, seed=99)[0][0]
        print(text, "\n" + "=" * 60)
    print(json.dumps(score_resume(text), indent=2))
