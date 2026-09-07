"""
Train and evaluate the resume scorer.

Run:  python train.py
Out:  models/resume_scorer.joblib   (+ metrics printed to stdout)

Pipeline shape
--------------
      raw resume text
             |
      +------+------+
      |             |
   TF-IDF      NumericFeatureExtractor -> StandardScaler
   (1-2 grams)      (25 handcrafted features)
      |             |
      +------+------+
             |   hstack (sparse)
        +----+----+
        |         |
  LogisticRegression   LightGBM
   (linear baseline)   (gradient-boosted trees)

We train BOTH on purpose. The linear model is the honesty check: if the fancy
model can't beat a well-tuned logistic regression, the extra complexity isn't
earning its keep. It also gives exact, additive explanations for free.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.metrics import ndcg_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from scipy.stats import spearmanr
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from data import GOOD_GRADE_THRESHOLD, GRADE_NAMES, load_graded_dataset
from features import NumericFeatureExtractor, redact_for_text_model

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:  # keeps the project runnable without a compiler toolchain
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    HAS_LGBM = False

MODEL_PATH = Path(__file__).parent / "models" / "resume_scorer.joblib"
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Feature pipeline
# --------------------------------------------------------------------------

def build_vectorizer() -> FeatureUnion:
    """Text -> sparse matrix combining TF-IDF and the handcrafted features.

    `sublinear_tf` dampens repeated words (saying "Python" 9 times shouldn't be
    9x the signal), `min_df=3` drops one-off typos that can't generalise.
    """
    tfidf = TfidfVectorizer(
        # Identity and dates are stripped before tokenising — see
        # features.redact_for_text_model for why this is not optional.
        preprocessor=redact_for_text_model,
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_features=5000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    numeric = Pipeline([
        ("extract", NumericFeatureExtractor()),
        # Scaling matters only for the linear model (trees are scale-invariant),
        # but it costs nothing and keeps LR coefficients comparable to each other.
        ("scale", StandardScaler()),
    ])
    return FeatureUnion([("tfidf", tfidf), ("numeric", numeric)])


def build_models(y_train: np.ndarray) -> dict:
    """Both classifiers, each told about the class imbalance."""
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    models = {
        # class_weight='balanced' re-weights the loss so the rare "good" class
        # isn't drowned out. Without it the model learns "always predict bad",
        # which scores 75% accuracy and is completely useless.
        "logistic_regression": LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced",
            solver="liblinear", random_state=RANDOM_STATE,
        ),
    }
    if HAS_LGBM:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=20, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=1.0,
            scale_pos_weight=pos_weight,      # the LightGBM spelling of the above
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
    else:
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
    return models


def build_ranker():
    """Regressor on the 0-4 grade — the model that produces the score.

    Regression on an ordinal grade (rather than classification) preserves the
    ORDER of the bands, which is what we actually need: the output only has to
    rank resumes correctly, because it's converted to a percentile afterwards.
    """
    if HAS_LGBM:
        return LGBMRegressor(
            n_estimators=500, learning_rate=0.04, num_leaves=15,
            min_child_samples=20, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.04, max_leaf_nodes=15, random_state=RANDOM_STATE)


def evaluate_ranking(name: str, y_grade: np.ndarray, pred: np.ndarray) -> dict:
    """Rank-quality metrics. Accuracy is meaningless for an ordinal target."""
    rho = float(spearmanr(y_grade, pred).statistic)
    # NDCG asks: if you read resumes in the model's order, how much of the
    # available quality do you see first? That IS the recruiter's workflow.
    ndcg10 = float(ndcg_score([y_grade], [pred], k=10))
    ndcg50 = float(ndcg_score([y_grade], [pred], k=50))
    binary = (y_grade >= GOOD_GRADE_THRESHOLD).astype(int)
    auc = float(roc_auc_score(binary, pred))
    mae = float(np.mean(np.abs(np.clip(pred, 0, 4) - y_grade)))
    print(f"\n--- {name} (ranking) ---")
    for k, v in (("spearman", rho), ("ndcg@10", ndcg10), ("ndcg@50", ndcg50),
                 ("auc(grade>=3)", auc), ("grade MAE", mae)):
        print(f"  {k:<16} {v:.4f}")
    return {"model": name, "spearman": rho, "ndcg@10": ndcg10, "ndcg@50": ndcg50,
            "roc_auc": auc, "grade_mae": mae}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(name: str, y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    m = {
        "model": name,
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # AUC-ROC is threshold-free: it asks "given one good and one bad resume,
        # how often does the model rank the good one higher?"
        "roc_auc": roc_auc_score(y_true, proba),
        # With 25% positives, PR-AUC is the more honest headline number.
        "pr_auc": average_precision_score(y_true, proba),
        # Are the probabilities themselves trustworthy? Matters because we turn
        # them straight into a 0-100 score.
        "brier": brier_score_loss(y_true, proba),
    }
    print(f"\n--- {name} ---")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:<12} {v:.4f}")
    print("  confusion (rows=true 0/1, cols=pred 0/1):")
    print("   ", confusion_matrix(y_true, y_pred).tolist())
    print(classification_report(y_true, y_pred, target_names=["bad", "good"], zero_division=0))
    return m


def pick_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Choose the cut-off that maximises F1 — on TRAIN folds only.

    Tuning a threshold on the test set is a subtle but real form of leakage:
    the reported metrics stop being an estimate of unseen performance.
    """
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def main() -> None:
    texts, grades = load_graded_dataset()
    X = np.asarray(texts, dtype=object)
    g = np.asarray(grades, dtype=int)
    y = (g >= GOOD_GRADE_THRESHOLD).astype(int)   # binary view, for the baseline

    print("\nGrade distribution:")
    for grade in sorted(set(g)):
        n = int((g == grade).sum())
        print(f"  {grade} {GRADE_NAMES[grade]:<12} {n:5d}  ({n / len(g):.1%})")

    # Stratify on the GRADE so every band is represented in both splits.
    X_tr, X_te, g_tr, g_te, y_tr, y_te = train_test_split(
        X, g, y, test_size=0.2, random_state=RANDOM_STATE, stratify=g)
    print(f"\nTrain {len(g_tr)} | Test {len(g_te)}")

    print("\nFitting vectorizer (TF-IDF + handcrafted features)...")
    vec = build_vectorizer()
    Xtr = vec.fit_transform(X_tr)   # fit on TRAIN ONLY
    Xte = vec.transform(X_te)
    print(f"Feature matrix: {Xtr.shape[0]} x {Xtr.shape[1]}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    # ---- 1. Binary classifiers (baselines + the confidence estimate) -----
    fitted_clf = {}
    for name, model in build_models(y_tr).items():
        print(f"\n=== Training {name} (binary baseline) ===")
        oof = cross_val_predict(model, Xtr, y_tr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        threshold = pick_threshold(y_tr, oof)
        model.fit(Xtr, y_tr)
        proba_te = model.predict_proba(Xte)[:, 1]
        m = evaluate(name, y_te, proba_te, threshold)
        m["spearman"] = float(spearmanr(g_te, proba_te).statistic)
        m["ndcg@10"] = float(ndcg_score([g_te], [proba_te], k=10))
        results.append(m)
        fitted_clf[name] = {
            "model": model, "threshold": threshold,
            "calibrator": IsotonicRegression(out_of_bounds="clip").fit(oof, y_tr),
        }

    # ---- 2. Graded ranker: the model that produces the score -------------
    print("\n=== Training graded ranker (LightGBM regression on 0-4 grade) ===")
    ranker = build_ranker()
    oof_rank = cross_val_predict(ranker, Xtr, g_tr, cv=cv, n_jobs=1)
    print(f"  out-of-fold spearman: {spearmanr(g_tr, oof_rank).statistic:.4f}")
    ranker.fit(Xtr, g_tr)
    pred_te = ranker.predict(Xte)
    rank_metrics = evaluate_ranking("graded_ranker", g_te, pred_te)
    results.append(rank_metrics)

    # The percentile reference: out-of-fold predictions on the training pool.
    # A new resume's score is its position in THIS distribution, which is why
    # the number means "better than X% of the pool" rather than "P(good)".
    percentile_reference = np.sort(oof_rank)

    print("\n" + "=" * 62)
    print("Comparison (test set):")
    for m in results:
        print(f"  {m['model']:<22} auc={m.get('roc_auc', 0):.4f}  "
              f"spearman={m.get('spearman', 0):.4f}  ndcg@10={m.get('ndcg@10', 0):.4f}")
    best_clf = max((m for m in results if m["model"] in fitted_clf),
                   key=lambda m: m["roc_auc"])
    print(f"Score comes from: graded_ranker (spearman {rank_metrics['spearman']:.3f})")
    print(f"Confidence comes from: {best_clf['model']} (calibrated P(grade>=3))")
    print("=" * 62)

    chosen = fitted_clf[best_clf["model"]]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "vectorizer": vec,
        "model": ranker,                      # explanations come from this one
        "ranker": ranker,
        "classifier": chosen["model"],
        "calibrator": chosen["calibrator"],
        "threshold": chosen["threshold"],
        "percentile_reference": percentile_reference,
        "model_name": "graded_ranker",
        "classifier_name": best_clf["model"],
        "feature_names": list(vec.get_feature_names_out()),
        "metrics": results,
        "grade_names": GRADE_NAMES,
    }, MODEL_PATH)
    print(f"\nSaved -> {MODEL_PATH}")
    (MODEL_PATH.parent / "metrics.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
