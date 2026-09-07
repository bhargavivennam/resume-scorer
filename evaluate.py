"""
Measure scoring accuracy against REAL human judgement.

Why this file matters more than any modelling change
----------------------------------------------------
Every metric printed by `train.py` is computed on synthetic resumes. It tells
you how well the model learned the generator — not whether it agrees with a
recruiter. You cannot improve what you cannot measure, and right now nothing in
this project measures the thing that matters.

This closes that loop with the cheapest possible labelling scheme: **pairwise
preferences**. Judging "is this resume a 3 or a 4?" is hard and inconsistent
even for professionals. Judging "is A better than B?" is fast, and it's what
recruiters actually do when they rank a stack. Pairwise labels are also exactly
what a ranking model wants.

Create `data/golden.jsonl`, one JSON object per line, in either form:

    {"better": "<resume A text>", "worse": "<resume B text>"}
    {"text": "<resume text>", "grade": 3}

Thirty pairs is enough to detect a real regression. Then:

    python evaluate.py

It reports pairwise agreement for the rubric AND the learned model, so you can
see which one actually tracks human preference — and whether a change helped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

GOLDEN_FILE = Path(__file__).parent / "data" / "golden.jsonl"


def load_golden(path: Path = GOLDEN_FILE) -> Tuple[List[Tuple[str, str]], List[Tuple[str, int]]]:
    """Return (pairs, graded). Graded rows are expanded into pairs as well."""
    if not path.exists():
        return [], []
    pairs: List[Tuple[str, str]] = []
    graded: List[Tuple[str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "better" in rec and "worse" in rec:
            pairs.append((rec["better"], rec["worse"]))
        elif "text" in rec and "grade" in rec:
            graded.append((rec["text"], int(rec["grade"])))

    # Every graded pair with different grades is also a preference pair.
    for i, (t1, g1) in enumerate(graded):
        for t2, g2 in graded[i + 1:]:
            if g1 > g2:
                pairs.append((t1, t2))
            elif g2 > g1:
                pairs.append((t2, t1))
    return pairs, graded


def pairwise_agreement(pairs: List[Tuple[str, str]], scorer: Callable[[str], float]) -> Dict[str, Any]:
    """Share of human-preferred pairs the scorer ranks the same way.

    0.5 is a coin flip. Below 0.5 means the scorer is actively inverted, which
    is a far more useful thing to learn than any synthetic AUC.
    """
    if not pairs:
        return {"n": 0, "agreement": None, "ties": 0}
    agree = ties = 0
    for better, worse in pairs:
        sb, sw = scorer(better), scorer(worse)
        if sb == sw:
            ties += 1
        elif sb > sw:
            agree += 1
    decided = len(pairs) - ties
    return {
        "n": len(pairs),
        "ties": ties,
        "agreement": round(agree / decided, 4) if decided else None,
        "n_agree": agree,
    }


def main() -> int:
    pairs, graded = load_golden()
    if not pairs:
        print(f"No golden data at {GOLDEN_FILE}.\n")
        print(__doc__.split("Create `data/golden.jsonl`")[1].split("It reports")[0].strip())
        print("\nUntil this file exists, every number this project prints is measured")
        print("against synthetic resumes and says nothing about real-world accuracy.")
        return 1

    from inference import load_bundle, score_resume
    from rubric import score_rubric

    bundle = load_bundle()
    scorers: Dict[str, Callable[[str], float]] = {
        "rubric (headline score)": lambda t: score_rubric(t)["quality_score"],
        "learned model (ml_percentile)": lambda t: score_resume(t, bundle=bundle)["ml_percentile"],
    }

    print(f"Golden set: {len(pairs)} preference pairs "
          f"({len(graded)} graded resumes expanded into pairs)\n")
    results = {}
    for name, fn in scorers.items():
        r = pairwise_agreement(pairs, fn)
        results[name] = r
        bar = "" if r["agreement"] is None else "#" * int(r["agreement"] * 40)
        print(f"  {name:32} {r['agreement']:.1%} agreement  "
              f"({r['n_agree']}/{r['n'] - r['ties']}, {r['ties']} ties)")
        print(f"  {'':32} {bar}")

    print("\n  0.50 = coin flip.  <0.50 = the scorer is inverted relative to humans.")
    best = max((r for r in results.values() if r["agreement"] is not None),
               key=lambda r: r["agreement"], default=None)
    if best and best["agreement"] < 0.6:
        print("\n  Neither scorer tracks human judgement well yet. More labelled")
        print("  pairs, or reweighting rubric.py's dimensions, is the next step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
