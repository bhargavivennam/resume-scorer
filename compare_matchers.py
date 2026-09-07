"""
Compare three ways of answering "does this resume match this job".

    1. lexical    ontology + evidence weighting, embeddings switched OFF
    2. embeddings the same, with semantic similarity switched ON
    3. llm        ask Claude to read both and judge

The question worth answering is not "which number is biggest" — none of them
is ground truth. It is where they DISAGREE, and whether the LLM even agrees
with itself when asked the same question twice.

    python compare_matchers.py <resume.pdf> <job.txt>
    python compare_matchers.py <resume.pdf> <job.txt> --llm-runs 5

NOTE: every --llm-runs costs a real API call (about a cent each).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import Any, Dict, List, Optional

import llm_matcher
import semantic
from documents import extract_text_from_upload
from inference import load_bundle


def read_any(path: str) -> str:
    """Accepts .pdf/.docx/.txt/.md — reuses the project's own extractor."""
    if path.lower().endswith((".pdf", ".docx")):
        with open(path, "rb") as fh:
            return extract_text_from_upload(path, fh.read())
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# 1. LEXICAL — written for you, as the worked example
# ---------------------------------------------------------------------------

def run_lexical(resume: str, job: str, bundle) -> Dict[str, Any]:
    """Rule-based matching with the embedding encoder forced off.

    `semantic.get_model` is monkeypatched rather than adding a flag to
    matching.py: the point of this script is to compare the system as it
    actually ships, not a special "comparison mode" of it.
    """
    from matching import match_resume_to_job

    real = semantic.get_model
    semantic.get_model = lambda: None            # force the lexical path
    try:
        m = match_resume_to_job(resume, job, bundle)
    finally:
        semantic.get_model = real                # always restore
        semantic._model.cache_clear()

    return {
        "name": "lexical",
        "score": m["match_score"],
        "addressed": m.get("n_requirements_addressed"),
        "total": m.get("n_requirements"),
        "missing": m["missing_required"],
    }


# ---------------------------------------------------------------------------
# 2. EMBEDDINGS — YOUR TURN
# ---------------------------------------------------------------------------

def run_embeddings(resume: str, job: str, bundle) -> Dict[str, Any]:
    """Same matcher, encoder ON. This is the default the app already uses.

    YOUR TURN. This is `run_lexical` minus the monkeypatch — call
    `match_resume_to_job(resume, job, bundle)` directly and return the same
    dict shape, with "name": "embeddings".

    Identical to run_lexical except that the encoder is left alone, so the
    difference between the two rows is exactly what semantic matching bought.
    """
    if not semantic.is_available():
        return None

    from matching import match_resume_to_job

    m = match_resume_to_job(resume, job, bundle)
    return {
        "name": "embeddings",
        "score": m["match_score"],
        "addressed": m.get("n_requirements_addressed"),
        "total": m.get("n_requirements"),
        "missing": m["missing_required"],
    }


# ---------------------------------------------------------------------------
# 3. LLM — YOUR TURN
# ---------------------------------------------------------------------------

def run_llm(resume: str, job: str, n_runs: int = 1) -> Optional[Dict[str, Any]]:
    """Ask Claude n_runs times and report the spread.

    YOUR TURN, and this is the interesting one.

    - `llm_matcher.match(resume, job)` returns a dict, or None with no creds.
    - Call it n_runs times, collecting each "match_score".
    - Return {"name": "llm", "score": <mean, rounded>, "runs": [...],
              "spread": max - min, "missing": <missing_required of run 1>}
    - Pass a different `temperature_note` per run (e.g. f" (attempt {i})") so
      the requests are not identical — otherwise you may be served a cached
      response and measure zero variance that is not real.

    Each run varies the prompt by one word. Identical requests can be served
    from cache, which would show zero variance that is not real — the point of
    asking n times is to measure whether the model agrees with itself.
    """
    if n_runs < 1 or not llm_matcher.is_available():
        return None

    scores: List[int] = []
    first: Optional[Dict[str, Any]] = None

    for i in range(n_runs):
        result = llm_matcher.match(resume, job, temperature_note=f" (attempt {i + 1})")
        if result is None:
            return None
        scores.append(int(result["match_score"]))
        if first is None:
            first = result

    return {
        "name": "llm",
        "score": round(statistics.mean(scores)),
        "runs": scores,
        "spread": max(scores) - min(scores),
        "missing": (first or {}).get("missing_required", []),
        "summary": (first or {}).get("summary", ""),
    }


# ---------------------------------------------------------------------------
# Report — written for you
# ---------------------------------------------------------------------------

def report(results: List[Optional[Dict[str, Any]]]) -> None:
    live = [r for r in results if r]
    print(f"\n{'matcher':<12} {'score':>6}   detail")
    print("-" * 68)
    for r in live:
        detail = ""
        if r.get("runs"):
            runs = ", ".join(str(x) for x in r["runs"])
            detail = f"runs: {runs}  (spread {r['spread']})"
        elif r.get("total"):
            detail = f"{r['addressed']}/{r['total']} requirements addressed"
        print(f"{r['name']:<12} {r['score']:>5}%   {detail}")

    scores = [r["score"] for r in live]
    if len(scores) > 1:
        print(f"\nspread across matchers: {max(scores) - min(scores)} points "
              f"(min {min(scores)}, max {max(scores)})")
        print("None of these is ground truth. To find out which tracks human")
        print("judgement, add labelled pairs to data/golden.jsonl and run evaluate.py.")

    for r in live:
        if r.get("missing"):
            print(f"\n{r['name']} says missing: {', '.join(r['missing'][:8])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("resume")
    ap.add_argument("job")
    ap.add_argument("--llm-runs", type=int, default=0,
                    help="how many times to ask the LLM (each one costs money)")
    args = ap.parse_args()

    resume, job = read_any(args.resume), read_any(args.job)
    bundle = load_bundle()

    results: List[Optional[Dict[str, Any]]] = [run_lexical(resume, job, bundle)]

    for fn, arg in ((run_embeddings, bundle), (run_llm, args.llm_runs)):
        try:
            results.append(fn(resume, job, arg) if fn is run_embeddings
                           else fn(resume, job, arg))
        except NotImplementedError as exc:
            print(f"[skipped] {fn.__name__}: {exc}")

    report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
