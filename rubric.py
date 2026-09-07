"""
Transparent, rule-based resume quality scoring.

Why this replaced the ML score as the default
---------------------------------------------
The learned quality model is trained on synthetic resumes. After removing three
separate shortcuts it still could not score a real resume, for a reason no
amount of feature engineering fixes: **real resumes are outside its training
distribution**. A measured example — 0.0% of training resumes reach 662 words,
which is an ordinary length for a real one. Gradient-boosted trees cannot
extrapolate past their splits, so the prediction is not just inaccurate, it is
undefined in any useful sense.

So quality is scored the way Resume Worded and Jobscan's content checks do it:
an explicit rubric with published thresholds. It has real advantages here.

  * It works on ANY resume today, with no labels.
  * Every point is traceable to a rule you can read and disagree with.
  * Every dimension comes with the specific fix that raises it.
  * No training distribution, so nothing to fall outside of.

The learned model is still computed and reported alongside as `ml_percentile`,
because it ranks resumes sensibly relative to each other. It just shouldn't be
the headline number until it's trained on real labelled data.

Thresholds below are stated as constants, not buried in expressions, so they can
be argued with and tuned against real hiring data when that exists.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from features import extract_features, parse_roles
from skills import EVIDENCE_QUANTIFIED, analyse_skills

# --- targets, all openly stated ------------------------------------------
TARGET_QUANTIFIED_RATE = 0.50   # half your bullets should carry a number
TARGET_DEMONSTRATED = 6         # skills proven in a bullet with a result
TARGET_ACTION_VERB_RATE = 0.60  # bullets opening with a strong verb
BULLET_WORDS_MIN, BULLET_WORDS_MAX = 8, 30
WORDS_MIN, WORDS_IDEAL_LO, WORDS_IDEAL_HI, WORDS_MAX = 250, 400, 900, 1400
TARGET_SKILL_GROUPS = 4

ACTION_VERB_RE = re.compile(
    r"^\s*[-•*‣●]?\s*(led|built|designed|architected|shipped|launched|optimi[sz]ed|"
    r"reduced|increased|improved|migrated|automated|scaled|delivered|owned|drove|"
    r"implemented|mentored|created|developed|established|introduced|rebuilt|"
    r"cut|grew|saved|eliminated|streamlined|spearheaded)\b", re.I)


def _band(value: float, floor: float, target: float) -> float:
    """Linear 0-100 between a floor and the target, clamped."""
    if target <= floor:
        return 100.0 if value >= target else 0.0
    return max(0.0, min(100.0, (value - floor) / (target - floor) * 100.0))


def score_rubric(resume_text: str) -> Dict[str, Any]:
    """Score resume quality 0-100 from explicit, inspectable rules."""
    text = resume_text or ""
    feats = extract_features(text)
    roles = parse_roles(text)
    signals = analyse_skills(text.lower(), roles=[(r.lower(), y) for r, y in roles])

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullets = [ln for ln in lines if re.match(r"^\s*[-•*‣●]", ln)]
    n_bullets = max(len(bullets), 1)

    quantified_bullets = sum(
        1 for b in bullets
        if re.search(r"(\d+(?:\.\d+)?\s*(?:%|percent|x\b|k\b|m\b|bn\b|million|billion)|[$€£]\s?\d|\b\d{2,}\b)", b, re.I))
    quant_rate = quantified_bullets / n_bullets
    verb_rate = sum(1 for b in bullets if ACTION_VERB_RE.match(b)) / n_bullets
    well_sized = sum(1 for b in bullets if BULLET_WORDS_MIN <= len(b.split()) <= BULLET_WORDS_MAX)
    n_demo = sum(1 for s in signals.values() if s.evidence >= EVIDENCE_QUANTIFIED)
    words = feats["word_count"]

    if words < WORDS_MIN:
        length_score = _band(words, 80, WORDS_MIN)
        length_detail = f"{int(words)} words — too thin to evidence your experience."
        length_fix = f"Aim for {WORDS_IDEAL_LO}-{WORDS_IDEAL_HI} words."
    elif words > WORDS_MAX:
        length_score = max(0.0, 100 - (words - WORDS_MAX) / 12)
        length_detail = f"{int(words)} words — long enough that reviewers will skim."
        length_fix = f"Trim toward {WORDS_IDEAL_HI} words; cut roles older than 10-15 years."
    else:
        length_score = 100.0
        length_detail = f"{int(words)} words — a good length."
        length_fix = "No change needed."

    structure = (
        30 * min(feats["n_sections"] / 3, 1.0)
        + 25 * feats["has_email"] + 15 * feats["has_phone"] + 15 * feats["has_links"]
        + 15 * (1.0 if roles else 0.0)
    )

    dims: List[Dict[str, Any]] = [
        {
            "name": "Quantified impact", "weight": 0.25,
            "score": round(_band(quant_rate, 0.0, TARGET_QUANTIFIED_RATE)),
            "detail": f"{quantified_bullets} of {len(bullets)} bullets carry a number "
                      f"({quant_rate:.0%}; target {TARGET_QUANTIFIED_RATE:.0%}).",
            "fix": "Add the result to each bullet: how much faster, cheaper, bigger, "
                   "or how many users. Numbers are what make a claim checkable.",
        },
        {
            "name": "Demonstrated skills", "weight": 0.20,
            "score": round(_band(n_demo, 0, TARGET_DEMONSTRATED)),
            "detail": (f"{n_demo} skills appear in a bullet that also states a result "
                       f"(target {TARGET_DEMONSTRATED}). You list "
                       f"{int(feats['n_skills'])} skills in total."),
            "fix": ("Name the technology inside your quantified bullets. Right now your "
                    "numbers and your tools are in different sentences, so neither one "
                    "proves the other." if n_demo == 0 else
                    "Tie more of your listed skills to a measured outcome."),
        },
        {
            "name": "Strong writing", "weight": 0.15,
            "score": round(0.7 * _band(verb_rate, 0.0, TARGET_ACTION_VERB_RATE)
                           + 0.3 * max(0, 100 - feats["n_weak_phrases"] * 20)),
            "detail": f"{verb_rate:.0%} of bullets open with a strong verb; "
                      f"{int(feats['n_weak_phrases'])} filler phrases "
                      f"(\"responsible for\", \"worked on\").",
            "fix": "Open every bullet with what you did — Led, Built, Cut, Shipped — "
                   "not what you were responsible for.",
        },
        {
            "name": "Bullet craft", "weight": 0.10,
            "score": round(_band(well_sized / n_bullets, 0.0, 0.85)),
            "detail": f"{well_sized} of {len(bullets)} bullets are "
                      f"{BULLET_WORDS_MIN}-{BULLET_WORDS_MAX} words "
                      f"(average {feats['avg_bullet_words']:.0f}).",
            "fix": f"One accomplishment per bullet, {BULLET_WORDS_MIN}-{BULLET_WORDS_MAX} words.",
        },
        {
            "name": "Structure & contact", "weight": 0.10,
            "score": round(min(structure, 100)),
            "detail": f"{int(feats['n_sections'])} standard sections, "
                      f"{len(roles)} dated roles, "
                      f"{'email' if feats['has_email'] else 'NO email'}, "
                      f"{'links' if feats['has_links'] else 'no links'}.",
            "fix": "Use plain EXPERIENCE / SKILLS / EDUCATION headings, dates under every "
                   "role, and an email plus LinkedIn or GitHub in the header.",
        },
        {
            "name": "Skill breadth & recency", "weight": 0.10,
            "score": round(0.5 * _band(feats["n_skill_groups"], 0, TARGET_SKILL_GROUPS)
                           + 0.5 * feats["avg_skill_recency"] * 100),
            "detail": f"{int(feats['n_skill_groups'])} skill areas; "
                      f"{int(feats['n_recent_skills'])} skills used recently.",
            "fix": "Make sure your current role names the skills you most want to be hired for.",
        },
        {
            "name": "Length", "weight": 0.10,
            "score": round(length_score), "detail": length_detail, "fix": length_fix,
        },
    ]

    total = sum(d["score"] * d["weight"] for d in dims)
    return {
        "quality_score": int(round(total)),
        "verdict": ("Strong" if total >= 75 else "Solid" if total >= 60
                    else "Needs work" if total >= 40 else "Weak"),
        # Worst-first: the top row is the highest-leverage thing to fix.
        "dimensions": sorted(dims, key=lambda d: d["score"]),
        "biggest_win": min(dims, key=lambda d: d["score"] * (1 - d["weight"]))["fix"],
    }
