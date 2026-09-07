"""
Resume <-> job description matching.

IMPORTANT distinction from the ML model
---------------------------------------
`inference.score_resume` answers "is this a well-built resume?" — it was trained
on quality labels and knows nothing about any particular job.

This module answers a different question: "does THIS resume fit THIS job?" It is
deliberately **rule-based, not learned**, because:

  * we have no labelled (resume, job, hired?) pairs to train on, and inventing
    them would just encode our own guesses with a veneer of ML;
  * a recruiter needs to see *which* required skill is missing, not a black-box
    number. Every component here decomposes into something you can act on.

The two scores are reported separately and then blended, so a great resume for
the wrong job doesn't get to hide behind its quality score.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import numpy as np

from features import extract_features, parse_roles
from requirements import score_requirements
from skills import (EVIDENCE_EXPERIENCE, EVIDENCE_QUANTIFIED, ONTOLOGY,
                    analyse_skills, find_skills, group_of)

# Phrases that mark a skill as a hard requirement rather than a nice-to-have.
_REQUIRED_MARKERS = re.compile(
    r"(required|requirement|must have|must-have|you have|we require|"
    r"minimum qualifications|basic qualifications|essential)", re.I)
_PREFERRED_MARKERS = re.compile(
    r"(preferred|nice to have|nice-to-have|bonus|plus|desirable|"
    r"preferred qualifications|good to have)", re.I)

# "5+ years", "at least 3 years", "3-5 years of experience"
_YEARS_RE = re.compile(
    r"(?:(?:at least|minimum(?: of)?|min\.?)\s*)?(\d{1,2})\s*(?:\+|-\s*\d{1,2})?\s*"
    r"(?:\+\s*)?years?", re.I)

# Skill groups that count as "soft"/behavioural rather than technical.
SOFT_GROUPS = {"soft", "practices"}

_DEGREE_LEVELS = [
    (r"\b(ph\.?\s?d|doctorate)\b", 3),
    (r"\b(master'?s?|m\.?\s?s\.?|m\.?\s?sc|m\.?\s?tech|mba)\b", 2),
    (r"\b(bachelor'?s?|b\.?\s?s\.?|b\.?\s?sc|b\.?\s?tech|b\.?\s?e\.?|undergraduate degree)\b", 1),
]

SENIORITY = {
    "intern": 0, "junior": 1, "entry": 1, "associate": 2, "mid": 3,
    "senior": 4, "staff": 5, "principal": 6, "lead": 5, "head": 6,
    "director": 7, "vp": 8,
}


def required_years(text: str) -> Optional[int]:
    """Smallest year-count the posting asks for.

    Postings often say "5+ years backend, 2+ years Kubernetes". The *minimum*
    bar is the smaller one — taking the max would reject people the company
    would happily interview.
    """
    lows = (text or "").lower()
    hits = [int(m.group(1)) for m in _YEARS_RE.finditer(lows) if 0 < int(m.group(1)) <= 30]
    return min(hits) if hits else None


def seniority_level(text: str) -> Optional[int]:
    head = (text or "").lower()[:400]  # the title lives at the top
    found = [v for k, v in SENIORITY.items() if re.search(rf"\b{k}\b", head)]
    return max(found) if found else None


def keyword_frequency(jd_text: str) -> Dict[str, int]:
    """How many times the posting mentions each skill.

    Jobscan states that high-frequency skills are the most impactful part of its
    match rate, and it matches intuition: a posting that says "Kubernetes" five
    times is telling you what the job is actually about. We weight required
    skills by log(1 + mentions) so repetition matters without letting one
    hammered keyword dominate everything else.
    """
    # Count OCCURRENCES, not distinct skills per line: "Kubernetes, Kubernetes"
    # is two mentions, and per-line set semantics would record one.
    from skills import _ALIAS_TO_CANONICAL, _SKILL_RE

    counts: Dict[str, int] = {}
    for m in _SKILL_RE.finditer(jd_text or ""):
        surface = m.group("long") or m.group("short")
        canonical = _ALIAS_TO_CANONICAL.get((surface or "").lower())
        if canonical:
            counts[canonical] = counts.get(canonical, 0) + 1
    return counts


def _degree_level(text: str) -> int:
    """0 = none found, 1 = bachelor's, 2 = master's, 3 = doctorate."""
    for pattern, level in _DEGREE_LEVELS:
        if re.search(pattern, (text or "").lower()):
            return level
    return 0


def education_match(resume_text: str, job_text: str) -> Optional[float]:
    """1.0 when the resume meets or exceeds the posting's degree requirement."""
    needed = _degree_level(job_text)
    if not needed:
        return None                      # no stated requirement -> don't score it
    have = _degree_level(resume_text)
    if have >= needed:
        return 1.0
    return 0.5 if have else 0.0          # a lesser degree is partial credit


def _title_tokens(text: str) -> set[str]:
    generic = {"senior", "junior", "staff", "principal", "lead", "ii", "iii", "iv",
               "i", "sr", "jr", "the", "and", "of", "a", "an", "for", "at"}
    return {t for t in re.findall(r"[a-z]+", (text or "").lower())
            if len(t) > 2 and t not in generic}


def job_title_match(resume_text: str, job_text: str) -> Optional[float]:
    """Overlap between the posting's title and the roles on the resume.

    Both Jobscan and Teal treat title alignment as a high-impact factor, and
    Jobright surfaces it as its own dimension. It's a strong signal because it
    captures domain fit that a skill list misses — a backend engineer and a
    sales engineer can share half their keywords.
    """
    lines = [ln.strip() for ln in (job_text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    jd_title = _title_tokens(lines[0])
    if not jd_title:
        return None

    # Compare against every dated role header on the resume, take the best.
    best = 0.0
    for role_text, _ in parse_roles(resume_text):
        header = role_text.splitlines()[0] if role_text.splitlines() else ""
        tokens = _title_tokens(header)
        if tokens:
            best = max(best, len(jd_title & tokens) / len(jd_title))
    # Fall back to the whole resume, at a discount — the words are there but
    # not necessarily as a job title.
    whole = len(jd_title & _title_tokens(resume_text)) / len(jd_title)
    return round(max(best, whole * 0.7), 3)


def _classify_requirements(jd_text: str) -> tuple[set[str], set[str]]:
    """Split JD skills into required vs preferred by the section they sit in.

    We walk the posting line by line and carry forward the most recent marker,
    because "Preferred qualifications:" applies to the bullets *under* it.
    """
    required: set[str] = set()
    preferred: set[str] = set()
    bucket = required  # anything before the first marker counts as required
    for line in (jd_text or "").splitlines():
        if _PREFERRED_MARKERS.search(line):
            bucket = preferred
        elif _REQUIRED_MARKERS.search(line):
            bucket = required
        bucket |= find_skills(line)
    preferred -= required  # a skill named in both is a hard requirement
    return required, preferred


def _tfidf_similarity(resume: str, jd: str, bundle: Optional[Dict[str, Any]]) -> Optional[float]:
    """Cosine similarity in the TF-IDF space the model was trained on.

    Reusing the trained vectorizer matters: its IDF weights come from a real
    corpus, so common filler ("team", "experience") is already down-weighted.
    Fitting a fresh vectorizer on just these two documents would give every
    word an identical, meaningless IDF.
    """
    if not bundle:
        return None      # unavailable, NOT zero — see the caller
    try:
        tfidf = dict(bundle["vectorizer"].transformer_list)["tfidf"]
        a, b = tfidf.transform([resume]), tfidf.transform([jd])
        denom = float(np.sqrt(a.multiply(a).sum()) * np.sqrt(b.multiply(b).sum()))
        return float(a.multiply(b).sum() / denom) if denom else None
    except Exception:
        return None


def match_resume_to_job(
    resume_text: str,
    job_text: str,
    bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score fit 0-100 and, more usefully, say exactly what's missing."""
    if not job_text or not job_text.strip():
        raise ValueError("job_text is empty")

    # Evidence-weighted skills, not a flat set. A skill demonstrated in a dated
    # bullet with a measured outcome counts far more than one typed into a
    # keyword list — which is exactly the gap ATS keyword matching leaves open.
    roles = parse_roles(resume_text)
    signals = analyse_skills(
        (resume_text or "").lower(), roles=[(r.lower(), y) for r, y in roles])
    resume_skills = set(signals)
    req_skills, pref_skills = _classify_requirements(job_text)

    matched_req = sorted(req_skills & resume_skills)
    missing_req = sorted(req_skills - resume_skills)
    matched_pref = sorted(pref_skills & resume_skills)

    # Split hard vs soft, as Jobscan and Teal both report them separately: they
    # are different kinds of evidence and a candidate fixes them differently.
    freq = keyword_frequency(job_text)
    hard_req = {s for s in req_skills if group_of(s) not in SOFT_GROUPS}
    soft_req = {s for s in req_skills if group_of(s) in SOFT_GROUPS}

    def _weighted_coverage(required: set[str]) -> Optional[float]:
        """Frequency-weighted coverage: skills the posting repeats count more."""
        if not required:
            return None
        total = earned = 0.0
        for sk in required:
            w = math.log1p(freq.get(sk, 1))
            total += w
            if sk in signals:
                earned += w * signals[sk].strength
        return earned / total if total else 0.0

    hard_coverage = _weighted_coverage(hard_req)
    soft_coverage = _weighted_coverage(soft_req)

    if req_skills:
        earned = sum(signals[sk].strength for sk in matched_req)
        coverage = earned / len(req_skills)
        raw_coverage = len(matched_req) / len(req_skills)
    else:
        coverage = raw_coverage = 0.6

    # Which matched skills are merely claimed rather than shown — the single
    # most actionable line of feedback we can give.
    weakly_evidenced = sorted(
        sk for sk in matched_req if signals[sk].evidence < EVIDENCE_EXPERIENCE)
    stale = sorted(sk for sk in matched_req if signals[sk].recency < 0.35)

    feats = extract_features(resume_text)
    have_years = feats["years_experience"]
    need_years = required_years(job_text)
    if need_years is None:
        years_component, years_note = 0.6, "No explicit experience requirement found."
    elif have_years >= need_years:
        years_component = 1.0
        years_note = f"Meets the {need_years}+ year requirement ({have_years:.1f} years)."
    else:
        # Partial credit, floored at 0 — 4 years against a 5-year ask is close,
        # 1 year against a 10-year ask is not.
        years_component = max(0.0, have_years / need_years)
        years_note = f"Short of the {need_years}-year requirement ({have_years:.1f} years)."

    similarity = _tfidf_similarity(resume_text, job_text, bundle)

    # Requirement-level coverage: scores every line of the posting, including
    # the majority that name no skill at all ("mentoring engineers", "owning
    # services in production"). Skill matching is blind to those.
    reqs = score_requirements(resume_text, job_text, signals, resume_years=have_years)
    req_coverage = reqs["coverage"]

    jd_level = seniority_level(job_text)
    cv_level = seniority_level(resume_text)
    if jd_level is None or cv_level is None:
        seniority_note = None
    elif cv_level >= jd_level:
        seniority_note = "Seniority in the resume matches or exceeds the posting."
    else:
        seniority_note = "Resume reads more junior than the posting's title."

    # Weights: skill coverage dominates because it's the thing recruiters filter
    # on first and the thing the candidate can most directly act on.
    #
    # Note on comparison with commercial tools: several weight the coarse
    # seniority bucket at ~60% of the headline, which makes a mid-level match
    # look like a strong overall match even when the skill sub-score is weak.
    # We weight the actionable dimension highest instead, so our headline sits
    # lower than theirs on the same posting while our skill number is often
    # HIGHER. Different weighting, not a different measurement.
    # If similarity couldn't be computed (no trained model loaded), redistribute
    # its weight across the other two components instead of treating it as a
    # zero. Scoring an *unmeasured* dimension as 0 silently drags the headline
    # down and looks identical to a genuinely irrelevant resume.
    sim_component = None if similarity is None else min(similarity * 2.5, 1.0)
    title = job_title_match(resume_text, job_text)
    education = education_match(resume_text, job_text)

    # Weights follow what the established tools actually score. Hard skills
    # dominate; job title is high-impact (Jobscan and Teal both say so, and
    # Jobright surfaces it as its own dimension); soft skills and education
    # matter but are rarely decisive.
    #
    # Only measurable components are weighted, then renormalised — an
    # unmeasured dimension is never scored as zero.
    parts: Dict[str, tuple] = {}
    if hard_coverage is not None:
        parts["hard_skills"] = (0.35, hard_coverage)
    parts["experience"] = (0.20, years_component)
    if title is not None:
        parts["title"] = (0.18, title)
    if soft_coverage is not None:
        parts["soft_skills"] = (0.12, soft_coverage)
    if education is not None:
        parts["education"] = (0.08, education)
    if sim_component is not None:
        parts["relevance"] = (0.07, sim_component)
    total_w = sum(w for w, _ in parts.values())
    score = 100 * sum(w * v for w, v in parts.values()) / total_w

    return {
        "match_score": int(round(min(max(score, 0), 100))),
        "skill_coverage": round(raw_coverage, 3),
        "evidence_weighted_coverage": round(coverage, 3),
        "weakly_evidenced": weakly_evidenced,
        "stale_skills": stale,
        "required_skills": sorted(req_skills),
        "matched_required": matched_req,
        "missing_required": missing_req,
        "matched_preferred": matched_pref,
        "missing_preferred": sorted(pref_skills - resume_skills),
        "required_years": need_years,
        "resume_years": have_years,
        "years_component": round(years_component, 3),
        "text_similarity": round(similarity, 4) if similarity is not None else None,
        # Sub-scores as percentages, so the UI can show ONE headline number
        # with a breakdown instead of several competing headline numbers.
        "requirement_coverage": req_coverage,
        "requirements": reqs["requirements"][:10],
        "unaddressed_requirements": reqs["unaddressed"],
        "n_requirements": reqs.get("n_requirements", 0),
        "n_requirements_addressed": reqs.get("n_addressed", 0),
        "hard_skills_required": sorted(hard_req),
        "soft_skills_required": sorted(soft_req),
        "keyword_frequency": {k: v for k, v in sorted(
            freq.items(), key=lambda kv: -kv[1]) if k in req_skills},
        "components": {
            "hard_skills": (int(round(hard_coverage * 100))
                            if hard_coverage is not None else None),
            "soft_skills": (int(round(soft_coverage * 100))
                            if soft_coverage is not None else None),
            "title": int(round(title * 100)) if title is not None else None,
            "education": int(round(education * 100)) if education is not None else None,
            "experience": int(round(years_component * 100)),
            "skills": int(round(coverage * 100)),
            "relevance": (int(round(sim_component * 100))
                          if sim_component is not None else None),
        },
        "notes": [n for n in [
            (f"Matches {len(matched_req)} of {len(req_skills)} required skills."
             if req_skills else "No recognisable technical skills found in the posting — "
                                "match is based on text similarity alone."),
            years_note,
            seniority_note,
            (f"Addresses {reqs['n_addressed']} of {reqs['n_requirements']} "
             f"stated requirements." if reqs.get("n_requirements") else None),
            (f"Not addressed anywhere: {reqs['unaddressed'][0]}"
             if reqs.get("unaddressed") else None),
            (f"Missing: {', '.join(missing_req[:6])}." if missing_req else None),
            (f"Listed but not demonstrated: {', '.join(weakly_evidenced[:4])} — "
             "back these with a bullet showing what you built and its result."
             if weakly_evidenced else None),
            (f"Last used years ago: {', '.join(stale[:4])}."
             if stale else None),
        ] if n],
    }


def combined_report(quality: Dict[str, Any], match: Dict[str, Any]) -> Dict[str, Any]:
    """One headline number, with the dimensions that produced it.

    Earlier this returned a third competing score next to "quality" and "match",
    which is genuinely confusing — three numbers, no indication which one to act
    on. Now `overall_score` is THE score and everything else is a labelled
    breakdown row underneath it, so there is one answer and a visible derivation.
    """
    overall = int(round(0.4 * quality["score"] + 0.6 * match["match_score"]))
    if match["missing_required"]:
        advice = (f"Add evidence of {', '.join(match['missing_required'][:3])} "
                  "if you genuinely have it.")
    elif quality["score"] < 45:
        advice = "Skills line up; the resume's writing is what's holding it back."
    else:
        advice = "Strong alignment on both quality and job fit."
    comp = match["components"]
    return {
        "overall_score": overall,
        "quality_score": quality["score"],
        "match_score": match["match_score"],
        "recommendation": advice,
        # Ordered worst-first: the top row is what to fix.
        # Dimensions mirror what Jobscan / Teal / Huntr actually report —
        # hard skills, soft skills, job title, education, experience — rather
        # than a per-responsibility checklist, which produced unusable output
        # like "4 of 25 addressed" on a posting full of duties and boilerplate.
        "breakdown": sorted([r for r in [
            ({"label": "Hard skills", "value": comp["hard_skills"],
              "detail": f"{len(match['matched_required'])} of "
                        f"{len(match['required_skills'])} required skills, "
                        f"weighted by evidence and how often the posting repeats them"}
             if comp.get("hard_skills") is not None else None),
            ({"label": "Soft skills", "value": comp["soft_skills"],
              "detail": "Mentoring, code review, agile, collaboration and similar"}
             if comp.get("soft_skills") is not None else None),
            ({"label": "Job title", "value": comp["title"],
              "detail": "Overlap between the posting's title and your role titles"}
             if comp.get("title") is not None else None),
            {"label": "Experience level", "value": comp["experience"],
             "detail": (f"{match['resume_years']:.1f} years vs "
                        f"{match['required_years']}+ required"
                        if match["required_years"] else "No stated requirement")},
            ({"label": "Education", "value": comp["education"],
              "detail": "Degree level against the posting's requirement"}
             if comp.get("education") is not None else None),
            {"label": "Resume quality", "value": quality["score"],
             "detail": f"{quality['verdict_quality']} — "
                       f"{quality['quality_dimensions'][0]['name'].lower()} is the "
                       f"weakest dimension"},
        ] if r], key=lambda r: r["value"]),
    }
