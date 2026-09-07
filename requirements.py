"""
Requirement-level matching.

The gap this closes
-------------------
`matching.py` extracts *skills* from a posting and checks whether the resume has
them. That misses most of what a job description actually asks for. Take a real
requirements list:

    - 6+ years of backend engineering experience      <- years, handled
    - Strong Python and Go                            <- skills, handled
    - Experience mentoring junior engineers           <- IGNORED
    - Track record designing scalable, reliable APIs  <- IGNORED
    - Comfortable owning services in production       <- IGNORED

Three of five requirements were invisible to skill matching. Recruiters weigh
them heavily, and "this requirement is unaddressed" is the most actionable
feedback the tool can give — far more than "you're missing the keyword etl".

So we match at the REQUIREMENT level: each requirement line is scored against
the resume independently, using skill evidence where the requirement names
skills and lexical similarity where it doesn't. Every requirement gets a
verdict, and the unaddressed ones are what the candidate should fix.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills import EVIDENCE_EXPERIENCE, SkillSignal, find_skills

# Lines that introduce a requirements block.
_SECTION_RE = re.compile(
    r"(requirements?|qualifications?|what you.{0,12}(bring|need|have)|"
    r"who you are|about you|must have|you will need|basic qualifications|"
    r"minimum qualifications|preferred qualifications|nice to have)",
    re.I)

# Responsibilities describe what you WILL do, not what you must already have.
# Scoring them as requirements produced output like "4 of 25 addressed", where
# the 21 misses were duties and corporate boilerplate no resume can "address".
_RESPONSIBILITY_RE = re.compile(
    r"^\s*(responsibilities|what you.{0,12}(do|ll do)|the role|day to day|"
    r"in this role|key duties|about the job)", re.I)

# Longest a line can be and still be a checkable requirement. Beyond this it is
# corporate prose ("Address a variety of responses to problems, questions, or
# situations by applying established criteria...") that means nothing concrete.
MAX_REQUIREMENT_WORDS = 22

# Boilerplate that looks like a requirement but isn't worth scoring.
_NOISE_RE = re.compile(
    r"(equal opportunity|eeo|salary|compensation|benefits|401|visa sponsorship|"
    r"we offer|perks|apply now|about (us|the company)|our mission|health insurance|"
    r"paid time off|dental|vision)", re.I)

# Words too generic to carry meaning when comparing a requirement to a resume.
_STOP = set("""a an the and or of to in on for with by at as is are be been being
have has had you your we our will would can could should must strong excellent
good great experience experienced work working works ability able years year
plus preferred required requirements skills knowledge understanding familiar
familiarity proficient proficiency demonstrated track record including etc such
this that these those from into across their them they it its more most other
some any all new using use used help helps team teams role position job
candidate candidates ideal successful""".split())

VERDICT_STRONG, VERDICT_PARTIAL, VERDICT_NONE = "addressed", "partial", "unaddressed"

# "6+ years", "at least 3 years", "5-7 years of experience"
_YEARS_REQ_RE = re.compile(
    r"(?:(?:at least|minimum(?: of)?|min\.?)\s*)?(\d{1,2})\s*(?:\+|-\s*\d{1,2})?\s*"
    r"(?:\+\s*)?years?", re.I)


def extract_requirements(job_text: str, max_items: int = 12) -> List[str]:
    """Pull the individual requirement lines out of a posting.

    Prefers bullets inside a requirements-style section; falls back to all
    bullets, then to sentences, so an unstructured posting still works.
    """
    lines = [ln.strip() for ln in (job_text or "").splitlines()]
    bullets: List[str] = []
    in_section = False

    for line in lines:
        if not line:
            continue
        if _RESPONSIBILITY_RE.search(line) and len(line.split()) < 12:
            in_section = False       # leave the requirements list
            continue
        if _SECTION_RE.search(line) and len(line.split()) < 12:
            in_section = True
            continue
        is_bullet = bool(re.match(r"^[-•*‣●·]|\s*\d+[.)]\s", line))
        if is_bullet or in_section:
            body = re.sub(r"^[-•*‣●·]\s*|^\d+[.)]\s*", "", line).strip()
            # A long prose paragraph means we've left the list.
            if len(body.split()) > 60:
                in_section = False
                continue
            if 3 <= len(body.split()) <= MAX_REQUIREMENT_WORDS and not _NOISE_RE.search(body):
                bullets.append(body)

    if not bullets:  # unstructured posting: fall back to sentences
        for sent in re.split(r"(?<=[.!?])\s+", job_text or ""):
            sent = sent.strip()
            if 5 <= len(sent.split()) <= 45 and not _NOISE_RE.search(sent):
                bullets.append(sent)

    # Keep only lines that state something checkable: a skill, a year count, or
    # a degree. A line we cannot verify against a resume should not be reported
    # as "unaddressed" — that's noise dressed up as a finding.
    concrete = []
    seen = set()
    for b in bullets:
        key = b.lower()
        if key in seen:
            continue
        seen.add(key)
        if (find_skills(b) or _YEARS_REQ_RE.search(b)
                or re.search(r"\b(degree|bachelor|master|ph\.?d|b\.?s\.?|m\.?s\.?)\b", b, re.I)):
            concrete.append(b)
    return concrete[:max_items]


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z+#.-]{2,}", (text or "").lower())
            if t not in _STOP}


def _resume_units(resume_text: str) -> List[str]:
    """Bullets if the resume has them, otherwise lines — the units we match against."""
    lines = [ln.strip() for ln in (resume_text or "").splitlines() if ln.strip()]
    bullets = [re.sub(r"^[-•*‣●]\s*", "", ln) for ln in lines
               if re.match(r"^[-•*‣●]", ln)]
    return bullets or lines


def score_requirements(
    resume_text: str,
    job_text: str,
    signals: Optional[Dict[str, SkillSignal]] = None,
    resume_years: Optional[float] = None,
) -> Dict[str, Any]:
    """Score every requirement in the posting against the resume."""
    reqs = extract_requirements(job_text)
    if not reqs:
        return {"coverage": None, "requirements": [], "unaddressed": []}

    units = _resume_units(resume_text)
    unit_tokens = [_content_tokens(u) for u in units]
    resume_tokens = _content_tokens(resume_text)
    signals = signals or {}

    scored: List[Dict[str, Any]] = []
    for req in reqs:
        req_skills = find_skills(req)
        req_tokens = _content_tokens(req)

        # 0. A years requirement is answered by the parsed timeline, not by
        #    word overlap. Matching "6+ years of backend engineering" against
        #    resume prose scored a candidate with 8 years at 0.15, which is
        #    simply the wrong tool for the question.
        years_match = _YEARS_REQ_RE.search(req)
        if years_match and resume_years is not None:
            needed = int(years_match.group(1))
            score = 1.0 if resume_years >= needed else max(0.0, resume_years / max(needed, 1))
            basis = "tenure"
            best = f"{resume_years:.1f} years of experience detected"
        # 1. If the requirement names skills, evidence decides it.
        elif req_skills:
            strengths = [signals[s].strength if s in signals else 0.0 for s in req_skills]
            score = sum(strengths) / len(strengths)
            basis = "skills"
            best = max(
                (u for u in units if find_skills(u) & req_skills),
                key=lambda u: len(find_skills(u) & req_skills), default=None)
        # 2. Otherwise fall back to word overlap with the best-matching bullet.
        #    Crude, but it's the difference between scoring the requirement and
        #    pretending it doesn't exist.
        else:
            best_i, best_overlap = -1, 0.0
            for i, toks in enumerate(unit_tokens):
                if not req_tokens:
                    continue
                overlap = len(req_tokens & toks) / len(req_tokens)
                if overlap > best_overlap:
                    best_i, best_overlap = i, overlap
            whole = len(req_tokens & resume_tokens) / len(req_tokens) if req_tokens else 0.0
            # Blend: a single bullet covering it is stronger evidence than the
            # same words scattered across the document.
            score = min(1.0, 0.7 * best_overlap * 2.0 + 0.3 * whole)
            basis = "language"
            best = units[best_i] if best_i >= 0 and best_overlap > 0.15 else None

        verdict = (VERDICT_STRONG if score >= 0.6 else
                   VERDICT_PARTIAL if score >= 0.3 else VERDICT_NONE)
        scored.append({
            "requirement": req,
            "score": round(float(score), 3),
            "verdict": verdict,
            "basis": basis,
            "evidence": (best[:160] if best and verdict != VERDICT_NONE else None),
        })

    coverage = sum(r["score"] for r in scored) / len(scored)
    return {
        "coverage": round(coverage, 3),
        "n_requirements": len(scored),
        "n_addressed": sum(1 for r in scored if r["verdict"] == VERDICT_STRONG),
        "n_partial": sum(1 for r in scored if r["verdict"] == VERDICT_PARTIAL),
        "requirements": sorted(scored, key=lambda r: r["score"]),
        "unaddressed": [r["requirement"] for r in scored if r["verdict"] == VERDICT_NONE][:6],
    }
