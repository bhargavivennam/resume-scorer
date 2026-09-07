"""
Feature extraction for resume scoring.

Design note
-----------
A resume is unstructured text, but the things recruiters actually judge are
*structured*: how long you worked, what you can do, what you finished, whether
there are unexplained holes in the timeline.  So we build TWO views of every
resume and let the model use both:

  1. HANDCRAFTED NUMERIC FEATURES (this file)  -> ~25 interpretable numbers.
     These generalise well from very little data and are what we show the user
     in the explanation ("8.5 years experience", "3 employment gaps").

  2. TF-IDF OF THE RAW TEXT (built in train.py) -> thousands of sparse columns.
     This catches vocabulary we never thought to hardcode ("kubernetes",
     "reduced latency", "responsible for" as a weak-language signal).

Handcrafted features are the backbone; TF-IDF is the safety net.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import skills as skills_mod
from skills import EVIDENCE_QUANTIFIED, analyse_skills, group_of

# --------------------------------------------------------------------------
# 1. Skill lexicon
# --------------------------------------------------------------------------
# Grouped so we can score BREADTH (how many groups are covered) separately from
# COUNT (how many skills total).  A candidate with 5 languages and nothing else
# is weaker than one with 2 languages + cloud + a database.

# The ontology in skills.py is now the source of truth (aliases, groups). These
# names stay for backwards compatibility with anything importing them.
ALL_SKILLS: List[str] = skills_mod.ALL_SKILLS
SKILL_GROUPS = {g: [s for s in skills_mod.ALL_SKILLS if group_of(s) == g]
                for g in skills_mod.GROUPS}

# Verbs that signal ownership rather than participation.
ACTION_VERBS = [
    "led", "built", "designed", "architected", "shipped", "launched",
    "optimized", "reduced", "increased", "improved", "migrated", "automated",
    "scaled", "delivered", "owned", "drove", "implemented", "mentored",
]

# Language that usually signals a weak, duty-list resume.
WEAK_PHRASES = [
    "responsible for", "duties included", "worked on", "helped with",
    "participated in", "assisted with", "familiar with", "exposure to",
]

DEGREE_PATTERNS = [
    r"\bph\.?\s?d\b", r"\bdoctorate\b",
    r"\bm\.?\s?tech\b", r"\bm\.?\s?sc\b", r"\bm\.?\s?s\b", r"\bmba\b",
    r"\bmaster'?s?\b",
    r"\bb\.?\s?tech\b", r"\bb\.?\s?sc\b", r"\bb\.?\s?e\b", r"\bb\.?\s?s\b",
    r"\bbachelor'?s?\b",
    r"\bassociate'?s? degree\b", r"\bdiploma\b",
]

CERT_PATTERNS = [
    r"aws certified", r"azure certified", r"google cloud certified",
    r"\bpmp\b", r"\bcissp\b", r"\bccna\b", r"\bckad\b", r"\bcka\b",
    r"scrum master", r"\bcsm\b", r"six sigma", r"\bcfa\b", r"\bcpa\b",
    r"certified [a-z ]{3,30}", r"certification",
]

SECTION_HEADERS = [
    "experience", "education", "skills", "projects", "summary", "objective",
    "certifications", "publications", "achievements", "awards",
]

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = "|".join(MONTHS)

# "Jan 2019 - Mar 2021", "January 2019 to Present", "2019 – 2021", "2019-present"
DATE_RANGE_RE = re.compile(
    rf"""(?ix)
    (?:(?P<m1>{_MONTH_RE})[a-z]*\.?\s*,?\s*)? (?P<y1>19\d{{2}}|20\d{{2}})
    \s*(?:-|–|—|to|until|through)\s*
    (?:
        (?P<present>present|current|now|ongoing)
      | (?:(?:(?P<m2>{_MONTH_RE})[a-z]*\.?\s*,?\s*)? (?P<y2>19\d{{2}}|20\d{{2}}))
    )
    """,
)

# A "today" that makes results reproducible across runs and machines.
REFERENCE_YEAR, REFERENCE_MONTH = 2026, 1


# --------------------------------------------------------------------------
# 2. Date / tenure logic
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    """A half-open employment window expressed in absolute months."""
    start: int
    end: int


def _to_months(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def parse_date_ranges(text: str) -> List[Interval]:
    """Pull every 'start - end' date range out of the resume."""
    intervals: List[Interval] = []
    for m in DATE_RANGE_RE.finditer(text):
        y1 = int(m.group("y1"))
        m1 = MONTHS.get((m.group("m1") or "jan").lower()[:3], 1)
        if m.group("present"):
            y2, m2 = REFERENCE_YEAR, REFERENCE_MONTH
        else:
            y2 = int(m.group("y2"))
            # No end month given -> assume December, i.e. the full year worked.
            m2 = MONTHS.get((m.group("m2") or "dec").lower()[:3], 12)
        start, end = _to_months(y1, m1), _to_months(y2, m2)
        if 0 < end - start <= 50 * 12:  # drop reversed / absurd ranges
            intervals.append(Interval(start, end))
    return intervals


def merge_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    """
    Merge overlapping windows.

    Why this matters: people hold concurrent roles (a job + a side contract).
    Naively summing durations inflates "years of experience". Merging first
    means we measure *calendar time employed*, which is what a recruiter means.
    """
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda i: i.start)
    merged = [ordered[0]]
    for cur in ordered[1:]:
        last = merged[-1]
        if cur.start <= last.end:  # touching or overlapping -> absorb
            merged[-1] = Interval(last.start, max(last.end, cur.end))
        else:
            merged.append(cur)
    return merged


# Headers that open the work-history section, and every header that closes it.
_EXP_HEADER_RE = re.compile(
    r"^\s*(work\s+)?(professional\s+)?(experience|employment|work\s+history|career)\b.*$",
    re.I | re.M,
)
_OTHER_HEADER_RE = re.compile(
    r"^\s*(education|skills|certifications?|projects?|publications?|awards?|"
    r"achievements?|summary|objective|interests|references)\b.*$",
    re.I | re.M,
)


def experience_block(text: str) -> Optional[str]:
    """The work-history block exactly as written, or None if there isn't one.

    Unlike `experience_section` this does NOT fall back to the whole document.
    Callers that need to know whether the *jobs* carry dates must use this —
    the fallback would otherwise let education dates stand in for employment
    dates and hide a resume the ATS can't build a timeline from.
    """
    start = _EXP_HEADER_RE.search(text or "")
    if not start:
        return None
    nxt = _OTHER_HEADER_RE.search(text, start.end())
    return text[start.end(): nxt.start() if nxt else len(text)]


def experience_section(text: str) -> str:
    """Return just the work-history block, or the whole text if there isn't one.

    Why this exists: "B.Tech, 2012 - 2016" is a date range too. Parsing the whole
    document counted degrees as jobs and inflated years of experience by ~60% on
    a typical resume. Scoping to the experience section fixes both the tenure
    number and the gap detection that depends on it.
    """
    start = _EXP_HEADER_RE.search(text)
    if not start:
        return text
    body_start = start.end()
    nxt = _OTHER_HEADER_RE.search(text, body_start)
    section = text[body_start: nxt.start() if nxt else len(text)]
    # A header with nothing under it means our section guess was wrong.
    return section if DATE_RANGE_RE.search(section) else text


def parse_roles(text: str) -> List[Tuple[str, Optional[int]]]:
    """Split the experience section into (role_text, end_year) pairs.

    Each date-range line starts a new role and everything up to the next one
    belongs to it. That gives every bullet a date, which is what lets
    `skills.analyse_skills` decay old skills and reward recent ones.
    """
    section = experience_section(text)
    lines = section.splitlines()
    starts = [i for i, ln in enumerate(lines) if DATE_RANGE_RE.search(ln)]
    if not starts:
        return []

    roles: List[Tuple[str, Optional[int]]] = []
    for idx, line_no in enumerate(starts):
        end_line = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        # Include the line above the date (usually the job title/company).
        body = "\n".join(lines[max(0, line_no - 1):end_line])
        ranges = parse_date_ranges(lines[line_no])
        end_year = (ranges[0].end // 12) if ranges else None
        roles.append((body, end_year))
    return roles


def experience_and_gaps(text: str, gap_threshold_months: int = 6) -> Tuple[float, int, float]:
    """Return (years_experience, n_gaps, longest_gap_months)."""
    merged = merge_intervals(parse_date_ranges(experience_section(text)))
    if not merged:
        return 0.0, 0, 0.0
    months = sum(i.end - i.start for i in merged)
    gaps = [
        nxt.start - cur.end
        for cur, nxt in zip(merged, merged[1:])
        if nxt.start - cur.end >= gap_threshold_months
    ]
    return months / 12.0, len(gaps), float(max(gaps, default=0))


# --------------------------------------------------------------------------
# 3. The feature vector
# --------------------------------------------------------------------------

def _count_patterns(text: str, patterns: Sequence[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text))


def _count_skill_hits(signals) -> Tuple[int, int, Dict[str, int]]:
    """Distinct canonical skills, groups covered, and per-group counts."""
    per_group: Dict[str, int] = {g: 0 for g in skills_mod.GROUPS}
    for skill in signals:
        g = group_of(skill)
        if g:
            per_group[g] += 1
    return len(signals), sum(1 for v in per_group.values() if v > 0), per_group


def extract_features(resume_text: str) -> Dict[str, float]:
    """Turn one resume into an ordered dict of interpretable numbers."""
    text = (resume_text or "").lower()
    words = re.findall(r"[a-z0-9+#.]+", text)
    n_words = len(words)
    lines = [ln.strip() for ln in (resume_text or "").splitlines() if ln.strip()]
    bullets = [ln for ln in lines if re.match(r"^[-•*•‣●]", ln)]

    # Parse the timeline exactly once — every tenure feature below reuses it.
    # (Calling experience_and_gaps/parse_date_ranges per feature re-ran the
    # regex over the whole document four times per resume, which dominated
    # training time on a few thousand rows.)
    exp_text = experience_section(text)
    raw_ranges = parse_date_ranges(exp_text)
    merged = merge_intervals(raw_ranges)
    years = sum(i.end - i.start for i in merged) / 12.0
    gaps = [nxt.start - cur.end for cur, nxt in zip(merged, merged[1:])
            if nxt.start - cur.end >= 6]
    n_gaps, longest_gap = len(gaps), float(max(gaps, default=0))

    # Grade every skill by WHERE it appears and HOW RECENTLY, not just whether
    # it appears — the difference between this and keyword search.
    roles = parse_roles(resume_text or "")
    signals = analyse_skills(text, roles=[(r.lower(), y) for r, y in roles])
    n_skills, n_groups, per_group = _count_skill_hits(signals)
    strengths = [s.strength for s in signals.values()] or [0.0]

    # Quantified impact: "reduced latency by 40%", "served 2M users".
    quantified = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|percent|x\b|k\b|m\b|bn\b|million|billion)", text))
    quantified += len(re.findall(r"[$€£]\s?\d", text))

    feats: Dict[str, float] = {
        # --- experience -----------------------------------------------
        "years_experience": round(years, 2),
        "n_roles": float(len(raw_ranges)),
        "n_employment_gaps": float(n_gaps),
        "longest_gap_months": longest_gap,
        "has_current_role": float(bool(re.search(r"\b(present|current|ongoing)\b", text))),
        "avg_tenure_years": round(years / max(len(merged), 1), 2),

        # --- skills ---------------------------------------------------
        "n_skills": float(n_skills),
        "n_skill_groups": float(n_groups),
        **{f"skills_{g}": float(c) for g, c in per_group.items()},
        # Evidence/recency aggregates: these are what a keyword-stuffed skills
        # list cannot fake, because they require dated bullets with outcomes.
        "skill_strength_mean": round(float(np.mean(strengths)), 3),
        "skill_strength_total": round(float(np.sum(strengths)), 3),
        "n_demonstrated_skills": float(
            sum(1 for s in signals.values() if s.evidence >= EVIDENCE_QUANTIFIED)),
        "n_recent_skills": float(sum(1 for s in signals.values() if s.recency >= 0.7)),
        "avg_skill_recency": round(
            float(np.mean([s.recency for s in signals.values()])) if signals else 0.0, 3),

        # --- credentials ----------------------------------------------
        "n_degrees": float(_count_patterns(text, DEGREE_PATTERNS)),
        "n_certifications": float(_count_patterns(text, CERT_PATTERNS)),
        # "M.S." / "M.Sc" / "MS" all need to match, but plain "ms" inside a
        # word must not, hence the explicit alternation rather than one \b group.
        "has_advanced_degree": float(bool(re.search(
            r"(\bph\.?\s?d\b|\bmaster'?s?\b|\bm\.?\s?tech\b|\bm\.?\s?sc\b|\bm\.\s?s\.?(?!\w)|\bms\b|\bmba\b|\bdoctorate\b)",
            text))),

        # --- writing quality ------------------------------------------
        "n_action_verbs": float(sum(text.count(v) for v in ACTION_VERBS)),
        "n_weak_phrases": float(sum(text.count(p) for p in WEAK_PHRASES)),
        "n_quantified_results": float(quantified),
        "quantified_per_bullet": round(quantified / max(len(bullets), 1), 3),

        # --- structure / completeness ---------------------------------
        "n_sections": float(sum(1 for h in SECTION_HEADERS if re.search(rf"^\s*{h}\b", text, re.M))),
        "n_bullets": float(len(bullets)),
        "word_count": float(n_words),
        "avg_bullet_words": round(
            np.mean([len(b.split()) for b in bullets]) if bullets else 0.0, 2
        ),
        "has_email": float(bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text))),
        "has_phone": float(bool(re.search(r"(\+?\d[\d\s().-]{7,}\d)", text))),
        "has_links": float(bool(re.search(r"(linkedin\.com|github\.com|https?://)", text))),
    }
    return feats


FEATURE_NAMES: List[str] = list(extract_features("placeholder").keys())


# --------------------------------------------------------------------------
# 4. Redaction for the text model
# --------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_URL_RE = re.compile(r"(https?://\S+|(?:www\.|linkedin\.com|github\.com)\S*)", re.I)
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def redact_for_text_model(text: str) -> str:
    """Strip identity and date tokens before TF-IDF sees the resume.

    Two reasons, and both matter:

    1. CORRECTNESS. Without this the model learns fragments of contact details
       — "linkedin.com/in/" tokenises into 'com in', which showed up as a top
       "reason" for a score. That is noise presented as an explanation.

    2. FAIRNESS. Names, email handles and graduation years are proxies for
       ethnicity, gender and age. A resume screener that keys on them is
       discriminating, whether or not anyone intended it. The structured
       features still capture what matters (years of experience is computed
       from parsed date ranges), so nothing of substance is lost — the model
       keeps the tenure, and loses the birth-year proxy.

    The first line is dropped because it is almost always the candidate's name.
    """
    lines = (text or "").splitlines()
    body = "\n".join(lines[1:]) if len(lines) > 1 else (text or "")
    body = _EMAIL_RE.sub(" ", body)
    body = _URL_RE.sub(" ", body)
    body = _PHONE_RE.sub(" ", body)
    body = _YEAR_RE.sub(" ", body)   # dates live in the structured features
    return body


class NumericFeatureExtractor(BaseEstimator, TransformerMixin):
    """sklearn wrapper so the handcrafted features live inside the Pipeline.

    Keeping this in the pipeline (rather than precomputing) means training and
    serving run the exact same code — the classic source of train/serve skew.
    """

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X):  # noqa: N803
        rows = [[extract_features(doc)[name] for name in FEATURE_NAMES] for doc in X]
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)
