"""
ATS parse-readiness checks.

A resume can be excellent and still get filtered out before a human sees it,
because the applicant tracking system couldn't parse it. Jobscan and Resume
Worded both sell this as a headline feature, and it's the one part of resume
scoring that is genuinely deterministic — no model needed, no labels required,
no bias risk. It's pure "will a machine read this correctly".

These checks run on the EXTRACTED TEXT, which is exactly what an ATS sees. That
matters: a two-column layout looks fine to a human and arrives as interleaved
nonsense in the parser.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from features import (DATE_RANGE_RE, experience_block, experience_section,
                      parse_roles)

# Severity drives ordering and colour in the UI.
BLOCKER, WARNING, TIP = "blocker", "warning", "tip"

STANDARD_HEADERS = ["experience", "education", "skills"]


def check_ats_readiness(text: str) -> Dict[str, Any]:
    """Return findings plus a 0-100 readiness score (100 = parses cleanly)."""
    findings: List[Dict[str, str]] = []
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    words = re.findall(r"[A-Za-z0-9+#.]+", text or "")

    def add(severity: str, issue: str, fix: str) -> None:
        findings.append({"severity": severity, "issue": issue, "fix": fix})

    # --- contact details: an unreachable candidate is auto-rejected -------
    if not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text or ""):
        add(BLOCKER, "No email address found.",
            "Put a plain-text email in the header — not inside an image or text box.")
    if not re.search(r"(\+?\d[\d\s().-]{7,}\d)", text or ""):
        add(WARNING, "No phone number found.", "Add a phone number in the header.")

    # --- section headers the parser looks for ----------------------------
    missing = [h for h in STANDARD_HEADERS
               if not re.search(rf"^\s*[^\n]*\b{h}\b", text or "", re.I | re.M)]
    if missing:
        add(BLOCKER if len(missing) > 1 else WARNING,
            f"Missing standard section heading(s): {', '.join(missing)}.",
            "Use plain headings — EXPERIENCE, EDUCATION, SKILLS. Creative labels "
            "like 'Where I've Made an Impact' are not recognised.")

    # --- dates the parser can read ---------------------------------------
    # Check the EXPERIENCE block specifically: education dates must not stand
    # in for employment dates, or a resume with no job timeline slips through.
    block = experience_block(text or "")
    if block is not None and not DATE_RANGE_RE.search(block):
        add(BLOCKER, "No employment dates found in the experience section.",
            "Put dates under every job title, e.g. 'Jan 2020 - Mar 2023'. "
            "Without them an ATS cannot compute your years of experience.")
    elif not DATE_RANGE_RE.search(text or ""):
        add(BLOCKER, "No parseable dates found anywhere.",
            "Write dates as 'Jan 2020 - Mar 2023' or '01/2020 - 03/2023'.")
    elif not parse_roles(text or ""):
        add(WARNING, "Dates found, but no role could be tied to them.",
            "Put each job's dates on the line directly under the job title.")

    # --- layout artefacts that survive extraction ------------------------
    # Multiple runs of 3+ spaces are the fingerprint of a table or two-column
    # layout collapsing into text. ATS parsers scramble these.
    columned = sum(1 for ln in lines if re.search(r"\S {3,}\S", ln))
    if columned > max(4, len(lines) * 0.18):
        add(BLOCKER, "Layout looks like a table or multi-column design.",
            "Rebuild as a single column. Columns interleave into unreadable "
            "text when the ATS extracts them.")

    if re.search(r"[⬛■●█]|�", text or ""):
        add(WARNING, "Unreadable characters found in the extracted text.",
            "Embed standard fonts, or export the PDF from Word/Google Docs.")

    # --- length ----------------------------------------------------------
    # Below ~100 words the text is almost certainly a failed extraction, not a
    # real resume. Between 100 and 250 it's just thin, which is a different
    # problem with a different fix — so don't call both a blocker.
    if len(words) < 100:
        add(BLOCKER, f"Only {len(words)} words extracted — likely a parse failure.",
            "If the file has more content than this, the ATS can't read it. "
            "Export a text-based PDF rather than a scan or an image.")
    elif len(words) < 250:
        add(WARNING, f"Thin resume ({len(words)} words).",
            "Add detail on recent roles — what you built, and what changed as a result.")
    elif len(words) > 1200:
        add(TIP, f"Long resume ({len(words)} words).",
            "Trim to the most recent 10-15 years; two pages is the practical cap.")

    # --- bullets ---------------------------------------------------------
    bullets = [ln for ln in lines if re.match(r"^\s*[-•*‣●]", ln)]
    if not bullets:
        add(WARNING, "No bullet points detected.",
            "Use simple hyphen or • bullets. Dense paragraphs are hard for both "
            "parsers and humans to skim.")
    else:
        long_bullets = [b for b in bullets if len(b.split()) > 45]
        if long_bullets:
            add(TIP, f"{len(long_bullets)} bullet(s) run very long.",
                "Keep bullets under about 30 words — one accomplishment each.")

    if experience_section(text or "") == (text or ""):
        add(TIP, "No distinct experience section could be isolated.",
            "A clear EXPERIENCE heading helps parsers separate jobs from education.")

    # Blockers cost 25, warnings 10, tips 3 — a resume with two blockers is
    # genuinely at risk of never being read, and the score should say so.
    penalty = sum({BLOCKER: 25, WARNING: 10, TIP: 3}[f["severity"]] for f in findings)
    score = max(0, 100 - penalty)
    order = {BLOCKER: 0, WARNING: 1, TIP: 2}
    findings.sort(key=lambda f: order[f["severity"]])

    return {
        "ats_score": score,
        "n_blockers": sum(1 for f in findings if f["severity"] == BLOCKER),
        "n_warnings": sum(1 for f in findings if f["severity"] == WARNING),
        "findings": findings,
        "verdict": ("Parses cleanly" if score >= 85 else
                    "Minor parsing risks" if score >= 60 else
                    "Likely to be mis-parsed or filtered out"),
    }
