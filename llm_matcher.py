"""
Resume/job matching by asking an LLM, as a third opinion alongside the
rule-based matcher (`matching.py`) and the embedding matcher (`semantic.py`).

Why this exists
---------------
The three approaches fail in different ways, and the point of having all three
is to be able to see that rather than argue about it:

    rules       deterministic, auditable, explains itself precisely,
                blind to anything outside the ontology
    embeddings  catches paraphrase, deterministic, cannot explain WHY
    LLM         reads the posting the way a recruiter would, cites evidence,
                and returns a different number every time you ask

That last property is the interesting one and the reason to measure rather than
assume. Run `compare_matchers.py` to see it.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...       # never commit this
    python llm_matcher.py resume.txt job.txt

Costs real money — roughly a cent per call at current Opus pricing on a
resume-sized input. Every entry point returns None instead of raising when the
SDK or credentials are missing, so the rest of the project keeps working.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

MODEL = "claude-opus-5"
MAX_TOKENS = 16000

SYSTEM = """You are screening a candidate's resume against a job description, \
the way an experienced technical recruiter would.

Rules you must follow:
- Judge ONLY on what the resume actually says. Never infer a skill the \
candidate did not mention, and never credit experience that is not written down.
- Every requirement you mark as addressed MUST quote the exact line from the \
resume that addresses it. If you cannot quote a line, it is not addressed.
- Distinguish a skill that is merely listed from one demonstrated with real \
work and a measurable outcome. A skills-section keyword is weak evidence.
- Be calibrated, not generous. A score of 80+ means the candidate could start \
this job now. Most real applicants are not that.
"""


class RequirementJudgement(BaseModel):
    requirement: str = Field(description="The requirement, quoted from the posting")
    verdict: Literal["addressed", "partial", "unaddressed"]
    evidence: Optional[str] = Field(
        default=None,
        description="Exact line quoted from the resume, or null if unaddressed")
    reasoning: str = Field(description="One sentence on why this verdict")


class LLMMatch(BaseModel):
    match_score: int = Field(ge=0, le=100, description="Overall fit, 0-100")
    requirements: List[RequirementJudgement]
    missing_required: List[str] = Field(
        description="Requirements the candidate clearly does not meet")
    strongest_point: str
    biggest_gap: str
    summary: str = Field(description="Two sentences a recruiter could act on")


def _ant_profile_active() -> bool:
    """Whether the `ant` CLI reports an active credential profile.

    Split out so tests can stub this one call. Patching `subprocess.run`
    globally instead segfaulted the suite — torch shells out during import,
    and a monkeypatched subprocess took it down with it.
    """
    try:
        import subprocess

        done = subprocess.run(["ant", "auth", "status"], capture_output=True,
                              timeout=5, text=True)
        return done.returncode == 0
    except Exception:
        return False


def _has_credentials() -> bool:
    """Whether a credential the SDK can use actually exists.

    Constructing `anthropic.Anthropic()` does NOT validate credentials — it
    succeeds with none and raises a TypeError later, at request time. An
    availability check built on construction alone therefore reports True on a
    machine that cannot make a single call, which is exactly the wrong answer
    for a caller deciding whether to try.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # `ant auth login` stores a profile the SDKs read automatically, so an
    # unset env var is not on its own proof that there are no credentials.
    return _ant_profile_active()


def _client():
    """Anthropic client, or None. Never raises."""
    try:
        import anthropic
    except ImportError:
        return None
    try:
        # Zero-arg constructor resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
        # or an `ant auth login` profile — so an unset env var does not
        # necessarily mean there are no credentials.
        return anthropic.Anthropic()
    except Exception:
        return None


def match(resume_text: str, job_text: str, temperature_note: str = "") -> Optional[Dict[str, Any]]:
    """Score a resume against a posting. Returns None if the LLM is unavailable.

    `temperature_note` is appended to the prompt purely so callers can vary the
    input when measuring run-to-run variance; leave it empty in normal use.
    """
    client = _client()
    if client is None:
        return None
    if not resume_text.strip() or not job_text.strip():
        raise ValueError("resume_text and job_text must both be non-empty")

    prompt = (
        f"JOB DESCRIPTION\n{'=' * 60}\n{job_text.strip()[:20000]}\n\n"
        f"RESUME\n{'=' * 60}\n{resume_text.strip()[:20000]}\n\n"
        "Judge every stated requirement in the posting, then give an overall "
        f"match score.{temperature_note}"
    )

    try:
        import anthropic

        response = client.messages.parse(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=LLMMatch,
        )
    except ImportError:
        return None
    except TypeError as exc:
        # The SDK raises a bare TypeError when it cannot resolve any credential.
        raise RuntimeError(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY in your shell "
            "(get one at console.anthropic.com), or install the `ant` CLI and run "
            "`ant auth login`."
        ) from exc
    except Exception as exc:  # typed chain below for the cases worth separating
        import anthropic as _a

        if isinstance(exc, _a.AuthenticationError):
            raise RuntimeError(
                "Anthropic rejected the credentials. Set ANTHROPIC_API_KEY or run "
                "`ant auth login`.") from exc
        if isinstance(exc, _a.RateLimitError):
            raise RuntimeError("Rate limited by the Anthropic API; retry shortly.") from exc
        if isinstance(exc, _a.APIStatusError):
            raise RuntimeError(f"Anthropic API error {exc.status_code}: {exc}") from exc
        if isinstance(exc, _a.APIConnectionError):
            raise RuntimeError("Could not reach the Anthropic API.") from exc
        raise

    # A refusal returns HTTP 200 with stop_reason "refusal" and no usable
    # content, so it has to be checked before reading the parsed output.
    if getattr(response, "stop_reason", None) == "refusal":
        detail = getattr(response, "stop_details", None)
        raise RuntimeError(f"Model declined to answer: {detail}")

    result = response.parsed_output.model_dump()
    result["model"] = MODEL
    result["usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return result


def is_available() -> bool:
    """True only when a call would actually be attempted with a credential."""
    return _client() is not None and _has_credentials()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        sys.exit("usage: python llm_matcher.py <resume.txt|pdf> <job.txt>")
    if not is_available():
        sys.exit("No Anthropic credentials. export ANTHROPIC_API_KEY=sk-ant-...")

    def _read(path: str) -> str:
        if path.lower().endswith((".pdf", ".docx")):
            from documents import extract_text_from_upload

            return extract_text_from_upload(path, open(path, "rb").read())
        return open(path, encoding="utf-8").read()

    out = match(_read(sys.argv[1]), _read(sys.argv[2]))
    print(json.dumps(out, indent=2))
