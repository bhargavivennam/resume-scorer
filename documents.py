"""
Getting plain text OUT of the things people actually have: PDFs, Word files,
and job-posting URLs.

Every scoring path downstream expects a plain string, so this module is the
single place where messy input becomes clean text.
"""

from __future__ import annotations

import io
import ipaddress
import re
import socket
from typing import Optional
from urllib.parse import urlparse

import json

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class ExtractionError(ValueError):
    """Raised when we can't get usable text out of the input."""


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------

# PDF fonts emit these as single glyphs. Left alone, "classiﬁcation" never
# matches the keyword "classification" — silently breaking skill matching on any
# resume exported from Word or a browser.
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
              "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "ft", "\ufb06": "st"}


def _normalise_glyphs(text: str) -> str:
    for lig, plain in _LIGATURES.items():
        text = text.replace(lig, plain)
    # Smart quotes and dashes trip exact-match keyword comparisons too.
    for fancy, plain in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'),
                         ("\u201d", '"'), ("\u00a0", " ")):
        text = text.replace(fancy, plain)
    return text


def _clean(text: str) -> str:
    """Normalise whitespace without destroying the line structure.

    Line breaks matter to us: `features.py` counts bullets and detects section
    headers with `^`-anchored regexes, so collapsing everything to one line
    would quietly zero out several features.
    """
    text = _normalise_glyphs(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)      # runs of spaces -> one space
    text = re.sub(r"\n{3,}", "\n\n", text)       # cap blank-line runs
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"Could not read PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")  # many resumes are "encrypted" with an empty password
        except Exception as exc:
            raise ExtractionError("PDF is password-protected.") from exc

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = _clean("\n".join(pages))

    if len(text) < 50:
        # Almost always a scanned/image-only resume. Say so plainly rather than
        # scoring an empty string and returning a confident, meaningless 0.
        raise ExtractionError(
            "No selectable text found — this looks like a scanned or image-only "
            "PDF. Export a text PDF from Word/Google Docs, or paste the text."
        )
    return text


def extract_docx(data: bytes) -> str:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"Could not read .docx: {exc}") from exc

    parts = [p.text for p in document.paragraphs]
    # Plenty of resume templates put the entire body inside a table.
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    text = _clean("\n".join(parts))
    if len(text) < 20:
        raise ExtractionError("The .docx appears to contain no text.")
    return text


def extract_text_from_upload(filename: str, data: bytes) -> str:
    """Dispatch on file extension. `.doc` (old binary Word) is not supported."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ExtractionError(f"File is larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return extract_pdf(data)
    if name.endswith(".docx"):
        return extract_docx(data)
    if name.endswith((".txt", ".md")):
        text = _clean(data.decode("utf-8", errors="replace"))
        if len(text) < 20:
            raise ExtractionError("File appears to be empty.")
        return text
    if name.endswith(".doc"):
        raise ExtractionError("Legacy .doc isn't supported — save as .docx or PDF.")
    raise ExtractionError(f"Unsupported file type. Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")


# --------------------------------------------------------------------------
# Job posting URLs
# --------------------------------------------------------------------------

# Boards that reliably serve bots a login wall or JS shell instead of the post.
KNOWN_BLOCKERS = ("linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com")


def _assert_public_url(url: str) -> None:
    """Refuse anything that isn't a public http(s) address.

    The server fetches a URL the user supplies, so without this check the
    endpoint would happily read internal services (SSRF) — 169.254.169.254,
    localhost admin panels, private subnets.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ExtractionError("Only http:// and https:// URLs are supported.")
    if not parsed.hostname:
        raise ExtractionError("That doesn't look like a valid URL.")
    try:
        resolved = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ExtractionError(f"Could not resolve {parsed.hostname}.") from exc
    for info in resolved:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ExtractionError("Refusing to fetch a private or internal address.")


def _html_to_text(raw: str) -> str:
    import html as html_mod

    from bs4 import BeautifulSoup

    # Greenhouse (and some others) return the description as an HTML-ESCAPED
    # string, so "<p>" arrives as "&lt;p&gt;". Unescaping first turns that back
    # into real markup; without it the tags end up as visible literal text.
    raw = raw or ""
    if "&lt;" in raw and "<" not in raw:
        raw = html_mod.unescape(raw)
    soup = BeautifulSoup(raw, "html.parser")
    # <li> and <br> carry the bullet structure our features count; without an
    # explicit separator BeautifulSoup would run them all together.
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return _clean(soup.get_text("\n"))


def _try_board_api(url: str, timeout: int) -> Optional[str]:
    """Use a job board's own JSON API when we recognise the URL shape.

    Far more reliable than scraping the page: these boards render the posting
    client-side, so the HTML we'd get back is a near-empty JavaScript shell
    while the API returns the real description.
    """
    import requests

    host = (urlparse(url).hostname or "").lower()
    parts = [p for p in urlparse(url).path.split("/") if p]

    try:
        # boards.greenhouse.io/<board>/jobs/<id>  |  job-boards.greenhouse.io/<board>/jobs/<id>
        if "greenhouse.io" in host and "jobs" in parts:
            board = parts[0]
            job_id = parts[parts.index("jobs") + 1]
            api = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}"
            data = requests.get(api, timeout=timeout).json()
            body = _html_to_text(data.get("content", ""))
            return f"{data.get('title', '')}\n\n{body}".strip() or None

        # jobs.lever.co/<company>/<id>
        if "lever.co" in host and len(parts) >= 2:
            api = f"https://api.lever.co/v0/postings/{parts[0]}/{parts[1]}"
            data = requests.get(api, timeout=timeout).json()
            chunks = [data.get("text", ""), _html_to_text(data.get("description", ""))]
            for lst in data.get("lists", []):
                chunks.append(lst.get("text", ""))
                chunks.append(_html_to_text(lst.get("content", "")))
            return _clean("\n\n".join(c for c in chunks if c)) or None
    except Exception:
        return None  # fall through to generic scraping
    return None


def _try_jsonld(soup) -> Optional[str]:
    """Pull schema.org JobPosting data out of the page.

    Most boards embed this for Google Jobs indexing, and it survives even when
    the visible page is JavaScript-rendered — so it's the single highest-yield
    generic extractor.
    """
    def walk(node):
        if isinstance(node, list):
            for item in node:
                yield from walk(item)
        elif isinstance(node, dict):
            types = node.get("@type")
            types = types if isinstance(types, list) else [types]
            if "JobPosting" in types:
                yield node
            for key in ("@graph", "itemListElement", "mainEntity"):
                if key in node:
                    yield from walk(node[key])

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(tag.string or "{}")
        except Exception:
            continue
        for posting in walk(payload):
            body = _html_to_text(posting.get("description", ""))
            if len(body) < 150:
                continue
            title = posting.get("title", "")
            org = posting.get("hiringOrganization") or {}
            org_name = org.get("name", "") if isinstance(org, dict) else ""
            header = " — ".join(p for p in (title, org_name) if p)
            return f"{header}\n\n{body}".strip()
    return None


def fetch_job_description(url: str, timeout: int = 12) -> str:
    """Best-effort extraction of a job posting, cheapest reliable method first."""
    import requests
    from bs4 import BeautifulSoup

    _assert_public_url(url)
    host = urlparse(url).hostname or ""

    # 1. The board's own API, if we recognise it.
    from_api = _try_board_api(url, timeout)
    if from_api and len(from_api) > 200:
        return from_api[:20_000]

    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ResumeScorer/1.0)"},
            allow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as exc:
        if any(b in host for b in KNOWN_BLOCKERS):
            raise ExtractionError(
                f"{host} blocks automated fetching (bot protection), so no tool can "
                "read it from a link. Open the posting, select the job description, "
                "and paste it into the box below — that works identically. Tip: many "
                f"{host} listings link to the company's own careers page, and those "
                "URLs usually do fetch."
            ) from exc
        raise ExtractionError(
            f"Could not reach {host}. Check the link, or paste the description instead."
        ) from exc

    soup = BeautifulSoup(resp.text, "html.parser")

    # 2. schema.org JobPosting — present on most boards, survives JS rendering.
    from_jsonld = _try_jsonld(soup)
    if from_jsonld:
        return from_jsonld[:20_000]

    # 3. Fall back to the visible text of the main content block.
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = _clean(main.get_text("\n"))

    if len(text) < 200:
        raise ExtractionError(
            f"Fetched {host} but found almost no text — the posting is probably "
            "rendered by JavaScript or behind a login. Paste the description instead."
        )
    if any(b in host for b in KNOWN_BLOCKERS):
        # We got *something*, but these sites often return a login wall that
        # still parses as text. Warn rather than silently scoring garbage.
        lowered = text.lower()
        if "sign in" in lowered[:600] or "join now" in lowered[:600]:
            raise ExtractionError(
                f"{host} returned a login wall rather than the posting. "
                "Paste the description text instead."
            )
    return text[:20_000]
