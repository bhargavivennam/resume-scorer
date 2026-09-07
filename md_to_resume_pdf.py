"""
Render the Markdown resume to an ATS-parseable PDF.

ATS-safety rules baked into the CSS (these are what `ats.py` checks for):
  * SINGLE COLUMN — no tables, no floats. Multi-column layouts interleave into
    unreadable text when a parser extracts them.
  * Real selectable text in standard fonts — never an image or a scan.
  * Plain section headings (SKILLS, EXPERIENCE, EDUCATION) that parsers look for.
  * Real <ul>/<li> bullets, which extract as "- " lines.
"""
import re
import subprocess
import sys
from pathlib import Path

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: Letter; margin: 0.5in 0.55in; }
* { box-sizing: border-box; }
body { font-family: Calibri, "Helvetica Neue", Arial, sans-serif;
       font-size: 10pt; line-height: 1.34; color: #1a1a1a; margin: 0;
       /* Ligatures render "fi" as a single glyph that extracts as U+FB01, so
          "classification" comes out as "classiﬁcation" and stops matching. */
       font-variant-ligatures: none; -webkit-font-variant-ligatures: none; }
h1 { font-size: 21pt; margin: 0 0 1pt; letter-spacing: -0.3pt; }
h1 + p { margin: 0 0 2pt; font-size: 11pt; font-weight: 600; color: #333; }
h2 { font-size: 11pt; text-transform: uppercase; letter-spacing: 0.6pt;
     border-bottom: 1px solid #999; padding-bottom: 2pt;
     margin: 13pt 0 6pt; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 9pt 0 1pt; page-break-after: avoid; }
h3 + p { margin: 0 0 3pt; font-style: italic; color: #555; font-size: 9pt; }
p { margin: 0 0 4pt; }
ul { margin: 3pt 0 6pt; padding-left: 0; list-style: none; }
li { margin-bottom: 2.5pt; page-break-inside: avoid;
     padding-left: 11pt; text-indent: -11pt; }
strong { font-weight: 700; }
hr { display: none; }
a { color: #1a1a1a; text-decoration: none; }
.contact { font-size: 9.5pt; color: #333; margin-bottom: 2pt; }
"""


def convert(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    # Strip the editing annotation — it must never reach an employer.
    text = re.sub(r"\s*←.*$", "", text, flags=re.M)

    html_body = markdown.markdown(text, extensions=["extra", "sane_lists"])
    # Chrome draws <li> markers as positioned glyphs that pypdf drops entirely,
    # so the extracted text had ZERO bullet lines — and every ATS heuristic that
    # counts bullets saw a wall of prose. Putting the character in the markup
    # makes it real text in the PDF content stream.
    html_body = html_body.replace("<li>", "<li>\u2022 ")
    # The two lines under the name are contact info, not prose.
    html_body = html_body.replace("<p>Irving, TX", '<p class="contact">Irving, TX')
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<style>{CSS}</style></head><body>{html_body}</body></html>")

    tmp_html = pdf_path.with_suffix(".tmp.html")
    tmp_html.write_text(html, encoding="utf-8")
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", tmp_html.as_uri()],
        check=True, capture_output=True, timeout=90)
    tmp_html.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python md_to_resume_pdf.py <resume.md> [out.pdf]")
    src = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else str(src.with_suffix(".pdf")))
    convert(src, out)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
