"""
Labelled resume dataset — graded, not binary.

What changed from v1 and why
----------------------------
v1 emitted a binary good/bad label. That is the wrong target for this problem,
and it showed: the model could only ever say "P(good)", so a genuinely strong
resume that didn't look like the generator's idea of "good" scored near zero.
There was no way to express "this is the 80th percentile".

Real systems don't use binary labels either. Recruiting data is naturally
*graded* and *relative*: recruiters rank a stack, and the usable signal is
funnel depth — screened, phone screen, onsite, offer. So we emit a 0-4 grade:

    0  auto-reject      1  weak      2  borderline      3  solid      4  strong

`train.py` then fits a regressor on the grade and converts predictions to a
PERCENTILE. "Better than 78% of the pool" is both honest and actionable, and it
doesn't collapse when the input doesn't look exactly like the training data.

The generator also varies surface form aggressively — header wording, date
formats, bullet glyphs, and industry shorthand ("k8s" vs "Kubernetes") — so the
model can't learn the template instead of the quality. That was the other half
of why v1 didn't transfer.

Real data: drop a JSONL at data/resumes.jsonl with {"text":..., "grade": 0-4}
(or {"label": 0|1}) per line and it's used instead.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

DATA_FILE = Path(__file__).parent / "data" / "resumes.jsonl"

GRADE_NAMES = {0: "auto-reject", 1: "weak", 2: "borderline", 3: "solid", 4: "strong"}
# A grade of 3+ is what "good" means when we need a binary view.
GOOD_GRADE_THRESHOLD = 3

FIRST = ["Alex", "Priya", "Jordan", "Wei", "Sam", "Nina", "Omar", "Elena", "Raj",
         "Chris", "Ana", "Yusuf", "Mei", "Tomas", "Ife", "Lena"]
LAST = ["Patel", "Kim", "Garcia", "Okafor", "Novak", "Silva", "Chen", "Haddad",
        "Rossi", "Doyle", "Nguyen", "Almeida", "Fischer", "Ortiz"]

TITLES = ["Software Engineer", "Senior Software Engineer", "Backend Engineer",
          "Data Engineer", "Machine Learning Engineer", "Full Stack Developer",
          "Platform Engineer", "Staff Engineer", "Site Reliability Engineer"]
COMPANIES = ["Northwind Systems", "Acme Corp", "Bluepeak Labs", "Vertex Analytics",
             "Corely", "Hexbyte", "Lumen Retail", "Kestrel Health", "Orbit Freight"]
SCHOOLS = ["State University", "Institute of Technology", "City College",
           "Metro University", "Riverside Tech"]

# Surface-form variation. If the model can only recognise one spelling of a
# section header, it has learned the template rather than the resume.
EXP_HEADERS = ["EXPERIENCE", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE",
               "Experience", "EMPLOYMENT HISTORY", "Work History"]
SKILL_HEADERS = ["SKILLS", "TECHNICAL SKILLS", "Skills", "CORE COMPETENCIES"]
EDU_HEADERS = ["EDUCATION", "Education", "EDUCATION & TRAINING"]
BULLETS = ["-", "•", "*", "‣"]

STRONG_BULLETS = [
    "Led migration of {n} microservices from a monolith to {k8s}, cutting deploy time by {p}%",
    "Designed a {kafka}-based event pipeline processing {n}M events/day at 99.9% uptime",
    "Optimized {pg} query plans, reducing p95 latency from 800ms to {q}ms",
    "Built a {torch} ranking model that lifted click-through rate by {p}%",
    "Automated {cicd} with Terraform, cutting release cycle from 2 weeks to {q} days",
    "Mentored {n} junior engineers and owned on-call for {n} production services",
    "Scaled {spark} ETL jobs to {n}TB/day, saving ${p}k annually in compute",
    "Shipped a {react} + FastAPI dashboard adopted by {n}00 internal users",
    "Cut {aws} spend {p}% by rightsizing workloads and adding autoscaling",
    "Introduced {tests}, raising coverage from 30% to {p}% and halving escaped defects",
]
MEDIUM_BULLETS = [
    "Developed and maintained REST APIs used by internal services",
    "Migrated legacy batch jobs to {spark} with the platform team",
    "Implemented {cicd} pipelines for three repositories",
    "Built internal tooling in {py} to support the data team",
    "Collaborated with product to deliver quarterly roadmap features",
]
WEAK_BULLETS = [
    "Responsible for writing code and fixing bugs",
    "Worked on various projects using different technologies",
    "Helped with testing and documentation tasks",
    "Participated in team meetings and code reviews",
    "Assisted with maintenance of existing applications",
    "Familiar with several programming languages",
    "Duties included supporting the development team",
]

# Same skill, different spellings — exercises the ontology's alias handling.
ALIASES = {
    "k8s": ["Kubernetes", "k8s", "K8s"],
    "kafka": ["Kafka", "Apache Kafka"],
    "pg": ["PostgreSQL", "Postgres"],
    "torch": ["PyTorch", "Torch"],
    "cicd": ["CI/CD", "GitHub Actions", "continuous integration"],
    "spark": ["Spark", "PySpark", "Apache Spark"],
    "react": ["React", "React.js"],
    "aws": ["AWS", "Amazon Web Services"],
    "tests": ["unit testing", "TDD", "pytest"],
    "py": ["Python", "Python3"],
}

STRONG_SKILLS = [
    "Python, Java, Go, TypeScript", "PostgreSQL, MongoDB, Redis, Elasticsearch",
    "AWS, Docker, k8s, Terraform, CI/CD", "Spark, Kafka, Airflow, SQL, Pandas",
    "PyTorch, sklearn, NLP, XGBoost", "React, FastAPI, Spring Boot, GraphQL, REST APIs",
    "Linux, Prometheus, Grafana, microservices",
]
WEAK_SKILLS = ["MS Office, Email, Teamwork", "HTML, CSS",
               "Communication, Time management", "Windows, Google Docs"]
CERTS = ["AWS Certified Solutions Architect - Associate (2023)",
         "Certified Kubernetes Administrator (CKA), 2022",
         "Professional Scrum Master I, 2021",
         "Google Cloud Professional Data Engineer, 2024"]

MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _fmt_date(rng: random.Random, year: int, month: int, style: int) -> str:
    if style == 0:
        return f"{MONTH[month - 1]} {year}"
    if style == 1:
        return f"{month:02d}/{year}"
    return str(year)


def _career(rng, n_roles: int, total_years: float, gaps: int):
    """Lay roles backwards from 2025 so the timeline stays consistent."""
    end_year, spans, remaining = 2025, [], total_years
    for i in range(n_roles):
        span = max(1, round(remaining / (n_roles - i)))
        start_year = end_year - span
        spans.append((start_year, rng.randint(1, 12), end_year, rng.randint(1, 12)))
        end_year = start_year
        remaining -= span
        if gaps > 0 and i < n_roles - 1 and rng.random() < 0.7:
            end_year -= rng.randint(1, 2)
            gaps -= 1
    return list(reversed(spans))


def _fill(rng: random.Random, template: str) -> str:
    """Fill numeric placeholders and pick a random spelling for each skill."""
    out = template.format(
        n=rng.randint(2, 40), p=rng.randint(15, 60), q=rng.randint(2, 90),
        **{k: rng.choice(v) for k, v in ALIASES.items()},
    )
    return out


def _p(rng: random.Random, base: float, slope: float, q: float) -> bool:
    """Include an optional section with probability base + slope*q.

    Why not `if q > 0.55`: a hard threshold makes section PRESENCE a near-perfect
    proxy for quality, and the model learns that shortcut instead of learning
    quality. It did exactly that — a one-line CERTIFICATIONS block moved a real
    resume from 27% to 56%, because in the old generator certifications only
    ever appeared above q=0.55.

    Keeping `base` large relative to `slope` means the section is weakly
    informative (as in real life) but never decisive, forcing the model onto the
    signals that actually matter: quantified bullets, skill evidence, tenure.
    """
    return rng.random() < base + slope * q


def _make_resume(rng: random.Random, q: float) -> str:
    """Build one resume whose richness scales with latent quality q in [0, 1]."""
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    # Verbosity is drawn INDEPENDENTLY of quality. Without this, length is a
    # proxy for the label (better resumes got more roles and more bullets), and
    # the model learns "longer = better" — which punishes a tight, strong resume
    # and rewards padding. In reality weak resumes are frequently the long ones.
    verbosity = rng.uniform(0.55, 1.8)
    date_style = rng.randint(0, 2)
    bullet = rng.choice(BULLETS)
    # Seniority is drawn INDEPENDENTLY of quality, on purpose.
    #
    # Tying years-of-experience to the quality label teaches the model that
    # "senior == good resume", which is both wrong and unfair: a strong junior
    # resume should be able to outscore a sloppy staff-engineer one. Experience
    # level is already scored separately, against the job's stated requirement,
    # in matching.py — that's the right place for it. Here, quality means how
    # well the work is evidenced, at whatever level.
    years = rng.uniform(1.0, 15.0)
    n_roles = max(1, min(5, round(years / rng.uniform(1.8, 4.0))))
    spans = _career(rng, n_roles, years, 0 if _p(rng, 0.45, 0.45, q) else rng.randint(0, 2))

    head = [name]
    if _p(rng, 0.75, 0.2, q):
        head.append(f"{name.split()[0].lower()}.{name.split()[1].lower()}@example.com"
                    f" | +1 (555) {rng.randint(100,999)}-{rng.randint(1000,9999)}")
    if _p(rng, 0.3, 0.45, q):
        head.append(f"linkedin.com/in/{name.split()[0].lower()}{rng.randint(10,99)}"
                    f" | github.com/{name.split()[0].lower()}dev")
    out = ["\n".join(head), ""]

    if _p(rng, 0.35, 0.4, q):
        out += [rng.choice(["SUMMARY", "PROFESSIONAL SUMMARY", "PROFILE"]),
                f"{rng.choice(TITLES)} with {int(years)} years building distributed "
                "systems. Owned services end to end, from design through on-call.", ""]

    out.append(rng.choice(EXP_HEADERS))
    for (sy, sm, ey, em) in spans:
        end = "Present" if ey >= 2025 else _fmt_date(rng, ey, em, date_style)
        out.append(f"{rng.choice(TITLES)} — {rng.choice(COMPANIES)}")
        out.append(f"{_fmt_date(rng, sy, sm, date_style)} – {end}")
        # Bullet COUNT comes from verbosity; bullet QUALITY comes from q.
        # Those must vary separately or count becomes a label leak.
        for _ in range(max(1, round(rng.randint(2, 4) * verbosity))):
            roll = rng.random()
            pool = STRONG_BULLETS if roll < q else (MEDIUM_BULLETS if roll < q + 0.3 else WEAK_BULLETS)
            out.append(f"{bullet} " + _fill(rng, rng.choice(pool)))
        out.append("")

    out.append(rng.choice(SKILL_HEADERS))
    # Mix the pools per line rather than switching wholesale: a weak resume can
    # still name real technologies, it just names fewer and demonstrates none.
    n_lines = max(1, round((1 + q * 2) * verbosity))
    chosen = [rng.choice(STRONG_SKILLS if rng.random() < 0.25 + 0.7 * q else WEAK_SKILLS)
              for _ in range(n_lines)]
    out.append(", ".join(dict.fromkeys(chosen)))
    out.append("")

    # Low-quality resumes often pad with generic filler — this is what makes
    # length genuinely uninformative about quality, as in real applicant pools.
    if q < 0.5 and verbosity > 1.2:
        out += [f"{bullet} " + rng.choice(WEAK_BULLETS)
                for _ in range(rng.randint(2, 5))] + [""]

    out.append(rng.choice(EDU_HEADERS))
    if _p(rng, 0.2, 0.4, q):
        out.append(f"M.S. Computer Science, {rng.choice(SCHOOLS)}, 2016 - 2018")
    out.append(f"B.Tech Computer Science, {rng.choice(SCHOOLS)}, 2012 - 2016")

    if _p(rng, 0.25, 0.4, q):
        out += ["", "CERTIFICATIONS"] + [f"{bullet} {c}"
                                         for c in rng.sample(CERTS, rng.randint(1, 3))]
    return "\n".join(out).strip()


def _grade_from_quality(q_noisy: float) -> int:
    """Cut the latent quality into 5 ordered bands.

    Boundaries are uneven on purpose: real applicant pools are bottom-heavy,
    so "strong" has to be rare or the grade carries no information.
    """
    for bound, grade in ((0.30, 0), (0.52, 1), (0.70, 2), (0.86, 3)):
        if q_noisy < bound:
            return grade
    return 4


def generate_graded_dataset(n: int = 1500, seed: int = 42) -> Tuple[List[str], List[int]]:
    """Return (texts, grades 0-4). Label noise keeps the bands overlapping."""
    rng = random.Random(seed)
    texts, grades = [], []
    for _ in range(n):
        q = rng.betavariate(2.2, 2.2) ** 0.75
        texts.append(_make_resume(rng, q))
        grades.append(_grade_from_quality(q + rng.gauss(0, 0.07)))
    return texts, grades


def generate_dataset(n: int = 1500, positive_rate: float = 0.25, seed: int = 42):
    """Binary view of the graded data, for the classification baseline."""
    texts, grades = generate_graded_dataset(n=n, seed=seed)
    return texts, [int(g >= GOOD_GRADE_THRESHOLD) for g in grades]


def load_graded_dataset(n: int = 1500, seed: int = 42):
    """Prefer real labelled data on disk; fall back to the generator."""
    if DATA_FILE.exists():
        texts, grades = [], []
        for line in DATA_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            texts.append(rec["text"])
            # Accept either schema; a binary label maps to the band edges.
            grades.append(int(rec["grade"]) if "grade" in rec
                          else (4 if int(rec["label"]) else 1))
        if texts:
            print(f"Loaded {len(texts)} labelled resumes from {DATA_FILE}")
            return texts, grades
    print(f"No {DATA_FILE.name} found — generating {n} synthetic resumes.")
    return generate_graded_dataset(n=n, seed=seed)


def load_dataset(n: int = 1500, seed: int = 42):
    texts, grades = load_graded_dataset(n=n, seed=seed)
    return texts, [int(g >= GOOD_GRADE_THRESHOLD) for g in grades]


if __name__ == "__main__":
    from collections import Counter
    texts, grades = generate_graded_dataset(1500)
    print("grade distribution:", sorted(Counter(grades).items()))
    for g in (0, 4):
        i = grades.index(g)
        print("=" * 70, f"\nGRADE {g} ({GRADE_NAMES[g]})\n", texts[i][:420])
