"""
Skill ontology with alias normalisation, evidence weighting and recency decay.

Why this module exists
----------------------
v1 matched skills by exact string. That fails the way every naive ATS fails:

  * "k8s", "K8s" and "Kubernetes" were three different things (two of them
    invisible), so a candidate got penalised for using the industry shorthand;
  * a skill listed once in a keyword-stuffed SKILLS block counted exactly as
    much as a skill demonstrated in a bullet with a measured outcome;
  * Python used in 2014 counted the same as Python used last year.

Commercial systems (Lightcast/EMSI, ESCO, O*NET) solve the first with a curated
skill taxonomy of aliases. The second and third are what separates a useful
matcher from keyword search, and they're what recruiters actually do when they
read a resume: they look for *evidence* and they look at *when*.

Three signals per skill
-----------------------
  evidence  0-3  where the skill appears (see EVIDENCE_* below)
  recency   0-1  exponential decay from the last role that mentions it
  count     int  raw mentions, kept for debugging (deliberately NOT scored -
                 rewarding raw frequency is what invites keyword stuffing)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REFERENCE_YEAR = 2026  # fixed "today" so scores are reproducible across runs

# Evidence tiers, strongest first. The gap between LISTED and QUANTIFIED is the
# whole point: anyone can type "Kubernetes" into a skills list.
EVIDENCE_QUANTIFIED = 3   # in a bullet that also states a measured result
EVIDENCE_EXPERIENCE = 2   # in a work-experience bullet
EVIDENCE_PROJECT = 1.5    # in a projects / education section
EVIDENCE_LISTED = 1       # only in a skills list or elsewhere
EVIDENCE_NONE = 0

# Credit awarded per evidence tier. See SkillSignal.strength for the reasoning.
EVIDENCE_STRENGTH = {
    EVIDENCE_NONE: 0.0,
    EVIDENCE_LISTED: 0.35,
    EVIDENCE_PROJECT: 0.55,
    EVIDENCE_EXPERIENCE: 0.80,
    EVIDENCE_QUANTIFIED: 1.0,
}

# Half-life of a skill in years: how fast unused experience stops counting.
RECENCY_HALFLIFE_YEARS = 4.0

# --------------------------------------------------------------------------
# The ontology: canonical name -> (group, aliases)
# --------------------------------------------------------------------------
# Aliases are matched case-insensitively with non-word boundaries, so "js"
# will NOT fire inside "json". Order within a group doesn't matter; the
# matcher sorts by length so "spring boot" wins over "spring".

ONTOLOGY: Dict[str, Tuple[str, List[str]]] = {
    # --- languages --------------------------------------------------------
    "python":      ("languages", ["python3", "py3"]),
    "java":        ("languages", ["java8", "java 8", "java 11", "java 17"]),
    "javascript":  ("languages", ["js", "ecmascript", "es6"]),
    "typescript":  ("languages", ["ts"]),
    "c++":         ("languages", ["cpp", "c plus plus"]),
    "c#":          ("languages", ["csharp", "c sharp", ".net", "dotnet"]),
    "go":          ("languages", ["golang"]),
    "rust":        ("languages", []),
    "scala":       ("languages", []),
    "kotlin":      ("languages", []),
    "ruby":        ("languages", ["ruby on rails", "rails"]),
    "php":         ("languages", ["laravel"]),
    "swift":       ("languages", ["swiftui"]),
    "r":           ("languages", []),
    "matlab":      ("languages", []),
    "bash":        ("languages", ["shell scripting", "shell script", "zsh"]),

    # --- data -------------------------------------------------------------
    "sql":         ("data", ["ansi sql", "t-sql", "tsql", "pl/sql", "plsql"]),
    "pandas":      ("data", []),
    "numpy":       ("data", []),
    "spark":       ("data", ["apache spark", "pyspark"]),
    "hadoop":      ("data", ["mapreduce", "hdfs"]),
    "airflow":     ("data", ["apache airflow"]),
    "dbt":         ("data", []),
    "etl":         ("data", ["elt", "data pipeline", "data pipelines"]),
    "tableau":     ("data", []),
    "power bi":    ("data", ["powerbi"]),
    "snowflake":   ("data", []),
    "databricks":  ("data", []),
    "kafka":       ("data", ["apache kafka"]),
    "flink":       ("data", ["apache flink"]),
    "data warehouse": ("data", ["data warehousing", "dwh"]),

    # --- ml ---------------------------------------------------------------
    "machine learning": ("ml", ["ml", "statistical learning"]),
    "deep learning":    ("ml", ["neural networks", "neural network"]),
    "pytorch":          ("ml", ["torch"]),
    "tensorflow":       ("ml", ["tf2", "keras"]),
    "scikit-learn":     ("ml", ["sklearn", "scikit learn"]),
    "nlp":              ("ml", ["natural language processing"]),
    "computer vision":  ("ml", ["cv", "image recognition"]),
    "llm":              ("ml", ["large language model", "large language models",
                                "gpt", "rag", "prompt engineering"]),
    "transformers":     ("ml", ["bert", "hugging face", "huggingface"]),
    "xgboost":          ("ml", []),
    "lightgbm":         ("ml", ["lgbm"]),
    "mlops":            ("ml", ["ml ops", "model deployment", "mlflow"]),
    "recommender systems": ("ml", ["recommendation systems", "recsys"]),

    # --- cloud / devops ---------------------------------------------------
    "aws":         ("cloud_devops", ["amazon web services", "ec2", "s3", "lambda"]),
    "azure":       ("cloud_devops", ["microsoft azure"]),
    "gcp":         ("cloud_devops", ["google cloud", "google cloud platform", "bigquery"]),
    "docker":      ("cloud_devops", ["containerization", "containers"]),
    "kubernetes":  ("cloud_devops", ["k8s", "eks", "gke", "aks"]),
    "terraform":   ("cloud_devops", ["infrastructure as code", "iac"]),
    "jenkins":     ("cloud_devops", []),
    "ci/cd":       ("cloud_devops", ["cicd", "ci cd", "continuous integration",
                                     "continuous delivery", "github actions", "gitlab ci"]),
    "ansible":     ("cloud_devops", []),
    "linux":       ("cloud_devops", ["unix", "ubuntu", "centos"]),
    "prometheus":  ("cloud_devops", []),
    "grafana":     ("cloud_devops", []),
    "microservices": ("cloud_devops", ["microservice", "service oriented architecture"]),
    "git":         ("cloud_devops", ["github", "gitlab", "version control"]),

    # --- databases --------------------------------------------------------
    "postgresql":  ("databases", ["postgres", "psql"]),
    "mysql":       ("databases", ["mariadb"]),
    "mongodb":     ("databases", ["mongo"]),
    "redis":       ("databases", []),
    "cassandra":   ("databases", []),
    "elasticsearch": ("databases", ["elastic search", "opensearch"]),
    "dynamodb":    ("databases", []),
    "oracle":      ("databases", ["oracle db"]),
    "sqlite":      ("databases", []),

    # --- web --------------------------------------------------------------
    "react":       ("web", ["react.js", "reactjs", "react native"]),
    "angular":     ("web", ["angularjs"]),
    "vue":         ("web", ["vue.js", "vuejs"]),
    "node.js":     ("web", ["nodejs", "node js"]),
    "django":      ("web", []),
    "flask":       ("web", []),
    "fastapi":     ("web", []),
    "spring boot": ("web", ["spring", "spring framework"]),
    "graphql":     ("web", []),
    "rest api":    ("web", ["restful", "rest apis", "restful api", "restful apis"]),
    "html":        ("web", ["html5"]),
    "css":         ("web", ["css3", "sass", "scss", "tailwind"]),
    "grpc":        ("web", []),

    # --- practices (things job posts ask for that aren't tools) -----------
    "agile":       ("practices", ["scrum", "kanban", "sprint planning"]),
    "on-call":     ("practices", ["on call", "oncall", "pagerduty", "incident response",
                                  "production support"]),
    "system design": ("practices", ["distributed systems", "architecture design"]),
    "testing":     ("practices", ["unit testing", "integration testing", "tdd",
                                  "test driven development", "pytest", "junit",
                                  "tested", "test coverage"]),
    "code review": ("practices", ["peer review", "reviewed code", "reviewing code"]),
    # Verb forms matter: resumes say "Mentored engineers", postings say
    # "experience mentoring". Without both spellings the requirement looks
    # unaddressed even when the resume states it outright.
    "mentoring":   ("practices", ["mentorship", "coaching", "onboarding engineers",
                                  "mentored", "mentors", "mentor"]),

    # --- soft skills (scored as their own category, as Jobscan/Teal do) ---
    "communication":   ("soft", ["written communication", "verbal communication",
                                 "presenting", "presentation skills", "documentation"]),
    "collaboration":   ("soft", ["cross-functional", "cross functional", "teamwork",
                                 "partnering", "worked with product", "stakeholders",
                                 "stakeholder management"]),
    "leadership":      ("soft", ["led team", "team lead", "tech lead", "leading",
                                 "ownership", "drove", "spearheaded"]),
    "problem solving": ("soft", ["troubleshooting", "debugging", "root cause",
                                 "analytical"]),
    "customer focus":  ("soft", ["customer facing", "user focused", "business context",
                                 "customer use cases", "end users"]),
    "adaptability":    ("soft", ["fast paced", "ambiguity", "self starter",
                                 "self-starter"]),
    "scalability": ("practices", ["scalable", "scaling", "high availability",
                                  "reliability", "reliable systems", "fault tolerant"]),
}

# --------------------------------------------------------------------------
# Skill implications: narrower skill -> broader skills it demonstrates
# --------------------------------------------------------------------------
# ESCO and Lightcast both model broader/narrower relations, and the omission
# showed: a resume with PostgreSQL was scored as "missing SQL", which no human
# reviewer would ever say. Implied skills are credited at REDUCED evidence —
# they're inferred, not stated, and shouldn't outrank a demonstrated mention.

IMPLIES: Dict[str, List[str]] = {
    "postgresql": ["sql"], "mysql": ["sql"], "oracle": ["sql"],
    "sqlite": ["sql"], "snowflake": ["sql"], "databricks": ["spark"],
    "pytorch": ["machine learning", "deep learning", "python"],
    "tensorflow": ["machine learning", "deep learning"],
    "scikit-learn": ["machine learning", "python"],
    "xgboost": ["machine learning"], "lightgbm": ["machine learning"],
    "transformers": ["nlp", "deep learning"], "llm": ["nlp"],
    "pandas": ["python"], "numpy": ["python"], "django": ["python"],
    "flask": ["python"], "fastapi": ["python"], "pyspark": ["python"],
    "spring boot": ["java"], "react": ["javascript"], "angular": ["javascript"],
    "vue": ["javascript"], "node.js": ["javascript"], "typescript": ["javascript"],
    "kubernetes": ["docker", "microservices", "linux"],
    "terraform": ["ci/cd"], "jenkins": ["ci/cd"],
    "spark": ["etl", "data warehouse"], "airflow": ["etl"], "dbt": ["etl", "sql"],
    "kafka": ["distributed systems", "system design"],
    "microservices": ["system design"], "grpc": ["rest api"],
    "prometheus": ["linux"], "grafana": ["linux"],
}

# Credit multiplier for a skill that was inferred rather than written down.
IMPLIED_EVIDENCE_FACTOR = 0.6

GROUPS: List[str] = sorted({g for g, _ in ONTOLOGY.values()})

# alias (lowercase) -> canonical
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _canon, (_group, _aliases) in ONTOLOGY.items():
    _ALIAS_TO_CANONICAL[_canon] = _canon
    for _a in _aliases:
        _ALIAS_TO_CANONICAL[_a] = _canon

# Longest first so "spring boot" is consumed before "spring", "google cloud
# platform" before "gcp", etc.
_SURFACE_FORMS: List[str] = sorted(_ALIAS_TO_CANONICAL, key=len, reverse=True)

# One compiled alternation instead of ~250 separate regex passes. The guards are
# lookarounds rather than \b because \b breaks on 'c++', 'node.js' and 'ci/cd'.
#
# Plurals are allowed only on forms of 5+ letters. Applying an optional "s"/"es"
# to everything would make "go" match "goes" and "r" match "rs" — the short
# skills are exactly the ones where a loose suffix creates false positives.
_LONG = [f for f in _SURFACE_FORMS if len(f) >= 5 and f[-1].isalpha()]
_SHORT = [f for f in _SURFACE_FORMS if f not in set(_LONG)]

_SKILL_RE = re.compile(
    r"(?<![\w+#.])(?:"
    + r"(?P<long>" + "|".join(re.escape(f) for f in _LONG) + r")(?:e?s)?"
    + r"|(?P<short>" + "|".join(re.escape(f) for f in _SHORT) + r")"
    + r")(?![\w+#.-])",
    re.IGNORECASE,
)

ALL_SKILLS: List[str] = sorted(ONTOLOGY)


def canonicalize(surface: str) -> Optional[str]:
    """Map any known surface form ('k8s', 'K8S', 'Kubernetes') to 'kubernetes'."""
    return _ALIAS_TO_CANONICAL.get(surface.strip().lower())


def find_skills(text: str) -> set[str]:
    """Every canonical skill mentioned anywhere in the text."""
    found: set[str] = set()
    for m in _SKILL_RE.finditer(text or ""):
        surface = m.group("long") or m.group("short")
        canonical = _ALIAS_TO_CANONICAL.get((surface or "").lower())
        if canonical:
            found.add(canonical)
    return found


def group_of(skill: str) -> Optional[str]:
    entry = ONTOLOGY.get(skill)
    return entry[0] if entry else None


# --------------------------------------------------------------------------
# Evidence + recency
# --------------------------------------------------------------------------

_QUANTIFIED_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:%|percent|x\b|k\b|m\b|bn\b|million|billion)|[$€£]\s?\d)", re.I)


@dataclass
class SkillSignal:
    """What we know about one skill on one resume."""
    skill: str
    evidence: float          # 0-3, see EVIDENCE_* constants
    recency: float           # 0-1, 1.0 = used in a current role
    count: int
    last_used_year: Optional[int]

    @property
    def strength(self) -> float:
        """Single 0-1 number combining where it appeared and how recently.

        The evidence curve is deliberately NOT linear in the tier number. A
        skill actually used on the job is worth most of full credit (0.80) —
        penalising it to 2/3 would punish ordinary good resumes that simply
        don't quantify every bullet. The steep drop is between "used" and
        "merely listed" (0.80 -> 0.35), which is the distinction that matters
        and the one keyword matching cannot make.
        """
        base = EVIDENCE_STRENGTH.get(self.evidence, self.evidence / EVIDENCE_QUANTIFIED)
        return base * (0.6 + 0.4 * self.recency)


def _recency_weight(last_year: Optional[int], reference_year: int = REFERENCE_YEAR) -> float:
    """Exponential decay on years since last use. No date -> neutral 0.5."""
    if last_year is None:
        return 0.5
    years_ago = max(0, reference_year - last_year)
    return math.pow(0.5, years_ago / RECENCY_HALFLIFE_YEARS)


# Projects are real evidence, but professional work carries a little more
# weight, so project evidence is scaled rather than counted identically.
PROJECT_EVIDENCE_FACTOR = 0.85


def analyse_skills(
    text: str,
    roles: Optional[Sequence[Tuple[str, Optional[int]]]] = None,
    skills_section: str = "",
    project_roles: Optional[Sequence[Tuple[str, Optional[int]]]] = None,
) -> Dict[str, SkillSignal]:
    """Grade every skill on the resume by evidence and recency.

    `roles` is (role_text, end_year) per position, newest end_year wins for
    recency. Passing it is what makes this better than keyword search; without
    it every skill degrades gracefully to "listed, unknown date".
    """
    signals: Dict[str, SkillSignal] = {}

    def bump(skill: str, evidence: float, year: Optional[int]) -> None:
        cur = signals.get(skill)
        if cur is None:
            signals[skill] = SkillSignal(skill, evidence, _recency_weight(year), 1, year)
            return
        cur.count += 1
        cur.evidence = max(cur.evidence, evidence)
        if year is not None and (cur.last_used_year is None or year > cur.last_used_year):
            cur.last_used_year = year
            cur.recency = _recency_weight(year)

    # Strongest signal first: skills demonstrated inside work-experience bullets.
    for role_text, end_year in (roles or []):
        for line in role_text.splitlines():
            found = find_skills(line)
            if not found:
                continue
            level = EVIDENCE_QUANTIFIED if _QUANTIFIED_RE.search(line) else EVIDENCE_EXPERIENCE
            for skill in found:
                bump(skill, level, end_year)

    # Project bullets earn evidence exactly like work bullets — a project
    # bullet stating a measured result proves the skill was used — scaled by
    # PROJECT_EVIDENCE_FACTOR.
    for project_text, year in (project_roles or []):
        for line in project_text.splitlines():
            found = find_skills(line)
            if not found:
                continue
            level = EVIDENCE_QUANTIFIED if _QUANTIFIED_RE.search(line) else EVIDENCE_EXPERIENCE
            for skill in found:
                bump(skill, level * PROJECT_EVIDENCE_FACTOR, year)

    # Anything else on the resume (skills list, summary, certifications).
    for skill in find_skills(text):
        bump(skill, EVIDENCE_LISTED, None)

    # Finally, credit the broader skills that the stated ones demonstrate.
    # Done last, and at reduced evidence, so an inferred skill can never
    # outrank the same skill actually written on the resume.
    for skill, sig in list(signals.items()):
        for implied in IMPLIES.get(skill, []):
            if implied in ONTOLOGY:
                bump(implied, sig.evidence * IMPLIED_EVIDENCE_FACTOR, sig.last_used_year)

    return signals


def group_coverage(signals: Dict[str, SkillSignal]) -> Dict[str, float]:
    """Total skill strength per ontology group — a breadth measure that
    can't be gamed by listing ten flavours of the same thing."""
    out = {g: 0.0 for g in GROUPS}
    for sig in signals.values():
        g = group_of(sig.skill)
        if g:
            out[g] += sig.strength
    return out
