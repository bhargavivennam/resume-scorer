# Resume Scorer

Scores resume quality 0–100 and matches resumes against job descriptions —
with an explanation for every point, an API, and a React front end.

Built to answer a question keyword-matching ATS can't: **is this skill
demonstrated, or just listed?** Two resumes containing 100% of a posting's
required keywords score 33 points apart here, because one proves the skills in
dated bullets with measured outcomes and the other only claims them.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py && pytest tests -q
uvicorn api:app --reload --port 8000     # http://localhost:8000/docs
cd ui && npm install && npm run dev       # http://localhost:5173
```

**119 tests.** Scoring methodology verified against how Jobscan, Teal and Huntr
actually score matches — see [What the match score measures](#what-the-match-score-measures-and-why).

### Things this project got wrong first, and what fixed them

Documented in full below, because they're the interesting part:

- **Shortcut learning.** A single `CERTIFICATIONS` header swung scores 29 points,
  because the data generator gated sections on quality. Three layers of this had
  to be removed before content outweighed structure.
- **Trees can't extrapolate.** 0% of training resumes reached real-resume length,
  so the learned model's absolute output was meaningless on real input — the
  headline score is now a transparent rubric.
- **Requirement matching that produced "4 of 25 addressed"** on a real posting,
  where the misses were job duties and corporate boilerplate no resume can address.

```
features.py    handcrafted feature extraction (experience, skills, gaps, …)
data.py        labelled dataset (synthetic generator, or your own JSONL)
train.py       trains Logistic Regression + LightGBM, evaluates, saves the winner
inference.py   score_resume(text) -> score, confidence, explanation
matching.py    rule-based resume <-> job-description fit scoring
documents.py   PDF / DOCX / TXT extraction + job-posting URL fetching
api.py         FastAPI: /health, /score, /extract, /fetch-job, /compare
ui/            Vite + React app (upload, job fit, side-by-side compare)
tests/         pytest suite for features, matching, extraction, and the API
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py                       # ~1 min, writes models/resume_scorer.joblib
pytest tests -q
uvicorn api:app --reload --port 8000  # http://localhost:8000/docs
```

Front end, in a second terminal:

```bash
cd ui && npm install && npm run dev   # http://localhost:5173
```

Vite proxies `/api` to `localhost:8000`, so both run on one origin in dev.

## v2: what changed and why

The first version had three flaws that any resume-matching product has to solve.

**1. Exact-string skill matching.** `k8s`, `K8s` and `Kubernetes` were three
different things, two of them invisible — so using industry shorthand cost you
points. Fixed with a **skill ontology** (`skills.py`): ~110 canonical skills,
~250 alias surface forms, one compiled alternation, lookaround guards so `js`
doesn't fire inside `json` and `go` doesn't fire inside `google`. This is the
same approach Lightcast/EMSI, ESCO and O*NET take.

**2. Every mention counted equally.** A keyword-stuffed skills list beat a
demonstrated achievement — the exact failure mode candidates exploit against
keyword ATS. Now every skill carries **evidence** and **recency**:

| evidence | where the skill appeared |
|---|---|
| 3 | in a bullet that also states a measured result |
| 2 | in a work-experience bullet |
| 1.5 | in a projects/education section |
| 1 | only in a skills list |

with an exponential recency decay (4-year half-life) from the role it appeared
in. Two resumes listing **100% of required keywords** now score 38 vs 70,
because one proves the skills and the other only claims them. An ATS ranks
those identically.

**3. Binary labels.** `P(good)` collapsed to ~0 on real resumes that didn't
match the generator's template. Real recruiting data is *graded* (funnel depth:
screened → phone → onsite → offer), so the dataset now emits a **0-4 grade** and
a LightGBM regressor is trained on it. The output is a **percentile**: "better
than 78% of the pool". That survives off-distribution input the way a
probability doesn't — the same strong resume went from **0 to 44**.

Ranking metrics are the honest ones here (accuracy is meaningless for an
ordinal target): **Spearman 0.876, NDCG@10 0.83, NDCG@50 0.93, AUC 0.95**.

**Also added:** PII redaction before TF-IDF (`features.redact_for_text_model`).
The model was learning fragments of contact details — `linkedin.com/in/`
tokenised into `com in`, which surfaced as a top *reason* for a score. Names,
email handles and graduation years are also proxies for ethnicity, gender and
age, so they're stripped. Nothing of substance is lost: years of experience is
still computed from parsed date ranges in the structured features.

**And an ATS parse-readiness check** (`ats.py`) — deterministic, model-free, no
labels, no bias risk. A resume can be excellent and never reach a human because
a two-column layout interleaved into nonsense during extraction. Checks contact
details, standard section headings, employment dates *inside the experience
block* (education dates must not stand in for them), multi-column artefacts,
length, and bullet structure. Every finding carries a fix.

## One score, with its derivation

The UI shows **one headline number** — overall fit for the job — and the
dimensions that produced it underneath, worst first, so the top row is what to
fix. An earlier version showed three competing scores (quality / match /
overall) with no indication which to act on; that was a design mistake.

```
49%  Moderate match          Overall fit for this job
     Overall relevance   36%   how closely the language tracks the posting
     Required skills     37%   5 of 8 required, evidence-weighted
     Resume quality      44%   better than 44% of the pool (borderline)
     Experience level   100%   10.5 years vs 2+ required
```

Underneath, two scores are still computed separately, because they answer
different questions and are produced different ways:

| | Quality score | Match score |
|---|---|---|
| Question | "Is this a well-built resume?" | "Does it fit *this* job?" |
| Method | Supervised ML (trained) | Rule-based (not trained) |
| Needs a job posting | No | Yes |

**Skill implications.** ESCO and Lightcast model broader/narrower relations, and
omitting them showed: a resume with PostgreSQL was reported as "missing SQL",
which no human reviewer would say. `skills.IMPLIES` credits the broader skill
(PostgreSQL → SQL, Kafka → system design, Django → Python) at **reduced**
evidence, so an inferred skill never outranks a stated one.

**Why this scores lower than commercial tools.** On the same posting a
commercial tool showed 83% where this showed 69%. Its own breakdown was
Experience 100%, Skill 61%, Industry 46% — its *skill* sub-score was lower than
ours. Solving `w₁·100 + w₂·61 + w₃·46 = 83` implies roughly 60% of that headline
is the coarse seniority bucket. We weight skill coverage highest instead,
because it's the actionable dimension. Different weighting, not a different
measurement.

The match score is deliberately **not** machine-learned. There's no labelled
`(resume, job, hired?)` data here, so a model would just encode our guesses with
a veneer of ML — and a recruiter needs to see *which* required skill is missing,
not a black-box number. It decomposes into skill coverage (50%), the years
requirement (25%), and TF-IDF similarity (25%), each individually inspectable.

`overall_score` blends them 40/60 but always reports both inputs, because a
great resume for the wrong job and a weak resume for the right job otherwise
look identical.

## Resume upload

`POST /extract` accepts **PDF, DOCX, TXT, MD** (10 MB cap) and returns plain
text, which the UI shows back to you before scoring. That round-trip is
deliberate: PDF extraction is lossy often enough that silently scoring whatever
fell out would be dishonest. Scanned/image-only PDFs are rejected with an
explanation rather than scored as an empty document. Legacy `.doc` is not
supported.

## Job postings by URL

A URL on its own is enough — `POST /score` takes `job_url` and fetches it during
scoring. (`POST /fetch-job` exists too, for previewing the text first.)

Extraction tries three methods, cheapest and most reliable first:

1. **The board's own JSON API**, for Greenhouse and Lever URLs. These boards
   render postings client-side, so scraping the HTML returns a near-empty
   JavaScript shell while the API returns the real description. Greenhouse
   returns it HTML-*escaped*, so it's unescaped before parsing — otherwise the
   tags survive as literal `<p>` text and pollute skill matching.
2. **schema.org `JobPosting` JSON-LD**, which most boards embed for Google Jobs
   indexing and which survives JS rendering. Highest-yield generic extractor.
3. **Visible text** of the page's `<main>`/`<article>` block.

**LinkedIn, Indeed, Glassdoor and ZipRecruiter block automated fetching** — they
return a login wall or a JS shell. Those cases are detected and reported as an
actionable error rather than scored as a sign-in page.

URLs are validated against private/loopback/link-local addresses first, so the
endpoint can't be used to reach internal services (SSRF).

## How it works

**Two views of every resume.** Handcrafted features (~28 interpretable numbers:
years of experience, skill breadth, degrees, employment gaps, quantified
achievements, filler-phrase count) are combined with TF-IDF over the raw text.
The handcrafted set generalises from little data and is what the UI explains;
TF-IDF catches vocabulary nobody hardcoded.

**Experience is calendar time, not summed tenures.** Date ranges are parsed,
overlapping roles are merged, and holes ≥ 6 months are counted as gaps. Summing
role durations naively double-counts concurrent jobs.

**Both models are trained on purpose.** Logistic Regression is the honesty check
— if gradient boosting can't beat a tuned linear model, the complexity isn't
earning its keep. On the synthetic data the linear model actually wins
(AUC ≈ 0.92 vs ≈ 0.90), and it gives exact additive explanations for free.

**Class imbalance is handled explicitly** (~25% positives) via
`class_weight="balanced"` / `scale_pos_weight`. Without it the model learns
"always predict bad", scores 75% accuracy, and is useless.

**No test-set leakage.** The vectorizer is fit on train only. The decision
threshold and the isotonic calibrator are fit on *out-of-fold* predictions
(`cross_val_predict`), never on the test set — tuning a threshold on test
quietly turns your reported metrics into training metrics.

**Score = calibrated probability × 100.** Raw classifier scores are not
probabilities; isotonic regression maps them to numbers you can honestly print
as "78/100". `confidence` is separate: `|p − 0.5| × 2`, so an uncertain 50 reads
differently from a confident 50.

**Explanations are exact, not approximate.** For the linear model the log-odds
*are* `Σ coefᵢ·xᵢ`, so contributions decompose perfectly. For LightGBM,
`pred_contrib=True` returns exact SHAP values. Positive = pushed the score up.
Note a below-average value on a negatively-weighted feature legitimately shows
as a positive contribution — that is the arithmetic, not a bug.

## What the match score measures (and why)

The dimensions were chosen by checking what the established tools actually
score, not by guessing:

| Tool | What it scores |
|---|---|
| [Jobscan](https://www.jobscan.co/blog/what-jobscan-match-rate-should-i-aim-for/) | Hard skills, soft skills, keywords — **frequency-weighted**; job title is high-impact |
| [Teal](https://www.tealhq.com/tool/resume-job-description-match) | Matched/missing keywords (hard + soft), plus **job title** and **education** alignment |
| [Huntr](https://help.huntr.co/en/articles/12241684-job-match-score) | Weighted **LLM semantic** analysis over qualifications, responsibilities, keywords, title |

So the score is built from those categories:

| dimension | weight | notes |
|---|---|---|
| Hard skills | 35% | evidence-weighted, and weighted by how often the posting repeats the skill |
| Experience level | 20% | parsed timeline vs. the stated year requirement |
| Job title | 18% | overlap with your role titles — catches domain fit a skill list misses |
| Soft skills | 12% | mentoring, code review, agile, collaboration |
| Education | 8% | degree level vs. requirement; unscored if the posting doesn't state one |
| Text relevance | 7% | TF-IDF cosine |

Only *measurable* dimensions are weighted, then renormalised — an unmeasured
dimension is never scored as zero.

**Keyword frequency weighting** follows Jobscan explicitly: a posting that says
"Kubernetes" five times is telling you what the job is about. Required skills
are weighted `log(1 + mentions)`, so repetition matters without one hammered
keyword dominating.

### A design that didn't survive contact with a real posting

An earlier version scored **every line** of a job description, including
responsibilities. On a real Citi-style posting that produced:

```
Key requirements — 4 of 25 addressed
  unaddressed  Address a variety of responses to problems, questions, or
               situations by applying established criteria to directly
               influence development outcomes...
  unaddressed  Negotiate features and associated priority and help the team
               and their customers reach consensus
```

Useless. Those are *duties*, not qualifications — no resume can "address" them,
and word-overlap on 60-word corporate prose is meaningless. None of the three
tools above does this; Huntr scores responsibilities but with an LLM doing
semantic comparison, which is a different technique entirely.

Requirement extraction now drops responsibility sections, caps lines at 22
words, and keeps only lines naming a **skill, a year count, or a degree** — the
things that can actually be checked against a resume. That same posting now
yields 5 concrete requirements instead of 25.

The genuinely useful part of the old approach was rescued differently: lines
like *"Conduct code reviews and mentor junior developers"* and *"Participate in
Agile ceremonies"* are now matched because `mentoring`, `code review` and
`agile` are **soft skills in the ontology**, scored as their own category — the
way Jobscan and Teal report them.

## Requirement-level matching

Skill matching only sees requirements that name a skill. On a real posting that
is usually the minority:

```
- 6+ years of backend engineering experience      <- years
- Strong Python and Go                            <- skills
- Experience mentoring junior engineers           <- invisible to skill matching
- Track record designing scalable, reliable APIs  <- invisible
- Comfortable owning services in production       <- invisible
```

`requirements.py` scores **every** line of the posting independently, choosing
the right tool per requirement:

| basis | used when | how it's scored |
|---|---|---|
| `tenure` | the line states a year count | parsed timeline vs. the requirement |
| `skills` | the line names skills | evidence/recency strength of those skills |
| `language` | neither | overlap with the best-matching resume bullet |

Each requirement gets `addressed` / `partial` / `unaddressed` plus the resume
line that satisfied it. "This requirement is unaddressed" is far more actionable
than "you're missing the keyword etl".

Two bugs this immediately surfaced, both now regression-tested:
`6+ years of backend engineering` scored **0.15** for a candidate with 8 years
(it was going through word overlap instead of the timeline), and
`Mentored 6 junior engineers` didn't match a requirement for `mentoring`,
because the ontology had no verb inflections or plurals. Plurals are now matched
for surface forms of 5+ letters only — applying an optional `s` to everything
would make `go` match `goes`.

## Measuring real accuracy: `evaluate.py`

Every metric `train.py` prints is computed on **synthetic** resumes. It measures
how well the model learned the generator, not whether it agrees with a
recruiter. You cannot improve what you cannot measure.

`evaluate.py` closes that loop using the cheapest labelling scheme that works —
**pairwise preferences**. "Is this a 3 or a 4?" is hard and inconsistent even
for professionals; "is A better than B?" is fast, and it's what recruiters
actually do. Create `data/golden.jsonl`:

```json
{"better": "<resume A text>", "worse": "<resume B text>"}
{"text": "<resume text>", "grade": 3}
```

Then `python evaluate.py` reports pairwise agreement for the rubric *and* the
learned model, so you can see which tracks human preference and whether a change
helped. **~30 pairs is enough to detect a regression.** Graded rows are expanded
into pairs automatically.

## Why quality is scored by a rubric, not the model

The learned model is **not** the headline quality score. It's reported as
`ml_percentile` for comparison and nothing else.

The reason is measured, not philosophical: **0.0% of the training resumes reach
662 words**, which is an ordinary length for a real one. Gradient-boosted trees
cannot extrapolate past their splits, so on a real resume the model's absolute
output isn't inaccurate so much as undefined. Chasing this produced three
successive shortcut fixes (below) and the score was *still* wrong, because the
remaining problem isn't a shortcut — it's the distribution.

So `rubric.py` scores quality from explicit, published thresholds, the way
Resume Worded and Jobscan's content checks do:

| dimension | weight | what it measures |
|---|---|---|
| Quantified impact | 25% | share of bullets carrying a number |
| Demonstrated skills | 20% | skills named inside a bullet that states a result |
| Strong writing | 15% | action verbs vs. filler phrases |
| Bullet craft | 10% | one accomplishment per bullet, 8-30 words |
| Structure & contact | 10% | standard headings, dated roles, contact details |
| Skill breadth & recency | 10% | areas covered, and how recently used |
| Length | 10% | 400-900 words |

Every dimension returns its score, the evidence behind it, and the fix that
raises it — sorted worst-first, so the top row is the highest-leverage change.
No labels needed, no training distribution to fall outside of, and every point
is a rule you can read and disagree with.

The most common real finding this surfaces: *"0 skills appear in a bullet that
also states a result, though you list 30 skills"* — the candidate's numbers and
their technologies are in different sentences, so neither proves the other.

## Shortcut learning: the bug that made every real resume score ~23%

Worth documenting, because it's the most instructive failure in this project.

Users reported the quality score was stuck near 23% for real resumes. It wasn't
stuck — the model had learned a **shortcut**. The generator gated whole sections
on hard quality thresholds:

```python
if q > 0.55:   out += ["CERTIFICATIONS", ...]     # only high-quality resumes
if q > 0.4:    out += ["SUMMARY", ...]
pool = STRONG_SKILLS if q > 0.35 else WEAK_SKILLS
```

So in training, "has a CERTIFICATIONS section" was a *perfect* predictor of
`q > 0.55`. Gradient boosting found that split immediately and stopped learning
anything harder. The symptom: adding a single certifications line moved a real
resume from **27% to 56%** — a 29-point swing from one section header — while
any resume without certifications was capped around 23%.

The fix is in the data, not the model. Section presence is now *probabilistic*
and only weakly correlated with quality (`_p(rng, base, slope, q)`), so it's
informative but never decisive:

| | old | new |
|---|---|---|
| certs in high-quality resumes | 100% | 54% |
| certs in low-quality resumes | 0% | 33% |

This is the general lesson for any synthetic-data ML pipeline: **if a surface
feature is a deterministic function of the label, the model will learn that
feature and nothing else.** Three regression tests now pin it — a certifications
block may not move the score more than 12 points, and removing quantified
results must cost more than adding a section.

## Using your own labelled data

Drop a JSONL file at `data/resumes.jsonl`:

```json
{"text": "Alex Chen\nSUMMARY...", "label": 1}
{"text": "Sam Doyle\nEXPERIENCE...", "label": 0}
```

`load_dataset()` picks it up automatically and skips the generator. Re-run
`python train.py`. Everything downstream is unchanged.

## API

```bash
curl -X POST localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"resume_text": "Alex Chen ... 10 years ... Python, AWS, Kubernetes"}'
```

```json
{
  "score": 81, "probability_good": 0.8072, "confidence": 0.6144,
  "verdict": "good", "decision_threshold": 0.31, "model": "logistic_regression",
  "top_features": [{"feature": "Quantified achievements", "impact": 0.94,
                    "direction": "positive", "value": 6.0}],
  "notes": ["10.2 years of experience — senior range."],
  "extracted": {"years_experience": 10.2, "n_skills": 17, "...": 0}
}
```

Add `job_description` (or `job_url`) to the same call to get job-fit analysis:

```json
{
  "score": 35,
  "match": {
    "match_score": 85, "skill_coverage": 1.0,
    "matched_required": ["aws", "docker", "go", "kafka", "kubernetes", "postgresql", "python"],
    "missing_required": [], "missing_preferred": ["elasticsearch", "rust"],
    "required_years": 6, "resume_years": 10.5,
    "notes": ["Matches 7 of 7 required skills.", "Meets the 6+ year requirement (10.5 years)."]
  },
  "combined": {"overall_score": 65, "quality_score": 35, "match_score": 85,
               "recommendation": "Skills line up; the resume's writing is what's holding it back."}
}
```

`POST /compare` takes `resume_a` / `resume_b` (plus an optional
`job_description`) and returns both scores, a winner, and a ranked diff of the
structured features. With a job description supplied, "winner" means better
*for that job*.

## Limitations — read before using this on real candidates

- The shipped model is trained on **synthetic** resumes. It has learned the
  generator's idea of quality, not any real hiring outcome. Retrain on your own
  labelled data before drawing any conclusion about a real person.
- **Absolute scores don't transfer off-distribution.** A hand-written strong
  resume in `tests/test_inference.py` scores 35 while a deliberately weak one
  scores 0 — the *ranking* is right, the absolute number is not, because the
  real text doesn't look like the generator's output. This is textbook
  distribution shift, and it is exactly why the tests assert
  `score(good) > score(bad)` rather than `score(good) > 70`. Real labelled data
  fixes it; nothing in the pipeline does.
- Resume-quality labels encode whatever bias existed in whoever labelled them.
  Automated candidate scoring is regulated in several jurisdictions (e.g. NYC
  Local Law 144 requires a bias audit). Audit before deploying.
- Treat the score as a ranking aid with a visible rationale, never an automatic
  reject.
