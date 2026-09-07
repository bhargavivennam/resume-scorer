import { useState } from 'react'

const API = import.meta.env.VITE_API_URL || '/api'
const ACCEPT = '.pdf,.docx,.txt,.md'

/** Score colour bands — one place so gauge, badge and bars never disagree. */
function bandFor(score) {
  if (score >= 70) return { label: 'Strong', color: '#16a34a' }
  if (score >= 45) return { label: 'Moderate', color: '#d97706' }
  return { label: 'Weak', color: '#dc2626' }
}

async function postJSON(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail?.[0]?.msg || data.detail || `Request failed (${res.status})`)
  return data
}

function FeatureBars({ features }) {
  const max = Math.max(...features.map(f => Math.abs(f.impact)), 0.0001)
  return (
    <div className="bars">
      {features.map((f, i) => (
        <div className="bar-row" key={i}>
          <div className="bar-label" title={f.feature}>
            {f.feature}
            {f.value !== null && f.value !== undefined && <span className="muted"> ({f.value})</span>}
          </div>
          <div className="bar-track">
            <div className="bar-mid" />
            <div className="bar-fill" style={{
              width: `${(Math.abs(f.impact) / max) * 50}%`,
              left: f.impact > 0 ? '50%' : undefined,
              right: f.impact < 0 ? '50%' : undefined,
              background: f.impact > 0 ? '#16a34a' : '#dc2626'
            }} />
          </div>
          <div className="bar-val">{f.impact > 0 ? '+' : ''}{f.impact.toFixed(2)}</div>
        </div>
      ))}
    </div>
  )
}

function SkillChips({ title, skills, tone }) {
  if (!skills?.length) return null
  return (
    <div className="chip-block">
      <div className="chip-title">{title}</div>
      <div className="chips">
        {skills.map(s => <span key={s} className={`chip ${tone}`}>{s}</span>)}
      </div>
    </div>
  )
}

function AtsPanel({ ats }) {
  if (!ats) return null
  const tone = ats.n_blockers ? 'bad' : ats.ats_score >= 85 ? 'ok' : 'warn'
  return (
    <div className={`ats ${tone}`}>
      <div className="ats-head">
        <b>ATS readiness: {ats.ats_score}/100</b>
        <span className="muted">{ats.verdict}</span>
      </div>
      {ats.findings.length === 0
        ? <div className="muted small">No parsing problems found.</div>
        : (
          <ul className="ats-list">
            {ats.findings.map((f, i) => (
              <li key={i}>
                <span className={`sev ${f.severity}`}>{f.severity}</span>
                <span> {f.issue} </span>
                <span className="muted">{f.fix}</span>
              </li>
            ))}
          </ul>
        )}
    </div>
  )
}

function QualityDetail({ result }) {
  if (!result.quality_dimensions) return null
  return (
    <>
      <h4>Resume quality: {result.score}/100 · {result.verdict_quality}</h4>
      <p className="hint">
        Scored from explicit rules, worst first — the top row is your highest-leverage fix.
      </p>
      <div className="dims">
        {result.quality_dimensions.map(d => {
          const band = bandFor(d.score)
          return (
            <div className="dim" key={d.name}>
              <div className="dim-head">
                <span className="dim-name">{d.name}</span>
                <span className="dim-weight muted">{Math.round(d.weight * 100)}% of score</span>
                <span className="dim-score" style={{ color: band.color }}>{d.score}</span>
              </div>
              <div className="brk-track">
                <div className="brk-fill" style={{ width: `${d.score}%`, background: band.color }} />
              </div>
              <div className="dim-detail">{d.detail}</div>
              {d.score < 85 && <div className="dim-fix">→ {d.fix}</div>}
            </div>
          )
        })}
      </div>
    </>
  )
}

function BreakdownRow({ row }) {
  const band = bandFor(row.value)
  return (
    <div className="brk-row">
      <div className="brk-label">{row.label}</div>
      <div className="brk-track">
        <div className="brk-fill" style={{ width: `${row.value}%`, background: band.color }} />
      </div>
      <div className="brk-val" style={{ color: band.color }}>{row.value}%</div>
      <div className="brk-detail muted">{row.detail}</div>
    </div>
  )
}

/** ONE headline number, with the dimensions that produced it underneath.
 *  Previously this showed three competing scores (quality / match / overall)
 *  with no indication which one to act on. */
function Headline({ result }) {
  const combined = result.combined
  const headline = combined ? combined.overall_score : result.score
  const band = bandFor(headline)
  const rows = combined ? combined.breakdown : result.quality_dimensions.map(d => ({
    label: d.name, value: d.score, detail: d.detail,
  }))

  return (
    <div className="headline">
      <div className="headline-score">
        <div className="hs-num" style={{ color: band.color }}>{headline}<span className="hs-pct">%</span></div>
        <div className="hs-band" style={{ background: band.color }}>
          {combined ? `${band.label} match` : result.verdict_quality}
        </div>
        <div className="muted small">
          {combined ? 'Overall fit for this job' : 'Resume quality (no job description)'}
        </div>
      </div>
      <div className="headline-breakdown">
        {rows.map(r => <BreakdownRow key={r.label} row={r} />)}
      </div>
    </div>
  )
}

function RequirementList({ match }) {
  if (!match?.requirements?.length) return null
  const tone = { addressed: 'ok', partial: 'warn', unaddressed: 'bad' }
  return (
    <>
      <h4>
        Key requirements — {match.n_requirements_addressed} of {match.n_requirements} addressed
      </h4>
      <p className="hint">
        Only the concrete, checkable lines: those naming a skill, a year count or a
        degree. Job duties and boilerplate are excluded — nothing in a resume can
        "address" them, so reporting them as gaps would be noise.
      </p>
      <div className="reqs">
        {match.requirements.map((r, i) => (
          <div className="req" key={i}>
            <div className="req-head">
              <span className={`sev ${tone[r.verdict]}`}>{r.verdict}</span>
              <span className="req-text">{r.requirement}</span>
            </div>
            {r.evidence && <div className="req-evidence">your resume: “{r.evidence}”</div>}
          </div>
        ))}
      </div>
    </>
  )
}

function MatchDetail({ match }) {
  if (!match) return null
  return (
    <>
      <RequirementList match={match} />
      <h4>Skill detail</h4>
      <ul className="notes">{match.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
      <SkillChips title="Required skills you have" skills={match.matched_required} tone="ok" />
      <SkillChips title="Required skills missing" skills={match.missing_required} tone="bad" />
      {/* The distinction an ATS can't make: claimed vs demonstrated. */}
      <SkillChips title="Listed but not demonstrated" skills={match.weakly_evidenced} tone="warn" />
      <SkillChips title="Not used in years" skills={match.stale_skills} tone="warn" />
      <SkillChips title="Preferred skills you have" skills={match.matched_preferred} tone="ok" />
      <SkillChips title="Preferred skills missing" skills={match.missing_preferred} tone="neutral" />
    </>
  )
}

function ResultPanel({ result, title }) {
  if (!result) return null
  return (
    <div className="panel">
      <h3>{title}</h3>

      <Headline result={result} />

      {result.combined && <p className="recommendation">{result.combined.recommendation}</p>}

      {result.job_fetch_error && (
        <div className="fetch-error">
          <b>Couldn't read that job link — scored resume quality only.</b>
          <div style={{ marginTop: 4 }}>{result.job_fetch_error}</div>
        </div>
      )}

      {!result.match && !result.job_fetch_error && (
        <div className="nudge">
          No job description supplied — this is resume quality only. Add a job posting
          above to score fit against a specific role.
        </div>
      )}

      <AtsPanel ats={result.ats} />

      <MatchDetail match={result.match} />

      <QualityDetail result={result} />

      {result.notes.length > 0 && (<>
        <h4>What stood out</h4>
        <ul className="notes">{result.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
      </>)}

      <details className="advanced">
        <summary>Experimental: learned model (not used for the score)</summary>
        <p className="hint">
          A LightGBM ranker trained on synthetic resumes ranks this one at the{' '}
          <b>{Math.round(result.ml_percentile)}th percentile</b>. It is shown for
          comparison only — real resumes fall outside its training distribution,
          so its absolute output isn't meaningful until it's trained on real
          labelled data. The score above comes from the rubric instead.
        </p>
        <p className="hint">Green pushes the model's estimate up, red pulls it down.</p>
        <FeatureBars features={result.top_features} />
        <h4>Extracted facts</h4>
        <div className="facts">
          {Object.entries(result.extracted).map(([k, v]) => (
            <div className="fact" key={k}>
              <span className="fact-val">{v}</span>
              <span className="fact-key">{k.replace(/_/g, ' ')}</span>
            </div>
          ))}
        </div>
        <p className="hint">Model: {result.model} · confidence {Math.round(result.confidence * 100)}%</p>
      </details>
    </div>
  )
}

function ResumeInput({ label, text, setText, disabled }) {
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState(null)

  async function handleFile(file) {
    if (!file) return
    setUploading(true); setUploadMsg(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API}/extract`, { method: 'POST', body: form })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) throw new Error(data.detail || `Upload failed (${res.status})`)
      setText(data.text)
      setUploadMsg(`Read ${data.n_chars.toLocaleString()} characters from ${data.filename}. Check it below before scoring.`)
    } catch (err) {
      setUploadMsg(`⚠ ${err.message}`)
    } finally {
      setUploading(false)
    }
  }

  const active = text.trim().length >= 20

  return (
    <div className={`editor step ${active ? 'on' : ''}`}>
      <div className="step-head">
        <label><span className="step-num">1</span> {label}</label>
        <span className={`status ${active ? 'on' : 'off'}`}>
          {active ? '✓ Ready' : 'Paste text or upload a file'}
        </span>
      </div>
      <textarea value={text} onChange={e => setText(e.target.value)} rows={14}
        placeholder="Paste your resume text, or upload a PDF / DOCX below…" disabled={disabled} />
      <div className="upload-row">
        <input type="file" accept={ACCEPT} disabled={uploading}
          onChange={e => handleFile(e.target.files[0])} />
        {uploading && <span className="hint">Extracting…</span>}
      </div>
      {uploadMsg && <p className="hint">{uploadMsg}</p>}
    </div>
  )
}

function JobInput({ jobText, setJobText, url, setUrl }) {
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)
  const [failed, setFailed] = useState(false)

  async function fetchJob() {
    if (!url.trim()) return
    setBusy(true); setMsg(null); setFailed(false)
    try {
      const data = await postJSON('/fetch-job', { url: url.trim() })
      setJobText(data.text)
      setMsg(`Fetched ${data.n_chars.toLocaleString()} characters. Trim it to the actual description for a better match.`)
    } catch (err) {
      // A failed fetch used to render as a small grey hint that was easy to
      // miss — you'd submit and silently get a quality-only score.
      setFailed(true)
      setMsg(err.message)
    } finally {
      setBusy(false)
    }
  }

  // A URL on its own is enough: the server fetches it during scoring, so you
  // don't have to remember to press Fetch first.
  const active = jobText.trim().length > 0 || url.trim().length > 0

  return (
    <div className={`editor job step ${active ? 'on' : ''}`}>
      <div className="step-head">
        <label><span className="step-num">2</span> Job description</label>
        {/* The whole point of this badge: you can see whether job-fit scoring
            is armed WITHOUT having to submit and discover it wasn't. */}
        <span className={`status ${active ? 'on' : 'off'}`}>
          {active ? '✓ Job-fit scoring ON' : 'Optional — paste a job post or URL to score the match'}
        </span>
      </div>
      <div className="url-row">
        <input type="url" value={url} onChange={e => setUrl(e.target.value)}
          placeholder="https://… job posting URL" />
        <button type="button" onClick={fetchJob} disabled={busy || !url.trim()}>
          {busy ? 'Fetching…' : 'Preview'}
        </button>
      </div>
      {failed ? (
        <div className="fetch-error">
          <b>Couldn't fetch that link.</b> {msg}
          <div className="fetch-error-fix">
            Open the posting in your browser, select the description, and paste it into the box below.
          </div>
        </div>
      ) : msg ? <p className="hint">{msg}</p> : (
        <p className="hint">
          A URL alone is enough — it's fetched when you score. "Preview" just shows you
          the text first. Works on Greenhouse, Lever and most company career pages;
          LinkedIn, Indeed, Glassdoor and ZipRecruiter block bots — paste those instead.
        </p>
      )}
      <textarea value={jobText} onChange={e => setJobText(e.target.value)} rows={10}
        placeholder="…or paste the job description here" />
    </div>
  )
}

export default function ResumeScorer() {
  const [mode, setMode] = useState('single')
  const [textA, setTextA] = useState('')
  const [textB, setTextB] = useState('')
  const [jobText, setJobText] = useState('')
  const [jobUrl, setJobUrl] = useState('')
  const [result, setResult] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true); setError(null); setResult(null); setComparison(null)
    try {
      let jd = jobText.trim() || undefined
      // Compare has no server-side URL fetch, so resolve the URL here first and
      // reuse the same text for both resumes.
      if (!jd && jobUrl.trim() && mode === 'compare') {
        try {
          jd = (await postJSON('/fetch-job', { url: jobUrl.trim() })).text
          setJobText(jd)
        } catch (err) {
          // Same rule as the server: a blocked link must not cost you the
          // comparison you asked for.
          setError(`Couldn't read the job link, comparing on quality only. ${err.message}`)
          jd = undefined
        }
      }
      if (mode === 'single') {
        setResult(await postJSON('/score', {
          resume_text: textA,
          job_description: jd,
          job_url: jd ? undefined : (jobUrl.trim() || undefined),
        }))
      } else {
        setComparison(await postJSON('/compare', { resume_a: textA, resume_b: textB, job_description: jd }))
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      // Results render below the fold; without this you click and nothing
      // visibly happens.
      requestAnimationFrame(() =>
        document.getElementById('results')?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
    }
  }

  const hasJob = jobText.trim().length > 0 || jobUrl.trim().length > 0
  const canSubmit = mode === 'single'
    ? textA.trim().length >= 20
    : textA.trim().length >= 20 && textB.trim().length >= 20

  return (
    <div className="app">
      <header>
        <h1>Resume Scorer</h1>
        <p className="muted">
          Scores resume quality 0–100 against an explicit rubric, and — if you add a
          job description — how well it fits that specific role.
        </p>
      </header>

      <div className="tabs">
        <button className={mode === 'single' ? 'active' : ''} onClick={() => setMode('single')}>Score one</button>
        <button className={mode === 'compare' ? 'active' : ''} onClick={() => setMode('compare')}>Compare two</button>
      </div>

      <form onSubmit={handleSubmit}>
        <div className={mode === 'compare' ? 'editors two' : 'editors'}>
          <ResumeInput label={mode === 'compare' ? 'Resume A' : 'Your resume'} text={textA} setText={setTextA} />
          {mode === 'compare' && <ResumeInput label="Resume B" text={textB} setText={setTextB} />}
        </div>

        <JobInput jobText={jobText} setJobText={setJobText} url={jobUrl} setUrl={setJobUrl} />

        <div className="submit-row">
          <button type="submit" className="primary" disabled={!canSubmit || loading}>
            {loading
              ? 'Scoring…'
              : hasJob
                ? (mode === 'single' ? 'Score resume vs. this job' : 'Compare both vs. this job')
                : (mode === 'single' ? 'Score resume quality' : 'Compare resume quality')}
          </button>
          <span className="submit-note">
            {hasJob
              ? 'You will get: quality score + job-match % + missing skills.'
              : 'You will get: quality score only. Add a job description or URL above for a match score.'}
          </span>
        </div>
        {!canSubmit && <p className="hint">Needs at least 20 characters of resume text.</p>}
      </form>

      <div id="results" />
      {error && <div className="error">{error}</div>}
      {result && <ResultPanel result={result} title="Result" />}

      {comparison && (<>
        <div className="verdict-bar">
          {comparison.winner === 'tie'
            ? 'Both resumes scored the same.'
            : `Resume ${comparison.winner.toUpperCase()} scores ${comparison.score_gap} points higher.`}
        </div>
        <div className="side-by-side">
          <ResultPanel result={comparison.a} title="Resume A" />
          <ResultPanel result={comparison.b} title="Resume B" />
        </div>
        <div className="panel">
          <h3>Where they differ</h3>
          <table className="diff">
            <thead><tr><th>Feature</th><th>A</th><th>B</th><th>B − A</th></tr></thead>
            <tbody>
              {comparison.differences.map(d => (
                <tr key={d.feature}>
                  <td>{d.feature.replace(/_/g, ' ')}</td>
                  <td>{d.a}</td><td>{d.b}</td>
                  <td style={{ color: d.delta > 0 ? '#16a34a' : '#dc2626' }}>
                    {d.delta > 0 ? '+' : ''}{d.delta}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>)}
    </div>
  )
}
