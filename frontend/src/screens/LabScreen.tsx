import { useMemo, useState } from 'react'
import { api } from '../api'
import { ColumnChart, HBarChart, LineChart } from '../components/charts'
import { Card, ErrorText, Metric, Spinner, Tag, VerdictLabel } from '../components/ui'
import { int, nice, num, pct, signed } from '../format'
import type { AnalysisBundle, ExperimentRecord, Tipping, VideoMeta } from '../types'

type Variable = 'hook_clip' | 'hypothetical_hook' | 'seeding' | 'policy'
const STRATS = ['mixed', 'random', 'relevance_biased', 'community_targeted']
const POLICIES = ['default', 'conservative', 'exploratory', 'interest_tight']

export default function LabScreen({ videoId, bundle, videos, experiment, onExperiment }: {
  videoId: string; bundle: AnalysisBundle; videos: VideoMeta[]
  experiment: ExperimentRecord | null; onExperiment: (e: ExperimentRecord) => void
}) {
  // alternate openings: same-topic versions first, then by prior
  const others = videos.filter((v) => v.analysed && v.video_id !== videoId).sort((a, b) =>
    Number(b.topic === bundle.profile.topic) - Number(a.topic === bundle.profile.topic) || (b.prior ?? 0) - (a.prior ?? 0))
  const [variable, setVariable] = useState<Variable>(others.length ? 'hook_clip' : 'hypothetical_hook')
  const [alt, setAlt] = useState(others[0]?.video_id ?? '')
  const [delta, setDelta] = useState(0.1)
  const [seedA, setSeedA] = useState('mixed'), [seedB, setSeedB] = useState('relevance_biased')
  const [polA, setPolA] = useState('default'), [polB, setPolB] = useState('exploratory')
  const [nSeeds, setNSeeds] = useState(30), [nSamples, setNSamples] = useState(10)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const runExp = async () => {
    setBusy(true); setError(null)
    const body: Record<string, unknown> = { variable, baseline_video_id: videoId, n_seeds: nSeeds, n_samples: nSamples }
    if (variable === 'hook_clip') body.variant_video_id = alt
    if (variable === 'hypothetical_hook') body.hypothetical_delta = delta
    if (variable === 'seeding') { body.baseline_seeding = seedA; body.variant_seeding = seedB }
    if (variable === 'policy') { body.baseline_policy = polA; body.variant_policy = polB }
    try { onExperiment(await api.experiment(body)) } catch (e) { setError((e as Error).message) }
    setBusy(false)
  }
  const exp = experiment && experiment.config.baseline.video_id === videoId ? experiment : null

  return (
    <div className="stack" style={{ gap: 16 }}>
      <Card title="Matched experiment" sub="Change one thing; hold the audience, graph, prior method, budget and random draws fixed">
        <div className="row" style={{ alignItems: 'flex-end', gap: 16 }}>
          <label className="field">Variable
            <select value={variable} onChange={(e) => setVariable(e.target.value as Variable)}>
              <option value="hook_clip" disabled={!others.length}>Real alternate opening (uploaded clip)</option>
              <option value="hypothetical_hook">Hypothetical stronger hook (labelled)</option>
              <option value="seeding">Seeding strategy</option>
              <option value="policy">Distribution policy</option>
            </select></label>
          {variable === 'hook_clip' && (
            <label className="field">Alternate clip
              <select value={alt} onChange={(e) => setAlt(e.target.value)}>
                {others.map((v) => <option key={v.video_id} value={v.video_id}>{v.filename}{v.prior != null ? ` (p ${num(v.prior)})` : ''}</option>)}
              </select></label>
          )}
          {variable === 'hypothetical_hook' && (
            <label className="field">Add to prior p<input type="number" step={0.05} min={0.01} max={0.5} value={delta} style={{ width: 90 }}
              onChange={(e) => setDelta(Number(e.target.value))} /></label>
          )}
          {variable === 'seeding' && <>
            <label className="field">Baseline<select value={seedA} onChange={(e) => setSeedA(e.target.value)}>{STRATS.map((s) => <option key={s} value={s}>{nice(s)}</option>)}</select></label>
            <label className="field">Variant<select value={seedB} onChange={(e) => setSeedB(e.target.value)}>{STRATS.map((s) => <option key={s} value={s}>{nice(s)}</option>)}</select></label>
          </>}
          {variable === 'policy' && <>
            <label className="field">Baseline<select value={polA} onChange={(e) => setPolA(e.target.value)}>{POLICIES.map((s) => <option key={s} value={s}>{nice(s)}</option>)}</select></label>
            <label className="field">Variant<select value={polB} onChange={(e) => setPolB(e.target.value)}>{POLICIES.map((s) => <option key={s} value={s}>{nice(s)}</option>)}</select></label>
          </>}
          <button className="btn primary" onClick={runExp} disabled={busy || (variable === 'hook_clip' && !alt)}>
            {busy ? 'Running…' : 'Run experiment'}</button>
          <details style={{ marginLeft: 'auto' }}>
            <summary>Advanced</summary>
            <div className="row" style={{ marginTop: 6 }}>
              <label className="field">Seeds<input type="number" min={2} max={60} value={nSeeds} style={{ width: 70 }} onChange={(e) => setNSeeds(Number(e.target.value))} /></label>
              <label className="field">Assumption samples<input type="number" min={2} max={20} value={nSamples} style={{ width: 70 }} onChange={(e) => setNSamples(Number(e.target.value))} /></label>
            </div>
          </details>
        </div>
        {variable === 'hypothetical_hook' && <p className="note" style={{ marginTop: 10 }}>Hypothetical: answers “what if the opening rated {num(delta)} higher”. It is not a real clip result.</p>}
        {busy && <div style={{ marginTop: 10 }}><Spinner text={`Running ${nSeeds} seeds × ${nSamples} assumption samples × 2 scenarios on shared random draws…`} /></div>}
        <ErrorText error={error} />
      </Card>
      {exp && <Verdict exp={exp} baselinePrior={bundle.prior.percentile} />}
      <TippingCard videoId={videoId} />
    </div>
  )
}

function Verdict({ exp, baselinePrior }: { exp: ExperimentRecord; baselinePrior: number }) {
  const r = exp.result
  const A = exp.assumption_samples.length
  const wins = Math.round(r.assumption_stability * A)
  const med = (a: number[]) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)]
  const baseMed = exp.baseline_values.map(med)
  const varMed = exp.variant_values.map(med)
  const diffs = useMemo(() => exp.variant_values.flatMap((row, a) => row.map((v, s) => v - exp.baseline_values[a][s])), [exp])
  const hist = useMemo(() => {
    // ties get their own bar; negative and positive differences are binned separately
    const bin = (vals: number[], k: number) => {
      if (!vals.length) return [] as { label: string; mid: number; c: number }[]
      const lo = Math.min(...vals), hi = Math.max(...vals), w = (hi - lo) / k || 1
      const counts = Array(k).fill(0)
      vals.forEach((d) => { counts[Math.min(k - 1, Math.floor((d - lo) / w))]++ })
      return counts.map((c, i) => ({ label: `${signed(lo + i * w)} to ${signed(lo + (i + 1) * w)}`, mid: lo + (i + 0.5) * w, c }))
    }
    const neg = diffs.filter((d) => d < 0), pos = diffs.filter((d) => d > 0), zero = diffs.length - neg.length - pos.length
    return [...bin(neg, neg.length ? 4 : 0), ...(zero ? [{ label: 'no difference', mid: 0, c: zero }] : []), ...bin(pos, 8)]
  }, [diffs])
  const sens = Object.entries(r.sensitivity).slice(0, 8)
  const isHyp = exp.config.variable === 'hypothetical_hook'
  const varLabel = String(exp.config.variant.label ?? 'variant')
  const pVar = exp.config.variable === 'hook_clip' ? (exp.config.variant.prior_percentile as number | undefined) : undefined
  return (
    <>
      <div className="grid g3">
        <Card title="Verdict" sub={`${nice(exp.config.variable)}: ${varLabel} vs baseline`} right={<Tag t="SIMULATED" />}>
          <div className="stack" style={{ gap: 10 }}>
            <VerdictLabel label={r.label} />
            <p>In this simulation the variant reaches more unique viewers in <b>{pct(r.win_rate)}</b> of {exp.seeds.length * A} paired
              runs, and its median beats the baseline's in <b>{wins} of {A}</b> assumption settings.</p>
            <p className="note">{r.note} Robust = both rates ≥ 80%; Likely = both ≥ 60%. A simulated scenario, not proof of a causal effect.</p>
            {isHyp && <p className="warnnote">Hypothetical experiment: the variant's prior is the baseline's plus a fixed amount. Not a real clip.</p>}
            {pVar != null && <p className="secondary small">Prior baseline {num(baselinePrior)} → variant {num(pVar)} <Tag t="MODEL_RATED" /></p>}
          </div>
        </Card>
        <Card className="span2" title="Effect on unique reach" sub="Paired difference, variant minus baseline (same seed, same assumptions)">
          <div className="grid g3">
            <Metric label="Win rate" value={pct(r.win_rate)} tag="SIMULATED" sub="share of paired runs" />
            <Metric label="Assumption stability" value={pct(r.assumption_stability)} tag="SIMULATED" sub={`${wins} of ${A} settings`} />
            <Metric label="Median effect" value={`${signed(r.effect_median)} viewers`} tag="SIMULATED"
              sub={`10th–90th pct: ${signed(r.effect_p10)} to ${signed(r.effect_p90)}`} />
          </div>
          <h4 style={{ margin: '14px 0 4px' }}>Distribution of paired differences</h4>
          <ColumnChart height={170} categories={hist.map((h) => h.label)} catLabel={() => ''}
            series={[{ name: 'Paired runs', color: 'var(--series-1)', values: hist.map((h) => h.c) }]}
            colorFor={(_, i) => (hist[i].mid > 0 ? 'var(--div-pos)' : hist[i].mid < 0 ? 'var(--div-neg)' : 'var(--text-muted)')} yFormat={(v) => int(v)} />
          <div className="legend" style={{ marginTop: 4 }}>
            <span className="key"><span className="swatch" style={{ background: 'var(--div-pos)' }} />variant ahead</span>
            <span className="key"><span className="swatch" style={{ background: 'var(--text-muted)' }} />no difference</span>
            <span className="key"><span className="swatch" style={{ background: 'var(--div-neg)' }} />baseline ahead</span>
          </div>
        </Card>
      </div>
      <div className="grid g2">
        <Card title="Median reach per assumption setting" sub="Each pair of columns is one draw of all assumed parameters" right={<Tag t="SIMULATED" />}>
          <ColumnChart height={220} categories={exp.assumption_samples.map((s) => `#${s.sample_id + 1}`)} xLabel="assumption sample"
            series={[{ name: 'Baseline', color: 'var(--series-1)', values: baseMed }, { name: varLabel, color: 'var(--series-2)', values: varMed }]}
            yFormat={(v) => int(v)} />
        </Card>
        <Card title="Which assumptions the verdict depends on" sub="Rank correlation between each sampled parameter and the paired effect" right={<Tag t="ASSUMED" />}>
          {sens.length ? (
            <>
              <div className="legend">
                <span className="key"><span className="swatch" style={{ background: 'var(--div-pos)' }} />higher value → larger effect</span>
                <span className="key"><span className="swatch" style={{ background: 'var(--div-neg)' }} />higher value → smaller effect</span>
              </div>
              <HBarChart labelWidth={240} domain={[-1, 1]} format={(v) => v.toFixed(2)}
                rows={sens.map(([k, v]) => ({ label: nice(k), value: v, color: v >= 0 ? 'var(--div-pos)' : 'var(--div-neg)', note: 'rank correlation' }))} />
              <p className="note" style={{ marginTop: 6 }}>Based on {A} assumption samples; with few samples, treat as indicative.</p>
            </>
          ) : <p className="secondary">No variation to rank (the arms gave identical results).</p>}
          <details style={{ marginTop: 10 }}><summary>Held fixed</summary>
            <p className="small secondary" style={{ marginTop: 4 }}>{exp.config.held_fixed.join(', ')} · seeds {exp.seeds[0]}–{exp.seeds[exp.seeds.length - 1]} · id {r.experiment_id}</p></details>
        </Card>
      </div>
    </>
  )
}

function TippingCard({ videoId }: { videoId: string }) {
  const [t, setT] = useState<Tipping | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const go = async () => {
    setBusy(true); setError(null)
    try { setT(await api.tipping(videoId)) } catch (e) { setError((e as Error).message) }
    setBusy(false)
  }
  const tt = t && t.video_p != null ? t : null
  return (
    <Card title="Tipping-point curve" sub="Share of simulated runs reaching the growth milestone as the prior p varies"
      right={tt ? <Tag t="SIMULATED" /> : <button className="btn" onClick={go} disabled={busy}>{busy ? 'Sweeping…' : 'Compute (≈40 s)'}</button>}>
      {busy && <Spinner text="Sweeping p across 15 points × 20 seeds, with and without assumption uncertainty…" />}
      <ErrorText error={error} />
      {tt && (
        <div className="grid g3">
          <div className="span2">
            <LineChart height={240} xLabel="prior p" yDomain={[0, 1]} yFormat={(v) => pct(v)} xFormat={(v) => v.toFixed(2)}
              markers={[{ x: tt.video_p, label: `this video (p ${num(tt.video_p)})` }]}
              series={[
                { name: 'Default assumptions', color: 'var(--series-1)', points: tt.points.map((p) => ({ x: p.p, y: p.milestone_rate })) },
                ...(tt.points_uncertain ? [{ name: 'Across assumption ranges', color: 'var(--series-2)',
                  points: tt.points_uncertain.map((p) => ({ x: p.p, y: p.milestone_rate })) }] : []),
              ]} />
          </div>
          <div className="stack" style={{ gap: 12 }}>
            <Metric label={`Runs reaching ${pct(tt.milestone_pct)} of population`} value={pct(tt.milestone_rate_at_video)} tag="SIMULATED" sub="default assumptions" />
            {tt.milestone_rate_at_video_uncertain != null &&
              <Metric label="Across assumption ranges" value={pct(tt.milestone_rate_at_video_uncertain)} tag="SIMULATED" />}
            <p className="note">{tt.message}</p>
            <p className="muted small">{tt.note}</p>
          </div>
        </div>
      )}
    </Card>
  )
}
