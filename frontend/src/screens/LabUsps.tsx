import { useEffect, useState } from 'react'
import { api, sourceUrl, waitForJob } from '../api'
import { HBarChart } from '../components/charts'
import { Card, ErrorText, Metric, Spinner, Tag, VerdictLabel } from '../components/ui'
import { int, nice, pct, signed } from '../format'
import type { Drivers, SeedingPlan, Surgeon, VOI } from '../types'

/** What to measure next: expected value of information per assumption. */
export function VoiPanel({ voi }: { voi: VOI }) {
  const rows = voi.per_parameter.filter((p) => p.evppi > (voi.floor ?? 0)).slice(0, 6)
  return (
    <Card title="Is this decision robust?" sub="Value of information: would knowing any assumption change the choice?"
      right={<Tag t="SIMULATED" />}>
      <div className="grid g3">
        <Metric label="Value of knowing everything" value={`${int(voi.evpi)} ${voi.unit}`} tag="SIMULATED"
          sub="average gain from perfect information" />
        <Metric label="Chance the other option is better" value={pct(voi.decision_risk)} tag="SIMULATED"
          sub="share of assumption settings" />
        <Metric label="Current choice" value={voi.decision === 'variant' ? 'Use the variant' : 'Keep the baseline'} tag="SIMULATED" />
      </div>
      <p className="warnnote" style={{ margin: '12px 0' }}>{voi.recommendation}</p>
      {rows.length === 0 && <p className="secondary small">No assumption is worth more than {int(voi.floor)} {voi.unit} to measure for this decision.</p>}
      {rows.length > 0 && (
        <>
          <HBarChart labelWidth={240} format={(v) => `${int(v)} ${voi.unit}`}
            rows={rows.map((r) => ({ label: nice(r.parameter), value: r.evppi,
              color: r.measurable ? 'var(--series-1)' : 'var(--axis)', note: r.measurable ? 'measurable by you' : 'platform-side' }))} />
          <div className="legend" style={{ marginTop: 4 }}>
            <span className="key"><span className="swatch" style={{ background: 'var(--series-1)' }} />you can measure it</span>
            <span className="key"><span className="swatch" style={{ background: 'var(--axis)' }} />platform-side, not measurable</span>
          </div>
          <ul className="clean small secondary" style={{ marginTop: 6 }}>
            {rows.filter((r) => r.hint).slice(0, 3).map((r) => <li key={r.parameter}><b>{nice(r.parameter)}:</b> check {r.hint}.</li>)}
          </ul>
        </>
      )}
      <p className="note" style={{ marginTop: 8 }}>{voi.note} The tool points to the one real number that would most improve its own advice.</p>
    </Card>
  )
}

/** Hook Surgeon: re-cuts of the creator's own shots, each tested in a matched experiment. */
export function SurgeonCard({ videoId, onUse, onTrace }: { videoId: string; onUse: (editVideoId: string) => void; onTrace: (editVideoId: string) => void }) {
  const [res, setRes] = useState<Surgeon | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  useEffect(() => { setRes(null); api.surgeonLatest(videoId).then(setRes).catch(() => {}) }, [videoId])
  const go = async () => {
    setError(null); setBusy('Starting…')
    try {
      const { job_id } = await api.surgeonStart(videoId)
      setRes(await waitForJob<Surgeon>(job_id, (m, s) => setBusy(`${m} · ${Math.round(s)}s`)))
    } catch (e) { setError((e as Error).message) }
    setBusy(null)
  }
  return (
    <Card title="Hook Surgeon" sub="Re-cuts of your own shots, each re-rated by the same prompt and tested on the same audience and dice"
      right={<button className="btn" onClick={go} disabled={!!busy}>{busy ? 'Working…' : res ? 'Re-run' : 'Find better edits (≈2 min)'}</button>}>
      {busy && <Spinner text={busy} />}
      <ErrorText error={error} />
      {!res && !busy && <p className="secondary small">Builds edits from detected cuts (payoff first, a short teaser, cutting the section the
        attention timeline flags as a drag…), renders real clips, and ranks them by robustness. Run the attention timeline first for the drag-based edits.</p>}
      {res && (
        <div className="stack" style={{ gap: 10 }}>
          <p className="warnnote">{res.message}</p>
          <div className="tablewrap" style={{ maxHeight: 'none' }}>
            <table className="t">
              <thead><tr><th>Edit</th><th className="num">Prior p <Tag t="MODEL_RATED" /></th><th>Verdict</th>
                <th className="num">Milestone chance <Tag t="SIMULATED" /></th><th className="num">Wins</th><th /></tr></thead>
              <tbody>{res.edits.map((e) => {
                const r = e.result
                const d = (r.milestone_variant ?? 0) - (r.milestone_baseline ?? 0)
                return (
                  <tr key={e.name}>
                    <td><b>{e.title}</b><div className="small secondary">{e.description}</div>
                      {open === e.name && <video src={sourceUrl(e.video_id)} controls autoPlay playsInline
                        style={{ width: 200, marginTop: 6, borderRadius: 6, background: '#000' }} />}</td>
                    <td className="num">{e.prior.toFixed(2)}</td>
                    <td><VerdictLabel label={r.label} /></td>
                    <td className="num">{pct(r.milestone_baseline)} → <b>{pct(r.milestone_variant)}</b>
                      <div className="small" style={{ color: d > 0 ? 'var(--good-ink)' : 'var(--text-muted)' }}>{signed(d * 100)} pts</div></td>
                    <td className="num">{pct(r.win_rate)}</td>
                    <td><div className="stack" style={{ gap: 4 }}>
                      <button className="btn" style={{ padding: '2px 8px' }} onClick={() => setOpen(open === e.name ? null : e.name)}>
                        {open === e.name ? 'Hide' : 'Watch'}</button>
                      <button className="btn" style={{ padding: '2px 8px' }} onClick={() => onUse(e.video_id)}>Full test</button>
                      <button className="btn" style={{ padding: '2px 8px' }} onClick={() => onTrace(e.video_id)}>Trace why</button>
                    </div></td>
                  </tr>
                )
              })}</tbody>
            </table>
          </div>
          <p className="note">{res.note} Milestone = share of runs reaching 25% of the population.</p>
        </div>
      )}
    </Card>
  )
}

/** Robust seeding plan: who to show it to first, by minimax regret. */
export function SeedingPlanCard({ videoId }: { videoId: string }) {
  const [res, setRes] = useState<SeedingPlan | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { setRes(null); api.seedingLatest(videoId).then(setRes).catch(() => {}) }, [videoId])
  const go = async () => {
    setError(null); setBusy('Starting…')
    try {
      const { job_id } = await api.seedingStart(videoId)
      setRes(await waitForJob<SeedingPlan>(job_id, (m, s) => setBusy(`${m} · ${Math.round(s)}s`)))
    } catch (e) { setError((e as Error).message) }
    setBusy(null)
  }
  const rec = res?.options.find((o) => o.key === res.recommended)
  return (
    <Card title="Robust seeding plan" sub="Who to show it to first, chosen to hold up across all assumption settings (minimax regret)"
      right={<button className="btn" onClick={go} disabled={!!busy}>{busy ? 'Working…' : res ? 'Re-run' : 'Plan seeding (≈20 s)'}</button>}>
      {busy && <Spinner text={busy} />}
      <ErrorText error={error} />
      {res && rec && (
        <div className="stack" style={{ gap: 10 }}>
          <div className="grid g3">
            <Metric label="Recommended first audience" value={nice(rec.label)} tag="SIMULATED" />
            <Metric label="Milestone chance" value={pct(rec.milestone_rate)} tag="SIMULATED"
              sub={`${signed(res.gain_vs_default * 100)} pts vs default`} />
            <Metric label="Worst-case regret" value={pct(rec.max_regret)} tag="SIMULATED" sub="behind the best option, worst setting" />
          </div>
          <p className="warnnote">{res.message}</p>
          <HBarChart labelWidth={190} domain={[0, 1]} format={(v) => pct(v)}
            rows={res.options.slice(0, 8).map((o) => ({ label: nice(o.label), value: o.milestone_rate,
              color: o.key === res.recommended ? 'var(--series-1)' : 'var(--axis)', note: `max regret ${pct(o.max_regret)}` }))} />
          <p className="note">{res.note} {res.n_seeds} seeds × {res.n_samples} assumption settings per option; {res.method}.</p>
        </div>
      )}
    </Card>
  )
}

/** What decides whether this video takes off: outcome drivers across many assumption samples. */
export function DriversCard({ videoId }: { videoId: string }) {
  const [res, setRes] = useState<Drivers | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { setRes(null); api.driversLatest(videoId).then(setRes).catch(() => {}) }, [videoId])
  const go = async () => {
    setError(null); setBusy('Starting…')
    try {
      const { job_id } = await api.driversStart(videoId)
      setRes(await waitForJob<Drivers>(job_id, (m, s) => setBusy(`${m} · ${Math.round(s)}s`)))
    } catch (e) { setError((e as Error).message) }
    setBusy(null)
  }
  return (
    <Card title="What decides whether it takes off" sub="Which uncertain assumptions explain most of the variation in reach for this video"
      right={<button className="btn" onClick={go} disabled={!!busy}>{busy ? 'Working…' : res ? 'Re-run' : 'Analyse (≈5 s)'}</button>}>
      {busy && <Spinner text={busy} />}
      <ErrorText error={error} />
      {res && (
        <div className="stack" style={{ gap: 10 }}>
          <p className="warnnote">{res.recommendation}</p>
          <HBarChart labelWidth={240} domain={[0, Math.max(...res.drivers.map((d) => d.share), 0.01)]} format={(v) => pct(v)}
            rows={res.drivers.slice(0, 8).map((d) => ({ label: nice(d.parameter), value: d.share,
              color: d.measurable ? 'var(--series-1)' : 'var(--axis)',
              note: `${d.rho > 0 ? 'higher helps' : 'lower helps'} (ρ ${d.rho.toFixed(2)})` }))} />
          <div className="legend">
            <span className="key"><span className="swatch" style={{ background: 'var(--series-1)' }} />you can measure it</span>
            <span className="key"><span className="swatch" style={{ background: 'var(--axis)' }} />platform-side assumption</span>
          </div>
          <p className="note">{res.note} Share of explained rank variation <Tag t="SIMULATED" /></p>
        </div>
      )}
    </Card>
  )
}
