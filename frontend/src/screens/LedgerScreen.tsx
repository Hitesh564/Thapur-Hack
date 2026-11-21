import { useEffect, useState } from 'react'
import { api } from '../api'
import { Card, ErrorText, Spinner, Tag, TAG_INFO } from '../components/ui'
import { nice } from '../format'
import type { EvidenceTag, Ledger } from '../types'

const TAGS: EvidenceTag[] = ['COMPUTED', 'MODEL_RATED', 'ASSUMED', 'SIMULATED', 'MEASURED']

const NICE_OR_LAST = (m: string) => { const n = nice(m); return n === m || n.includes('.') ? nice(m.split('.').slice(-1)[0]) : n }

const fmt = (v: number | string | null) =>
  v == null ? '—' : typeof v === 'number' ? (Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : Number(v.toFixed(4)).toString()) : v

export default function LedgerScreen({ videoId, runId, experimentId }: { videoId: string | null; runId: string | null; experimentId: string | null }) {
  const [led, setLed] = useState<Ledger | null>(null)
  const [filter, setFilter] = useState<EvidenceTag | 'ALL'>('ALL')
  const [q, setQ] = useState('')
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    api.ledger({ video_id: videoId ?? undefined, run_id: runId ?? undefined, experiment_id: experimentId ?? undefined })
      .then(setLed).catch((e) => setErr((e as Error).message))
  }, [videoId, runId, experimentId])
  if (err) return <ErrorText error={`Could not load the ledger: ${err}`} />
  if (!led) return <Spinner text="Loading ledger…" />
  const counts = Object.fromEntries(TAGS.map((t) => [t, led.entries.filter((e) => e.tag === t).length]))
  const rows = led.entries.filter((e) => (filter === 'ALL' || e.tag === filter) && (!q || e.metric.toLowerCase().includes(q.toLowerCase())))
  const validated = !led.validation.startsWith('Not validated')
  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="grid g2">
        <Card title="What each number is" sub="Every metric in the app carries one of these tags">
          <div className="stack" style={{ gap: 8 }}>
            {TAGS.map((t) => (
              <div key={t} className="row" style={{ alignItems: 'flex-start' }}>
                <div style={{ width: 120 }}><Tag t={t} /></div>
                <span className="small secondary" style={{ flex: 1 }}>{TAG_INFO[t]}</span>
                <b style={{ minWidth: 30, textAlign: 'right' }}>{counts[t]}</b>
              </div>
            ))}
          </div>
        </Card>
        <Card title="Validation" sub="We have no outcome dataset, so we do not claim predictive accuracy">
          <p style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>
            {validated ? <Tag t="MEASURED" /> : null} {led.validation}
          </p>
          <table className="t">
            <thead><tr><th>Layer</th><th>What it shows</th><th>Method</th></tr></thead>
            <tbody>{led.layers.map((l) => <tr key={l.layer}><td>{l.layer}</td><td>{l.shows}</td><td className="secondary">{l.method}</td></tr>)}</tbody>
          </table>
        </Card>
      </div>
      <Card title="Ledger" sub={`${led.entries.length} metrics for the current video, run and experiment`}
        right={<div className="row">
          <input type="text" placeholder="Search metric" value={q} onChange={(e) => setQ(e.target.value)} />
          <select value={filter} onChange={(e) => setFilter(e.target.value as EvidenceTag | 'ALL')}>
            <option value="ALL">All tags</option>{TAGS.map((t) => <option key={t} value={t}>{t}</option>)}</select>
        </div>}>
        <div className="tablewrap" style={{ maxHeight: 560 }}>
          <table className="t">
            <thead><tr><th>Metric</th><th className="num">Value</th><th>Tag</th><th className="num">Range</th><th>Note</th></tr></thead>
            <tbody>{rows.map((e) => (
              <tr key={e.metric}>
                <td title={e.metric}>{NICE_OR_LAST(e.metric)} <span className="muted small mono">{e.metric}</span></td>
                <td className="num">{fmt(e.value)}</td>
                <td><Tag t={e.tag} /></td>
                <td className="num secondary">{e.range ? `${fmt(e.range[0])} – ${fmt(e.range[1])}` : ''}</td>
                <td className="secondary small">{e.note}</td>
              </tr>))}</tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}
