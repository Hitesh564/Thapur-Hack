import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { HBarChart, Legend, LineChart } from '../components/charts'
import { Card, Metric, Tag } from '../components/ui'
import { int, num, pct, signed } from '../format'
import type { CounterfactualTrace, WorldLayout } from '../types'

const SIGNAL_NAME: Record<string, string> = { stay: 'Staying past the opening', watch: 'Watch time', like: 'Likes', share: 'Shares' }

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888'
}

/** Counterfactual Trace: why the variant wins, viewer by viewer, on the same dice. */
export default function TraceCard({ trace, onClose }: { trace: CounterfactualTrace; onClose: () => void }) {
  const t = trace
  const top = useRef<HTMLDivElement>(null)
  useEffect(() => { top.current?.scrollIntoView({ block: 'start' }) }, [trace])
  const d = t.decomposition
  const sp = d.share_parts
  const g = t.rec_growth
  const parts = [
    { label: 'Recommendation, same rounds', value: d.rec_common_rounds },
    { label: "Recommendation, rounds the original never reached", value: d.rec_extra_rounds },
    { label: 'Shares from new sharers', value: sp.from_new_sharers },
    { label: 'Shares lost (sharers only in original)', value: sp.lost_from_baseline_only_sharers },
    { label: 'Shares from sharers in both', value: sp.change_from_sharers_in_both },
    { label: 'Seed audience', value: d.seed },
  ].filter((p) => p.value !== 0)
  const col = (v: number) => (Math.abs(v) < 0.005 ? 'var(--axis)' : v > 0 ? 'var(--div-pos)' : 'var(--div-neg)')
  const fmtLog = (v: number) => (Math.abs(v) < 0.005 ? '0.00' : (v > 0 ? '+' : '') + v.toFixed(2))
  const signals = Object.entries(g.log_batch_difference_by_signal)
  const scan = t.seed_scan
  return (
    <div ref={top}>
    <Card title="Counterfactual Trace: why it wins, viewer by viewer"
      sub={`Original vs ${t.variant_label}, on the same seed, the same assumptions and the same per-viewer dice`}
      right={<div className="row"><Tag t="SIMULATED" /><button className="btn" style={{ padding: '2px 8px' }} onClick={onClose}>Close</button></div>}>
      <div className="stack" style={{ gap: 14 }}>
        <div className="grid g4">
          <Metric label="Reach on this seed" value={`${int(t.reach_baseline)} → ${int(t.reach_variant)}`} tag="SIMULATED"
            sub={`${signed(t.delta_reach)} viewers`} />
          <Metric label="Viewers who saw both" value={int(t.exposure.both)} tag="SIMULATED"
            sub={`${int(t.exposure.only_variant)} only variant · ${int(t.exposure.only_baseline)} only original`} />
          <Metric label="Stayed only with variant" value={int(t.flips.stay?.gained)} tag="SIMULATED"
            sub={`${int(t.flips.stay?.lost)} stayed only with the original`} />
          <Metric label="Seed used" value={t.seed} tag="SIMULATED"
            sub={scan ? `median of ${scan.seeds.length} seeds; variant ahead in ${pct(scan.share_positive)}` : 'chosen seed'} />
        </div>
        <p className="warnnote">{t.message}</p>

        <div className="grid g2">
          <div>
            <h4 style={{ marginBottom: 6 }}>Where the extra viewers came from</h4>
            <HBarChart labelWidth={260} format={(v) => signed(v)}
              rows={parts.map((p) => ({ label: p.label, value: p.value, color: col(p.value), note: 'viewers' }))} />
            <p className="small secondary" style={{ marginTop: 4 }}>
              Sum: {signed(parts.reduce((a, p) => a + p.value, 0))} = reach difference {signed(t.delta_reach)}
              {d.check ? ' ✓ adds up exactly' : ''}</p>
          </div>
          <div>
            <h4 style={{ marginBottom: 6 }}>Why recommendation grew faster</h4>
            <HBarChart labelWidth={190} format={fmtLog}
              rows={signals.map(([k, v]) => ({ label: SIGNAL_NAME[k] ?? k, value: v, color: col(v), note: 'log batch size gap' }))} />
            <p className="small secondary" style={{ marginTop: 4 }}>
              Exact split of the gap in log batch size over {g.rounds_used} shared rounds: batches ended ×{num(g.batch_ratio, 2)} larger.
              {g.capped ? ' The batch hit its cap, so the split is approximate there.' : ''}</p>
          </div>
        </div>

        <div className="grid g2">
          <div>
            <h4 style={{ marginBottom: 6 }}>Same dice, different threshold</h4>
            <p className="small secondary" style={{ marginBottom: 6 }}>
              Each viewer rolls the same number in both runs. They stay if it is below their stay probability, so a viewer
              flips exactly when the variant lifts the threshold past their roll.</p>
            <div className="tablewrap" style={{ maxHeight: 260 }}>
              <table className="t">
                <thead><tr><th>Viewer</th><th className="num">Dice</th><th className="num">P(stay) original → variant</th><th>Result</th></tr></thead>
                <tbody>{t.examples.map((e) => (
                  <tr key={e.viewer}>
                    <td>#{e.viewer}<div className="small muted">{e.persona} · {e.community}</div></td>
                    <td className="num">{num(e.dice, 3)}</td>
                    <td className="num">{num(e.p_stay_baseline, 3)} → <b>{num(e.p_stay_variant, 3)}</b></td>
                    <td className="small">{e.stayed_variant && !e.stayed_baseline ? 'stayed only with variant' : 'stayed only with original'}
                      {e.shared_variant ? ', then shared' : ''}</td>
                  </tr>))}</tbody>
              </table>
            </div>
          </div>
          <div>
            <h4 style={{ marginBottom: 6 }}>Expansion score per round</h4>
            <LineChart height={210} xLabel="round" yFormat={(v) => v.toFixed(2)} series={[
              { name: 'Original', color: 'var(--series-1)', points: t.rounds.filter((r) => r.S_baseline != null).map((r) => ({ x: r.round, y: r.S_baseline! })) },
              { name: t.variant_label, color: 'var(--series-2)', points: t.rounds.filter((r) => r.S_variant != null).map((r) => ({ x: r.round, y: r.S_variant! })) },
              { name: 'Growth threshold S0', color: 'var(--text-muted)', points: t.rounds.map((r) => ({ x: r.round, y: g.S0 })) },
            ]} />
          </div>
        </div>

        <DiffReplay trace={t} />
        <p className="note">{t.note} Assumptions at config defaults <Tag t="ASSUMED" />.</p>
      </div>
    </Card>
    </div>
  )
}

let layoutCache: WorldLayout | null = null

function DiffReplay({ trace }: { trace: CounterfactualTrace }) {
  const [layout, setLayout] = useState<WorldLayout | null>(layoutCache)
  const maxRound = Math.max(trace.rounds_baseline, trace.rounds_variant) - 1
  const [round, setRound] = useState(maxRound)
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => { if (!layoutCache) api.layout().then((l) => { layoutCache = l; setLayout(l) }).catch(() => {}) }, [])
  useEffect(() => setRound(maxRound), [maxRound, trace.seed])
  const idx = useMemo(() => {
    const m = (c: typeof trace.cascade_baseline) => {
      const r = new Map<number, { round: number; stay: number; share: number }>()
      c.viewer.forEach((v, i) => r.set(v, { round: c.round[i], stay: c.stay[i], share: c.share[i] }))
      return r
    }
    return { b: m(trace.cascade_baseline), v: m(trace.cascade_variant) }
  }, [trace])
  const counts = useMemo(() => {
    let both = 0, onlyV = 0, onlyB = 0, flip = 0
    const ids = new Set([...idx.b.keys(), ...idx.v.keys()])
    ids.forEach((id) => {
      const b = idx.b.get(id), v = idx.v.get(id)
      const inB = !!b && b.round <= round, inV = !!v && v.round <= round
      if (inB && inV) { both++; if (b!.stay !== v!.stay) flip++ } else if (inV) onlyV++; else if (inB) onlyB++
    })
    return { both, onlyV, onlyB, flip }
  }, [idx, round])

  useLayoutEffect(() => {
    const cv = ref.current
    if (!cv || !layout) return
    const dpr = window.devicePixelRatio || 1
    const size = cv.clientWidth
    cv.width = size * dpr; cv.height = size * dpr
    const ctx = cv.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size, size)
    const P = (x: number, y: number) => [(x * 0.5 + 0.5) * size, (y * 0.5 + 0.53) * size]
    const c = { un: cssVar('--unexposed'), both: cssVar('--text-muted'), gain: cssVar('--div-pos'), lost: cssVar('--div-neg'),
      ink: cssVar('--text-primary'), surface: cssVar('--surface-1'), label: cssVar('--text-secondary') }
    for (let v = 0; v < layout.x.length; v++) {
      const b = idx.b.get(v), w = idx.v.get(v)
      const inB = !!b && b.round <= round, inV = !!w && w.round <= round
      const [px, py] = P(layout.x[v], layout.y[v])
      let color = c.un, r = 1.5
      if (inB && inV) { color = c.both; r = 2 } else if (inV) { color = c.gain; r = 2.6 } else if (inB) { color = c.lost; r = 2.6 }
      ctx.fillStyle = color
      ctx.beginPath(); ctx.arc(px, py, r, 0, 7); ctx.fill()
      if (inB && inV && b!.stay !== w!.stay) {
        ctx.strokeStyle = w!.stay ? c.gain : c.lost; ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.arc(px, py, 4.5, 0, 7); ctx.stroke()
      }
    }
    ctx.font = '600 12px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.lineWidth = 4
    layout.community_centres.forEach(([cx, cy], k) => {
      const [px, py] = P(cx, cy - 0.27)
      const text = layout.community_names[k].split(' & ')[0]
      const wdt = ctx.measureText(text).width / 2 + 4
      const x = Math.min(size - wdt, Math.max(wdt, px)), y = Math.max(14, py)
      ctx.strokeStyle = c.surface; ctx.strokeText(text, x, y); ctx.fillStyle = c.label; ctx.fillText(text, x, y)
    })
  })

  return (
    <div>
      <h4 style={{ marginBottom: 6 }}>Difference replay</h4>
      <Legend items={[
        { name: `Reached only by the variant (${int(counts.onlyV)})`, color: 'var(--div-pos)' },
        { name: `Reached only by the original (${int(counts.onlyB)})`, color: 'var(--div-neg)' },
        { name: `Reached by both (${int(counts.both)})`, color: 'var(--text-muted)' },
        { name: 'Not reached', color: 'var(--unexposed)' },
      ]} />
      <div className="small secondary" style={{ marginBottom: 6 }}>Ringed = saw both versions but decided differently ({int(counts.flip)} so far).</div>
      <canvas ref={ref} className="replay-canvas" style={{ maxHeight: 520 }} />
      <div className="row" style={{ marginTop: 8 }}>
        <input type="range" min={0} max={maxRound} value={round} style={{ flex: 1 }} onChange={(e) => setRound(Number(e.target.value))} />
        <span className="small secondary">Round {round} of {maxRound} <Tag t="SIMULATED" /></span>
      </div>
    </div>
  )
}
