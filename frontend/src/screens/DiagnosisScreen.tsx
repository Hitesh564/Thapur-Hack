import { HBarChart, LineChart, SegmentedBar } from '../components/charts'
import { Card, Metric, Tag } from '../components/ui'
import { int, nice, num, pct } from '../format'
import type { RunPayload } from '../types'

const fmtFactor = (f: string, v: number | null | undefined) =>
  v == null ? '—' : f === 'share_given_stay' || f === 'stay_rate' ? pct(v, 1) : num(v, 2)

export default function DiagnosisScreen({ run }: { run: RunPayload }) {
  const d = run.diagnosis
  const b = d.binding
  const pooled = d.pooled
  const rounds = d.rounds.filter((r) => r.has_next)
  const sb = d.share_breakdown
  const comms = Object.entries(d.by_community)
  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="grid g3">
        <Card title="Reproduction number R" sub="New exposures each exposed viewer leads to in the next round (pooled over observed rounds)">
          <Metric hero label="R" value={num(d.R)} tag="SIMULATED"
            sub={<>recommendation {num(pooled.R_rec)} + shares {num(pooled.R_share)}</>} />
          <p className="note" style={{ marginTop: 10 }}>{d.stop.message}</p>
        </Card>
        <Card title="Binding constraint" sub="The factor furthest below its 'healthy' reference value" className="span2">
          <div className="row" style={{ alignItems: 'flex-start', gap: 28 }}>
            <Metric label="Weak factor" value={nice(b.factor)} tag="SIMULATED"
              sub={<>measured {fmtFactor(b.factor, b.measured)} vs reference {fmtFactor(b.factor, b.reference)} <Tag t="ASSUMED" /></>} />
            <Metric label="Ratio to reference" value={pct(b.ratio)} tag="SIMULATED" />
            <Metric label="Needed for R = 1" value={b.needed_value_for_R1 != null ? fmtFactor(b.factor, b.needed_value_for_R1) : 'R ≥ 1 already'}
              tag="SIMULATED" sub={b.needed_multiplier_for_R1 != null ? `×${num(b.needed_multiplier_for_R1, 1)} on any share factor, others fixed` : undefined} />
          </div>
          <div style={{ marginTop: 14 }}>
            <HBarChart labelWidth={190} format={(v) => pct(v)} refLine={{ value: 1, label: 'reference' }}
              rows={d.factors.map((f) => ({ label: nice(f.factor) + (f.factor === b.factor ? ' (binding)' : ''), value: f.ratio,
                color: f.factor === b.factor ? 'var(--series-1)' : 'var(--axis)', note: `measured ${fmtFactor(f.factor, f.measured)}` }))} />
          </div>
          <p className="note" style={{ marginTop: 8 }}>
            In this simulated run. References are assumed “healthy” settings from config; every multiplicative share factor
            needs the same multiplier to reach R = 1, so the reference gap — not the multiplier — names the weak link.
          </p>
        </Card>
      </div>

      <div className="grid g2">
        <Card title="R by round" sub="Rounds whose onward spread was observed" right={<Tag t="SIMULATED" />}>
          {rounds.length > 0 ? (
            <LineChart height={210} xLabel="round" yFormat={(v) => v.toFixed(2)} series={[
              { name: 'R total', color: 'var(--series-3)', points: rounds.map((r) => ({ x: r.round, y: r.R ?? 0 })) },
              { name: 'R recommendation', color: 'var(--series-1)', points: rounds.map((r) => ({ x: r.round, y: r.R_rec ?? 0 })) },
              { name: 'R shares', color: 'var(--series-2)', points: rounds.map((r) => ({ x: r.round, y: r.R_share })) },
            ]} />
          ) : <p className="secondary">The run stopped before any onward spread was observed.</p>}
        </Card>
        <Card title="Why shares spread (or not)" sub="R from shares = stay rate × share given stay × exposures per sharer (exact)" right={<Tag t="SIMULATED" />}>
          <div className="grid g3">
            <Metric label="Stay rate" value={pct(pooled.stay_rate, 1)} tag="SIMULATED" />
            <Metric label="Share given stay" value={pct(pooled.share_given_stay, 2)} tag="SIMULATED" />
            <Metric label="Exposures per sharer" value={num(pooled.exposures_per_sharer, 1)} tag="SIMULATED" />
          </div>
          <h4 style={{ margin: '14px 0 6px' }}>Exposures per sharer, broken down</h4>
          <div className="grid g2">
            <Metric label="Followers / sharer" value={num(sb.followers_per_sharer, 1)} tag="SIMULATED" />
            <Metric label="× unseen" value={pct(sb.unseen_fraction)} tag="SIMULATED" />
            <Metric label="× accept" value={pct(sb.acceptance)} tag="SIMULATED" sub={<>accept prob <Tag t="ASSUMED" /></>} />
            <Metric label="× cap & overlap" value={pct(sb.cap_and_overlap)} tag="SIMULATED" />
          </div>
        </Card>
      </div>

      <Card title="Bottlenecks" sub="Rules with thresholds from config; each cites its numbers">
        {d.bottlenecks.length === 0 ? <p className="secondary">No bottleneck rule fired in this simulated run.</p> : (
          <div className="stack" style={{ gap: 10 }}>
            {d.bottlenecks.map((x) => (
              <div key={x.code + x.message} className="persona">
                <div className="row"><b>{nice(x.code)}</b><Tag t={x.evidence_tag} /><span className="spacer" />
                  <span className="small muted">severity</span>
                  <div className="bar-inline" style={{ width: 90 }}><div style={{ width: `${x.severity * 100}%`, background: 'var(--serious)' }} /></div>
                  <span className="small">{pct(x.severity)}</span></div>
                <p style={{ margin: '4px 0' }}>{x.message}</p>
                <div className="chips">{Object.entries(x.evidence).map(([k, v]) => (
                  <span key={k} className="chip">{k.replace(/_/g, ' ')}: {Math.abs(v) < 1 && v !== 0 ? num(v, 3) : int(v)}</span>))}</div>
              </div>
            ))}
          </div>
        )}
      </Card>

      <div className="grid g2">
        <Card title="Reach attribution" sub="Where exposures came from: channel → source community → source role" right={<Tag t="SIMULATED" />}>
          <SegmentedBar parts={[
            { name: 'Recommendation', value: run.attribution.by_channel.rec ?? 0, color: 'var(--series-1)' },
            { name: 'Share', value: run.attribution.by_channel.share ?? 0, color: 'var(--series-2)' },
            { name: 'Seed', value: run.attribution.by_channel.seed ?? 0, color: 'var(--series-3)' },
          ]} />
          <p className="secondary small" style={{ margin: '8px 0' }}>
            {pct(run.attribution.share_cross_community)} of share exposures crossed into another community.</p>
          <div className="tablewrap" style={{ maxHeight: 280 }}>
            <table className="t">
              <thead><tr><th>Channel</th><th>Source</th><th>Role</th><th>Landed in</th><th className="num">Exposures</th></tr></thead>
              <tbody>{run.attribution.table.slice(0, 40).map((r, i) => (
                <tr key={i}><td>{r.channel}</td><td>{r.source_community}</td><td>{r.source_role}</td><td>{r.target_community}</td>
                  <td className="num">{int(r.exposures)}</td></tr>))}</tbody>
            </table>
          </div>
        </Card>
        <div className="stack" style={{ gap: 16 }}>
          <Card title="Pathway" sub={run.pathways.note} right={<Tag t="SIMULATED" />}>
            {run.pathways.primary ? (
              <div className="stack">
                <div className="row"><b style={{ fontSize: 16 }}>{nice(run.pathways.primary)}</b>
                  <span className="secondary small">{run.pathways.descriptions[run.pathways.primary]}</span></div>
                {run.pathways.matches.length > 1 && <div className="small secondary">Also matches: {run.pathways.matches.slice(1).map(nice).join(', ')}</div>}
              </div>
            ) : <p className="secondary">No pathway rule matched.</p>}
          </Card>
          <Card title="Factors by community" sub="Share factors where the community's viewers were exposed" right={<Tag t="SIMULATED" />}>
            <div className="tablewrap" style={{ maxHeight: 260 }}>
              <table className="t">
                <thead><tr><th>Community</th><th className="num">Exposed</th><th className="num">Stay</th><th className="num">Share | stay</th><th className="num">R shares</th><th>Weak factor</th></tr></thead>
                <tbody>{comms.map(([name, c]) => (
                  <tr key={name}><td>{name}</td><td className="num">{int(c.pooled.exposed)}</td><td className="num">{pct(c.pooled.stay_rate)}</td>
                    <td className="num">{pct(c.pooled.share_given_stay, 1)}</td><td className="num">{num(c.pooled.R_share, 2)}</td>
                    <td>{nice(c.binding.factor)}</td></tr>))}</tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
