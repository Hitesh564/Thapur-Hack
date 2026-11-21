import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { MouseEvent } from 'react'
import type { RunOptions } from '../App'
import { api } from '../api'
import { ColumnChart, LineChart, Legend } from '../components/charts'
import { Card, ErrorText, Metric, Spinner, Tag } from '../components/ui'
import { int, nice, num, pct } from '../format'
import type { Cascade, RunPayload, WorldLayout } from '../types'

const CH_COLOR = ['var(--series-3)', 'var(--series-1)', 'var(--series-2)'] // seed, rec, share
const CH_NAME = ['Seed', 'Recommendation', 'Share']
const ROLE = ['other', 'hub', 'bridge']

let layoutCache: WorldLayout | null = null

function cssVar(name: string): string {
  const v = name.match(/var\((--[^)]+)\)/)?.[1] ?? name
  return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || '#888'
}

export default function ReplayScreen({ run, running, error, opts, onRun, onNext }: {
  run: RunPayload | null; running: boolean; error: string | null; opts: RunOptions
  onRun: (o: RunOptions) => void; onNext: () => void
}) {
  const [draft, setDraft] = useState<RunOptions>(opts)
  useEffect(() => setDraft(opts), [opts])
  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="row">
        <label className="field">Seed<input type="number" value={draft.seed} style={{ width: 80 }}
          onChange={(e) => setDraft({ ...draft, seed: Number(e.target.value) })} /></label>
        <label className="field">Seeding strategy
          <select value={draft.seeding_strategy ?? ''} onChange={(e) => setDraft({ ...draft, seeding_strategy: e.target.value || null })}>
            <option value="">Mixed (default)</option><option value="random">Random</option>
            <option value="relevance_biased">Relevance-biased</option><option value="community_targeted">Community-targeted</option>
          </select></label>
        <label className="field">Distribution policy
          <select value={draft.policy_variant} onChange={(e) => setDraft({ ...draft, policy_variant: e.target.value })}>
            <option value="default">Default</option><option value="conservative">Conservative (higher threshold)</option>
            <option value="exploratory">Exploratory (more random)</option><option value="interest_tight">Interest-tight (no random)</option>
          </select></label>
        <button className="btn primary" style={{ alignSelf: 'flex-end' }} disabled={running} onClick={() => onRun(draft)}>
          {running ? 'Running…' : 'Run simulation'}</button>
        <span className="spacer" />
        {run && <button className="btn" style={{ alignSelf: 'flex-end' }} onClick={onNext}>Diagnose this run →</button>}
      </div>
      <ErrorText error={error} />
      {running && !run && <Card><Spinner text="Simulating 4,000 viewers…" /></Card>}
      {run && <ReplayBody run={run} />}
    </div>
  )
}

function ReplayBody({ run }: { run: RunPayload }) {
  const s = run.summary
  const rounds = run.replay
  return (
    <div className="grid" style={{ gridTemplateColumns: 'minmax(0, 1.25fr) minmax(0, 1fr)' }}>
      <CascadeCanvas run={run} />
      <div className="stack" style={{ gap: 16 }}>
        <Card title="This simulated run" sub={`seed ${run.config.seed} · ${nice(run.config.seeding_strategy)} seeding · ${run.config.policy_variant} policy`}>
          <div className="grid g3">
            <Metric label="Unique reach" value={int(s.unique_reach)} tag="SIMULATED" sub={`${pct(s.reach_pct, 1)} of population`} />
            <Metric label="Rounds" value={s.rounds} tag="SIMULATED" sub={nice(run.stop_reason)} />
            <Metric label="Shares" value={int(s.total_shares)} tag="SIMULATED" sub={`share depth ${s.depth}`} />
          </div>
          <p className="note" style={{ marginTop: 10 }}>{run.diagnosis.stop.message}</p>
          {run.pathways.primary && (
            <div className="row" style={{ marginTop: 8 }}>
              <span className="secondary small">Pathway</span>
              {run.pathways.matches.map((m) => <span key={m} className="chip">{nice(m)}</span>)}<Tag t="SIMULATED" />
            </div>
          )}
        </Card>
        <Card title="New exposures per round, by channel" right={<Tag t="SIMULATED" />}>
          <ColumnChart height={200} stacked categories={rounds.map((r) => String(r.round))} xLabel="round"
            series={['seed', 'rec', 'share'].map((c, i) => ({ name: CH_NAME[i], color: CH_COLOR[i], values: rounds.map((r) => r.by_channel[c] ?? 0) }))}
            yFormat={(v) => int(v)} />
        </Card>
        <Card title="Engagement signals per round" sub="The policy grows the recommendation batch when the expansion score beats its threshold" right={<Tag t="SIMULATED" />}>
          <LineChart height={190} xLabel="round" yDomain={[0, 1]} yFormat={(v) => v.toFixed(2)}
            series={[
              { name: 'Stay rate', color: 'var(--series-1)', points: s.rounds_metrics.map((m) => ({ x: m.round, y: m.stay_rate })) },
              { name: 'Mean watch fraction (stayers)', color: 'var(--series-2)', points: s.rounds_metrics.map((m) => ({ x: m.round, y: m.mean_watch_fraction })) },
              { name: 'Expansion score', color: 'var(--series-3)', points: s.rounds_metrics.map((m) => ({ x: m.round, y: m.expansion_score })) },
            ]} />
        </Card>
      </div>
    </div>
  )
}

function CascadeCanvas({ run }: { run: RunPayload }) {
  const [layout, setLayout] = useState<WorldLayout | null>(layoutCache)
  const [cas, setCas] = useState<Cascade | null>(null)
  const [round, setRound] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [picked, setPicked] = useState<number | null>(null)
  const [loadErr, setLoadErr] = useState<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const maxRound = run.summary.rounds - 1

  useEffect(() => {
    if (!layoutCache) api.layout().then((l) => { layoutCache = l; setLayout(l) }).catch((e) => setLoadErr((e as Error).message))
  }, [])
  useEffect(() => {
    setCas(null); setRound(0); setPicked(null)
    api.cascade(run.run_id).then((c) => { setCas(c); setRound(0); setPlaying(true) }).catch((e) => setLoadErr((e as Error).message))
  }, [run.run_id])
  useEffect(() => {
    if (!playing) return
    if (round >= maxRound) { setPlaying(false); return }
    const t = setTimeout(() => setRound((r) => r + 1), 850)
    return () => clearTimeout(t)
  }, [playing, round, maxRound])

  const idx = useMemo(() => {
    const m = new Map<number, number>()
    cas?.viewer.forEach((v, i) => m.set(v, i))
    return m
  }, [cas])

  const toPx = (x: number, y: number, size: number) => [(x * 0.5 + 0.5) * size, (y * 0.5 + 0.53) * size] as const

  useLayoutEffect(() => {
    const cv = canvasRef.current
    if (!cv || !layout) return
    const dpr = window.devicePixelRatio || 1
    const size = cv.clientWidth
    cv.width = size * dpr; cv.height = size * dpr
    const ctx = cv.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size, size)
    const col = CH_COLOR.map(cssVar)
    const unexposed = cssVar('--unexposed'), ink = cssVar('--text-primary'), muted = cssVar('--text-muted'), surface = cssVar('--surface-1')
    const n = layout.x.length
    const drawLabels = () => {
      ctx.font = '600 12px system-ui, sans-serif'; ctx.textAlign = 'center'
      ctx.lineWidth = 4; ctx.strokeStyle = surface; ctx.fillStyle = muted
      layout.community_centres.forEach(([cx, cy], c) => {
        const [px, py] = toPx(cx, cy - 0.27, size)
        const label = layout.community_names[c].split(' & ')[0]
        const w = ctx.measureText(label).width / 2 + 4
        const x = Math.min(size - w, Math.max(w, px)), y = Math.max(14, py)
        ctx.strokeText(label, x, y); ctx.fillText(label, x, y)
      })
    }
    // unexposed
    ctx.fillStyle = unexposed
    for (let v = 0; v < n; v++) {
      if (cas && idx.has(v) && cas.round[idx.get(v)!] <= round) continue
      const [px, py] = toPx(layout.x[v], layout.y[v], size)
      const r = layout.role[v] === 1 ? 3.5 : 1.6
      ctx.beginPath(); ctx.arc(px, py, r, 0, 7); ctx.fill()
    }
    if (!cas) { drawLabels(); return }
    // share links of the current round
    ctx.strokeStyle = col[2]; ctx.globalAlpha = 0.55; ctx.lineWidth = 1
    cas.viewer.forEach((v, i) => {
      if (cas.round[i] !== round || cas.channel[i] !== 2 || cas.source[i] < 0) return
      const [ax, ay] = toPx(layout.x[cas.source[i]], layout.y[cas.source[i]], size)
      const [bx, by] = toPx(layout.x[v], layout.y[v], size)
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke()
    })
    ctx.globalAlpha = 1
    // exposed nodes up to this round
    cas.viewer.forEach((v, i) => {
      if (cas.round[i] > round) return
      const [px, py] = toPx(layout.x[v], layout.y[v], size)
      const current = cas.round[i] === round
      const role = layout.role[v]
      const r = (role === 1 ? 4.5 : role === 2 ? 3 : 2.2) + (current ? 0.8 : 0)
      ctx.globalAlpha = cas.stay[i] ? 1 : 0.35
      ctx.fillStyle = col[cas.channel[i]]
      ctx.beginPath(); ctx.arc(px, py, r, 0, 7); ctx.fill()
      ctx.globalAlpha = 1
      if (role === 2) { ctx.strokeStyle = ink; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(px, py, r + 1.5, 0, 7); ctx.stroke() }
      if (cas.share[i]) { ctx.strokeStyle = surface; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(px, py, r * 0.45, 0, 7); ctx.stroke() }
    })
    if (picked != null) {
      const [px, py] = toPx(layout.x[picked], layout.y[picked], size)
      ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(px, py, 8, 0, 7); ctx.stroke()
    }
    drawLabels()
  })

  const onClick = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!layout || !cas) return
    const rect = e.currentTarget.getBoundingClientRect()
    const size = rect.width
    let best = -1, bd = 14 * 14
    cas.viewer.forEach((v, i) => {
      if (cas.round[i] > round) return
      const [px, py] = toPx(layout.x[v], layout.y[v], size)
      const d = (px - (e.clientX - rect.left)) ** 2 + (py - (e.clientY - rect.top)) ** 2
      if (d < bd) { bd = d; best = v }
    })
    setPicked(best >= 0 ? best : null)
  }

  const exposedSoFar = cas ? cas.round.filter((r) => r <= round).length : 0
  const pi = picked != null ? idx.get(picked) : undefined
  return (
    <Card title="Cascade replay" sub="Each dot is a fictional viewer, grouped by community. Click an exposed dot to inspect it."
      right={<Tag t="SIMULATED" />}>
      <Legend items={[
        { name: 'Seed', color: CH_COLOR[0] }, { name: 'Recommendation', color: CH_COLOR[1] },
        { name: 'Share', color: CH_COLOR[2] }, { name: 'Not exposed', color: 'var(--unexposed)' },
      ]} />
      <div className="small secondary" style={{ marginBottom: 6 }}>
        Faded = skipped the opening · hollow centre = shared · large = hub · ringed = bridge · lines = shares this round
      </div>
      <div className="replay-wrap">
        {loadErr ? <ErrorText error={`Could not load the replay: ${loadErr}`} /> : (!layout || !cas) && <Spinner text="Loading cascade…" />}
        <canvas ref={canvasRef} className="replay-canvas" onClick={onClick} />
      </div>
      <div className="row" style={{ marginTop: 10 }}>
        <button className="btn" onClick={() => { if (round >= maxRound) setRound(0); setPlaying(!playing) }}>
          {playing ? 'Pause' : round >= maxRound ? 'Replay' : 'Play'}</button>
        <input type="range" min={0} max={Math.max(0, maxRound)} value={round} style={{ flex: 1 }}
          onChange={(e) => { setPlaying(false); setRound(Number(e.target.value)) }} />
        <span className="secondary small" style={{ minWidth: 170, textAlign: 'right' }}>
          Round {round} of {maxRound} · {int(exposedSoFar)} exposed <Tag t="SIMULATED" /></span>
      </div>
      {picked != null && pi != null && cas && layout && (
        <div className="persona" style={{ marginTop: 10 }}>
          <div className="row"><b>Viewer {picked}</b><span className="chip">{layout.community_names[layout.community[picked]]}</span>
            <span className="chip">{ROLE[layout.role[picked]]}</span><span className="muted small">{layout.followers[picked]} followers</span>
            <span className="spacer" /><button className="btn" style={{ padding: '2px 8px' }} onClick={() => setPicked(null)}>Close</button></div>
          <div className="small secondary" style={{ marginTop: 6 }}>
            Exposed in round {cas.round[pi]} via <b>{CH_NAME[cas.channel[pi]].toLowerCase()}</b>
            {cas.source[pi] >= 0 && <> from viewer {cas.source[pi]}</>}. {cas.stay[pi] ? 'Stayed past the opening' : 'Skipped the opening'}
            {cas.stay[pi] ? (cas.complete[pi] ? ', watched to the end' : ', partial view') : ''}
            {cas.like[pi] ? ', liked' : ''}{cas.share[pi] ? ', shared' : ''}. <Tag t="SIMULATED" />
          </div>
        </div>
      )}
      <p className="muted small" style={{ marginTop: 8 }}>Prior used: p = {num(run.config.assumption.prior_percentile)} ({run.config.prior_method}).</p>
    </Card>
  )
}
