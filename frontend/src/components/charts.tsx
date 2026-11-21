import { useLayoutEffect, useRef, useState } from 'react'
import type { PointerEvent as RPointerEvent, ReactNode, RefObject } from 'react'

// ---------- shared ----------
function useWidth<T extends HTMLElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T>(null)
  const [w, setW] = useState(600)
  useLayoutEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setW(Math.max(200, e.contentRect.width)))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, w]
}

function niceTicks(lo: number, hi: number, n = 5): number[] {
  if (hi === lo) hi = lo + 1
  const span = hi - lo
  const step0 = span / n
  const mag = 10 ** Math.floor(Math.log10(step0))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) ?? mag * 10
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v / step) * step)
  return out
}

/** Bar path: square at the baseline, 4px rounded data-end. Works for up/down/left/right. */
function barPath(x0: number, y0: number, x1: number, y1: number, end: 'top' | 'bottom' | 'left' | 'right', r = 4): string {
  const w = Math.abs(x1 - x0), h = Math.abs(y1 - y0)
  const L = Math.min(x0, x1), T = Math.min(y0, y1), R = L + w, B = T + h
  const rr = Math.min(r, end === 'top' || end === 'bottom' ? Math.min(w / 2, h) : Math.min(h / 2, w))
  if (rr <= 0.5) return `M${L},${T}H${R}V${B}H${L}Z`
  switch (end) {
    case 'top': return `M${L},${B}V${T + rr}Q${L},${T} ${L + rr},${T}H${R - rr}Q${R},${T} ${R},${T + rr}V${B}Z`
    case 'bottom': return `M${L},${T}H${R}V${B - rr}Q${R},${B} ${R - rr},${B}H${L + rr}Q${L},${B} ${L},${B - rr}Z`
    case 'right': return `M${L},${T}H${R - rr}Q${R},${T} ${R},${T + rr}V${B - rr}Q${R},${B} ${R - rr},${B}H${L}Z`
    case 'left': return `M${R},${T}V${B}H${L + rr}Q${L},${B} ${L},${B - rr}V${T + rr}Q${L},${T} ${L + rr},${T}Z`
  }
}

export function Legend({ items }: { items: { name: string; color: string; kind?: 'line' | 'rect' }[] }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.name} className="key">
          <span className={`swatch${it.kind === 'line' ? ' line' : ''}`} style={{ background: it.color }} />
          {it.name}
        </span>
      ))}
    </div>
  )
}

type Tip = { x: number; y: number; title: ReactNode; rows: { name: string; color?: string; value: string }[] } | null

function Tooltip({ tip, width }: { tip: Tip; width: number }) {
  if (!tip) return null
  const left = tip.x > width - 180 ? tip.x - 170 : tip.x + 12
  return (
    <div className="tooltip" style={{ left, top: Math.max(0, tip.y - 10) }}>
      <div className="t-title">{tip.title}</div>
      {tip.rows.map((r) => (
        <div key={r.name} className="t-row">
          {r.color ? <span className="swatch line" style={{ background: r.color }} /> : null}
          <span>{r.name}</span>
          <b>{r.value}</b>
        </div>
      ))}
    </div>
  )
}

function TableToggle({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  const [open, setOpen] = useState(false)
  return (
    <div style={{ marginTop: 6 }}>
      <button className="btn small" style={{ padding: '2px 8px', fontSize: 11 }} onClick={() => setOpen(!open)}>
        {open ? 'Hide table' : 'Table'}
      </button>
      {open && (
        <div className="tablewrap" style={{ marginTop: 6, maxHeight: 240 }}>
          <table className="t">
            <thead><tr>{head.map((h, i) => <th key={h} className={i ? 'num' : ''}>{h}</th>)}</tr></thead>
            <tbody>{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className={j ? 'num' : ''}>{c}</td>)}</tr>)}</tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ---------- line chart ----------
export interface LineSeries { name: string; color: string; points: { x: number; y: number }[]; band?: { lo: number; hi: number }[] }

export function LineChart({ series, height = 220, xLabel, yFormat = (v) => String(v), xFormat = (v) => String(v),
  yDomain, markers = [], legend = true }: {
  series: LineSeries[]; height?: number; xLabel?: string; yFormat?: (v: number) => string; xFormat?: (v: number) => string
  yDomain?: [number, number]; markers?: { x: number; label: string }[]; legend?: boolean
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip>(null)
  const [hx, setHx] = useState<number | null>(null)
  const m = { l: 44, r: 16, t: 10, b: xLabel ? 34 : 22 }
  const xs = series.flatMap((s) => s.points.map((p) => p.x))
  const ys = series.flatMap((s) => [...s.points.map((p) => p.y), ...(s.band?.flatMap((b) => [b.lo, b.hi]) ?? [])])
  const x0 = Math.min(...xs), x1 = Math.max(...xs)
  const [y0, y1] = yDomain ?? [Math.min(0, ...ys), Math.max(...ys) * 1.05 || 1]
  const W = width - m.l - m.r, H = height - m.t - m.b
  const sx = (v: number) => m.l + ((v - x0) / (x1 - x0 || 1)) * W
  const sy = (v: number) => m.t + H - ((v - y0) / (y1 - y0 || 1)) * H
  const yt = niceTicks(y0, y1, 4)
  const xt = Array.from(new Set(xs)).sort((a, b) => a - b)
  const xticks = xt.length <= 12 ? xt : niceTicks(x0, x1, 6)
  const onMove = (e: RPointerEvent<SVGRectElement>) => {
    const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
    const px = e.clientX - rect.left
    const xv = x0 + ((px - m.l) / W) * (x1 - x0)
    const near = xt.reduce((a, b) => (Math.abs(b - xv) < Math.abs(a - xv) ? b : a), xt[0])
    setHx(near)
    setTip({
      x: sx(near), y: e.clientY - rect.top, title: `${xLabel ?? 'x'} ${xFormat(near)}`,
      rows: series.map((s) => {
        const p = s.points.find((q) => q.x === near)
        return { name: s.name, color: s.color, value: p ? yFormat(p.y) : '—' }
      }),
    })
  }
  return (
    <div className="chart" ref={ref}>
      {legend && series.length > 1 && <Legend items={series.map((s) => ({ name: s.name, color: s.color, kind: 'line' }))} />}
      <svg width={width} height={height} role="img">
        {yt.map((t) => (
          <g key={t}>
            <line className="gridline" x1={m.l} x2={width - m.r} y1={sy(t)} y2={sy(t)} />
            <text x={m.l - 6} y={sy(t) + 4} textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}
        <line className="axisline" x1={m.l} x2={width - m.r} y1={sy(y0)} y2={sy(y0)} />
        {xticks.map((t) => <text key={t} x={sx(t)} y={height - m.b + 14} textAnchor="middle">{xFormat(t)}</text>)}
        {xLabel && <text x={m.l + W / 2} y={height - 4} textAnchor="middle">{xLabel}</text>}
        {series.map((s) => s.band && (
          <path key={s.name + 'b'} fill={s.color} opacity={0.1} d={
            `M${s.points.map((p, i) => `${sx(p.x)},${sy(s.band![i].hi)}`).join('L')}` +
            `L${[...s.points].reverse().map((p, i) => `${sx(p.x)},${sy(s.band![s.points.length - 1 - i].lo)}`).join('L')}Z`} />
        ))}
        {markers.map((mk) => (
          <g key={mk.label}>
            <line x1={sx(mk.x)} x2={sx(mk.x)} y1={m.t} y2={m.t + H} stroke="var(--text-secondary)" strokeWidth={1} />
            <text x={sx(mk.x) + 4} y={m.t + 10} className="label-ink">{mk.label}</text>
          </g>
        ))}
        {series.map((s) => (
          <path key={s.name} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"
            d={'M' + s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join('L')} />
        ))}
        {hx != null && <line x1={sx(hx)} x2={sx(hx)} y1={m.t} y2={m.t + H} stroke="var(--axis)" strokeWidth={1} />}
        {hx != null && series.map((s) => {
          const p = s.points.find((q) => q.x === hx)
          return p ? <circle key={s.name} cx={sx(p.x)} cy={sy(p.y)} r={4} fill={s.color} stroke="var(--surface-1)" strokeWidth={2} /> : null
        })}
        <rect x={m.l} y={m.t} width={W} height={H} fill="transparent" onPointerMove={onMove}
          onPointerLeave={() => { setTip(null); setHx(null) }} />
      </svg>
      <Tooltip tip={tip} width={width} />
      <TableToggle head={[xLabel ?? 'x', ...series.map((s) => s.name)]}
        rows={xt.map((x) => [xFormat(x), ...series.map((s) => { const p = s.points.find((q) => q.x === x); return p ? yFormat(p.y) : '—' })])} />
    </div>
  )
}

// ---------- column chart (vertical; grouped, stacked or signed) ----------
export interface ColSeries { name: string; color: string; values: number[] }

export function ColumnChart({ categories, series, stacked = false, height = 220, yFormat = (v) => String(v),
  colorFor, xLabel, catLabel }: {
  categories: string[]; series: ColSeries[]; stacked?: boolean; height?: number; yFormat?: (v: number) => string
  colorFor?: (v: number, i: number) => string; xLabel?: string; catLabel?: (c: string) => string
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip>(null)
  const [hi, setHi] = useState<number | null>(null)
  const m = { l: 48, r: 12, t: 10, b: xLabel ? 40 : 28 }
  const W = width - m.l - m.r, H = height - m.t - m.b
  const totals = categories.map((_, i) => series.reduce((a, s) => a + Math.max(0, s.values[i]), 0))
  const all = series.flatMap((s) => s.values)
  const ymax = stacked ? Math.max(...totals, 0) : Math.max(...all, 0)
  const ymin = stacked ? 0 : Math.min(...all, 0)
  const pad = (ymax - ymin) * 0.08 || 1
  const yt = niceTicks(ymin < 0 ? ymin - pad : 0, ymax + pad, 4)
  const lo = Math.min(yt[0], ymin), top = Math.max(yt[yt.length - 1], ymax)
  const sy = (v: number) => m.t + H - ((v - lo) / (top - lo || 1)) * H
  const band = W / categories.length
  const groups = stacked ? 1 : series.length
  const bw = Math.min(24, (band * 0.7) / groups)
  const gap = 2
  return (
    <div className="chart" ref={ref}>
      {series.length > 1 && <Legend items={series.map((s) => ({ name: s.name, color: s.color }))} />}
      <svg width={width} height={height} role="img">
        {yt.map((t) => (
          <g key={t}>
            <line className="gridline" x1={m.l} x2={width - m.r} y1={sy(t)} y2={sy(t)} />
            <text x={m.l - 6} y={sy(t) + 4} textAnchor="end">{yFormat(t)}</text>
          </g>
        ))}
        {categories.map((c, i) => {
          const cx = m.l + band * i + band / 2
          const groupW = groups * bw + (groups - 1) * gap
          let acc = 0
          return (
            <g key={c} opacity={hi == null || hi === i ? 1 : 0.55}>
              {series.map((s, k) => {
                const v = s.values[i]
                if (!v) return null
                let x: number, yA: number, yB: number
                if (stacked) {
                  x = cx - bw / 2
                  yA = sy(acc) - (acc > 0 ? gap : 0)
                  acc += v
                  yB = sy(acc)
                } else {
                  x = cx - groupW / 2 + k * (bw + gap)
                  yA = sy(0); yB = sy(v)
                }
                const isTop = stacked ? k === series.map((ss) => ss.values[i] > 0).lastIndexOf(true) : true
                const end = v >= 0 ? 'top' : 'bottom'
                const d = isTop ? barPath(x, yA, x + bw, yB, end) : `M${x},${Math.min(yA, yB)}h${bw}v${Math.abs(yB - yA)}h${-bw}Z`
                return <path key={s.name} d={d} fill={colorFor ? colorFor(v, i) : s.color} />
              })}
              {categories.length <= 16 && (
                <text x={cx} y={height - m.b + 14} textAnchor="middle">{catLabel ? catLabel(c) : c}</text>
              )}
              <rect x={m.l + band * i} y={m.t} width={band} height={H} fill="transparent"
                onPointerMove={(e) => {
                  const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
                  setHi(i)
                  setTip({ x: cx, y: e.clientY - rect.top, title: c,
                    rows: series.map((s) => ({ name: s.name, color: colorFor ? colorFor(s.values[i], i) : s.color, value: yFormat(s.values[i]) })) })
                }}
                onPointerLeave={() => { setTip(null); setHi(null) }} />
            </g>
          )
        })}
        <line className="axisline" x1={m.l} x2={width - m.r} y1={sy(0)} y2={sy(0)} />
        {xLabel && <text x={m.l + W / 2} y={height - 4} textAnchor="middle">{xLabel}</text>}
      </svg>
      <Tooltip tip={tip} width={width} />
      <TableToggle head={['', ...series.map((s) => s.name)]} rows={categories.map((c, i) => [c, ...series.map((s) => yFormat(s.values[i]))])} />
    </div>
  )
}

// ---------- horizontal bars (signed, with optional reference line) ----------
export function HBarChart({ rows, format = (v) => String(v), domain, refLine, rowHeight = 30, labelWidth = 170 }: {
  rows: { label: string; value: number; color: string; note?: string }[]
  format?: (v: number) => string; domain?: [number, number]; refLine?: { value: number; label: string }
  rowHeight?: number; labelWidth?: number
}) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const [tip, setTip] = useState<Tip>(null)
  const valW = 64
  const m = { l: labelWidth, r: valW, t: refLine ? 18 : 4, b: 4 }
  const W = Math.max(60, width - m.l - m.r)
  const vals = rows.map((r) => r.value)
  const [d0, d1] = domain ?? [Math.min(0, ...vals), Math.max(0, ...vals, refLine?.value ?? 0) || 1]
  const sx = (v: number) => m.l + ((v - d0) / (d1 - d0 || 1)) * W
  const height = m.t + rows.length * rowHeight + m.b
  const bh = Math.min(16, rowHeight - 10)
  const maxChars = Math.floor((labelWidth - 10) / 6.4)
  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img">
        {rows.map((r, i) => {
          const y = m.t + i * rowHeight + (rowHeight - bh) / 2
          const v = Math.max(d0, Math.min(d1, r.value))
          const end = v >= 0 ? 'right' : 'left'
          return (
            <g key={r.label + i}>
              <text x={m.l - 8} y={y + bh / 2 + 4} textAnchor="end" className="label-ink">
                {r.label.length > maxChars ? r.label.slice(0, maxChars - 1) + '…' : r.label}
              </text>
              {Math.abs(sx(v) - sx(0)) > 0.5 && <path d={barPath(sx(0), y, sx(v), y + bh, end)} fill={r.color} />}
              <text x={v >= 0 ? Math.max(sx(v), sx(0)) + 6 : sx(0) + 6} y={y + bh / 2 + 4} className="label-ink">{format(r.value)}</text>
              <rect x={0} y={m.t + i * rowHeight} width={width} height={rowHeight} fill="transparent"
                onPointerMove={(e) => {
                  const rect = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect()
                  setTip({ x: e.clientX - rect.left, y: e.clientY - rect.top, title: r.label,
                    rows: [{ name: r.note ?? 'value', color: r.color, value: format(r.value) }] })
                }}
                onPointerLeave={() => setTip(null)} />
            </g>
          )
        })}
        <line className="axisline" x1={sx(0)} x2={sx(0)} y1={m.t - 2} y2={height - m.b} />
        {refLine && (
          <g>
            <line x1={sx(refLine.value)} x2={sx(refLine.value)} y1={m.t - 4} y2={height - m.b} stroke="var(--text-secondary)" strokeWidth={1} />
            <text x={sx(refLine.value)} y={10} textAnchor="middle" className="label-ink">{refLine.label}</text>
          </g>
        )}
      </svg>
      <Tooltip tip={tip} width={width} />
      <TableToggle head={['', 'value']} rows={rows.map((r) => [r.label, format(r.value)])} />
    </div>
  )
}

/** Thin segmented bar: one whole split into parts (no text inside). */
export function SegmentedBar({ parts }: { parts: { name: string; value: number; color: string }[] }) {
  const total = parts.reduce((a, p) => a + p.value, 0) || 1
  return (
    <div>
      <div style={{ display: 'flex', gap: 2, height: 8 }}>
        {parts.filter((p) => p.value > 0).map((p) => (
          <div key={p.name} title={`${p.name}: ${(100 * p.value / total).toFixed(0)}%`}
            style={{ flex: p.value / total, background: p.color, borderRadius: 4 }} />
        ))}
      </div>
      <div className="legend" style={{ marginTop: 8, marginBottom: 0 }}>
        {parts.map((p) => (
          <span key={p.name} className="key"><span className="swatch" style={{ background: p.color }} />
            {p.name} <b style={{ color: 'var(--text-primary)' }}>{(100 * p.value / total).toFixed(0)}%</b></span>
        ))}
      </div>
    </div>
  )
}
