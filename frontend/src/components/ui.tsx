import type { ReactNode } from 'react'
import type { EvidenceTag } from '../types'

export const TAG_INFO: Record<EvidenceTag, string> = {
  COMPUTED: 'Computed: a deterministic function of the video or inputs (e.g. embedding relevance).',
  MODEL_RATED: "Model-rated: a model's judgement (the GPT rubric prior). Not trained or validated.",
  ASSUMED: 'Assumed: a parameter we chose, shown with its range. Verdicts are tested across the range.',
  SIMULATED: 'Simulated: an output of the simulation under stated assumptions. Not a prediction.',
  MEASURED: 'Measured: checked against real data (mini backtest only, anecdotal).',
}

const SHORT: Record<EvidenceTag, string> = {
  COMPUTED: 'COMPUTED', MODEL_RATED: 'MODEL-RATED', ASSUMED: 'ASSUMED', SIMULATED: 'SIMULATED', MEASURED: 'MEASURED',
}

export function Tag({ t }: { t: EvidenceTag | string }) {
  const tag = (t in SHORT ? t : 'SIMULATED') as EvidenceTag
  return <span className={`tag ${tag}`} title={TAG_INFO[tag]}>{SHORT[tag]}</span>
}

export function Metric({ label, value, tag, sub, hero }: { label: ReactNode; value: ReactNode; tag: EvidenceTag | string; sub?: ReactNode; hero?: boolean }) {
  return (
    <div className={`metric${hero ? ' hero' : ''}`}>
      <div className="label">{label} <Tag t={tag} /></div>
      <div className="value">{value}</div>
      {sub ? <div className="sub">{sub}</div> : null}
    </div>
  )
}

export function Card({ title, sub, right, children, className }: { title?: ReactNode; sub?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className ?? ''}`}>
      {(title || right) && (
        <div className="card-head">
          <div className="stack" style={{ gap: 2 }}>
            {title ? <h3>{title}</h3> : null}
            {sub ? <div className="card-sub">{sub}</div> : null}
          </div>
          {right}
        </div>
      )}
      {children}
    </section>
  )
}

const LABEL_STATUS: Record<string, { color: string; icon: string }> = {
  Robust: { color: 'var(--good)', icon: '✓' },
  Likely: { color: 'var(--warning)', icon: '◐' },
  Inconclusive: { color: 'var(--serious)', icon: '?' },
}

export function VerdictLabel({ label }: { label: string }) {
  const s = LABEL_STATUS[label] ?? LABEL_STATUS.Inconclusive
  return (
    <span className="status" style={{ background: 'var(--surface-2)' }}>
      <span className="dot" style={{ background: s.color }} aria-hidden />
      <span aria-hidden>{s.icon}</span> {label}
    </span>
  )
}

export function Spinner({ text }: { text?: string }) {
  return <span className="row secondary small"><span className="spinner" /> {text}</span>
}

export function ErrorText({ error }: { error: string | null }) {
  return error ? <div className="error">{error}</div> : null
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}
