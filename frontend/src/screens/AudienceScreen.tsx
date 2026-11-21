import { useEffect, useState } from 'react'
import { api } from '../api'
import { ColumnChart } from '../components/charts'
import { Card, ErrorText, Spinner, Tag } from '../components/ui'
import { int, num } from '../format'
import type { Audience } from '../types'

export default function AudienceScreen({ videoId }: { videoId: string }) {
  const [aud, setAud] = useState<Audience | null>(null)
  const [comm, setComm] = useState(0)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { api.audience(videoId).then(setAud).catch((e) => setErr((e as Error).message)) }, [videoId])
  if (err) return <ErrorText error={`Could not load the audience: ${err}`} />
  if (!aud) return <Spinner text="Loading audience…" />
  const personas = aud.archetypes.filter((a) => a.community === comm)
  const top = [...aud.communities].sort((a, b) => (b.mean_relevance ?? 0) - (a.mean_relevance ?? 0))[0]
  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="grid g2">
        <Card title="Mean relevance by community" sub="CLIP similarity between persona interests and the video's frames" right={<Tag t="COMPUTED" />}>
          <ColumnChart height={230} categories={aud.communities.map((c) => c.name)} catLabel={(c) => c.split(' ')[0]}
            series={[{ name: 'Mean relevance', color: 'var(--series-1)', values: aud.communities.map((c) => c.mean_relevance ?? 0) }]}
            yFormat={(v) => v.toFixed(2)} />
          <p className="note" style={{ marginTop: 8 }}>
            Highest: {top?.name} ({num(top?.mean_relevance)}). A zero-shot proxy for interest, not a measurement of real people.
          </p>
        </Card>
        <Card title="Synthetic population" sub={`${int(aud.n_viewers)} fictional viewers · population ${aud.population_version} · graph ${aud.graph_version}`}
          right={<Tag t="ASSUMED" />}>
          <div className="tablewrap">
            <table className="t">
              <thead><tr><th>Community</th><th className="num">Viewers</th><th className="num">Hubs</th><th className="num">Bridges</th><th className="num">Relevance <Tag t="COMPUTED" /></th></tr></thead>
              <tbody>{aud.communities.map((c) => (
                <tr key={c.community} style={{ cursor: 'pointer', background: c.community === comm ? 'var(--surface-2)' : undefined }}
                  onClick={() => setComm(c.community)}>
                  <td>{c.name}</td><td className="num">{int(c.viewers)}</td><td className="num">{c.hubs}</td><td className="num">{c.bridges}</td>
                  <td className="num">{num(c.mean_relevance)} </td>
                </tr>))}
              </tbody>
            </table>
          </div>
          <p className="note" style={{ marginTop: 8 }}>{aud.note} Hubs and bridges are computed from the generated follower graph.</p>
        </Card>
      </div>
      <Card title={`Personas in ${aud.communities[comm]?.name}`} sub="Archetypes generated once by GPT and validated (no sensitive traits). Click a community above to switch."
        right={<select value={comm} onChange={(e) => setComm(Number(e.target.value))}>
          {aud.communities.map((c) => <option key={c.community} value={c.community}>{c.name}</option>)}</select>}>
        <div className="grid g3">
          {personas.map((a) => (
            <div key={a.name} className="persona stack" style={{ gap: 8 }}>
              <div className="row"><b>{a.name}</b><span className="spacer" /><span className="chip">{a.influence_class} influence</span></div>
              <div className="stack" style={{ gap: 4 }}>
                {a.interests.map((i) => (
                  <div key={i.phrase} className="small">
                    <div className="row"><span>{i.phrase}</span><span className="spacer" /><span className="muted">{num(i.strength, 1)}</span></div>
                    <div className="bar-inline"><div style={{ width: `${i.strength * 100}%`, background: 'var(--series-1)' }} /></div>
                  </div>
                ))}
              </div>
              <div className="small secondary">
                {Object.entries(a.traits).map(([k, [lo, hi]]) => (
                  <span key={k} style={{ marginRight: 10 }}>{k} {num(lo, 2)}–{num(hi, 2)}</span>))}
                <Tag t="ASSUMED" />
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
