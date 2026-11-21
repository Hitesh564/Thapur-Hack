import { useRef, useState } from 'react'
import { api, frameUrl, sourceUrl } from '../api'
import { ColumnChart, HBarChart } from '../components/charts'
import { Card, Empty, ErrorText, Metric, Spinner, Tag } from '../components/ui'
import { nice, num } from '../format'
import type { AnalysisBundle, Attention, VideoMeta } from '../types'

export default function VideoScreen({ videos, videoId, bundle, onSelect, onRefresh, onAnalysed, onNext }: {
  videos: VideoMeta[]; videoId: string | null; bundle: AnalysisBundle | null
  onSelect: (id: string) => void; onRefresh: () => void; onAnalysed: (b: AnalysisBundle) => void; onNext: () => void
}) {
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const upload = async (f: File) => {
    setBusy('Uploading and validating…'); setError(null)
    try {
      const meta = await api.upload(f)
      await onRefresh()
      onSelect(meta.video_id)
    } catch (e) { setError((e as Error).message) }
    setBusy(null)
  }
  const analyse = async () => {
    if (!videoId) return
    setBusy('Analysing: frames, cuts, transcript, GPT rubric ×3, hook analysis, CLIP relevance (about a minute for a new video)…')
    setError(null)
    try { onAnalysed(await api.analyze(videoId)) } catch (e) { setError((e as Error).message) }
    setBusy(null)
  }
  const meta = videos.find((v) => v.video_id === videoId)

  return (
    <div className="grid" style={{ gridTemplateColumns: 'minmax(220px, 300px) minmax(0, 1fr)' }}>
      <Card title="Videos" sub="Pick a draft or upload one (MP4, 3–90 s)">
        <div className="stack">
          <input ref={fileRef} type="file" accept="video/*" hidden
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
          <button className="btn primary" onClick={() => fileRef.current?.click()} disabled={!!busy}>Upload video</button>
          <div className="vidlist" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
            {videos.map((v) => (
              <button key={v.video_id} className={`vidcard${v.video_id === videoId ? ' selected' : ''}`} onClick={() => onSelect(v.video_id)}>
                {v.thumb ? <img src={frameUrl(v.video_id, v.thumb)} alt="" /> : <div style={{ aspectRatio: '9/12', background: 'var(--surface-2)' }} />}
                <div className="meta">
                  <div style={{ fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v.filename}</div>
                  <div className="muted">{v.duration_s.toFixed(1)} s{v.prior != null ? <> · p {num(v.prior)} <Tag t="MODEL_RATED" /></> : ' · not analysed'}</div>
                </div>
              </button>
            ))}
          </div>
          {!videos.length && <div className="muted small">No videos yet.</div>}
        </div>
      </Card>

      <div className="stack" style={{ gap: 16 }}>
        {busy && <Card><Spinner text={busy} /></Card>}
        <ErrorText error={error} />
        {!videoId && <Card><Empty>Select a video on the left to see its analysis.</Empty></Card>}
        {videoId && !bundle && !busy && (
          <Card title={meta?.filename} sub="Not analysed yet">
            <div className="row">
              <button className="btn primary" onClick={analyse}>Analyse video</button>
              <span className="secondary small">Runs once per file; results are cached by file hash.</span>
            </div>
          </Card>
        )}
        {videoId && bundle && <Analysis videoId={videoId} bundle={bundle} onNext={onNext} />}
      </div>
    </div>
  )
}

function Analysis({ videoId, bundle, onNext }: { videoId: string; bundle: AnalysisBundle; onNext: () => void }) {
  const { profile: p, prior } = bundle
  const f = p.features
  const rated = prior.method === 'vlm_rubric'
  return (
    <>
      <div className="grid g2">
        <Card title={p.topic || 'Video'} sub={p.category ? `${p.category} · ${p.format}` : undefined}
          right={<button className="btn primary" onClick={onNext}>Simulate spread →</button>}>
          <video className="preview" src={sourceUrl(videoId)} controls playsInline />
          <div className="grid g3" style={{ marginTop: 12 }}>
            <Metric label="Duration" value={`${p.duration_s.toFixed(1)} s`} tag="COMPUTED" />
            <Metric label="Scene cuts" value={p.cut_times_s.length} tag="COMPUTED" sub={`${num(f.cut_rate, 2)} per s`} />
            <Metric label="Opening motion" value={num(f.opening_motion, 3)} tag="COMPUTED" sub={`overall ${num(f.motion_energy, 3)}`} />
          </div>
        </Card>
        <Card title="Prior: how strong the opening looks" sub={rated ? `GPT rubric, ${prior.ensemble.length} samples (${prior.model})` : 'Heuristic fallback'}>
          <div className="row" style={{ alignItems: 'flex-end', gap: 24 }}>
            <Metric hero label="Prior p" value={num(prior.percentile)} tag={prior.evidence_tag}
              sub={prior.ensemble.length ? `samples ${prior.ensemble.map((e) => num(e)).join(' · ')}` : undefined} />
          </div>
          <p className="note" style={{ margin: '10px 0 14px' }}>
            {prior.note} {rated && 'The spread across samples is the model uncertainty the Lab samples from.'}
          </p>
          {Object.keys(prior.rubric_scores).length > 0 && (
            <>
              <h4 style={{ marginBottom: 6 }}>Rubric scores (1–10) <Tag t="MODEL_RATED" /></h4>
              <HBarChart labelWidth={130} domain={[0, 10]} format={(v) => v.toFixed(1)}
                rows={Object.entries(prior.rubric_scores).map(([k, v]) => ({ label: nice(k), value: v, color: 'var(--series-7)' }))} />
            </>
          )}
          {prior.rationale && <p className="secondary small" style={{ marginTop: 8 }}>“{prior.rationale}”</p>}
        </Card>
      </div>

      <div className="grid g2">
        <Card title="Hook breakdown" sub="Explanation only: these fields never drive the simulation" right={<Tag t="MODEL_RATED" />}>
          {p.analysis_source === 'none' ? <p className="secondary">Qualitative analysis unavailable for this video.</p> : (
            <div className="stack" style={{ gap: 10 }}>
              <div className="chips">{p.hook.categories.map((c) => <span key={c} className="chip">{c}</span>)}</div>
              <div><h4>Strengths</h4><ul className="clean">{p.hook.strengths.map((s) => <li key={s}>{s}</li>)}</ul></div>
              <div><h4>Weaknesses</h4><ul className="clean">{p.hook.weaknesses.map((s) => <li key={s}>{s}</li>)}</ul></div>
              <div><h4>Evidence</h4><ul className="clean small secondary">{p.hook.evidence.map((s) => <li key={s}>{s}</li>)}</ul></div>
              <div className="warnnote"><b>Suggested improvement: </b>{p.hook.suggested_improvement}</div>
              <div className="muted small">Model confidence {num(p.hook.confidence)} <Tag t="MODEL_RATED" /> · opening window {p.hook.opening_window_s}s</div>
            </div>
          )}
        </Card>
        <Card title="What the video is" sub={p.analysis_source.startsWith('vlm') || p.analysis_source.startsWith('cache') ? 'GPT analysis' : ''}>
          <div className="stack" style={{ gap: 10 }}>
            <p>{p.summary || '—'}</p>
            {p.narrative_structure && <p className="secondary small">{p.narrative_structure}</p>}
            <div><h4 style={{ marginBottom: 4 }}>Audience interest tags</h4>
              <div className="chips">{p.audience_interest_tags.map((t) => <span key={t} className="chip">{t}</span>)}</div></div>
            {p.transcript && <div><h4>Transcript</h4><p className="secondary small">{p.transcript}</p></div>}
            <div className="row">{p.frame_paths.slice(0, 8).map((fp, i) => (
              <img key={fp} src={frameUrl(videoId, fp)} alt={`frame at ${p.frame_times_s[i]}s`} title={`${p.frame_times_s[i]}s`}
                style={{ width: 48, height: 64, objectFit: 'cover', borderRadius: 4 }} />))}
            </div>
            {p.notes.length > 0 && <div className="warnnote">{p.notes.join(' ')}</div>}
          </div>
        </Card>
      </div>
      <AttentionCard videoId={videoId} rated={rated} />
    </>
  )
}

function AttentionCard({ videoId, rated }: { videoId: string; rated: boolean }) {
  const [att, setAtt] = useState<Attention | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const go = async () => {
    setBusy(true); setError(null)
    try { setAtt(await api.attention(videoId)) } catch (e) { setError((e as Error).message) }
    setBusy(false)
  }
  return (
    <Card title="Attention-risk timeline" sub="Where the model's rating is sensitive — not where viewers drop off"
      right={att ? <Tag t={att.evidence_tag} /> : <button className="btn" onClick={go} disabled={busy || !rated}>
        {busy ? 'Rating 16 variants…' : 'Compute (≈20 s)'}</button>}>
      {!att && !busy && <p className="secondary small">{rated
        ? 'Re-rates the video with each of 8 time windows removed, using the same rubric prompt.'
        : 'Needs the GPT rubric prior (not available for heuristic ratings).'}</p>}
      {busy && <Spinner text="Re-running the rubric without each window…" />}
      <ErrorText error={error} />
      {att && (
        <>
          <div className="legend">
            <span className="key"><span className="swatch" style={{ background: 'var(--div-pos)' }} />Supports the rating (drops without it)</span>
            <span className="key"><span className="swatch" style={{ background: 'var(--div-neg)' }} />Drag (rating higher without it)</span>
          </div>
          <ColumnChart height={180} categories={att.windows.map((w) => `${w.start_s.toFixed(1)}–${w.end_s.toFixed(1)}s`)}
            series={[{ name: 'Change in rating', color: 'var(--div-pos)', values: att.windows.map((w) => w.change) }]}
            colorFor={(v) => (v >= 0 ? 'var(--div-pos)' : 'var(--div-neg)')} yFormat={(v) => v.toFixed(3)} />
          <p className="note" style={{ marginTop: 8 }}>{att.message} Sampling noise ≈ {att.noise.toFixed(3)}.</p>
        </>
      )}
    </Card>
  )
}
