import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { Tag } from './components/ui'
import { num, pct } from './format'
import AudienceScreen from './screens/AudienceScreen'
import DiagnosisScreen from './screens/DiagnosisScreen'
import LabScreen from './screens/LabScreen'
import LedgerScreen from './screens/LedgerScreen'
import ReplayScreen from './screens/ReplayScreen'
import VideoScreen from './screens/VideoScreen'
import type { AnalysisBundle, ExperimentRecord, RunPayload, VideoMeta } from './types'

export type TabId = 'video' | 'audience' | 'replay' | 'diagnosis' | 'lab' | 'ledger'
const TABS: { id: TabId; label: string; needs?: 'analysis' | 'run' }[] = [
  { id: 'video', label: 'Video & hook' },
  { id: 'audience', label: 'Audience', needs: 'analysis' },
  { id: 'replay', label: 'Replay', needs: 'analysis' },
  { id: 'diagnosis', label: 'Diagnosis', needs: 'run' },
  { id: 'lab', label: 'Lab', needs: 'analysis' },
  { id: 'ledger', label: 'Evidence ledger' },
]

export interface RunOptions { seed: number; seeding_strategy: string | null; policy_variant: string }

export default function App() {
  const [tab, setTab] = useState<TabId>('video')
  const [videos, setVideos] = useState<VideoMeta[]>([])
  const [videoId, setVideoId] = useState<string | null>(null)
  const [bundle, setBundle] = useState<AnalysisBundle | null>(null)
  const [run, setRun] = useState<RunPayload | null>(null)
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  const [runOpts, setRunOpts] = useState<RunOptions>({ seed: 0, seeding_strategy: null, policy_variant: 'default' })
  const [experiment, setExperiment] = useState<ExperimentRecord | null>(null)
  const [backendOk, setBackendOk] = useState<boolean | null>(null)

  const refreshVideos = useCallback(async () => {
    try { setVideos(await api.videos()); setBackendOk(true) } catch { setBackendOk(false) }
  }, [])
  useEffect(() => { refreshVideos() }, [refreshVideos])
  useEffect(() => { // retry while the backend is still starting
    if (backendOk !== false) return
    const t = setInterval(refreshVideos, 3000)
    return () => clearInterval(t)
  }, [backendOk, refreshVideos])

  const selectVideo = useCallback(async (id: string) => {
    setVideoId(id); setBundle(null); setRun(null); setRunError(null)
    try { setBundle(await api.analysis(id)) } catch { /* not analysed yet */ }
  }, [])

  const runSim = useCallback(async (opts: RunOptions = runOpts) => {
    if (!videoId) return
    setRunning(true); setRunError(null); setRunOpts(opts)
    try { setRun(await api.simulate({ video_id: videoId, ...opts })) } catch (e) { setRunError(String((e as Error).message)) }
    setRunning(false)
  }, [videoId, runOpts])

  // demo path: opening Replay or Diagnosis runs the baseline simulation automatically
  useEffect(() => {
    if ((tab === 'replay' || tab === 'diagnosis') && bundle && videoId && !running && (!run || run.config.video_id !== videoId) && !runError) {
      runSim()
    }
  }, [tab, bundle, videoId, run, running, runSim, runError])

  const enabled = (t: (typeof TABS)[number]) =>
    !t.needs || (t.needs === 'analysis' ? !!bundle : !!bundle)
  const meta = videos.find((v) => v.video_id === videoId)

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand"><h1>Viralyst</h1><span>Propagation Lab</span></div>
        <p className="disclaimer">
          A what-if simulation of a synthetic audience under stated assumptions. It does not predict views and does not
          reproduce any platform's algorithm.
        </p>
      </header>
      <nav className="tabs">
        {TABS.map((t, i) => (
          <button key={t.id} className={`tab${tab === t.id ? ' active' : ''}`} disabled={!enabled(t)}
            onClick={() => setTab(t.id)}>
            <span className="num">{i + 1}</span>{t.label}
          </button>
        ))}
      </nav>
      {backendOk === false && (
        <div className="warnnote" style={{ marginBottom: 16 }}>
          Backend not reachable. Start it with <span className="mono">cd backend; uvicorn app.api.main:app --port 8000</span>
        </div>
      )}
      {bundle && meta && tab !== 'video' && (
        <div className="context">
          <b style={{ color: 'var(--text-primary)' }}>{bundle.profile.topic || meta.filename}</b>
          <span>·</span><span>prior p = {num(bundle.prior.percentile)}</span><Tag t={bundle.prior.evidence_tag} />
          {run && <><span>·</span><span>run seed {run.config.seed}: reach {pct(run.summary.reach_pct, 1)}</span><Tag t="SIMULATED" /></>}
        </div>
      )}
      {tab === 'video' && (
        <VideoScreen videos={videos} videoId={videoId} bundle={bundle} onSelect={selectVideo}
          onRefresh={refreshVideos} onAnalysed={(b) => { setBundle(b); refreshVideos() }} onNext={() => setTab('replay')} />
      )}
      {tab === 'audience' && videoId && <AudienceScreen videoId={videoId} />}
      {tab === 'replay' && bundle && (
        <ReplayScreen run={run} running={running} error={runError} opts={runOpts} onRun={runSim}
          onNext={() => setTab('diagnosis')} />
      )}
      {tab === 'diagnosis' && (run ? <DiagnosisScreen run={run} /> : <div className="empty">{running ? 'Running the simulation…' : runError}</div>)}
      {tab === 'lab' && bundle && videoId && (
        <LabScreen videoId={videoId} bundle={bundle} videos={videos} experiment={experiment} onExperiment={setExperiment} />
      )}
      {tab === 'ledger' && <LedgerScreen videoId={videoId} runId={run?.run_id ?? null} experimentId={experiment?.result.experiment_id ?? null} />}
    </div>
  )
}
