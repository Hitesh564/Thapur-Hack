import type {
  AnalysisBundle, Attention, Audience, Cascade, Drivers, ExperimentRecord, Job, Ledger, RunPayload, SeedingPlan, Surgeon, Tipping,
  VideoMeta, WorldLayout,
} from './types'

const BASE = '/api'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init)
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* not JSON */ }
    throw new Error(msg)
  }
  return res.json() as Promise<T>
}

const post = <T>(path: string, body: unknown = {}) =>
  req<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  health: () => req<{ status: string }>('/health'),
  videos: () => req<VideoMeta[]>('/videos'),
  upload: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return req<VideoMeta>('/videos', { method: 'POST', body: fd })
  },
  analyze: (id: string) => post<AnalysisBundle>(`/videos/${id}/analyze`, {}),
  analysis: (id: string) => req<AnalysisBundle>(`/videos/${id}/analysis`),
  attention: (id: string) => post<Attention>(`/videos/${id}/attention`),
  audience: (id?: string) => req<Audience>(`/audience${id ? `?video_id=${id}` : ''}`),
  simulate: (body: { video_id: string; seed?: number; seeding_strategy?: string | null; policy_variant?: string; prior_override?: number | null }) =>
    post<RunPayload>('/simulations', body),
  cascade: (runId: string) => req<Cascade>(`/simulations/${runId}/cascade`),
  layout: () => req<WorldLayout>('/world/layout'),
  experiment: (body: Record<string, unknown>) => post<ExperimentRecord>('/experiments', body),
  tipping: (videoId: string) => post<Tipping>('/tipping', { video_id: videoId }),
  surgeonStart: (id: string) => post<{ job_id: string }>(`/videos/${id}/surgeon`),
  surgeonLatest: (id: string) => req<Surgeon>(`/videos/${id}/surgeon`),
  seedingStart: (id: string) => post<{ job_id: string }>('/seeding-plan', { video_id: id }),
  seedingLatest: (id: string) => req<SeedingPlan>(`/seeding-plan/${id}`),
  driversStart: (id: string) => post<{ job_id: string }>(`/videos/${id}/drivers`),
  driversLatest: (id: string) => req<Drivers>(`/videos/${id}/drivers`),
  job: <T>(jobId: string) => req<Job<T>>(`/jobs/${jobId}`),
  ledger: (q: { video_id?: string; run_id?: string; experiment_id?: string }) => {
    const qs = new URLSearchParams(Object.entries(q).filter(([, v]) => v) as [string, string][]).toString()
    return req<Ledger>(`/ledger${qs ? `?${qs}` : ''}`)
  },
}

export const frameUrl = (videoId: string, path: string) =>
  `${BASE}/videos/${videoId}/frames/${encodeURIComponent(path.split(/[\\/]/).pop() ?? '')}`
export const sourceUrl = (videoId: string) => `${BASE}/videos/${videoId}/source`

/** Poll a background job until it finishes; reports progress text. */
export async function waitForJob<T>(jobId: string, onProgress: (msg: string, elapsed: number) => void): Promise<T> {
  for (;;) {
    const j = await api.job<T>(jobId)
    onProgress(j.progress, j.elapsed_s)
    if (j.state === 'done') return j.result as T
    if (j.state === 'error') throw new Error(j.error ?? 'job failed')
    await new Promise((r) => setTimeout(r, 1500))
  }
}
