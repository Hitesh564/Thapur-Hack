# Viralyst frontend

React + Vite + TypeScript, no UI or chart libraries (hand-built SVG charts and a canvas replay).

```
# terminal 1 (backend)
cd backend && uvicorn app.api.main:app --port 8000
# terminal 2 (frontend)
cd frontend && npm install && npm run dev   # http://localhost:5173, /api proxied to :8000
```

Screens: 1 Video & hook · 2 Audience · 3 Replay · 4 Diagnosis · 5 Lab · 6 Evidence ledger.
Demo path (cached data): pick video → Replay (auto-runs) → Diagnosis → Lab → Run experiment → Ledger.

`src/types.ts` mirrors `backend/schemas.py`; change them together.
Every number on screen carries an evidence tag (`<Tag t=... />`).
