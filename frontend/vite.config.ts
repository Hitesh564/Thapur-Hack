import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// /api/* is proxied to the FastAPI backend (uvicorn app.api.main:app --port 8000)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
})
