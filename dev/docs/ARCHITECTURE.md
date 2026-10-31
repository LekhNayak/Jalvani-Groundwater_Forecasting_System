# Architecture

```
          ┌──────────────── project (read-only for dev/) ────────────────┐
          │ data/clean, data/raw (CGWB, boundaries)                       │
          │ 04_results/tft_v4_3/dashboard_export   (Colab model run)      │
          │   fallback: 04_results/tft_v4_2/simple (backtests only)       │
          └───────────────────────────┬───────────────────────────────────┘
                                      │ python dev/pipeline/build_dashboard_data.py
                                      ▼
          dev/backend/data/  manifest.json · districts.json · history/<district>.json · geo_districts.json · stations.json
                                      │ loaded once, read-only (services/store.py)
                                      ▼
   FastAPI (dev/backend/app/main.py)                                  Groq API (chat, optional)
     /api/meta /api/districts /api/districts/{d} /api/districts/{d}/history  ◀──▶  services/chat.py
     /api/state/overview /api/state/priorities /api/export/priorities.csv           (offline guide if no key)
     /api/geo/districts /api/geo/stations /api/chat /api/speech/stt|tts  ◀──▶  Gnani (speech, later)
                                      │ JSON
                                      ▼
   React app (dev/frontend): StartPage · FarmerPage · GovPage · GovDistrictPage · HelpPage
     ChatWidget + Tour (data-tour ids) · i18next (en / mr / hi / gu, lazy-loaded) · light / dark theme · Recharts · Leaflet (no tile server)
```

## Why it is built this way
- **Batch model, file-based API.** Forecasts change only when CGWB publishes readings, so there is no need for a live model server. It is cheaper, works offline, and every number can be traced to a file (the manifest holds the source md5s).
- **The rules live in the backend** (`advisories.py`, `priorities.py`) and return translation keys plus values. Every language therefore shows the same logic, and the tests check the rules directly.
- **Two run modes.**
  - Development: Vite on :5173, proxying `/api` to :8000.
  - "Single": FastAPI serves the built app on :8000. This is also the shape for later hosting.
- **The site is split by page.** The farmer page is small. Government pages, the map and the charts load only when opened.
