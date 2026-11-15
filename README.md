# Field Shift

Western Kansas dryland wheat fields often include long fallow periods. Soil erodes, groundwater drops, and a single dry year hits income hard. **Field Shift** is a decision-support web app for **Finney County, Kansas** that compares three real crop rotations (from USDA Cropland Data Layer history on 1,000 sample fields) using **13 years of NASA weather and satellite data**, farmer priorities, and soil properties. It shows expected wheat greenness (MODIS NDVI) with uncertainty bands, water and erosion trade-offs, bare months, evidence, risks, and data gaps—without replacing local expert judgment.

## Datasets

| Dataset | Role | Link |
|---------|------|------|
| NASA POWER | Daily temperature, rain, humidity, radiation, ET0 | [power.larc.nasa.gov](https://power.larc.nasa.gov/) |
| NASA MODIS MOD13Q1 | 250 m, 16-day NDVI (peak season target) | [MOD13Q1](https://lpdaac.usgs.gov/products/mod13q1v061/) |
| NASA SMAP SPL3SMP_E | Soil moisture at planting | [SMAP](https://smap.jpl.nasa.gov/) |
| NASA SRTM | Elevation / slope context | [SRTM](https://www2.jpl.nasa.gov/srtm/) |
| USDA Cropland Data Layer | Crop history and rotation labels | [CropScape](https://nassgeodata.gmu.edu/CropScape/) |
| USDA SSURGO (SDA) | Texture, organic matter, K factor, AWC | [Soil Data Access](https://sdmdataaccess.sc.egov.usda.gov/) |
| USDA NASS QuickStats | County wheat yield validation | [QuickStats](https://quickstats.nass.usda.gov/) |
| Crop table | Kc, prices, forage flags (FAO-56, USDA AMS, K-State) | `data/crop_table.csv` and `crop_table_notes.pdf` |

## Model results (Finney County, shipped model)

- **Leave-one-year-out MAE:** 0.118 NDVI (MC p50) vs **0.129** for a fair no-weather baseline.
- **Interval coverage:** ~**80%** of observations inside conformal-adjusted p10–p90.
- **NASS sanity check:** county peak NDVI vs winter wheat yield **r = 0.86** (**n = 6** years)—not proof of farm-level accuracy.

Details: `reports/model_card.md`, `models/final_model.json`, `reports/nass_validation.png`.

## Run locally

**Requirements:** Python 3.10+ (see `backend/requirements.txt`; GDAL may be needed for raster steps when rebuilding raw data).

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt

cd backend
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open **[http://localhost:8000/](http://localhost:8000/)** (one process serves the API and the web UI from `frontend/`).

Optional static UI on port 5500: `python -m http.server 5500` from `frontend/` — `config.js` points API calls to `http://localhost:8000`.

**Tests:** run from `backend/`.

```bash
cd backend && python -m pytest tests/
```

**Note:** `earthengine-api` is only needed to re-download data from Google Earth Engine (`src/03b_gee_extract.py`). On machines without a prebuilt `cryptography` wheel it needs a Rust toolchain; skip it if you are not re-downloading.

## Deploy

The repo splits into a static `frontend/` and a Dockerised `backend/`, so they deploy separately.

**Backend — Render (Docker):**

1. In Render: **New → Blueprint**, connect this repo.
2. Render reads `render.yaml` (builds `backend/Dockerfile` with the repo root as context, health check `/health`, free plan).
3. Set the `ALLOWED_ORIGINS` env var to your Vercel URL (comma separated for several). Any `*.vercel.app` origin is already allowed by default.
4. Artifacts (`backend/models/final_model.json`, `backend/data/processed/*.parquet`) are committed—`backend/data/raw/` is not needed at runtime.

The container serves the API *and* the UI, so the Render URL alone is a complete working app.

**Frontend — Vercel (static):**

1. In Vercel: **Add New → Project**, import this repo.
2. `vercel.json` publishes `frontend/` as-is (no build step, no framework).
3. After the backend is live, set `window.API_URL` in `frontend/config.js` to the Render URL (e.g. `https://field-shift.onrender.com`, no trailing slash) and redeploy.

## Rebuild `data/raw/` and pipeline

`data/raw/` is excluded from the submission zip (large). To reproduce from scratch:

1. Copy `.env.example` → `.env` and set `NASS_API_KEY` / `GEE_PROJECT_ID` only if you re-download.
2. Place downloads per `CURSOR_CONTEXT.md` §4–8 (`cdl/`, `power/`, `appeears/` or GEE exports, `soil/`, `nass/`, etc.).
3. Run `src/` scripts in pipeline order (see `CURSOR_CONTEXT.md` §8): crop history → weather → features → `06_train_final.py` → `07_validate_nass.py` → `08_build_feature_store.py` → `09_field_calendar.py`.
4. Confirm `GET /health` returns `artifacts_ready: true`.

## Limitations

- Trained on **dryland** winter wheat fields; irrigated pivots are flagged but not part of training.
- **Rotation C** (forage) has very few pure CDL sequences—low confidence.
- **NDVI** is relative crop health, not bu/acre revenue.
- Nearest sample point within **3 km** proxies soil and climate for clicked locations.
- SMAP ~9 km; MODIS 250 m; NASS county yields only.

## AI use statement

Code was written with help from **Cursor** and **Claude**; analysis, validation, and product decisions were reviewed by the team. NASA data processing and final metrics were verified against held-out years and documented in `CURSOR_CONTEXT.md`.

## Push to GitHub

```bash
git init
git add .
git commit -m "Field Shift final submission"
git branch -M main
git remote add origin https://github.com/<username>/field-shift.git
git push -u origin main
```

Replace `<username>` with your GitHub username or organization.

## Project layout

```text
frontend/            Plain HTML/CSS/JS UI (static; deploys to Vercel)
  index.html
  app.js, style.css, config.js

backend/             Python API, model and pipeline (Docker; deploys to Render)
  api/               FastAPI (serves ../frontend at /)
  src/               Numbered pipeline and compare logic
  data/crop_table.csv
  data/processed/    feature_store, calendar, sample_points (runtime)
  data/raw/          Source downloads (git-ignored, not needed at runtime)
  models/            final_model.json
  reports/           Figures + model_card.md (+ archive/ for older metrics)
  scripts/           Release zip and debug helpers
  tests/             pytest
  requirements.txt
  Dockerfile

render.yaml          Render blueprint (backend)
vercel.json          Vercel config (frontend)
```

## License and disclaimer

This tool **supports, and does not replace**, the judgment of farmers and local agricultural experts.