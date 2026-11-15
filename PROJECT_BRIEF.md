# Field Shift — NASA Space Apps 2026

Decision support for Finney County, Kansas farmers comparing three crop rotations using NASA Earth observations, soil data, crop history, and stated priorities. Shows predicted outcomes, trade-offs, risks, and uncertainty. Supports farmer judgment; it does not replace it.

## Rotations

- **A:** Wheat → Fallow (baseline)
- **B:** Wheat → Sorghum → Fallow
- **C:** Wheat → Field Pea

## Data (`data/raw`)

| Folder | Contents |
|--------|----------|
| `cdl` | USDA CDL GeoTIFFs (Finney, 2013–2024) |
| `appeears` | AppEEARS MOD13Q1 NDVI, SMAP soil moisture CSVs; SRTM GeoTIFF |
| `power` | NASA POWER daily CSV |
| `imerg` | GPM IMERG monthly precipitation (Giovanni) |
| `nass` | NASS QuickStats county yields |
| `soil` | SSURGO point properties |
| `nwis` | USGS groundwater (optional) |

## Model

- Unit: sample point × year
- Features: crop history (2–3 years), season weather, heat/frost days, reference ET, planting soil moisture, soil properties, slope
- Target: peak-season NDVI
- LightGBM, time split (train ≤ 2021, test ≥ 2022), baseline comparison, quantiles (p10/p50/p90), SHAP, NASS validation

## API

FastAPI: `GET /health`, `POST /field-info`, `POST /compare`, `GET /trends`, `GET /model-info`

## Stack

Python, pandas, LightGBM, FastAPI, Streamlit. Numbered scripts in `src/` write to `data/processed`. Secrets in `.env`. Missing values marked `DATA_GAP` — never invent numbers.