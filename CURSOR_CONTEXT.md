# CURSOR_CONTEXT.md: Field Shift (NASA Space Apps 2026)

This file is the single source of truth for this project. Any AI agent (Cursor or other) must read it fully before writing code. It contains the challenge, the team lead's framework, every final decision, current status, the exact next steps, and the contracts other teammates depend on. Keep PROJECT_BRIEF.md too, but if they disagree, this file wins.

Owner of this file: Shaz (team data scientist). Update the "Status log" section after every completed step.

---

## 1. How the agent should work

- Work one step at a time from section 9. Do not jump ahead.
- After each step: run the script, print a short summary, and stop so the human can check output.
- Small readable scripts in `src/`, numbered in run order. Each script reads from `data/raw` or `data/processed` and writes to `data/processed`, `models` or `reports`, so any step can be rerun alone.
- Never invent numbers (yields, prices, Kc values, accuracy). If a value is missing, mark it as a data gap.
- Secrets live in `.env`, never in code.
- Distances and areas in metres or acres from real data. Never fake a calibration.
- Prefer simple and explainable over fancy. Time is very limited (see section 12).

---

## 2. The challenge (NASA Space Apps 2026, "Field Shift: Adapting Farms with NASA Data")

Build a decision support tool that combines NASA Earth observations with local soil information, crop characteristics and farmer priorities so farmers can explore crop rotation strategies that improve soil health and resilience to changing weather and water conditions.

Must do:
- Compare at least three rotation strategies, including the existing rotation.
- For each strategy explain: why it was suggested, what evidence supports it, benefits, information gaps, and risks needing local review by farmers or agricultural experts.
- Highlight times when fields are bare and seasons when current crops are more vulnerable.
- Make clear the tool supports and never replaces farmer judgment and local expertise.

Judging criteria (equal weight): Influence (impact), Creativity, Validity, Relevance, Presentation. Target award: Best Use of Data (also fits Best Use of Science).

---

## 3. Team lead's framework (verbatim intent)

Four inputs to leverage:
1. NASA Earth observations
2. Soil information
3. Crop characteristics
4. Farmer priorities

Problems farmers face: temperature shifts, rainfall changes, soil health decline. Other problems: water shortages, erosion, nutrient depletion, extreme weather events.

Farmer priorities: water use, forage value, nitrogen fixing potential, erosion reduction, organic matter improvement, market suitability.

Our decision support tool must: compare 3 rotations, help understand risks, help understand trade-offs, make uncertainty visible.

Output goal: help farmers explore rotation strategies that strengthen soil health and adapt their farms to changing conditions.

### How we cover each point

| Team lead point | Our implementation |
|---|---|
| NASA Earth observations | NASA POWER, GPM IMERG, SMAP, MODIS NDVI, SRTM |
| Soil information | USDA SSURGO via Soil Data Access API, editable by farmer |
| Crop characteristics | USDA CDL rotation history + researcher's sourced crop table |
| Farmer priorities | Top 3 priority picks in the app, turned into weights |
| Temperature shift | POWER trend, heat days, "hot year" scenario (+2 C) |
| Rainfall | IMERG and POWER rainfall, "dry year" scenario (-20% rain) |
| Soil health decline | Bare days per year + organic matter |
| Water shortage | Water balance (Kc x ET0 minus effective rain) + USGS groundwater trend (optional) |
| Erosion | Bare days + SSURGO K factor + SSURGO slope |
| Nutrient depletion | Nitrogen fixing flag from crop table (estimate, confirm with soil test) |
| Extreme weather | Heavy rain days, heat days, frost days |
| Compare 3 rotations | `/compare` returns A, B, C |
| Understand risks | risks + data_gaps per rotation |
| Understand trade-offs | ranking changes when priorities change |
| Uncertainty visible | p10 / p50 / p90 range + confidence badge |

---

## 4. Final decisions (locked)

- Region: Finney County, Kansas, USA. FIPS 20055. Centre point lat 37.98, lon -100.86. Main town Garden City, KS 67846.
- Focus: dryland (rain fed) winter wheat systems. Irrigated pivots (corn, alfalfa, soybeans) are excluded from training.
- Rotations:
  - A: Wheat then Fallow (existing baseline)
  - B: Wheat then Sorghum then Fallow
  - C: Wheat then Forage (triticale, other hay; peas included if present). Peas were changed out because CDL 2024 shows almost zero peas in Finney County. Treat C as low confidence unless sample counts grow.
- Model target: peak season NDVI of winter wheat (crop health proxy).
- Model: LightGBM, time based split, quantile models, SHAP.
- Deadline: target submit Sunday 15 Nov 2026, 10 pm local time. Official close is assumed 11:59 pm local event time; confirm with Local Lead.

### Datasets used

| Dataset | Use | How obtained | Folder |
|---|---|---|---|
| USDA CDL 2013 to 2025 | Crop history, rotation labels, sample points | CropScape GetCDLFile service, FIPS 20055 | data/raw/cdl |
| MODIS MOD13Q1 NDVI (250 m, 16 day) | Target (peak NDVI), bare days | NASA AppEEARS point request | data/raw/appeears |
| SMAP SPL3SMP_E soil moisture (9 km, 2015 onward) | Soil moisture at planting | NASA AppEEARS point request | data/raw/appeears |
| SRTM SRTMGL1 (30 m) | Elevation per point | NASA AppEEARS point request | data/raw/appeears |
| NASA POWER daily | Temperature, rain, humidity, wind, radiation, ET0, heat and frost days | POWER API URL | data/raw/power |
| GPM IMERG monthly | Rainfall trend and variability | NASA Giovanni area averaged CSV | data/raw/imerg |
| USDA NASS QuickStats | County wheat and sorghum yields for validation | QuickStats website | data/raw/nass |
| USDA SSURGO | Organic matter, sand, silt, clay, AWC, K factor | Soil Data Access API script | data/raw/soil |
| USGS NWIS (optional) | Groundwater trend chart | waterdata.usgs.gov | data/raw/nwis |
| USDA AMS prices (researcher) | Market column in crop table | researcher | data/crop_table.csv |

Skipped on purpose: ECOSTRESS, PRISM, STATSGO2, NAIP, NLCD, NLDAS/GLDAS, GAEZ, CoCoRaHS, OpenET, LANDFIRE, GEDI, Sentinel-2, ERA5, Ag Data Commons.

Presentation note: highlight NASA datasets first; USDA data plays a supporting role.

---

## 5. Architecture

```
Offline (once)
  data/raw/*  ->  src/01..07 build features  ->  src/08 train  ->  models/*.pkl + data/processed/feature_store.parquet

Online (each farmer request)
  Streamlit app  ->  FastAPI (api/main.py)  ->  models + feature store in memory  ->  JSON  ->  app
```

Rule: never call any external data service inside an API request. Everything the API needs is precomputed.

Feature store simplification: the API uses the nearest of the 1000 sample points (within 3 km) as the source of soil, slope and climate features for a clicked location. No separate grid is needed.

---

## 6. Repo structure

```
field-shift/
  CURSOR_CONTEXT.md        (this file)
  PROJECT_BRIEF.md
  requirements.txt  .env  .env.example  .gitignore  README.md
  data/raw/{cdl,appeears,power,imerg,nass,soil,nwis}
  data/processed/
  data/crop_table.csv       (from researcher)
  src/                      (numbered pipeline scripts)
  api/main.py               (FastAPI)
  api/mock_compare.json     (for the app developer)
  app/streamlit_app.py      (developer)
  models/  reports/  tests/
```

---

## 7. Status log (update after each step)

- [x] Project setup, venv, requirements
- [x] `src/00_download_cdl.py`: CDL 2013 to 2025 for FIPS 20055 downloaded to data/raw/cdl (about 5 MB each). Note: 2020 and 2022 rasters are aligned to the 2024 grid with nearest neighbour.
- [x] CDL 2024 inspected: grassland 275k acres, winter wheat 154k, corn 115k, sorghum 98k, fallow 95k, alfalfa 41k, triticale 16k, double crop wheat/sorghum 3k, soybeans 2k, peas about 0.
- [x] `src/01_crop_history.py` first run: filters gave 2,073,160 then 708,480 then 696,721 pixels; 20,000 random subset and 300 m spacing gave 5,219 candidates. Final points: wheat_fallow 66, wheat_sorghum_fallow 350, wheat_forage 32, other 552. 4,350 wheat-year rows. Problem: point level A rule too strict.
- [x] `src/01_crop_history.py` update: all 696,721 filtered pixels through 300 m spacing gave 10,981 candidates. Final points: wheat_fallow 306, wheat_sorghum_fallow 350, wheat_forage 156, other 188 (total 1000). Wheat-year rotation_context counts (all years / 2022 onward): A 1,542 / 523, B 982 / 383, C 70 / 25, other 1,780 / 276. Decision: C stays in the app as Low confidence and is listed as a data gap (fewer than 100 wheat-year rows). "other" rows stay in training as their own category.
- [x] NASA POWER daily downloaded and verified: data/raw/power/power_finney_daily.csv, 1 Jan 2000 to 31 Oct 2025, 9,436 days, no missing days, no -999 values. Header is 15 lines (use skiprows=15). Columns are YEAR, DOY (day of year, not month/day), T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M, WS2M, ALLSKY_SFC_SW_DWN. Rain and temperature come from MERRA-2, radiation from CERES. Annual rain is mostly 370 to 760 mm; 2022 is a severe drought year (282 mm), 2011 and 2012 also dry. 2025 is partial (to 31 Oct).
- [x] AppEEARS point request "finney_points" submitted (Sat 14 Nov, 3:29 pm PKT): 1000 points, 07-01-2012 to 12-31-2025, MOD13Q1.061 layers _250m_16_days_NDVI and _250m_16_days_pixel_reliability. Earthdata user: tayyab60. Waiting for email.
- [x] AppEEARS SMAP point request "finney_smap" submitted (ID cff4ea84-2e8e-4c78-8b1e-b63dda08c8c8): layer Soil_Moisture_Retrieval_Data_AM_soil_moisture (SPL3SMP_E.006), 09-15-2017 to 10-31-2024 (continuous daily). Shorter range because of the AppEEARS value limit. So SMAP planting-time moisture exists for harvest years 2018 to 2025 only; earlier years are NaN. Check in Explore whether it used all 1000 points or only part1 (500); if 500, submit part2 too. Split files exist: data/processed/sample_points_part1.csv and _part2.csv.
- [x] AppEEARS SRTM point request "finney_srtm" submitted (ID cae67938-53b4-4dbd-bfda-4ed97e2d958f): SRTMGL1_DEM (SRTMGL1_NC.003) at the 1000 points, gives elevation per point. Decision: slope comes from SSURGO component slope_r (representative slope %), not from SRTM. No area request needed.
- [x] Soil: src/02_soil.py done (SDA mukey by point, dominant component, 0 to 30 cm weighted om, sand, silt, clay, awc, kwfact, plus slope_r and hydgrp, cached per point). data/raw/soil/soil_points.csv has 1000 rows, 976 with data; 24 points have no SSURGO horizons (IDs 10, 40, 53, 104, 107, 189, 193, 216, 230, 247, 255, 412, 566, 631, 735, 842, 917, 920, 922, 942, 961, 982, 985, 992). Keep them with NaN soil features.
- [x] Weather: src/04_weather.py done, data/processed/weather_by_year.csv and reports/weather_by_year.png for harvest years 2013 to 2025 (fallow_rain_mm, season_rain_mm, heavy_rain_days, heat_days, frost_days, et0_mm via pyet pm_fao56, water_balance_mm, mean_tmax_c). 2022 drought check: season rain 166 mm, 32 heat days, water balance -923 mm.
- [x] NASS county wheat yields saved. Available years only 2013, 2014, 2015, 2020, 2021, 2022 (bu/acre: 23.9, 25.5, 41, 40.7, 66.5, 30.6). 2022 drought halves yield vs 2021.
- [x] NASS winter wheat yield (bu/acre), Finney County, county level: only available 2013 to 2022. Save as data/raw/nass/nass_finney_wheat_yield.csv. Optional backup: Southwest Kansas agricultural district yields (data/raw/nass/nass_sw_district_wheat_yield.csv) for later years. Validation in step 7 uses 2013 to 2022 county values (includes the 2022 drought year).
- [ ] IMERG (optional)
- [x] AppEEARS status at 4:35 pm Sat: finney_points 25% processing, finney_smap and finney_srtm pending.
- [x] Scripts written and self-tested, not yet run on real data: src/03_parse_appeears.py, src/05_build_table.py, src/06_train.py (each has --self-test). When AppEEARS lands: download CSVs to data/raw/appeears, then run 03, 05, 06 in order.
- [x] Backup for satellite data (AppEEARS stuck at 25% for over an hour): Google Earth Engine. Shaz's old Google Cloud project (n8n-connection) was suspended for leaked credentials, unrelated to this project; registering EE with a new Gmail. Teammate Abrar also asked to run gee_finney_export.js in the EE Code Editor (uploads sample_points as an asset, exports finney_ndvi_gee.csv with ID, date, NDVI, SummaryQA; finney_smap_gee.csv with ID, date, first; finney_srtm_gee.csv with ID, first to Google Drive folder field_shift). Put those CSVs in data/raw/gee and update 03_parse_appeears.py to read them (NDVI x 0.0001, SummaryQA 0 or 1). Use whichever source arrives first.
- [x] Sat 10 pm: AppEEARS finney_points only 29% after 6.5 hours, smap and srtm still pending, so AppEEARS will not finish before the deadline. Abrar has been sent full step by step instructions in the team group to run gee_finney_export.js; waiting for his 3 CSVs. Plan C if no satellite data by about midnight: rule based scoring with weather, soil and rotation data, and present the satellite pipeline as ready.
- [x] GEE data received from Abrar (Sat 11:54 pm), verified: finney_ndvi_gee.csv 310,000 rows (1000 points x 310 MODIS composites, 2012-07-11 to 2025-12-19), no nulls, raw NDVI needs x 0.0001, SummaryQA 0: 291,591, 1: 8,468, 2: 2,644, 3: 7,297. finney_smap_gee.csv 589,000 rows (1000 points x 589 days, Sep and Oct of 2015 to 2024), column 'first' is soil moisture, about 51% NaN which is normal because SMAP passes every 2 to 3 days; every point-year has 19 to 25 valid values in the 15 Sep to 31 Oct window; range 0.035 to 0.365. So SMAP planting moisture now exists for harvest years 2016 to 2025. finney_srtm_gee.csv 1000 rows, elevation 754 to 924 m, no nulls. Files go in data/raw/gee/.
- [x] Features: model_table_v2, feature_store.parquet, ndvi_points.parquet, weather_by_year.csv (GEE NDVI + POWER + SSURGO + GEE SMAP/SRTM).
- [x] Final model trained and validated: models/final_model.json, 07_validate_nass.py, reports/nass_validation.png, LOO MAE 0.118.
- [x] Compare + absolute scores + field calendar: src/compare.py, src/09_field_calendar.py, GET /calendar, bare_months on cards.
- [x] API: api/main.py (health, field-info, compare, trends, model-info, calendar). Dockerfile. pytest 12 passed.
- [x] Web: app/web (map, compare, rotation cards, calendar strips). Live API on :8000 or static + field_calendar.json fallback.
- [x] data/crop_table.csv from researcher notes (crop_table_notes.pdf); blanks documented; proxies flagged in notes column.
- [ ] Deploy API (Render) + static app hosting + public demo link (human).
- [ ] Submission form + 30 s video / 7 slides (human, section 13).

---


### Model results and final decision (Sun 15 Nov, 00:20 PKT)
- v1 LightGBM (all features, fixed split 2022+): MAE 0.132 vs baseline 0.139, R2 0.05, p10 to p90 coverage 0.53. Weak because weather is one county point per year (only 9 training years) and MODIS 250 m pixels mix wheat and fallow strips.
- v2 (src/06_train_v2.py): purity filter (share of wheat in 9x9 CDL window >= 0.7) keeps 1,490 rows (A 526, B 536, C 7, other 421). Two part model. Year layer linear regression of county mean peak NDVI on season rain and heat days: intercept 0.648, rain +0.059 per SD, heat -0.003, leave-one-year-out MAE 0.072 (works, explains drought years). Field layer LightGBM on anomaly: LOYO MAE 0.096 vs rotation-mean baseline 0.092, so it does NOT beat the simple baseline. Conformal widening delta 0.0905 brings coverage 0.46 to 0.80. Observed A minus B anomaly -0.036, predicted -0.036. Top SHAP field features: sm_planting, elevation_m, prev_year_bare_days, prev2_crop.
- DECISION: ship the simpler model that won cross-validation. Final model = year layer (linear, weather scenarios) + empirical anomaly distribution per rotation_context (median, p10, p90, n), combined by Monte Carlo for the range. LightGBM is reported as explored but not better. C uses purity >= 0.5 rows (23) and is labelled Low confidence or insufficient data. Saved as a human readable models/final_model.json so the API needs no LightGBM.
- FINAL MODEL RESULT (00:27 PKT, after fixing two LOO evaluation bugs: clipping during LOO and wrong residual pool): leave-one-year-out MAE 0.118 (Monte Carlo p50) and 0.119 deterministic, vs fair baseline 0.126 (training mean NDVI plus rotation median anomaly, no weather) and rotation-mean only 0.128. So the final model beats the fair baseline. p10 to p90 coverage 0.80 after conformal delta 0.0085. Year layer LOO MAE 0.0715. Per-year MAE examples: 2016 0.091, 2022 0.124, 2024 0.187. The earlier 0.092 baseline used the true held-out county mean and is not a fair comparison. Rotation medians (anomaly vs county mean): A -0.019 (n 526, high), B +0.020 (n 536, high), C +0.076 (n 23, low), other +0.007 (n 287, medium). compare.py uses models/final_model.json live. pytest 6 passed.
- To check Sunday morning: the LOO year prediction for 2024 was -0.04 before clipping, which is implausible for a linear model with intercept 0.648; inspect 2024 season_rain_mm and heat_days and the coefficients when 2024 is held out. Also print full-fit predicted county mean NDVI for every year and confirm all are between 0.4 and 0.8.
- 00:38 PKT: Fixed a scaling bug in _predict_year_mu (heat_days was standardized with season_rain stats); this affected manual and API predictions, sklearn pipeline was fine. Year coefficients now fit on purity >= 0.7, 2013 to 2025; NDVI clip [0.4, 0.8]. Full-fit county mean observed vs predicted all within range (2013 0.538/0.548, 2022 0.492/0.532, 2024 0.746/0.591). 2024 is under-predicted by the two feature linear model (recovery year).
- NASS validation (src/07_validate_nass.py, reports/nass_validation.png): observed county mean peak NDVI vs NASS winter wheat yield r = 0.859 (n = 6 years: 2013, 2014, 2015, 2020, 2021, 2022); year-model predicted county mean vs yield r = 0.547. Sanity check only because n = 6. Stored in final_model.json under nass_validation.
- FINAL NUMBERS after scaling fix (00:40 PKT): LOO MAE 0.1182 deterministic (MC p50 0.1177) vs fair baseline 0.1290; p10 to p90 coverage 0.772 before and 0.800 after conformal delta 0.0100; year layer LOO county mean MAE 0.0644; beats fair baseline. Use these numbers in the presentation and submission.
- 01:19 PKT compare fixes: empirical Bayes shrinkage k = 100 (shrunk normal effects A -0.0157, B +0.0166, C +0.0143; dry-year effects A -0.0204 n 183, B +0.0124 n 202, C +0.0045 n 10). Dry and hot scenarios use dry effects for A and B; C falls back with a data gap. Low confidence rotations cannot be best_match. Bare days per year by rotation cycle: A 160.6, B 126.5, C 32.8. Scores are on fixed absolute scales (soil = 10 x (1 - bare/365) + 1 if N fixing; erosion = 10 x (1 - risk), risk = 0.5 bare/365 + 0.3 kwfact/0.6 + 0.2 slope/10; health = (p50 - 0.4)/0.4 x 10; water from Kc only; income, forage, market only from crop_table.csv), null scores dropped with weight renormalization. Confidence is n based (A, B High; C Low), irrigated one step lower. Point 0: normal best_match B, totals A 5.5, B 6.3, C 7.9 (excluded, low); dry A 5.2, B 6.0. Finding: in dry years wheat after fallow only falls further below the county mean while wheat after sorghum and fallow stays above it.

## 8. Model specification

### Unit and target
- One row = one sample point x one wheat year (CDL 24 or 236 in that year).
- Target `peak_ndvi`: maximum of MOD13Q1 NDVI between 1 March and 30 June of the harvest year. Apply scale factor 0.0001. Use only observations with pixel reliability 0 or 1. If fewer than 3 good observations, drop the row.

### Features (no leakage: nothing from the target season's NDVI)
- Rotation: `rotation_context` (A, B, C, other) as categorical, `prev1_crop`, `prev2_crop`.
- Fallow season rain: precipitation from 1 July two years before through 30 September before planting (captures stored moisture, the whole point of fallow). Use POWER PRECTOTCORR.
- Growing season rain: 1 October (planting year) to 30 June.
- Heavy rain days: days with more than 25 mm in the growing season.
- Heat days: days with T2M_MAX above 32 C from 1 April to 30 June.
- Frost days after green-up: days with T2M_MIN below -2 C from 15 March to 30 April.
- ET0 sum for the growing season: FAO-56 Penman-Monteith via `pyet` from POWER T2M, T2M_MAX, T2M_MIN, RH2M, WS2M, ALLSKY_SFC_SW_DWN.
- Water balance: growing season rain minus ET0.
- Soil moisture at planting: mean SMAP soil moisture 15 September to 31 October before harvest year (requested range 09-15-2017 to 10-31-2024, so available for harvest years 2018 to 2025; NaN otherwise, LightGBM handles NaN). SMAP revisits each place every 2 to 3 days, so many days are empty per point: average whatever valid values exist in the window, drop fill values (-9999) and values outside 0 to 0.6 m3/m3, and record n_obs. If n_obs is below 3, set NaN.
- Soil: organic matter, sand, silt, clay, AWC, K factor.
- Slope (percent) from SSURGO slope_r; elevation from SRTM.
- Previous year bare days: days with NDVI below 0.2 in the year before (from MOD13Q1 interpolated to daily).
- Do not use year as a feature.

### Training and checks
- Split: train harvest years up to 2021, test 2022 onward. No random split.
- Baseline: mean peak NDVI per rotation_context from the training years. The model must beat the baseline MAE on test, otherwise report that honestly.
- If test R2 is above 0.95, suspect leakage and investigate.
- Quantile models: LightGBM objective quantile at alpha 0.1, 0.5, 0.9. Save as models/model_p10.pkl, model_p50.pkl, model_p90.pkl. Fix crossing quantiles by sorting.
- SHAP: TreeExplainer on the p50 model. Save summary plot to reports/shap_summary.png and keep the explainer for the API.
- Validation against NASS: average predicted p50 by harvest year across points, correlate with NASS county winter wheat yield (bu/acre). Report Pearson r and a scatter plot in reports/nass_validation.png. Small n, so present it as a sanity check, not proof.
- Save metrics to reports/model_metrics.json (MAE baseline, MAE model, R2, coverage of the p10 to p90 interval on test, NASS r, sample counts per rotation_context).

### Confidence rule
- High: rotation_context has at least 300 training rows and p90 minus p10 below 0.12.
- Medium: at least 100 rows or interval below 0.18.
- Low: otherwise. C is expected to be Low.

---

## 9. Next steps with exact prompts (run in order)

### Step 1: Fix crop history (pending)
```
Update src/01_crop_history.py:
1. Draw candidates from all pixels that pass the filters (about 697k), not a 20000 random subset, then apply the 300 m spacing.
2. In wheat_years.csv add a column rotation_context for each wheat year:
   A if prev1 is fallow (61) and prev2 is wheat (24 or 236)
   B if prev1 is fallow (61) and prev2 is sorghum (4)
   C if prev1 is triticale (205), other hay (37) or peas (53)
   other otherwise
3. Relax the point-level system label: A wheat_fallow if at least 60% of the pixel's wheat years have rotation_context A and it never grew sorghum; keep B and C rules as they are.
4. Re-run the stratified selection: up to 350 points each for A, B and C, the rest from other, total at most 1000.
5. Print: final points per system, and wheat-year counts per rotation_context (overall and for test years 2022 onward).
```
Check: if C has fewer than 100 wheat-year rows, keep C as low confidence and record it as a data gap.

### Step 2: AppEEARS (manual, human does this)
1. Log in at appeears.earthdatacloud.nasa.gov with an Earthdata account.
2. Point Sample: upload data/processed/sample_points.csv. Dates 2012-07-01 to 2025-12-31. Products: MOD13Q1 (latest version) layers 250m_16_days_NDVI and 250m_16_days_pixel_reliability; SPL3SMP_E (latest version) soil moisture AM layer. Output CSV.
3. Area Sample: draw a box around Finney County. Product SRTMGL1 (latest version). Output GeoTIFF.
4. Download results into data/raw/appeears/. If the point limit errors, split into two requests of 500.

### Step 3: Weather, rainfall, yields (manual, human does this)
- POWER: open in browser and save as data/raw/power/power_finney_daily.csv
  `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,WS2M,ALLSKY_SFC_SW_DWN&community=AG&longitude=-100.86&latitude=37.98&start=20000101&end=20251031&format=CSV`
- IMERG: giovanni.gsfc.nasa.gov, Time Series Area-Averaged, IMERG monthly final precipitation, box around Finney County, 2001 to 2025, download CSV to data/raw/imerg/imerg_finney_monthly.csv
- NASS: quickstats.nass.usda.gov, Survey, Crops, Field Crops, Wheat then Sorghum, Yield, County, Kansas, Finney, 2013 to 2025. Save to data/raw/nass/.

### Step 4: Soil
```
Read CURSOR_CONTEXT.md. Write src/02_soil.py: for every point in data/processed/sample_points.csv query the USDA Soil Data Access API (SSURGO) for organic matter, sand, silt, clay, available water capacity and K factor (kwfact) of the dominant component, weighted over the top 30 cm, plus the component representative slope (slope_r, percent) and hydrologic soil group. Cache responses in data/raw/soil/cache, save data/raw/soil/soil_points.csv, and report points with no data.
```

### Step 5: Build features
```
Read CURSOR_CONTEXT.md section 8. Write src/03_parse_appeears.py to tidy the AppEEARS CSVs into data/processed/ndvi_points.parquet (point_id, date, ndvi, reliability) and smap_points.parquet (point_id, date, sm), and extract elevation per point from the SRTM point CSV into data/processed/elev_points.csv (slope comes from SSURGO slope_r). Then write src/04_weather.py to compute per harvest year the weather features from POWER (fallow season rain, growing season rain, heavy rain days, heat days, frost days, ET0 with pyet, water balance) into data/processed/weather_by_year.csv. Then write src/05_build_table.py to join wheat_years, NDVI target, previous year bare days, SMAP, soil (including slope_r), elevation and weather into data/processed/model_table.parquet. Print row counts, missing value share per column, and rows per rotation_context and year.
```

### Step 6: Train
```
Read CURSOR_CONTEXT.md section 8. Write src/06_train.py exactly as specified: time split, baseline, LightGBM p10/p50/p90, SHAP, metrics json, and plots. Print a short metrics table.
```
Check: model MAE below baseline MAE; R2 not suspiciously high; SHAP top features make agronomic sense.

### Step 7: Validate with NASS
```
Write src/07_validate_nass.py as specified in section 8 and update reports/model_metrics.json.
```

### Step 8: Compare logic
```
Read CURSOR_CONTEXT.md sections 8, 10 and 11. Write src/compare.py with a function compare(request_dict) that returns the /compare response. For the nearest sample point, build one feature row per rotation (set rotation_context and prev crops to A, B, C), apply the scenario to weather features, predict p10/p50/p90, compute water need, bare days, erosion risk, priority scores and total score, SHAP based plain language reasons, risks, data gaps and confidence. Also write data/processed/feature_store.parquet with the latest features per point and api/mock_compare.json with a realistic example response. Add tests in tests/test_compare.py.
```

### Step 9: API
```
Read CURSOR_CONTEXT.md section 11. Build api/main.py with FastAPI and Pydantic models exactly matching the contracts. Load models, SHAP explainer, feature store and crop table once at startup. Endpoints GET /health, POST /field-info, POST /compare, GET /trends, GET /model-info. Enable CORS. Return 404 with a clear message when the location is more than 3 km from any sample point. Add tests/test_api.py and a Dockerfile. Run locally with uvicorn and open /docs.
```

### Step 10: Connect and deploy
- Developer switches the app from api/mock_compare.json to the live API.
- Deploy API on Render (free) and the app on Streamlit Community Cloud. Public GitHub repo.
- Backup: if the API is not deployed by Sunday 2 pm, the app imports src/compare.py directly.

---

## 10. Scoring, crop table and farmer input mapping

### Crop table (researcher fills, every number needs a source URL)
File `data/crop_table.csv`, one row per crop (winter_wheat, sorghum, triticale, other_hay, field_pea, fallow):
`crop, kc_ini, kc_mid, kc_end, season_start, season_end, nitrogen_fixing (yes/no), forage_value (low/medium/high), residue_level (low/medium/high), price_usd_per_unit, price_unit, price_source, notes`

### Per rotation metrics
- Water need (mm per year): sum over the crops in the rotation cycle of Kc x ET0 during each crop's season, minus effective rain, divided by cycle length in years.
- Bare days per year: from historical NDVI of points in that rotation_context (median), adjusted by cycle.
- Erosion risk: score = bare_days_norm x 0.5 + K_factor_norm x 0.3 + slope_norm x 0.2. Low below 0.33, medium below 0.66, high otherwise.
- Nitrogen: yes if the cycle includes a nitrogen fixing crop.

### Priority scores (each 0 to 10, normalised across the 3 rotations)
- water: lower water need is better
- soil: fewer bare days, higher residue, nitrogen fixing
- income: from crop table prices and predicted crop health (state clearly it is relative, not a profit forecast)
- erosion: lower erosion score is better
- forage: crop table forage value
- market: crop table price and local market notes

### Total score
- Weights from farmer's top 3 picks: first 3, second 2, third 1, all others 0.5. If none picked: water, soil, income equal.
- priority_score = weighted mean of the six scores.
- health_score = (p50 - 0.4) / 0.5 x 10, clipped 0 to 10.
- total = 0.7 x priority_score + 0.3 x health_score, rounded to 1 decimal.

### Farmer inputs to request fields
| App input | Field | Values | Default |
|---|---|---|---|
| Field location (map click) | lat, lon | numbers | required |
| What do you grow now | current_rotation | A, B, C, other | from CDL |
| Water source | water_source | dryland, irrigated | dryland |
| Soil type (prefilled, editable) | soil_texture | light, medium, heavy | from SSURGO |
| Top 3 priorities | priorities | list from water, income, soil, erosion, forage, market | water, soil, income |
| Year type | year_type | normal, dry, hot | normal |
| Soil test organic matter (advanced) | soil_om_pct | number or null | SSURGO |
| Machinery (advanced) | machinery | sorghum_planter, forage_equipment booleans | both true |

Scenario mapping: dry = growing season rain x 0.8; hot = rain x 0.8 and temperatures +2 C (recompute heat days and ET0). Irrigated: flag that the model is trained on dryland fields, set confidence one level lower, and add a data gap note.
Machinery: missing sorghum planter adds a risk to B and lowers its priority score by 1; missing forage equipment does the same for C.

---

## 11. API contracts

### GET /health
`{"status": "ok", "model_version": "v1"}`

### POST /field-info
Request: `{"lat": 37.98, "lon": -100.86}`
Response:
```json
{
  "nearest_point_id": 123,
  "distance_km": 0.8,
  "soil": {"texture_class": "medium", "organic_matter_pct": 1.6, "awc": 0.18, "k_factor": 0.37},
  "slope_pct": 1.2,
  "current_rotation_guess": "A",
  "crop_history": {"2021": "winter_wheat", "2022": "fallow", "2023": "winter_wheat"},
  "bare_days_last_year": 210,
  "climate_summary": {"mean_growing_season_rain_mm": 330, "mean_heat_days": 18}
}
```

### POST /compare
Request:
```json
{
  "lat": 37.98, "lon": -100.86,
  "current_rotation": "A",
  "water_source": "dryland",
  "soil_texture": "medium",
  "priorities": ["water", "soil", "income"],
  "year_type": "dry",
  "soil_om_pct": null,
  "machinery": {"sorghum_planter": true, "forage_equipment": false}
}
```
Response:
```json
{
  "location": {"nearest_point_id": 123, "distance_km": 0.8},
  "scenario": {"year_type": "dry", "rain_change_pct": -20, "temp_change_c": 0},
  "rotations": [
    {
      "id": "B",
      "name": "Wheat, Sorghum, Fallow",
      "is_current": false,
      "crop_health": {"p10": 0.58, "p50": 0.64, "p90": 0.71},
      "water_need_mm": 410,
      "bare_days_per_year": 120,
      "erosion_risk": "medium",
      "nitrogen_fixing": false,
      "scores": {"water": 6, "soil": 6, "income": 8, "erosion": 6, "forage": 6, "market": 6, "health": 4.8, "total": 6.9},
      "confidence": "medium",
      "why": ["Fallow before wheat stores soil moisture", "Sorghum is drought tolerant and adds income"],
      "evidence": ["350 fields in Finney County follow this rotation (USDA CDL)", "Model MAE 0.05 on unseen years 2022 to 2025"],
      "risks": ["Confirm local sorghum price"],
      "data_gaps": ["Market price is a regional average"]
    }
  ],
  "best_match": "B",
  "disclaimer": "This tool supports, and does not replace, the judgment of farmers and local agricultural experts."
}
```
All numbers above are format examples only.

### GET /trends?lat=&lon=
Yearly series for charts: `years`, `growing_season_rain_mm`, `mean_max_temp_c`, `heat_days`, `peak_ndvi`, `bare_days`, and optional `groundwater_depth_m`.

### GET /model-info
Contents of reports/model_metrics.json plus a list of data sources and known limitations.

---

## 12. Timeline (Pakistan time, UTC+5)

| When | Data scientist | Rest of team |
|---|---|---|
| Sat afternoon | Steps 1 to 3 | Developer builds app on mock JSON, researcher builds crop table, presenter drafts script |
| Sat night | Steps 4 and 5 | continue |
| Sun 9 am to 2 pm | Steps 6 to 9 | app polish, slides |
| Sun 2 pm to 7 pm | Step 10, fixes | deploy, record video |
| Sun 7 pm to 10 pm | submission form | final checks, submit by 10 pm |

Backup plans:
- If AppEEARS results are not back by Saturday night, train with weather + soil + rotation features only and use county level NDVI later.
- If the model does not beat the baseline, ship the rule based scoring and present the model results honestly as a finding.

---

## 13. Submission checklist

- Demo: video up to 30 seconds with English subtitles, or up to 7 slides including title, in English. Exceeding limits makes the project ineligible for Global Judging. Host on a public link with no login.
- Form: project name, high level summary, demo link, final project link (GitHub repo and deployed app), detailed description, NASA data used and how, other data, use of AI.
- List every dataset, library and tool used.
- AI use statement: code written with help from Cursor and Claude; any AI images or video carry a visible watermark; no NASA logos in AI generated content.
- All team members must be confirmed at the same Local Event.

30 second video script:
"In Western Kansas, wheat fields can sit bare for over a year. Soil blows away, and groundwater keeps falling. Field Shift learns from thousands of real fields, using NASA satellite data on rain, heat, soil moisture and crop health. A farmer picks their field and priorities. We compare three rotations, show why each was suggested, and how sure we are. Field Shift supports the farmer's decision. It never replaces it."

7 slide outline: title and team; problem with a chart; solution flow; NASA data used; app screenshot with 3 rotation cards; validation and honest limits; impact and next step (Pakistan version using global datasets).

---

## 14. Honesty rules and known data gaps

- Show confidence and data gaps on every rotation.
- Never claim exact yield or profit.
- Known gaps: peas almost absent in Finney County; forage rotation has few samples; SMAP only from 2015 and 9 km; MODIS NDVI is 250 m so small fields mix with neighbours; NASS yields are county level; model trained on dryland fields only; prices are regional averages.
- Nutrient and organic matter effects are estimates; tell farmers to confirm with a soil test.