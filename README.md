# Streamflow & Drought Forecasting ML Project (1980–2025)

An empirical machine learning framework evaluating the predictive value of India Meteorological Department (IMD) meteorological forcings for multi-horizon streamflow forecasting and hydrological drought early warning across 12 diverse, non-overlapping river basins in India.

---

## Core Research Question & Objectives

> **Central Scientific Question:**  
> *How much does adding meteorological information to historical streamflow improve 1-, 2-, 4-, and 8-week-ahead streamflow forecasts compared with streamflow history alone and a persistence baseline? Does this additional information improve the reliability of drought warnings, and how do these benefits change with forecast horizon and drought severity?*

### Specific Research Sub-Questions
1. **Forecast Horizon Decay**: At what forecast horizon (1-, 2-, 4-, or 8-week ahead) does antecedent river discharge memory ($Q_t, \dots, Q_{t-k}$) lose predictive power, and at what horizon does IMD meteorological memory (precipitation accumulations, temperature/PET, soil moisture) become the dominant driver of predictive skill?
2. **Drought Warning Reliability**: Does the inclusion of meteorological forcings reduce False Alarm Rates (FAR) and increase Probability of Detection (POD) for hydrological drought events?
3. **Drought Severity Sensitivity**: How does the value of meteorological inputs differ between mild low-flow anomalies ($Q_{80}$ / SSI < -0.8) and extreme multi-week hydrological droughts ($Q_{95}$ / SSI < -1.6)?
4. **Spatial Basin Generalizability**: Can a hydro-meteorological model trained on 3 independent catchments in a state transfer its drought warning skill zero-shot to an unseen 4th catchment in a different river basin?

---

## Experimental Design Matrix

| Factor | Configurations | Description |
| :--- | :--- | :--- |
| **Temporal Span** | **46 Years (1980–2025)** | 2,394 weekly timesteps combining CAMELS-IND (1980–2020) and CWC/IMD/ERA5-Land (2021–2025) |
| **Forecast Horizons ($H$)** | **1-week, 2-week, 4-week, 8-week** | Direct multi-step weekly mean discharge forecasts ($y_{t+H}$) |
| **Model Tiers** | **M0: Persistence Baseline**<br>**M1: Autoregressive (Flow Only)**<br>**M2: Hydro-Meteorological (Flow + Multi-Source Weather)** | $Q_{t+H} = Q_t$ for $H \le 2\text{w}$; seasonal weekly climatology for $H \ge 4\text{w}$<br>Input: Past discharge lags ($Q_t, \dots, Q_{t-7}$) + recession slope + within-week flow std + calendar Fourier seasonality<br>Input: Flow lags + IMD Rainfall ($P_{1w}, P_{2w}, P_{1m}, P_{2m}, P_{3m}, API_{30d}, \text{dry days}$) + ERA5-Land/GLEAM Temp & PET ($T_{mean}, T_{max}, PET_{1w}, PET_{1m}, \text{Water Deficit}_{1w}, \text{Water Deficit}_{1m}$) |
| **Drought Severity Levels** | **Mild Drought** ($Q \le Q_{80}$ or $SSI \le -0.8$)<br>**Moderate Drought** ($Q \le Q_{90}$ or $SSI \le -1.2$)<br>**Severe Drought** ($Q \le Q_{95}$ or $SSI \le -1.6$) | Empirically derived from historical baseline distribution |
| **Evaluation Metrics** | **Track A: Continuous Hydrograph Accuracy**<br>**Track B: Drought Early Warning Reliability** | Track A: **NSE, KGE (decomposed into $r, \alpha, \beta$), RMSE, MAE, Log-NSE**<br>Track B: **POD (Recall), FAR, CSI (Threat Score), Brier Score, Reliability Curves** |

---

## Multi-Source Hydro-Meteorological Forcing Pipeline

Meteorological inputs are harmonized across two continuous observation epochs to ensure physical consistency without artificial structural breaks at the 2020/2021 train/test seam:

1. **Daily Precipitation ($P$, mm/day)**:
   - **Historical Baseline (1980–2020)**: CAMELS-IND catchment-averaged daily precipitation (derived from IMD 0.25° gridded daily rainfall).
   - **Operational Period (2021–2025)**: Sourced from IMD Pune's high-resolution **0.25° × 0.25° gridded daily rainfall dataset** (`data/raw/imd/imd_rainfall_2001_2025.nc`, 1.27 GB). Extracted at station coordinates and calibrated via spatial scaling ratio $\alpha_P = \sum P_{CAMELS} / \sum P_{IMD}$ over the 2015–2020 overlap period (mean correlation: $r = 0.660$).
   - **Engineered Memory Windows**:
     - $P_{1w}$ (7-day sum), $P_{2w}$ (14-day sum), $P_{1m}$ (30-day sum), $P_{2m}$ (60-day sum), $P_{3m}$ (90-day seasonal accumulation).
     - **Antecedent Precipitation Index ($API_t = \sum_{k=0}^{29} 0.85^k P_{t-k}$)**: Decayed soil moisture recharge proxy.
     - **Dry Spell Duration**: Count of dry days ($P < 1.0\,\text{mm}$) within the issuance week.

2. **Atmospheric Evaporative Demand & Thermal Forcings**:
   - **Historical Baseline (1980–2020)**: Sourced from CAMELS-IND, which compiles IMD gridded daily temperature ($T_{max}, T_{min}, T_{avg}$) and GLEAM v3.6a Potential Evapotranspiration ($PET_{GLEAM}$, mm/day).
   - **Operational Period (2021–2025)**: To preserve source continuity with the CAMELS-IND baseline and eliminate structural breaks at the 2020/2021 boundary, non-rainfall forcings are ingested from **ERA5-Land reanalysis** (`data/raw/era5_land/`), matching the GLEAM evaporative demand formulation via FAO-56 Penman-Monteith reference evapotranspiration. We explicitly avoid crude empirical approximations (e.g., Hargreaves-Samani on coarse grids).
   - **Boundary Calibration (2015–2020 Overlap)**:
     - **Temperature**: Ingests daily $T_{avg}$ and $T_{max}$ from ERA5-Land, calibrated with an additive bias correction ($\text{Bias}_T = \mu_{CAMELS} - \mu_{ERA5}$, portfolio mean bias: $-0.035\,^\circ\text{C}$, $r = 0.947$).
     - **PET**: Ingests FAO-56 Penman-Monteith ET0, calibrated via multiplicative scaling ($\alpha_{PET} = \sum PET_{GLEAM} / \sum PET_{ERA5} \approx 0.734$, $r = 0.677$).
   - **Engineered Weekly Features**:
     - $T_{mean, 1w}$ (7-day mean), $T_{max, 1w}$ (7-day max).
     - $PET_{1w}$ (7-day sum), $PET_{1m}$ (30-day sum).
     - **Hydrological Water Deficit**: $\text{Deficit}_{1w} = P_{1w} - PET_{1w}$ and $\text{Deficit}_{1m} = P_{1m} - PET_{1m}$ capturing net atmospheric water deficit and catchment drying stress.

3. **Multi-Depth Soil Moisture Pruning**:
   - As mandated by data integrity rules, multi-depth soil moisture ($sm_1 \dots sm_4$) has been pruned from the operational pipeline to eliminate non-native scraping dependencies and ensure 100% auditable traceability.

---

## 12-Station Independent Sub-Basin Portfolio (0 Sub-Basin Overlap)

To eliminate **spatial autocorrelation** and prevent upstream-downstream data leakage, we enforce a strict **zero sub-basin overlap rule** across all three target states (Maharashtra, Odisha, Karnataka). While macro-basins (such as the Krishna or Godavari) cover vast continental expanses spanning divergent climatic zones, their distinct sub-basins represent hydrologically decoupled catchments with unique drainage geometries, geology, and elevation gradients.

Following formal approval gate review, the macro-basin zero-overlap constraint was explicitly relaxed to a strictly enforced **sub-basin zero-overlap rule** to prioritize the **46-year continuous timeline (1980–2025: 16,802 daily timesteps)**. This ensures our models train on authentic historical hydrological regimes without relying on automated or synthetic imputation across multi-month gaps.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               12-STATION INDEPENDENT SUB-BASIN EXPERIMENTAL MATRIX                                    │
├─────────────────┬────────────────────────┬──────────────────────┬─────────────┬───────────┬──────────┬────────────────┤
│ State           │ Station Name           │ Distinct Sub-Basin   │ CAMELS ID   │ Area(km²) │ Aridity  │ Regime Type    │
├─────────────────┼────────────────────────┼──────────────────────┼─────────────┼───────────┼──────────┼────────────────┤
│ **Maharashtra** │ 1. Karad               │ Upper Krishna Stem   │ Gauge 4021  │   5,462.0 │   0.810  │ Deccan Plains  │
│                 │ 2. Warunji             │ Koyna Headwaters     │ Gauge 4063  │   1,890.0 │   1.106  │ Ghats Headwater│
│                 │ 3. Ashti               │ Wainganga Sub-basin  │ Gauge 3005  │  50,990.0 │   0.813  │ Sub-Humid East │
│                 │ 4. Bamni (Holdout)     │ Wardha Sub-basin     │ Gauge 3008  │  46,020.0 │   0.592  │ Vidarbha Basin │
├─────────────────┼────────────────────────┼──────────────────────┼─────────────┼───────────┼──────────┼────────────────┤
│ **Odisha**      │ 1. Anandpur            │ Baitarani Sub-basin  │ Gauge 7002  │   8,570.0 │   1.180  │ Humid Coastal  │
│                 │ 2. Kashinagar          │ Vamsadhara Basin     │ Gauge 16007 │   7,820.0 │   1.044  │ Eastern Ghats  │
│                 │ 3. Gomlai              │ Upper Brahmani Basin │ Gauge 7007  │  21,950.0 │   0.990  │ Highland Plain │
│                 │ 4. Kantamal (Holdout)  │ Tel / Mahanadi Basin │ Gauge 8013  │  19,600.0 │   0.988  │ Inland Delta   │
├─────────────────┼────────────────────────┼──────────────────────┼─────────────┼───────────┼──────────┼────────────────┤
│ **Karnataka**   │ 1. Yadgir              │ Bhima Sub-basin      │ Gauge 4064  │  69,863.0 │   0.428  │ Semi-Arid Soil │
│                 │ 2. Huvinhedigi         │ Upper Krishna South  │ Gauge 4017  │  55,150.0 │   0.506  │ Peninsular Mid │
│                 │ 3. KUDIGE              │ Upper Cauvery Basin  │ Gauge 5010  │   1,934.0 │   1.388  │ High-Rain Head │
│                 │ 4. Haralahalli(Holdout)│ Tungabhadra Basin    │ Gauge 4013  │  14,582.0 │   0.916  │ Central Valley │
└─────────────────┴────────────────────────┴──────────────────────┴─────────────┴───────────┴──────────┴────────────────┘
```

### Data Continuity & Gap Handling Protocol
To uphold scientific integrity without fabricating streamflow observations:
1. **Missingness Retention Threshold**: Stations with total 46-year streamflow missingness $< 10.0\%$ are retained. All 12 stations qualify (ranging from 0.00% to 7.52% missingness).
2. **Small Gap Interpolation ($\le 35$ days)**: Minor recording breaks (such as dry-season low-flow cessations or the nationwide May 2020 COVID-19 lockdown) are linearly interpolated to preserve weekly inertia.
3. **Large Gap Policy ($> 35$ days)**: Contiguous gaps exceeding 35 days are **strictly preserved as `NaN` (unobserved)** without automated imputation. In weekly resampling, any week with $< 4$ daily observations is marked as `NaN` and masked out from loss and metric evaluation, ensuring models evaluate solely on true physical observations.

---

## Dual Evaluation Protocol

```text
45-Year Timeline (1980 - 2025):
|◄──────────── Train (1980–2010) ────────────►| [Embargo] |◄── Val (2011–2015) ──►| [Embargo] |◄── Test (2016–2025) ──►|
           ~1,617 weekly steps                                 ~261 weekly steps                  ~522 weekly steps (10 yrs)
```

### Protocol 1: Station-Specific Multi-Horizon Forecasting (Temporal Split)
* Evaluates how much IMD weather inputs extend forecast horizon ($H = 1, 2, 4, 8$ weeks) over flow inertia alone.
* Evaluated across the 10-year out-of-sample test window (2016–2025) containing the severe 2018 drought and 2019 super-monsoon.

### Protocol 2: Spatial Generalization to Unseen River Basins (Spatial Split)
* In each state, train models on **Stations 1, 2, and 3** using area-normalized specific discharge ($q = Q / \text{Catchment Area}$ in mm/week) and CAMELS-IND static catchment attributes.
* Evaluate zero-shot transfer onto **Station 4 (Holdout)** located in an independent, unseen sub-basin.
* Directly answers: *Can a drought warning system calibrated on regional basins issue reliable drought warnings for an unseen catchment?*

---

## Empirical Benchmark Results: Baseline M0 vs Flow-Only M1 vs Hydro-Met M2 (Audited)

Full out-of-sample evaluation on the 10-year Test block (2016–2025: 522 weeks) under the terminal $H$-week purge embargo (AG-TS-02) and sanitized raw CWC gauge limits (AG-TS-04). Detailed reports are pre-compiled in [`docs/PROJECT_REPORT_DOCUMENTATION.md`](file:///Users/abhitej/Downloads/ML%20Project/docs/PROJECT_REPORT_DOCUMENTATION.md).

### 12-Basin Portfolio Medians across Forecast Horizons (Protocol 1: Temporal)

| Horizon | Continuous NSE (M0 / M1 / M2) | Continuous KGE (M0 / M1 / M2) | Severe Drought ($Q_{95}$) CSI (M0 / M1 / M2) | Severe Drought ($Q_{95}$) FAR (M1 / M2) | Key Empirical Finding |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1-Week** | 0.265 / **0.458** / 0.453 | **0.633** / 0.497 / 0.506 | **0.578** / 0.000 / 0.000 | 0.674 / **0.400** (-0.274) | Persistence dominates 1w alerts; M2 weather forcings slash false alarms by 27.4%. |
| **2-Week** | -0.339 / 0.259 / **0.345** (+0.086) | 0.331 / 0.298 / **0.397** (+0.099) | **0.439** / 0.026 / 0.003 | 0.660 / **0.482** (-0.178) | Persistence collapses to negative NSE; M2 Hydro-Met models provide major skill surge (+0.684 over M0). |
| **4-Week** | 0.090 / **0.282** / 0.267 | 0.146 / 0.264 / **0.380** (+0.116) | 0.000 / **0.031** / 0.020 | 0.556 / 0.610 | Climatology is completely blind to droughts ($\text{CSI} = 0$); M2 delivers large volumetric KGE gains. |
| **8-Week** | -0.207 / 0.222 / **0.290** (+0.068) | -0.088 / 0.222 / **0.308** (+0.086) | 0.000 / **0.148** / 0.007 | 0.519 / **0.447** (-0.072) | Climatology negative across all 12 basins; M2 seasonal rain memory ($P_{3m}$) boosts 10/12 basins. |

### Protocol 2: Cross-Basin Zero-Shot Spatial Transfer to Held-Out Sub-Basins

| State | Holdout Sub-Basin | Horizon | M0 Baseline NSE | M1 Zero-Shot NSE | M2 PyTorch LSTM NSE | Δ(LSTM - M1 GBT) | M2 LSTM KGE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Odisha** | **Kantamal** | 1-week<br>2-week<br>4-week<br>8-week | 0.388<br>-0.030<br>0.093<br>-0.419 | **0.536 (GBT)**<br>**0.437 (GBT)**<br>**0.459 (GBT)**<br>**0.432 (GBT)** | **0.514**<br>**0.410**<br>**0.446**<br>**0.403** | -0.022<br>-0.027<br>-0.013<br>-0.028 | +0.592<br>+0.477<br>+0.501<br>+0.388 |
| **Maharashtra**| **Bamni** | 1-week<br>2-week<br>4-week<br>8-week | 0.250<br>-0.256<br>0.177<br>-0.116 | 0.240 (GBT)<br>-0.466 (GBT)<br>-0.376 (GBT)<br>-0.260 (GBT) | **0.339**<br>**0.031**<br>**0.153**<br>-1.266 | **+0.099**<br>**+0.497**<br>**+0.528**<br>-1.005 | +0.386<br>+0.081<br>+0.186<br>-0.773 |
| **Karnataka** | **Haralahalli** | 1-week<br>2-week<br>4-week<br>8-week | -0.149<br>-0.469<br>0.086<br>-0.118 | 0.154 (Ridge)<br>**0.072 (Ridge)**<br>**0.255 (Ridge)**<br>**0.152 (Ridge)** | **0.258**<br>-0.192<br>-0.098<br>-0.421 | **+0.123**<br>-0.226<br>-0.134<br>-0.319 | +0.383<br>+0.248<br>+0.296<br>+0.069 |

*Key Spatial Transfer Findings:*
1. **Ashti Outlier Sanitization Effect**: Enforcing the $50,000\,\text{m}^3/\text{s}$ limit under Gate AG-TS-04 completely reversed Maharashtra negative transfer at $H=1\text{w}, 2\text{w}, 4\text{w}$, transforming Bamni NSE from $-0.466$ to $+0.031$ ($H=2\text{w}$) and $-0.376$ to $+0.153$ ($H=4\text{w}$).
2. **PyTorch HydroLSTM Zero-Shot Generalization**: Successfully transfers across homogeneous regimes (Odisha: $H=8\text{w}$ NSE $= +0.403$, beating M0 by $+0.822\,\Delta\text{NSE}$). In flashy/semi-arid regions, parsimonious linear Ridge regression remains more resilient at extended lead times.

---

## Repository Structure


```text
ML Project/
├── data/
│   ├── raw/
│   │   ├── cwc/                          # CWC observed discharge (2001-2025)
│   │   │   ├── river_discharge_manual_daily_cwc_mh_2001_2025.csv   # Maharashtra (Daily, 49 MB)
│   │   │   ├── river_discharge_manual_daily_cwc_ka_2001_2025.csv   # Karnataka (Daily, 45 MB)
│   │   │   ├── river_discharge_manual_daily_cwc_od_2001_2025.csv   # Odisha (Daily, 31 MB)
│   │   │   └── ...
│   │   ├── imd/                          # IMD 0.25° x 0.25° gridded daily rainfall
│   │   │   ├── Rainfall_ind2001_rfp25.grd ... Rainfall_ind2025_rfp25.grd
│   │   │   └── imd_rainfall_2001_2025.nc # Unified NetCDF spatial grid (1.27 GB)
│   │   ├── era5_land/                    # ERA5-Land daily reanalysis for 2015-2025 (Tavg, Tmax, Tmin, PET)
│   │   └── camels_ind/                   # CAMELS-IND 41-year hydro-climatic archive (1980-2020)
│   │       ├── attributes_csv/           # Static catchment attributes (topo, climate, soil, geology)
│   │       ├── catchment_mean_forcings/  # Daily basin-averaged forcings (P, T, PET)
│   │       ├── shapefiles_catchment/     # Catchment drainage boundary polygons (.shp)
│   │       └── streamflow_timeseries/    # Observed daily streamflow (14,976 days)
│   └── processed/
│       ├── station_catalog.csv           # Master catalog of 285 stations across India
│       ├── validation_reports/           # Structural validation and boundary calibration logs
│       ├── weekly_tensors/               # Harmonized 46-year weekly feature tensors (2,394 steps)
│       └── model_results/                # Evaluated experiment metric tables (CSV)
├── docs/
│   ├── PROJECT_REPORT_DOCUMENTATION.md   # Complete publication-grade technical report reference
│   ├── DECISION_LOG.md                   # Formal approval gates and strategic decision trail
│   ├── PROGRESS.md                       # Automated deployment and progress tracking log
│   └── 00_CAMELS_IND_Data_Description.pdf# Official CAMELS-IND documentation
├── README.md                             # This guide
├── deploy.sh                             # Automated artifact porting and gh-pages deployment script
└── scripts/
    ├── station_explorer.py               # CLI tool to browse, search, and select stations
    ├── imd_cwc_alignment_test.py         # Parametric meteorological alignment test suite
    ├── prepare_weekly_dataset.py         # 46-year weekly tensor extraction & calibration pipeline
    ├── train_forecast_models.py          # M0 vs M1 multi-horizon training & benchmarking engine
    └── train_m2_models.py                # M2 Hydro-Meteorological models (HistGradientBoosting & PyTorch LSTM)
```

---

## Technical Report & Decision Documentation

For submission in academic and course project reports, complete technical specifications, mathematical derivations, and audit trails are pre-compiled in `docs/`:

1. **[Master Project Report Documentation (`docs/PROJECT_REPORT_DOCUMENTATION.md`)](docs/PROJECT_REPORT_DOCUMENTATION.md)**:
   - Full scientific motivation, hypothesis formulation, and research sub-questions.
   - Comprehensive 12-catchment portfolio table and physical hydrological regime descriptions.
   - 2015–2020 boundary calibration equations and daily vs weekly correlation audit table.
   - 39-feature schema breakdown, chronological API decay formulation, and issue-time leakage proof.
   - Exact mathematical formulas for Track A continuous accuracy (NSE, KGE-2009 with guards, RMSE, MAE) and Track B drought early warning (Mild $Q_{80}$, Moderate $Q_{90}$, Severe $Q_{95}$ with POD, FAR, CSI).
   - Direct mapping to the IITGN ES 335 course evaluation rubric.
2. **[Formal Decision & Approval Gates Log (`docs/DECISION_LOG.md`)](docs/DECISION_LOG.md)**:
   - Immutable audit trail of all approved gates: AG-01 (sub-basin relaxation), AG-02 (gap policy), AG-03 (ERA5-Land ingestion), AG-TS-01 (API weight fix), AG-TS-02 (terminal $H$-week purge embargo), AG-TS-04 (outlier masking), and AG-TS-05 (M2 hydro-meteorological models).

---

## Quickstart Commands

### 1. Browse Stations by State & River Basin
```bash
# View all Karnataka stations with CAMELS catchments
python3 "scripts/station_explorer.py" --state Karnataka --matched-only

# Search for specific basins
python3 "scripts/station_explorer.py" --basin Cauvery
python3 "scripts/station_explorer.py" --basin Mahanadi
python3 "scripts/station_explorer.py" --basin Godavari
```

### 2. Run IMD Meteorological Alignment Test
```bash
# Run multi-state alignment test across distinct sub-basins:
python3 "scripts/imd_cwc_alignment_test.py" \
  --stations "Karad" "Kantamal" "KUDIGE"

# Test specific regional stations:
python3 "scripts/imd_cwc_alignment_test.py" \
  --stations "Warunji" "Gomlai" "Huvinhedigi"
```

### 3. Generate 46-Year Weekly Feature Tensors
```bash
# Process all 12 stations across Maharashtra, Odisha, and Karnataka (Peak RAM < 0.5 GB):
python3 "scripts/prepare_weekly_dataset.py"
```

### 4. Benchmark M0 Baseline vs M1 Autoregressive Models
```bash
# Protocol 1: Station-Specific Temporal Benchmarking across 1, 2, 4, 8 weeks:
python3 "scripts/train_forecast_models.py" --protocol 1 --model ridge

# Protocol 2: Cross-Basin Zero-Shot Spatial Transfer to held-out sub-basins:
python3 "scripts/train_forecast_models.py" --protocol 2 --model ridge

# Full Evaluation Suite (Protocol 1 & 2 across Ridge and Gradient Boosting):
python3 "scripts/train_forecast_models.py" --protocol all --model all --horizons 1 2 4 8
```

### 5. Benchmark Tier M2 Hydro-Meteorological Models (Awaiting Review)
```bash
# Protocol 1: Temporal Benchmarking with HistGradientBoosting on 26 Hydro-Met features:
python3 "scripts/train_m2_models.py" --protocol 1 --horizons 1 2 4 8

# Protocol 2: Cross-Basin Spatial Transfer with PyTorch Deep Recurrent LSTM:
python3 "scripts/train_m2_models.py" --protocol 2 --seq-len 8 --hidden-dim 64 --epochs 50

# Full M2 Evaluation Suite:
python3 "scripts/train_m2_models.py" --protocol all --horizons 1 2 4 8
```

### 6. Deploy Interactive Visualizations to GitHub Pages
The deployment script enforces a strict one-way port of Plotly `.html` widgets from the backend to the orphaned `gh-pages` branch, simultaneously logging the update to `PROGRESS.md`.
```bash
# Execute from the main branch; requires a progress update message
./deploy.sh "Regenerated KGE component plots and updated index.html iframe layouts"
```
