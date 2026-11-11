# Streamflow & Drought Forecasting ML Project (1980–2025)

An empirical machine learning framework evaluating the predictive value of India Meteorological Department (IMD) meteorological forcings for multi-horizon streamflow forecasting and hydrological drought early warning at 12 gauges across peninsular Indian subcatchments, including shared Godavari and Krishna drainages and the nested Warunji–Karad–Huvinhedigi cascade.

---

## Core Research Question & Objectives

> **Central Scientific Question:**  
> *How much does adding meteorological information to historical streamflow improve 1-, 2-, 4-, and 8-week-ahead streamflow forecasts compared with streamflow history alone and a persistence baseline? Does this additional information improve the reliability of drought warnings, and how do these benefits change with forecast horizon and drought severity?*

### Specific Research Sub-Questions
1. **Forecast Horizon Decay**: At what forecast horizon (1-, 2-, 4-, or 8-week ahead) does antecedent river discharge memory ($Q_t, \dots, Q_{t-k}$) lose predictive power, and at what horizon does IMD meteorological memory (precipitation accumulations, temperature/PET, soil moisture) become the dominant driver of predictive skill?
2. **Drought Warning Reliability**: Does the inclusion of meteorological forcings reduce False Alarm Ratio (FAR) and increase Probability of Detection (POD) for hydrological drought events?
3. **Drought Severity Sensitivity**: How does the value of meteorological inputs differ between mild low-flow anomalies ($Q_{80}$, training-period 20th percentile) and extreme multi-week hydrological droughts ($Q_{95}$, training-period 5th percentile)?
4. **Spatial Basin Generalizability**: Can a hydro-meteorological model trained on three other catchments in a state transfer its drought warning skill zero-shot to an unseen 4th catchment at a held-out gauge?

---

## Experimental Design Matrix

| Factor | Configurations | Description |
| :--- | :--- | :--- |
| **Temporal Span** | **46 Years (1980–2025)** | 2,394 weekly timesteps combining CAMELS-IND (1980–2020) and CWC/IMD/ERA5-Land (2021–2025) |
| **Forecast Horizons ($H$)** | **1-week, 2-week, 4-week, 8-week** | Direct multi-step weekly mean discharge forecasts ($y_{t+H}$) |
| **Model Tiers** | **M0: Persistence Baseline**<br>**M1: Autoregressive (Flow Only)**<br>**M2: Hydro-Meteorological (Flow + Multi-Source Weather)** | $Q_{t+H} = Q_t$ for $H \le 2\text{w}$; seasonal weekly climatology for $H \ge 4\text{w}$<br>Input: Past discharge lags ($Q_t, \dots, Q_{t-7}$) + recession slope + within-week flow std + calendar Fourier seasonality<br>Input: Flow lags + IMD Rainfall ($P_{1w}, P_{2w}, P_{1m}, P_{2m}, P_{3m}, API_{30d}, \text{dry days}$) + ERA5-Land/GLEAM Temp & PET ($T_{mean}, T_{max}, PET_{1w}, PET_{1m}, \text{Water Deficit}_{1w}, \text{Water Deficit}_{1m}$) |
| **Drought Severity Levels** | **Mild Drought** ($Q \le Q_{80}$)<br>**Moderate Drought** ($Q \le Q_{90}$)<br>**Severe Drought** ($Q \le Q_{95}$) | Empirically derived from historical baseline distribution |
| **Evaluation Metrics** | **Track A: Continuous Hydrograph Accuracy**<br>**Track B: Drought Early Warning Reliability** | Track A: **NSE, KGE (decomposed into $r, \alpha, \beta$), RMSE, MAE**<br>Track B: **POD (Recall), FAR, CSI (Threat Score), PR-AUC**, scored across Severe ($Q_{95}$), Moderate ($Q_{90}$), and Mild ($Q_{80}$) drought tiers (audited in `scripts/audit_drought_sensitivity.py`, artifact: `data/processed/model_results/drought_severity_sensitivity.csv`; full comparison in `docs/PROJECT_REPORT_DOCUMENTATION.md` §10.3.1). Log-NSE is computed and reported; Brier score and reliability curves are not computed. |

---

## Multi-Source Hydro-Meteorological Forcing Pipeline

Meteorological inputs are harmonized across two continuous observation epochs to ensure physical consistency without artificial structural breaks at the 2020/2021 train/test seam:

1. **Daily Precipitation ($P$, mm/day)**:
   - **Historical Baseline (1980–2020)**: CAMELS-IND catchment-averaged daily precipitation (derived from IMD 0.25° gridded daily rainfall).
   - **Operational Period (2021–2025)**: Reconstructed as catchment area-weighted means from IMD Pune's high-resolution **0.25° × 0.25° gridded daily rainfall dataset** (`data/raw/imd/imd_rainfall_2001_2025.nc`, 1.27 GB) using polygon-cell intersection weights against CAMELS catchment shapefiles (`camels_ind/shapefiles_catchment/catchments.shp`). Evaluated across the 2015–2020 overlap period against CAMELS-IND historical series, the catchment-averaged reconstruction achieves physical parity with portfolio mean scale ratio $\bar{\alpha}_P = 1.0009$ (vs 1.134 at nearest grid point), daily Pearson $r = 0.9994$ (vs 0.660), and weekly Pearson $r = 0.9998$ (vs 0.697).
   - **Engineered Memory Windows**:
     - $P_{1w}$ (7-day sum), $P_{2w}$ (14-day sum), $P_{1m}$ (30-day sum), $P_{2m}$ (60-day sum), $P_{3m}$ (90-day seasonal accumulation).
     - **Antecedent Precipitation Index ($API_t = \sum_{k=0}^{29} 0.85^k P_{t-k}$)**: Decayed soil moisture recharge proxy.
     - **Dry Spell Duration**: Count of dry days ($P < 1.0\,\text{mm}$) within the issuance week.

2. **Atmospheric Evaporative Demand & Thermal Forcings**:
   - **Historical Baseline (1980–2020)**: Sourced from CAMELS-IND, which compiles IMD gridded daily temperature ($T_{max}, T_{min}, T_{avg}$) and GLEAM v3.6a Potential Evapotranspiration ($PET_{GLEAM}$, mm/day; Priestley-Taylor formulation).
   - **Operational Period (2021–2025)**: To preserve source continuity with the CAMELS-IND baseline and eliminate structural breaks at the 2020/2021 boundary, non-rainfall forcings are ingested from **ERA5-Land reanalysis** (gauge-point extractions calibrated to GLEAM), matching the evaporative demand formulation via reference evapotranspiration.
   - **Boundary Calibration (2015–2020 Overlap)**:
     - **Temperature**: Ingests daily $T_{avg}$ and $T_{max}$ from ERA5-Land, calibrated with an additive bias correction ($\text{Bias}_T = \mu_{CAMELS} - \mu_{ERA5}$; Mean Absolute Bias across portfolio: $0.84\,^\circ\text{C}$, signed portfolio mean: $-0.035\,^\circ\text{C}$, $r = 0.947$).
     - **PET**: Ingests FAO-56 Penman-Monteith ET0, calibrated via multiplicative scaling ($\alpha_{PET} = \sum PET_{GLEAM} / \sum PET_{ERA5} \approx 0.734$, $r = 0.677$).
   - **Engineered Weekly Features**:
     - $T_{mean, 1w}$ (7-day mean), $T_{max, 1w}$ (7-day max).
     - $PET_{1w}$ (7-day sum), $PET_{1m}$ (30-day sum).
     - **Hydrological Water Deficit**: $\text{Deficit}_{1w} = P_{1w} - PET_{1w}$ and $\text{Deficit}_{1m} = P_{1m} - PET_{1m}$ capturing net atmospheric water deficit and catchment drying stress.
   - **Aridity Index Definition**: Catchment aridity used in this project is the Budyko index $\overline{\text{PET}} / \bar{P}$, computed from the weekly tensors over 1980–2020 (values $> 1.0$ indicate water-limited regimes; values $< 1.0$ indicate energy-limited/humid regimes).

3. **Multi-Depth Soil Moisture Pruning**:
   - As mandated by data integrity rules, multi-depth soil moisture ($sm_1 \dots sm_4$) has been pruned from the operational pipeline to eliminate non-native scraping dependencies and ensure 100% auditable traceability.

---

## 12-Station Catchment Portfolio

The experimental portfolio encompasses **12 CWC/CAMELS-IND river gauging catchments** spanning three distinct macro-climatic and physiographic zones of peninsular and eastern India (Maharashtra, Odisha, Karnataka) across a 46-year span (1980–2025: 2,394 weekly timesteps).

### Shared drainages and nested gauges
The portfolio includes shared major drainages and a nested gauge cascade. Distinct station names do not establish statistical independence:
1. **Shared major drainages**: CAMELS-IND station metadata places Wainganga and Wardha within Godavari, and Koyna, Bhima, Tungabhadra, and the Krishna main-stem gauges within Krishna. The portfolio also includes Tel/Mahanadi, Baitarani, Brahmani, Vamsadhara, and Cauvery subcatchments.
2. **1 Three-Gauge Nested Cascade (Upper Krishna Basin)**: Three stations form an upstream-to-downstream nested drainage hierarchy:
   - **Warunji (Gauge 4063)**: Headwater catchment on the Koyna River ($1,890.0\,\text{km}^2$) in the Western Ghats high-rainfall crest.
   - **Karad (Gauge 4021)**: Upper Krishna mainstem gauge ($5,462.0\,\text{km}^2$), directly receiving Koyna outflow. Warunji represents **34.6% of Karad's drainage area**.
   - **Huvinhedigi (Gauge 4017)**: Downstream regional trunk gauge on the peninsular Krishna in Karnataka ($55,150.0\,\text{km}^2$). Warunji comprises **3.4%** and Karad comprises **9.9%** of Huvinhedigi's upstream catchment.
   - *Audit Disclosure*: We explicitly retire the former 'Zero Sub-Basin Overlap' claim. Models evaluated on Warunji, Karad, and Huvinhedigi evaluate multi-scale hydrological propagation along a 55,000 km² drainage cascade rather than mutually independent catchments.

### Unmeasured Low Flows Recorded as Zero (Karad, Warunji, Yadgir, Huvinhedigi)
After about 2010 CWC operated Karad, Warunji and Yadgir as monsoon-season stations; weeks that were not measured are logged as exactly $0.0000\,\text{m}^3/\text{s}$ (see `docs/AUDIT_LOG.md`, Issue C). Share of weeks recorded as zero:

| Station | Train 1980–2010 | Val 2011–2015 | Test 2016–2025 | Test zeros falling in Jun–Oct |
| :-- | --: | --: | --: | --: |
| Karad | 2.1% | 76.6% | 69.0% | 21% |
| Warunji | 4.0% | 77.0% | 70.0% | 22% |
| Yadgir | 19.9% | 65.1% | 62.3% | 18% |
| Huvinhedigi | 0.0% | 0.0% | 14.1% | 28% |

These are missing measurements, not droughts (about a fifth fall inside the monsoon). At these four stations every week whose issue-time or target flow is a recorded zero is excluded from scoring for all models (`ZERO_AS_MISSING_STATIONS` in `scripts/audit_metrics.py`). This lowers the pooled severe-drought base rate from 0.238 to about 0.065, close to the nominal 0.05. Haralahalli's zeros (dry season only, also present in training) are kept. Drought thresholds are fitted on the 1980–2010 training fold (Gate AG-TS-07). All models are trained and tuned with zero-flow masking applied to both training (1980–2010) and validation (2011–2015) folds at these four stations, ensuring spurious telemetry zeros do not distort model fitting or early stopping. In addition, log-target regression variants (`M1-log`, `M2-log`) fit specific runoff in log space to mitigate heteroskedasticity and preserve low-flow sensitivity.


```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               12-STATION EXPERIMENTAL MATRIX: REGIMES & ATTRIBUTES                                    │
├─────────────────┬────────────────────────┬──────────────────────┬─────────────┬───────────┬──────────┬────────────────┤
│ State           │ Station Name           │ Sub-Basin / River    │ CAMELS ID   │ Area(km²) │ Aridity* │ Regime Type    │
├─────────────────┼────────────────────────┼──────────────────────┼─────────────┼───────────┼──────────┼────────────────┤
│ **Maharashtra** │ 1. Karad (Cascade 2)   │ Upper Krishna Stem   │ Gauge 4021  │   5,462.0 │   0.977  │ Deccan Plains  │
│                 │ 2. Warunji (Cascade 1) │ Koyna Headwaters     │ Gauge 4063  │   1,890.0 │   0.747  │ Ghats Headwater│
│                 │ 3. Ashti               │ Wainganga Sub-basin  │ Gauge 3005  │  50,990.0 │   0.852  │ Sub-Humid East │
│                 │ 4. Bamni (Holdout)     │ Wardha Sub-basin     │ Gauge 3008  │  46,020.0 │   1.110  │ Vidarbha Basin │
├─────────────────┼────────────────────────┼──────────────────────┼─────────────┼───────────┼──────────┼────────────────┤
│ **Odisha**      │ 1. Anandpur            │ Baitarani Sub-basin  │ Gauge 7002  │   8,570.0 │   0.674  │ Humid Coastal  │
│                 │ 2. Kashinagar          │ Vamsadhara Basin     │ Gauge 16007 │   7,820.0 │   0.770  │ Eastern Ghats  │
│                 │ 3. Gomlai              │ Upper Brahmani Basin │ Gauge 7007  │  21,950.0 │   0.751  │ Highland Plain │
│                 │ 4. Kantamal (Holdout)  │ Tel / Mahanadi Basin │ Gauge 8013  │  19,600.0 │   0.740  │ Inland Delta   │
├─────────────────┼────────────────────────┼──────────────────────┼─────────────┼───────────┼──────────┼────────────────┤
│ **Karnataka**   │ 1. Yadgir              │ Bhima Sub-basin      │ Gauge 4064  │  69,863.0 │   1.440  │ Semi-Arid Soil │
│                 │ 2. Huvinhedigi(Casc. 3)│ Upper Krishna Main   │ Gauge 4017  │  55,150.0 │   1.367  │ Peninsular Mid │
│                 │ 3. KUDIGE              │ Upper Cauvery Basin  │ Gauge 5010  │   1,934.0 │   0.725  │ High-Rain Head │
│                 │ 4. Haralahalli(Holdout)│ Tungabhadra Basin    │ Gauge 4013  │  14,582.0 │   0.901  │ Central Valley │
└─────────────────┴────────────────────────┴──────────────────────┴─────────────┴───────────┴──────────┴────────────────┘
* Aridity is the Budyko aridity index PET/P over 1980–2020, computed from the weekly tensors; > 1.0 = water-limited.
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
* Evaluate zero-shot transfer onto **Station 4 (Holdout)** at a held-out gauge; some training and held-out gauges share major drainages.
* Directly answers: *Can a drought warning system calibrated on regional basins issue reliable drought warnings for an unseen catchment?*

---

## Empirical Benchmark Results (Audited)

<!-- README_RESULTS:START -->
### Audited results (test block 2016–2025, 12 stations)

Generated by `scripts/render_report_results.py`; full tables in `docs/PROJECT_REPORT_DOCUMENTATION.md` §10. Recorded zeros at Karad, Warunji, Yadgir and Huvinhedigi are treated as unmeasured weeks. Δ = paired median of per-station differences; *p* = uncorrected exploratory Wilcoxon signed-rank. Across four NSE lead comparisons, the Bonferroni reference is 0.0125. The one-week value 0.027 does not meet it, while the two-week value 0.007 does. Shared drainages and nested gauges also limit inference.

| Lead | NSE M0 / M1 / M2 | ΔNSE (M2−M1), *p* | Log-NSE M0 / M2-log | Q95 CSI M0 / M2 / M2-log | Climatology or persistence beats M2 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **1w** | +0.284 / +0.458 / +0.486 | +0.045, 0.027 | +0.829 / +0.844 | 0.529 / 0.000 / 0.229 | 0 of 12 |
| **2w** | -0.185 / +0.282 / +0.331 | +0.034, 0.007 | +0.649 / +0.746 | 0.379 / 0.000 / 0.060 | 0 of 12 |
| **4w** | +0.364 / +0.273 / +0.292 | -0.007, 0.677 | +0.623 / +0.647 | 0.000 / 0.000 / 0.000 | 9 of 12 |
| **8w** | +0.353 / +0.237 / +0.271 | -0.014, 0.569 | +0.622 / +0.609 | 0.000 / 0.000 / 0.000 | 11 of 12 |

**Key findings.**
- **Weather inputs improve short-lead NSE:** M2 exceeds M1 by +0.045 paired median NSE at 1 week and +0.034 at 2 weeks. Reported KGE tests do not show significant gains at these leads.
- **Seasonal averages remain important:** Weekly climatology beats M2 at 9 and 11 of 12 stations at 4 and 8 weeks.
- **Flow accuracy and warning skill differ:** Persistence has median Q95 CSI 0.529 and 0.379 at 1 and 2 weeks. The additional 4-week lagged-persistence diagnostic has pooled CSI 0.251, versus M2 0.043. Official M0 at 4 weeks is climatology. Log-target M2 improves one-week median CSI to 0.229 but lowers continuous NSE from 0.486 to 0.464.
- **Zero-shot spatial transfer:** Pooled HydroLSTM beats the held-out station's own baseline only at 1–2 weeks.
- **Known limitations:** 2021–2025 temperature and PET remain gauge-point extractions from ERA5-Land rather than catchment-averaged means. Dynamic permutation feature importance shows that shuffling the atmospheric-demand group increases 1-week MSE by only +3.1% (versus +114% when shuffling precipitation features; single-feature share ~2.6%). Catchment-averaging rainfall resolved the dominant spatial mismatch while avoiding error propagation from uncalibrated reanalysis ET0 grids.

<!-- README_RESULTS:END -->

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
    ├── train_m2_models.py                # M2 Hydro-Meteorological models (HistGradientBoosting & PyTorch LSTM)
    ├── eval_config.py                    # Shared evaluation settings (stations whose recorded zeros are unmeasured)
    ├── audit_metrics.py                  # Scores every saved model on one shared mask (source of all published numbers)
    ├── export_benchmark_data.py          # Writes assets/benchmark-data.js for the website
    ├── generate_interactive_plots.py     # Plotly chart sources from the scored predictions
    ├── uncertainty_tests.py              # Wilcoxon tests and bootstrap confidence intervals
    └── render_report_results.py          # Regenerates report sections 10-12 and the README results
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
   - Immutable audit trail of all approved gates: AG-01 (sub-basin relaxation), AG-02 (gap policy), AG-03 (ERA5-Land ingestion), AG-TS-01 (API weight fix), AG-TS-02 (terminal $H$-week purge embargo), AG-TS-04 (outlier masking), AG-TS-05 (M2 hydro-meteorological models), AG-TS-06 (nested cascade reframe H1), AG-TS-07 (Karad regime shift & threshold retention C1), and AG-TS-08 (objective statistical reporting & climatology pivot).

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
> **Warning:** steps 4 and 5 retrain the models and overwrite `models/`. Retrained models, especially the LSTM, may not reproduce the published numbers exactly. To reproduce the published results from the committed models, skip to step 6.
```bash
# Protocol 1: Station-Specific Temporal Benchmarking across 1, 2, 4, 8 weeks:
python3 "scripts/train_forecast_models.py" --protocol 1 --model ridge

# Protocol 2: Cross-Basin Zero-Shot Spatial Transfer to held-out sub-basins:
python3 "scripts/train_forecast_models.py" --protocol 2 --model ridge

# Full Evaluation Suite (Protocol 1 & 2 across Ridge and Gradient Boosting):
python3 "scripts/train_forecast_models.py" --protocol all --model all --horizons 1 2 4 8
```

### 5. Benchmark Tier M2 Hydro-Meteorological Models (Implemented & Evaluated)
```bash
# Protocol 1: Temporal Benchmarking with HistGradientBoosting on 26 Hydro-Met features:
python3 "scripts/train_m2_models.py" --protocol 1 --horizons 1 2 4 8

# Protocol 2: Cross-Basin Spatial Transfer with PyTorch Deep Recurrent LSTM:
python3 "scripts/train_m2_models.py" --protocol 2 --seq-len 8 --hidden-dim 64 --epochs 50

# Full M2 Evaluation Suite:
python3 "scripts/train_m2_models.py" --protocol all --horizons 1 2 4 8
```

### 6. Score the Saved Models and Rebuild All Published Numbers
These steps use only the committed weekly tensors and models, and reproduce every number on the website and in the report.
```bash
python3 scripts/audit_metrics.py              # score every model on the shared mask -> audited_metrics_recomputed.csv, test_predictions.csv
python3 scripts/export_benchmark_data.py      # site data file assets/benchmark-data.js (medians, paired deltas, Wilcoxon tests)
python3 scripts/generate_interactive_plots.py # Plotly chart sources in visualizations/source/ and feature-importance CSVs
python3 scripts/uncertainty_tests.py          # Wilcoxon + bootstrap confidence intervals (Issue K)
python3 scripts/render_report_results.py      # regenerate report sections 10-12 and the README results section
```

### 7. Deploy Interactive Visualizations to GitHub Pages
`deploy.sh` runs step 6, commits `main`, then updates the `gh-pages` worktree (`site/`, created if missing): it copies the chart sources, site data and docs, rebuilds `figdata.js`, the chart pages and the benchmark table, then commits and pushes `gh-pages`.
```bash
# Execute from the main branch; requires a progress update message
./deploy.sh "Regenerated audited metrics, charts and report tables"
```
