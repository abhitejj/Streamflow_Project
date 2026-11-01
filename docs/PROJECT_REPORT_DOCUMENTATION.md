# 🌊 Empirical Streamflow & Hydrological Drought Forecasting (1980–2025)
## Comprehensive Technical Documentation & Course Project Report Reference

**Author / Team:** ML Course Project Team  
**Institution:** Indian Institute of Technology Gandhinagar (IITGN) — ES 335  
**Version:** 2.0 (Audited)  
**Date of Current Audit:** October 31, 2026  
**Repository Working Directory:** `ML Project/`

---

## 1. Executive Summary & Research Motivation

Predicting river discharge ($Q$) and providing actionable early warnings for hydrological droughts across peninsular India is vital for reservoir operation, water allocation, and disaster preparedness. Conventional hydrological modeling relies heavily on either purely physical conceptual models (which require dense, poorly available parameterizations) or short-term statistical persistence. 

This project establishes an empirical, leakage-free machine learning benchmark spanning **46 continuous years (1980–2025: 16,802 daily timesteps $\to$ 2,394 weekly timesteps)** across **12 hydrologically decoupled sub-basins** in three climatically diverse states: **Maharashtra (Deccan Plateau / Western Ghats)**, **Odisha (Sub-Humid Eastern Ghats / Deltaic Plains)**, and **Karnataka (Semi-Arid Peninsular Basin / Upper Cauvery)**.

### Core Scientific Questions
1. **Forecast Horizon Decay**: At what forecast horizon ($H \in \{1, 2, 4, 8\}$ weeks ahead) does antecedent river discharge memory ($Q_t, \dots, Q_{t-7}$) lose predictive skill, and at what lead time do hydro-meteorological forcings (precipitation memory, evaporative demand, and climatic water deficit $P - PET$) become the dominant predictor of streamflow?
2. **Drought Early Warning Reliability**: Does the inclusion of meteorological forcings reduce False Alarm Ratios (FAR) and increase Critical Success Indices (CSI / Threat Score) for drought events compared to flow-only autoregression and seasonal climatology?
3. **Severity Sensitivity**: How does forecasting skill vary across standardized drought severity tiers:
   - Mild Low-Flow Anomalies ($Q \le Q_{80}$ / 20th percentile of historical baseline)
   - Moderate Drought ($Q \le Q_{90}$ / 10th percentile)
   - Severe Drought ($Q \le Q_{95}$ / 5th percentile)
4. **Cross-Basin Zero-Shot Spatial Transfer**: Can a model trained on three independent catchments within a state transfer its streamflow and drought prediction capability zero-shot to an unseen fourth held-out sub-basin using area-normalized specific runoff ($q = Q \cdot 604.8 / \text{Area}_{\text{km}^2}$ in mm/week)?

---

## 2. Experimental Matrix & Station Portfolio

To eliminate spatial autocorrelation and prevent upstream-downstream leakage, we enforce a **zero sub-basin overlap constraint**. Stations belong to distinct, decoupled tributaries with independent drainage geometries and elevation gradients.

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

---

## 3. Data Lineage & Provenance Contract

Data ingestion spans two observation eras, harmonized to ensure continuous physical forcing without synthetic fabrication or structural breaks:

| Variable | Temporal Span | Primary Source & Resolution | Processing & Extraction Rule |
| :--- | :--- | :--- | :--- |
| **Streamflow ($Q$)** | 1980–2020 | CAMELS-IND observed streamflow (CWC gauge records) | Extracted from gauge ID timeseries column |
| | 2001–2025 | Central Water Commission (CWC) daily manual discharge | Matched by station string name; normalized to m³/s |
| | Combined | Stitched series (1980–2025: 16,802 daily timesteps) | Conservative linear interpolation on gaps $\le 35$ days; gaps $> 35$ days remain `NaN` |
| **Precipitation ($P$)**| 1980–2020 | CAMELS-IND daily catchment-averaged precipitation | Zonal mean derived from IMD 0.25° gridded rainfall |
| | 2021–2025 | IMD Pune 0.25° × 0.25° Gridded NetCDF (`imd_rainfall_2001_2025.nc`) | Nearest-grid point extraction at station coordinates; scaled by $\alpha_P$ |
| **Temperature ($T$)** | 1980–2020 | CAMELS-IND daily $T_{\text{max}}, T_{\text{min}}, T_{\text{avg}}$ | Zonal mean derived from IMD 1.0° gridded temperature |
| | 2021–2025 | ERA5-Land Reanalysis (0.1° resolution daily CSVs) | Point extraction at station coordinates; corrected via additive bias $\text{Bias}_T$ |
| **Evaporation (PET)** | 1980–2020 | CAMELS-IND GLEAM v3.6a PET (`pet_gleam(mm/day)`) | Priestley-Taylor catchment zonal mean (0 NaNs in 1980) |
| | 2021–2025 | ERA5-Land FAO-56 Penman-Monteith reference ET0 | Point extraction at station coordinates; scaled by $\alpha_{\text{PET}}$ |

---

## 4. Boundary Calibration Audit (2015–2020 Overlap)

To prevent step-change artifacts at the train/test interface (December 31, 2020 / January 1, 2021), a formal boundary calibration was executed over the **2,192 overlapping daily timesteps (2015–2020)**.

### Calibration Formulations
1. **Precipitation Multiplicative Scaling**:
   $$\alpha_P = \frac{\sum_{t} P_{\text{CAMELS}}(t)}{\sum_{t} P_{\text{IMD}}(t)}, \quad P_{\text{calibrated}}(t) = \alpha_P \cdot P_{\text{IMD}}(t)$$
2. **Temperature Additive Bias Correction**:
   $$\text{Bias}_{T} = \mu(T_{\text{CAMELS}}) - \mu(T_{\text{ERA5}}), \quad T_{\text{calibrated}}(t) = T_{\text{ERA5}}(t) + \text{Bias}_{T}$$
3. **Evaporative Demand Multiplicative Scaling**:
   $$\alpha_{\text{PET}} = \frac{\sum_{t} \text{PET}_{\text{GLEAM}}(t)}{\sum_{t} \text{PET}_{\text{ERA5}}(t)}, \quad \text{PET}_{\text{calibrated}}(t) = \alpha_{\text{PET}} \cdot \text{PET}_{\text{ERA5}}(t)$$

### Comprehensive 12-Station Boundary Calibration Audit Table
*Artifact Log:* [`data/processed/validation_reports/structural_break_calibration.md`](file:///Users/abhitej/Downloads/ML%20Project/data/processed/validation_reports/structural_break_calibration.md)

| Station | Gauge | Area (km²) | $\alpha_P$ | $P$ Daily $r$ | $P$ Wkly $r$ | $T_{\text{avg}}$ Bias ($^\circ$C) | $T_{\text{avg}}$ Daily $r$ | $T_{\text{avg}}$ Wkly $r$ | $\alpha_{\text{PET}}$ | PET Daily $r$ | PET Wkly $r$ | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Karad** | `4021` | 5,462 | `1.5616` | 0.823 | **0.897** | $+1.07$ | 0.943 | **0.961** | `0.7196` | 0.760 | **0.781** | `PASS` |
| **Warunji** | `4063` | 1,890 | `2.1797` | 0.767 | **0.841** | $+1.45$ | 0.933 | **0.950** | `0.7260` | 0.796 | **0.825** | `PASS` |
| **Ashti** | `3005` | 50,990 | `0.7824` | 0.645 | **0.857** | $-1.47$ | 0.949 | **0.961** | `0.6937` | 0.708 | **0.736** | `PASS` |
| **Bamni** | `3008` | 46,020 | `0.8548` | 0.691 | **0.893** | $-0.47$ | 0.976 | **0.984** | `0.7351` | 0.688 | **0.706** | `PASS` |
| **Anandpur** | `7002` | 8,570 | `0.8995` | 0.779 | **0.896** | $+0.44$ | 0.973 | **0.987** | `0.7913` | 0.776 | **0.785** | `PASS` |
| **Kashinagar**| `16007` | 7,820 | `1.1447` | 0.678 | **0.806** | $-0.09$ | 0.964 | **0.980** | `0.7652` | 0.789 | **0.811** | `PASS` |
| **Gomlai** | `7007` | 21,950 | `0.8509` | 0.745 | **0.850** | $-0.48$ | 0.965 | **0.974** | `0.7430` | 0.713 | **0.720** | `PASS` |
| **Kantamal** | `8013` | 19,600 | `1.0135` | 0.638 | **0.821** | $-0.65$ | 0.956 | **0.967** | `0.6965` | 0.724 | **0.732** | `PASS` |
| **Yadgir** | `4064` | 69,863 | `0.8560` | 0.571 | **0.808** | $-0.88$ | 0.950 | **0.966** | `0.6529` | 0.532 | **0.533** | `Weekly PASS` |
| **Huvinhedigi**| `4017` | 55,150 | `0.9987` | 0.349 | **0.549** | $-1.17$ | 0.945 | **0.966** | `0.6410` | 0.543 | **0.542** | `Weekly PASS` |
| **KUDIGE** | `5010` | 1,934 | `2.1191` | 0.678 | **0.801** | $+2.35$ | 0.872 | **0.924** | `0.9030` | 0.522 | **0.522** | `Weekly PASS` |
| **Haralahalli**| `4013` | 14,582 | `1.6131` | 0.553 | **0.676** | $-0.52$ | 0.935 | **0.960** | `0.7445` | 0.568 | **0.568** | `Weekly PASS` |

### Critical Physical Insights Disclosed in Report
1. **Scale Integration Effect**: In macro-catchments (e.g., Yadgir, $69,863\,\text{km}^2$), localized storms at the gauge outlet have weak daily correlation with upper-catchment rainfall ($r = 0.349\text{--}0.571$). Aggregation to weekly operational timesteps (`W-MON`) eliminates high-frequency spatial mismatch, elevating portfolio mean precipitation correlation to **$\bar{r}_{P, \text{weekly}} = 0.808$**.
2. **Evaporative Demand Formulation Disparity**: The correlation ceiling in semi-arid Karnataka ($r_{\text{PET}} \approx 0.53$) reflects genuine physical physics: GLEAM is radiation-constrained Priestley-Taylor, whereas ERA5-Land Penman-Monteith incorporates aerodynamic drying (vapor pressure deficit and wind speed). Multiplicative scaling ($\alpha_{\text{PET}} \approx 0.734$) aligns the volumetric evaporative demand seamlessly across the boundary.

---

## 5. Feature Engineering Schema & Leakage Prevention

The pipeline generates **2,394 weekly timesteps** across **39 audited features and targets** in [`data/processed/weekly_tensors/`](file:///Users/abhitej/Downloads/ML%20Project/data/processed/weekly_tensors/):

### Feature Schema (39 Columns)
1. **Contemporaneous Streamflow State (Issue Time $t$)**:
   - `flow_target_m3s`: Mean weekly discharge $Q_t$ (set to `NaN` if observed daily records $< 4$).
   - `flow_target_specific_mm`: Specific runoff $q_t = Q_t \cdot 604.8 / \text{Area}_{\text{km}^2}$ (mm/week).
   - `flow_lag_0` to `flow_lag_7`: Autoregressive lags ($Q_t, Q_{t-1}, Q_{t-2}, Q_{t-3}, Q_{t-4}, Q_{t-7}$).
   - `flow_std_7d`: Within-week daily flow standard deviation (flashiness proxy).
   - `flow_min_7d`, `flow_max_7d`: Within-week daily baseflow minimum and peak flow maximum.
   - `flow_recession_slope`: Weekly hydrograph recession delta $Q_t - Q_{t-1}$.
2. **Antecedent Precipitation Memory**:
   - `rain_sum_1w`: 7-day cumulative rainfall.
   - `rain_sum_2w`: 14-day cumulative rainfall (`rolling(14).sum()`).
   - `rain_sum_1m`: 30-day cumulative rainfall (`rolling(30).sum()`).
   - `rain_sum_2m`: 60-day cumulative rainfall (`rolling(60).sum()`).
   - `rain_sum_3m`: 90-day seasonal cumulative rainfall (`rolling(90).sum()`).
   - `api_30d`: Decayed Antecedent Precipitation Index ($\sum_{k=0}^{29} 0.85^k P_{t-k}$ with chronologically reversed weights).
   - `dry_days_count`: Count of zero-rain days ($P < 1.0\,\text{mm}$) during the 7 days of week $t$.
3. **Atmospheric Demand & Thermal Drivers**:
   - `temp_mean_1w`: 7-day mean of daily average temperature ($T_{\text{avg}}$).
   - `temp_max_1w`: 7-day maximum of daily maximum temperature ($T_{\text{max}}$).
   - `temp_min_1w`: 7-day minimum of daily minimum temperature ($T_{\text{min}}$).
   - `pet_sum_1w`: 7-day cumulative Potential Evapotranspiration.
   - `pet_sum_1m`: 30-day cumulative Potential Evapotranspiration.
4. **Climatic Water Deficit (Catchment Drying Stress)**:
   - `water_deficit_1w`: Net weekly atmospheric water balance ($P_{1w} - \text{PET}_{1w}$).
   - `water_deficit_1m`: Net 30-day atmospheric water balance ($P_{1m} - \text{PET}_{1m}$).
5. **Calendar Seasonality & Attributes**:
   - `week_of_year`: ISO week integer (1 to 53).
   - `sin_week`, `cos_week`: Cyclical Fourier harmonics:
     $$\sin\left(\frac{2\pi \cdot \text{week}}{52.1775}\right), \quad \cos\left(\frac{2\pi \cdot \text{week}}{52.1775}\right)$$
   - `catchment_area_km2`: Topographic drainage area.
6. **Multi-Horizon Targets (Future Lead Shifts)**:
   - Volumetric targets ($y_{t+H} \in \text{m}^3/\text{s}$): `target_H1`, `target_H2`, `target_H4`, `target_H8`.
   - Specific runoff targets ($q_{t+H} \in \text{mm/week}$): `target_specific_H1`, `target_specific_H2`, `target_specific_H4`, `target_specific_H8`.

### Leakage Prevention Guarantees
* **Zero Future Weather**: Predictors terminate strictly on Monday 00:00 UTC of issue week $t$. Absolutely no meteorological observations from $t+1 \dots t+H$ enter the feature set.
* **Warmup Alignment**: The initial 7 weeks (`1980-01-01` to `1980-02-18`) are dropped to initialize the 7-week autoregressive lag window, producing exactly 2,394 aligned weekly timesteps.

---

## 6. Evaluation Protocols & Guardrails

```text
46-Year Temporal Partitioning:
|◄────────── Train (1980-02-25 to 2010-12-31) ──────────►|◄── Val (2011–2015) ──►|◄── Test (2016–2025) ──►|
|◄─────────────── Active Train ───────────────►|[H Purge]|◄ Active Val ►|[H Purge]|◄──── Final Test ─────►|
```

### 6.1 Terminal $H$-Week Purge Embargo (Approval Gate AG-TS-02)
For forecast horizon $H \in \{1, 2, 4, 8\}$ weeks:
* **Train Set**: Drops the final $H$ weeks before `2010-12-31` (`df_train.iloc[:-H]`).
* **Val Set**: Drops the final $H$ weeks before `2015-12-31` (`df_val.iloc[:-H]`).
* **Test Set**: Un-purged final 10-year out-of-sample block (`2016-01-01` to `2025-12-31`).
* **Result**: Zero target label leakage across split seams.

### 6.2 Identical Row Evaluation Contract
To eliminate missingness and selection bias across model tiers (M0 vs M1 vs M2):
$$\mathcal{M}_{\text{eval}} = \left\{ t \;\middle|\; y_t \ne \text{NaN} \;\land\; Q_t \ne \text{NaN} \;\land\; \hat{y}_{\text{M0}, t} \ne \text{NaN} \;\land\; \hat{y}_{\text{M1}, t} \ne \text{NaN} \right\}$$
All models are evaluated on the exact identical row intersection per catchment and horizon.

### 6.3 Track A: Continuous Hydrograph Metrics
* **Nash-Sutcliffe Efficiency (NSE)**:
  $$\text{NSE} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  *Guard*: In dry-season zero-variance regimes ($SS_{\text{tot}} < 10^{-6}$), returns $1.0$ if $SS_{\text{res}} < 10^{-6}$; otherwise clamps lower bound at $-10.0$ and flags `is_degenerate = True`.
* **Kling-Gupta Efficiency (KGE-2009)**:
  $$\text{KGE} = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}$$
  where $r$ is Pearson correlation, $\alpha = \sigma_{\hat{y}} / \sigma_y$ is flow variability, and $\beta = \mu_{\hat{y}} / \mu_y$ is volume bias ratio.
* **RMSE & MAE**: Computed with physical non-negativity constraint $\hat{y} = \max(\hat{y}, 0.0)$.

### 6.4 Track B: Drought Early Warning Metrics
* **Threshold Fitting Scope**: Mild ($Q_{80}$), Moderate ($Q_{90}$), and Severe ($Q_{95}$) thresholds are fitted **strictly on the purged Training fold (1980–2010)**. Never fit on Validation or Test folds.
* **Contingency Formulations**:
  - Probability of Detection: $\text{POD} = \frac{H}{H + M}$
  - False Alarm Ratio: $\text{FAR} = \frac{FA}{H + FA}$ (defined as $\text{NaN}$ if $H+FA=0$ and events occurred)
  - Critical Success Index: $\text{CSI} = \frac{H}{H + M + FA}$ (returns $1.0$ if $H+M+FA=0$, complete null agreement)

---

## 7. Model Architecture Hierarchy

```text
Model Tiers:
┌────────────────────────┬─────────────────────────────────────────────────────────────────────────────┐
│ Tier                   │ Mathematical / Feature Formulation                                          │
├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ M0: Baseline           │ H <= 2w: Pure Persistence (Q_{t+H} = Q_t)                                   │
│                        │ H >= 4w: Seasonal Weekly Climatology from Training Fold                     │
├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ M1: Flow-Only          │ Ridge (alpha-tuned) & HistGradientBoostingRegressor                         │
│                        │ Features: Past flow lags (Q_t..Q_{t-7}) + recession slope + Fourier calendar│
│                        │ Strictly 0 weather features.                                                │
├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ M2: Hydro-Meteorology  │ Multi-Source Weather Model (To be trained in next phase)                    │
│                        │ Features: Flow lags + IMD Rainfall memory + ERA5 Temp/PET + Water Deficit   │
└────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Formal Human Approval Gates Log

Every strategic, architectural, and data alteration was paused at a formal Human Approval Gate before code modification:

| Gate ID | Decision Topic | Status | Formal Decision & Evidence |
| :--- | :--- | :---: | :--- |
| **AG-01** | Macro-Basin to Sub-Basin Relaxation | **APPROVED** | Relaxed macro-basin constraint to strictly enforced sub-basin zero overlap to guarantee 46-year continuous data without synthetic multi-month gap imputation. |
| **AG-02** | Contiguous Gap Policy | **APPROVED** | Relaxed 30-day discard rule: retain stations with $<10\%$ missingness; linearly interpolate gaps $\le 35$ days; leave gaps $> 35$ days as unobserved `NaN`. |
| **AG-03** | Dropping Artificial IMD-Only Constraint | **APPROVED** | Ingested ERA5-Land reanalysis for 2021–2025 non-rainfall forcings ($T$, $\text{PET}$) to eliminate the 2020/2021 train/test structural break. |
| **AG-TS-01**| API Weight Vector Chronological Reversal | **APPROVED** | Corrected inverted array weighting in `calc_daily_api` using `weights[::-1]`, assigning weight 1.0 to day $t$ and decaying backwards. |
| **AG-TS-02**| Terminal $H$-Week Purge Embargo | **APPROVED** | Purged terminal $H$ weeks from Training (2010) and Validation (2015) sets to eliminate target leakage into subsequent folds. |

---

## 9. Alignment with Course Evaluation Rubric (IITGN ES 335)

| Course Rubric Criteria | Project Implementation & Evidence Artifact |
| :--- | :--- |
| **1. Scientific Problem Formulation** | Central research question decomposing forecast horizon decay, drought reliability, severity sensitivity, and zero-shot cross-basin transfer. |
| **2. Data Lineage & Engineering Rigor** | 46-year timeline (1980–2025) harmonizing CWC discharge, CAMELS-IND, IMD 0.25° NetCDF, and ERA5-Land. Memory operations constrained $<0.5\,\text{GB}$ via lazy loading. |
| **3. Methodological Honesty & Leakage Prevention** | Strictly backward-looking windows; terminal $H$-week purge embargo; identical row evaluation mask across missing flow intervals. |
| **4. Rigorous Baseline Benchmarking** | M0 baseline enforced: pure persistence for $H \le 2\text{w}$, climatology for $H \ge 4\text{w}$. Halt guardrail if persistence attempted $>2\text{w}$. |
| **5. Dual-Track Metric Evaluation** | Evaluated on continuous hydrograph accuracy (NSE, KGE, RMSE, MAE) and categorical drought early warning (CSI, POD, FAR). |
| **6. Reproducibility & Traceability** | Static parameterizations, deterministic seed setting, pre-cached tensor pipelines, and fully logged validation reports. |

