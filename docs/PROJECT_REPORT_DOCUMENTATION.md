# 🌊 Empirical Streamflow & Hydrological Drought Forecasting (1980–2025)
## Comprehensive Technical Documentation & Course Project Report Reference

**Author / Team:** ML Course Project Team  
**Institution:** Indian Institute of Technology Gandhinagar (IITGN) — ES 335  
**Version:** 2.1 (Audited; external review response)  
**Date of Current Audit:** November 4, 2026  
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

## 2. Experimental Matrix & Station Portfolio: 10 River Systems (9 Independent Gauges + 1 Three-Gauge Cascade)

Rigorous hydrological and spatial audit establishes the exact topological structure of the 12-catchment portfolio:
1. **9 Independent Gauges**: Nine catchments are hydrologically decoupled, non-overlapping drainage networks across the Godavari (Wainganga, Wardha), Mahanadi (Tel), Baitarani, Vamsadhara, Brahmani, Cauvery, and Tungabhadra river basins.
2. **1 Three-Gauge Nested Cascade (Upper Krishna Basin)**: Three stations form an upstream-to-downstream nested drainage hierarchy:
   - **Warunji (Gauge 4063)**: Headwater catchment on the Koyna River ($1,890.0\,\text{km}^2$) in the Western Ghats high-rainfall crest.
   - **Karad (Gauge 4021)**: Upper Krishna mainstem gauge ($5,462.0\,\text{km}^2$), directly receiving Koyna outflow. Warunji represents **34.6% of Karad's drainage area**.
   - **Huvinhedigi (Gauge 4017)**: Downstream regional trunk gauge on the peninsular Krishna in Karnataka ($55,150.0\,\text{km}^2$). Warunji comprises **3.4%** and Karad comprises **9.9%** of Huvinhedigi's upstream catchment.
   - *Audit Disclosure (Gate AG-TS-06 / Option H1)*: We explicitly retire the former 'Zero Sub-Basin Overlap' claim. Models evaluated on Warunji, Karad, and Huvinhedigi evaluate multi-scale hydrological propagation along a 55,000 km² drainage cascade rather than mutually independent catchments.

### Unmeasured Low Flows Recorded as Zero (Karad, Warunji, Yadgir, Huvinhedigi)
After about 2010 CWC operated Karad, Warunji and Yadgir as monsoon-season stations; weeks that were not measured are logged as exactly $0.0000\,\text{m}^3/\text{s}$ (see `docs/AUDIT_LOG.md`, Issue C). Share of weeks recorded as zero:

| Station | Train 1980–2010 | Val 2011–2015 | Test 2016–2025 | Test zeros falling in Jun–Oct |
| :-- | --: | --: | --: | --: |
| Karad | 2.1% | 76.6% | 69.0% | 21% |
| Warunji | 4.0% | 77.0% | 70.0% | 22% |
| Yadgir | 19.9% | 65.1% | 62.3% | 18% |
| Huvinhedigi | 0.0% | 0.0% | 14.1% | 28% |

These are missing measurements, not droughts (about a fifth fall inside the monsoon). At these four stations every week whose issue-time or target flow is a recorded zero is excluded from scoring for all models (`ZERO_AS_MISSING_STATIONS` in `scripts/audit_metrics.py`). This lowers the pooled severe-drought base rate from 0.238 to about 0.065, close to the nominal 0.05. Haralahalli's zeros (dry season only, also present in training) are kept. Drought thresholds are still fitted on the 1980–2010 training fold (Gate AG-TS-07); the models were not retrained, so their inputs still contain the recorded zeros.

**Data Lineage & Overlap Audit (2011–2020)**: To verify that pipeline stitching (`prepare_weekly_dataset.py`) did not overwrite valid discharge, we evaluated all 8,740 station-days where CWC recorded zero flow over the 2011–2020 overlap across the four affected stations. On 82.8% of those days (7,236 days), published CAMELS-IND already recorded exact 0.000 m³/s, and 17.2% (1,499 days) were unmeasured NaNs; CAMELS had positive flow on only 5 days (0.06%). This proves that post-2010 zero inflation is an upstream reporting artifact inherited directly from CWC by CAMELS-IND, rather than a pipeline merge defect.


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
* Aridity is the Budyko aridity index PET/P over 1980–2020, computed from the weekly tensors (GLEAM PET / IMD-based P); > 1.0 = water-limited.
```

### Outlier Imputation & Data Quality Protocol
To prevent anomalous clerical errors from distorting loss gradients (e.g. Ashti's unclipped peak of $1,829,351\,\text{m}^3/\text{s}$ on `05-10-2025`):
- A physical clipping ceiling of $> 50,000\,\text{m}^3/\text{s}$ is enforced.
- Values exceeding this threshold are masked to `NaN` and immediately undergo **temporal linear interpolation with a limit of $\le 2$ days** before weekly aggregation. This prevents monsoon storm peaks from being replaced with dry-season medians.

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
3. **Temperature Bias Structure**: While the arithmetic signed mean bias across all 12 stations is $-0.035\,^\circ\text{C}$ ($r = 0.947$), this reflects cancellation between positive biases in high-elevation Western Ghats headwaters (e.g., Kudige $+2.35\,^\circ\text{C}$, Warunji $+1.45\,^\circ\text{C}$) and negative biases in inland plains (e.g., Ashti $-1.47\,^\circ\text{C}$, Huvinhedigi $-1.17\,^\circ\text{C}$). The true portfolio **Mean Absolute Bias (MAB) is $0.84\,^\circ\text{C}$** ($\text{RMSE} = 1.15\,^\circ\text{C}$), which is fully corrected via station-specific additive bias calibration.

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
$$\mathcal{M}_{\text{eval}} = \left\{ t \;\middle|\; y_t, Q_t, \hat{y}_{\text{M0}, t}, \hat{y}_{\text{M1}, t}, \hat{y}_{\text{M2}, t} \ne \text{NaN} \;\land\; \lnot\big(\text{station} \in \mathcal{Z} \land (y_t = 0 \lor Q_t = 0)\big) \right\}$$
with $\mathcal{Z}$ = {Karad, Warunji, Yadgir, Huvinhedigi} (recorded zeros = unmeasured weeks, §2). All models are evaluated on the identical row set per catchment and horizon; the scored rows are flagged in `data/processed/model_results/test_predictions.csv`.

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
│ M2: Hydro-Meteorology  │ Protocol 1: HistGradientBoostingRegressor (26 hydro-met features)           │
│                        │ Protocol 2: Deep Recurrent PyTorch HydroLSTM (L=8 lookback, 64 hidden units)│
│                        │ Features: Flow lags + IMD Rainfall memory + ERA5 Temp/PET + Water Deficit   │
│                        │ Status: Fully trained, evaluated, and audited across 12 catchments.         │
└────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Formal Human Approval Gates Log

Every strategic, architectural, and data alteration was paused at a formal Human Approval Gate before code modification:

| Gate ID | Decision Topic | Status | Formal Decision & Evidence |
| :--- | :--- | :---: | :--- |
| **AG-01** | Macro-Basin to Sub-Basin Relaxation | **APPROVED** | Relaxed macro-basin constraint to distinct sub-basin networks to guarantee 46-year continuous data without synthetic multi-month gap imputation. |
| **AG-02** | Contiguous Gap Policy | **APPROVED** | Relaxed 30-day discard rule: retain stations with $<10\%$ missingness; linearly interpolate gaps $\le 35$ days; leave gaps $> 35$ days as unobserved `NaN`. |
| **AG-03** | Dropping Artificial IMD-Only Constraint | **APPROVED** | Ingested ERA5-Land reanalysis for 2021–2025 non-rainfall forcings ($T$, $\text{PET}$) to eliminate the 2020/2021 train/test structural break. |
| **AG-TS-01**| API Weight Vector Chronological Reversal | **APPROVED** | Corrected inverted array weighting in `calc_daily_api` using `weights[::-1]`, assigning weight 1.0 to day $t$ and decaying backwards. |
| **AG-TS-02**| Terminal $H$-Week Purge Embargo | **APPROVED** | Purged terminal $H$ weeks from Training (2010) and Validation (2015) sets to eliminate target leakage into subsequent folds. |
| **AG-TS-04**| Upper Physical Discharge Clipping & Interpolation | **APPROVED** | Enforced $50,000\,\text{m}^3/\text{s}$ ceiling and $\le 2$ days linear interpolation to eliminate CWC clerical typos (e.g. Ashti $1.8\times 10^6\,\text{m}^3/\text{s}$). |
| **AG-TS-05**| Tier M2 Hydro-Meteorological Models Execution | **APPROVED** | Evaluated HistGradientBoosting (Protocol 1) and PyTorch HydroLSTM (Protocol 2) under strict $H$-week purge embargoes and identical-row masks. |
| **AG-TS-06**| Basin Topology Reframe (Option H1) | **APPROVED** | Retired "Zero Overlap" claim; reframed as 10 river systems: 9 independent gauges and 1 three-gauge nested cascade ($\text{Warunji} \subset \text{Karad} \subset \text{Huvinhedigi}$). |
| **AG-TS-07**| Karad Regime Shift & Threshold Retention (Option C1) | **APPROVED** | Retained training $Q_{95}=5.994\,\text{m}^3/\text{s}$; documented post-2010 Koyna diversion/shutdown (69.2% zero-flow test weeks) and empirical 0.238 PR baseline. Scoring now treats recorded zeros at Karad, Warunji, Yadgir and Huvinhedigi as unmeasured (AUDIT_LOG §7). |
| **AG-TS-08**| Strategic Pivot to Objective Statistical Storytelling | **APPROVED** | Explicitly reported that M2 vs M1 is not statistically significant ($p > 0.20$); highlighted seasonal climatology winning at $H \ge 4\text{w}$. |

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

---

<!-- REPORT_RESULTS:START -->
## 10. Empirical Results (Audited, Zero-Flow Mask Applied)

> These sections are generated by `scripts/render_report_results.py` from `data/processed/model_results/audited_metrics_recomputed.csv` and `test_predictions.csv` (written by `scripts/audit_metrics.py`). Do not edit by hand; rerun the script.

**Evaluation set.** Test block 2016–2025 (up to 521 scored weeks per station). All models are scored on the same weeks. At Karad, Warunji, Yadgir and Huvinhedigi, weeks whose issue-time or target flow is recorded as exactly 0 m³/s are treated as unmeasured and excluded (see §2). Weeks removed per station (at the longest lead): Huvinhedigi 121, Karad 434, Warunji 439, Yadgir 400. M0 is persistence at 1–2 weeks and the training-period weekly climatology at 4–8 weeks. Δ values are paired: the median over the 12 stations of each station's own difference. *p* = Wilcoxon signed-rank test, *n* = 12.

### 10.1 Protocol 1: 12-Station Medians

| Lead | NSE M0 / M1 / M2 | ΔNSE (M2−M1), *p* | KGE M0 / M1 / M2 | ΔKGE (M2−M1), *p* | r M0 / M1 / M2 | Climatology or persistence beats M2 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **1w** | +0.284 / +0.458 / +0.423 | -0.018, 0.380 | +0.643 / +0.543 / +0.494 | -0.062, 0.007 | 0.643 / 0.686 / 0.671 | 2 of 12 |
| **2w** | -0.185 / +0.282 / +0.294 | +0.012, 0.233 | +0.408 / +0.355 / +0.354 | +0.000, 0.850 | 0.408 / 0.547 / 0.571 | 0 of 12 |
| **4w** | +0.364 / +0.273 / +0.262 | -0.017, 0.233 | +0.366 / +0.328 / +0.311 | -0.012, 0.424 | 0.616 / 0.541 / 0.548 | 11 of 12 |
| **8w** | +0.353 / +0.237 / +0.278 | +0.010, 0.622 | +0.367 / +0.327 / +0.348 | -0.003, 0.850 | 0.602 / 0.534 / 0.546 | 8 of 12 |

### 10.2 Station-by-Station NSE (Protocol 1)

Each cell: M0 / M1 GBT / M2 GBT. † = recorded zeros treated as unmeasured.

| Station | Scored weeks (1w) | 1w | 2w | 4w | 8w |
| :-- | --: | :-- | :-- | :-- | :-- |
| Karad † | 142 | -0.157 / +0.370 / +0.432 | -0.943 / +0.082 / +0.230 | +0.133 / +0.078 / -0.069 | +0.084 / +0.049 / +0.147 |
| Warunji † | 133 | -0.165 / +0.334 / +0.326 | -1.017 / +0.056 / +0.145 | +0.080 / +0.121 / +0.037 | +0.044 / +0.132 / +0.046 |
| Ashti | 521 | +0.407 / +0.525 / +0.535 | -0.011 / +0.378 / +0.366 | +0.417 / +0.365 / +0.311 | +0.416 / +0.365 / +0.321 |
| Bamni (Ballarsha) | 521 | +0.250 / +0.349 / +0.414 | -0.256 / +0.199 / +0.285 | +0.234 / +0.184 / +0.180 | +0.233 / +0.138 / +0.190 |
| Anandpur | 521 | +0.410 / +0.558 / +0.563 | +0.157 / +0.336 / +0.432 | +0.409 / +0.310 / +0.341 | +0.408 / +0.254 / +0.317 |
| Kashinagar | 521 | +0.298 / +0.473 / +0.293 | -0.069 / +0.418 / +0.403 | +0.419 / +0.413 / +0.394 | +0.417 / +0.348 / +0.361 |
| Gomlai | 510 | +0.469 / +0.578 / +0.647 | +0.258 / +0.499 / +0.490 | +0.482 / +0.457 / +0.461 | +0.481 / +0.453 / +0.480 |
| Kantamal | 507 | +0.388 / +0.444 / +0.381 | -0.030 / +0.312 / +0.304 | +0.483 / +0.319 / +0.406 | +0.481 / +0.402 / +0.298 |
| Yadgir † | 179 | +0.173 / +0.318 / +0.282 | -0.532 / +0.163 / +0.088 | -0.028 / +0.092 / +0.013 | -0.059 / -0.043 / -0.038 |
| Huvinhedigi † | 421 | +0.270 / +0.473 / +0.445 | -0.478 / +0.252 / +0.249 | +0.346 / +0.235 / +0.267 | +0.326 / +0.205 / +0.259 |
| KUDIGE | 489 | +0.314 / +0.524 / +0.450 | -0.113 / +0.407 / +0.433 | +0.381 / +0.357 / +0.257 | +0.379 / +0.359 / +0.346 |
| Haralahalli | 415 | -0.149 / +0.298 / +0.211 | -0.469 / +0.194 / +0.222 | +0.210 / +0.204 / +0.190 | +0.206 / +0.219 / +0.226 |

### 10.3 Track B: Severe Drought (Q95) Early Warning

Thresholds are the training-period (1980–2010) 5th percentile at each station. CSI and FAR are medians over the stations where they are defined (*n* in brackets). PR-AUC is average precision on all scored weeks pooled.

| Lead | Pooled base rate (events / weeks) | CSI M0 | CSI M1 | CSI M2 | FAR M1 | FAR M2 | PR-AUC M0 / M1 / M2 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **1w** | 0.065 (319 / 4880) | 0.529 (n=9) | 0.000 (n=9) | 0.000 (n=10) | 1.000 (n=3) | 0.773 (n=5) | 0.561 / 0.160 / 0.159 |
| **2w** | 0.064 (308 / 4819) | 0.379 (n=8) | 0.026 (n=10) | 0.000 (n=9) | 0.731 (n=9) | 0.695 (n=6) | 0.409 / 0.193 / 0.175 |
| **4w** | 0.066 (313 / 4725) | 0.000 (n=9) | 0.014 (n=12) | 0.000 (n=11) | 0.750 (n=11) | 0.733 (n=9) | 0.193 / 0.176 / 0.144 |
| **8w** | 0.067 (305 / 4552) | 0.000 (n=9) | 0.111 (n=12) | 0.000 (n=9) | 0.696 (n=12) | 0.660 (n=5) | 0.194 / 0.217 / 0.178 |

### 10.4 Protocol 2: Zero-Shot Transfer to Held-Out Basins (NSE)

M0 uses the held-out station's own record. Ridge and GBT are flow-only models pooled over the state's other three stations in specific runoff; the HydroLSTM adds weather inputs.

| Held-out basin | Lead | M0 | Pooled Ridge | Pooled GBT | HydroLSTM | LSTM − M0 |
| :-- | :-- | --: | --: | --: | --: | --: |
| Bamni (Ballarsha) | 1w | +0.250 | +0.244 | +0.240 | +0.339 | +0.089 |
|  | 2w | -0.256 | -0.241 | -0.466 | +0.031 | +0.287 |
|  | 4w | +0.234 | -0.393 | -0.376 | +0.153 | -0.082 |
|  | 8w | +0.233 | -0.843 | -0.260 | -1.266 | -1.499 |
| Kantamal | 1w | +0.388 | +0.529 | +0.536 | +0.514 | +0.126 |
|  | 2w | -0.030 | +0.362 | +0.437 | +0.410 | +0.440 |
|  | 4w | +0.483 | +0.348 | +0.459 | +0.446 | -0.038 |
|  | 8w | +0.481 | +0.326 | +0.432 | +0.403 | -0.078 |
| Haralahalli | 1w | -0.149 | +0.154 | +0.135 | +0.258 | +0.407 |
|  | 2w | -0.469 | +0.072 | +0.034 | -0.191 | +0.278 |
|  | 4w | +0.210 | +0.255 | +0.035 | -0.096 | -0.306 |
|  | 8w | +0.206 | +0.152 | -0.102 | -0.428 | -0.634 |

### 10.5 What the M2 Model Uses (Permutation Importance)

Computed on the scored test weeks of the 8 stations with complete low-flow records. *Group* = increase in MSE (as a fraction of the baseline MSE, station median) when every feature in the family is shuffled together, which avoids correlated features hiding each other's contribution.

| Feature family | Group ΔMSE/MSE at 1w | at 8w |
| :-- | --: | --: |
| Streamflow Memory | +22% | +1% |
| Precipitation & Recharge | +84% | +3% |
| Atmospheric Demand | +3% | +2% |
| Climatic Water Deficit | +0% | +0% |
| Calendar Seasonality | +3% | +86% |

Single-feature shares at 1 week: `api_30d` 52.7%, current flow `flow_lag_0` 4.6%, all ten flow features 22.0%. At 8 weeks the calendar harmonics take 86.9%.

## 11. Answers to the Research Sub-Questions

### 11.1 Forecast horizon decay
Persistence has a median NSE of +0.284 at 1 week and -0.185 at 2 weeks, so it is no longer useful beyond one week. The trained models keep positive skill at 2 weeks (M1 +0.282, M2 +0.294). From 4 weeks the station's weekly climatology is the strongest model (median NSE +0.364 and +0.353), beating M2 at 11 and 8 of 12 stations.

### 11.2 Value of meteorological inputs
M2 is not significantly better than M1 on NSE at any lead (*p* = 0.380, 0.233, 0.233, 0.622; M2 better at 5, 6, 4 and 8 of 12 stations). On KGE it is significantly worse at 1 week (*p* = 0.007). The importance analysis (§10.5) explains why. At 1 week M2 leans on recent rainfall rather than current flow, which carries much of the same information. At 8 weeks it relies almost entirely on the calendar, which makes it a noisier version of climatology (median r 0.546 vs 0.602).

### 11.3 Drought warning reliability and severity
No model gives reliable severe-drought warnings. M2's median Q95 CSI is 0.000 at every lead (pooled over all stations 0.03–0.11), and its pooled PR-AUC is at or below M1's at every lead (e.g. 0.175 vs 0.193 at 2 weeks). Persistence is the best warning at 1 and 2 weeks (median CSI 0.529 (n=9), 0.379 (n=8); medians are over the stations where CSI is defined, so the three models' station sets differ slightly; pooled CSI 0.48 and 0.37). At 4 and 8 weeks skill is weak but not zero. Climatology issues no alerts at all, because a weekly climatology almost never falls below the 5th-percentile flow. The best model, M1, detects 28% of severe-drought weeks at 8 weeks (pooled CSI 0.19, PR-AUC 0.22 against a base rate of 0.067), and M2 does worse (pooled CSI 0.11). Paired FAR comparisons are possible at only 2–9 stations and show no consistent reduction. Mild and moderate thresholds were not re-scored in this audit.

### 11.4 Zero-shot spatial transfer
The pooled HydroLSTM beats each held-out station's own baseline at 1 and 2 weeks (Kantamal +0.514 vs +0.388 at 1 week). At 4 and 8 weeks climatology wins at all three basins (Kantamal +0.483 / +0.481 vs +0.446 / +0.403; Bamni +0.234 vs +0.153 at 4 weeks), and at Bamni the LSTM fails badly at 8 weeks (NSE -1.266). At Kantamal the pooled three-feature flow-only GBT matches the LSTM: NSE differs by only 0.01–0.03 (GBT higher at every lead), within sampling noise at 1–4 weeks on a 52-week block bootstrap, and the LSTM has the higher KGE at 1 and 4 weeks. The weather inputs and the recurrent model therefore add no clear gain there. At Haralahalli the 2-week advantage is between two negative scores (-0.191 vs -0.469). A positive zero-shot NSE should therefore not be read as skill. The held-out basins are gauged, so the baseline uses their own history; an ungauged basin would have no such baseline.

## 12. Conclusions and Limitations

1. **Baselines are hard to beat.** Persistence wins at 1 week for drought warnings, and climatology wins at 4–8 weeks for continuous flow. Sub-seasonal models should always be reported against both.
2. **Past weather adds no significant skill** over flow history in this setup, at any lead.
3. **Severe-drought skill beyond 2 weeks is weak.** The best model (M1) catches 28% of severe-drought weeks at 8 weeks (pooled CSI 0.19), and the weather inputs make it worse, not better. Useful longer-lead warning would need information not used here (reservoir operations, groundwater, seasonal rainfall forecasts).
4. **Zero-shot transfer works only at 1–2 weeks** relative to the held-out station's own baseline.

**Limitations.**
- At Karad, Warunji and Yadgir, recorded zeros leave only 75–180 scored weeks and almost no severe-drought events. Their drought metrics are effectively undefined.
- The models were trained before the zero-flow mask was adopted. Their training data and inputs still contain the recorded zeros, which also enter the pooled LSTM training for Maharashtra and Karnataka.
- There are only 12 stations (10 river systems, three of them nested), so the Wilcoxon tests have little power.
- Bootstrap intervals (`scripts/uncertainty_tests.py`, `docs/AUDIT_LOG.md` Issue K) agree with the tests above. The portfolio 95% CI of the median ΔNSE spans zero at every lead. Station by station (52-week block bootstrap), M2 is clearly better in 4 of 48 station-lead cases and M1 in 7. Intervals for the short masked series are indicative only.

<!-- REPORT_RESULTS:END -->

---

## 13. Interactive Visualization & Web Deployment Architecture

To communicate the multi-horizon forecasts and spatial transfer results without exposing the Python backend or raw datasets, the project utilizes a static web architecture that is completely isolated from the computational pipeline.

### 13.1 Standalone Interactive Widgets
Instead of static PNGs, the evaluation pipeline generates self-contained, interactive Plotly `.html` widgets. These include:
* **Interactive Hydrographs:** Featuring zoom sliders to isolate specific drought events (e.g., the 2018–2019 monsoon failure).
* **Precision-Recall (PR) Curves:** Featuring hover data mapping the tradeoff dynamics for Severe Drought ($Q_{95}$) thresholds.
* **KGE Component Decomposition:** Visualizing the trajectories of correlation ($r$), variability ($\alpha$), and bias ($\beta$) across the $H$-week horizons.

### 13.2 Orphaned Branch Deployment Isolation
The website is a static page on an orphaned `gh-pages` branch that holds only frontend assets: `index.html`, `assets/` (CSS, JS, `benchmark-data.js`, `figdata.js`) and the chart pages in `visualizations/`. Training code, model weights and data stay on `main`.

Data flow, run by `deploy.sh`:
1. `scripts/audit_metrics.py` scores every saved model on the shared mask and writes `audited_metrics_recomputed.csv` and `test_predictions.csv`.
2. `scripts/export_benchmark_data.py` writes `assets/benchmark-data.js` (station metrics, paired medians, Wilcoxon tests).
3. `scripts/generate_interactive_plots.py` writes the Plotly chart sources to `visualizations/source/`.
4. On `gh-pages`, `tools/build_site_data.py` decodes those sources into `assets/figdata.js`, writes the standalone chart pages and regenerates the benchmark table in `index.html`. The page renders every chart from these two data files with `assets/charts.js`.
