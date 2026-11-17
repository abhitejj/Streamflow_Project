# Streamflow and Hydrological Drought Forecasting
## ES 335 project report — streamflow and drought benchmark

**Team:** Abhitej Singh Bhullar and Dhruv Jalan

**Institution:** Indian Institute of Technology Gandhinagar
**Date:** 16 November 2026

This report explains the benchmark methods, results and limitations. Numerical sections are generated from saved evaluation artifacts. The run manifest identifies the inputs, fitted models and outputs used together in this experiment.

## 1. Question, scope and unit of analysis

Does adding antecedent weather information to river-flow history improve one-, two-, four- and eight-week forecasts? Does better discharge prediction also improve low-flow drought warnings? A separate spatial-transfer experiment tests models at three gauges excluded from fitting.

One row represents **one gauge at one weekly issue date**. For each horizon H, its target is the mean measured discharge in the week ending H weeks after that date, subject to the observation-coverage rule below. Discharge Q is in m³/s. Specific runoff q = Q × 604.8 / catchment area is in mm/week and allows comparisons across catchment sizes; it is based on a weekly mean and does not establish a fully measured seven-day volume.

The study is a **retrospective hindcast** using historical observations and meteorological inputs. It does not establish a real-time service, source release availability at Monday 00:00 UTC, or causal effects of weather. Differences between trained models are predictive comparisons.

## 2. Gauges, sources and access

The portfolio contains 12 gauges in Maharashtra, Odisha and Karnataka. Warunji, Karad and Huvinhedigi form a nested cascade; other gauges share major drainages. Twelve station names do not constitute twelve independent river systems. The website and exported station metadata give gauge IDs, drainage areas and model roles.

| Input | Historical component | Later component and qualification |
| --- | --- | --- |
| Discharge | CAMELS-IND historical records | CWC manual daily records; source precedence and date matching are implemented in the dataset builder |
| Rainfall | CAMELS-IND catchment-mean forcings through 2020 | IMD 0.25° daily grids for 2021–2025, averaged using catchment/grid intersection weights |
| Temperature | CAMELS-IND catchment forcings through 2020 | Supplied Open-Meteo Archive daily mean/max/min temperature, requested at gauge coordinates |
| Evaporative demand | CAMELS-IND GLEAM series through 2020 | Open-Meteo FAO reference ET₀, magnitude-scaled using the training overlap; this is not the same physical quantity as GLEAM PET |
| Geography | CAMELS-IND attributes and catchment polygons | Static catchment area is used for runoff conversion and rainfall extraction |

The weather acquisition script omits the model-selection parameter. Open-Meteo documents its default **Best Match** as combining weather products; the supplied files are not verified homogeneous ERA5-Land data. Their filenames retain an earlier naming convention. The reconstructed request uses catalog coordinates rounded to four decimal places, daily variables, GMT and 2001–2025 dates. Exact provider-returned grid/elevation metadata were not retained, so a nearest-cell or uniform-resolution claim is not established.

All 12 supplied weather CSVs contain 9,131 daily rows for 2001–2025, with unique dates, finite fields, ordered temperature extrema and nonnegative ET₀. These are integrity checks, not independent verification of the provider's physical measurements or product identity.

Karnataka daily discharge was recovered from the processed merged file. The recreated archive matches its Karnataka original-field records apart from row order and serial numbering. The dataset builder explicitly identifies the derived input and records its hash; it does not claim recovery of the original CWC archive.

Raw inputs remain local and excluded from Git. Access and redistribution rights must be checked with the source providers. See [data requirements](DATA_ACCESS.md), the local source inventory and each tensor's provenance sidecar. The [Open-Meteo source documentation](https://open-meteo.com/en/docs/historical-weather-api) explains model selection and reference evapotranspiration.

The local CAMELS-IND disclaimer requires citation of its data-description article: [Mangukiya et al. (2025), *CAMELS-IND: hydrometeorological time series and catchment attributes for 228 catchments in Peninsular India*](https://essd.copernicus.org/articles/17/461/2025/). The article links the [dataset repository](https://doi.org/10.5281/zenodo.14005378); the local-file hashes identify the bundle actually used and should not be mistaken for verification of a particular downloaded release.

## 3. Cleaning and measured-target provenance

Invalid discharge above the existing 50,000 m³/s ceiling is masked. This ceiling is a coarse clerical-error screen, not proof of physical plausibility for every catchment. **No daily discharge or weather series is filled by two-sided temporal interpolation.**

Under the inclusion rule, a station is retained if raw daily discharge missingness over the full record is below 15%. This descriptive cohort rule is distinct from the weekly target-eligibility rule. Recorded zeros can additionally represent unmeasured flow, so the raw non-missing percentage alone does not establish usable low-flow coverage.

A weekly discharge target requires at least **four original measured daily values**. The builder records the measured-day count before any model feature imputation. Weeks below this minimum remain missing and are not used as observed targets. A four-day mean is not a complete seven-day measurement.

At Karad, Warunji, Yadgir and Huvinhedigi, the configured scoring policy excludes rows with recorded-zero issue or target discharge. The corresponding training/validation exclusion policy is applied in the trainers. Haralahalli's dry-season zeros are retained under the existing station-specific policy. Seasonal patterns support the unmeasured-zero interpretation but do not independently establish an institutional explanation for every zero.

The current station-integrity report and generated results provide observed and scored counts. Old discard reports based on the former 10% rule are historical, not current station-status decisions.

## 4. Training-only harmonization

Weather source corrections are fitted on **paired finite daily observations from 2001–2010**, entirely within the primary training period. The available rainfall overlap begins in 2001. Each fitted variable requires at least 365 paired days.

Rainfall and evaporative-demand multipliers are ratios of paired training sums. Temperature offsets are paired training mean differences. The builder rejects invalid denominators and records fit dates, paired counts, coefficients and input hashes. Later years must not change these fitted quantities.

These corrections adjust average magnitude. They do not establish identical physical formulations, remove all structural breaks, or prove that a changing weather-product blend is homogeneous. Temperature and ET₀ remain gauge-point inputs rather than catchment averages. Low permutation importance cannot prove their spatial mismatch is negligible.

The structural-break report records the training-period coefficients and fit diagnostics. Calibration uses no observations from the validation or test periods.

## 5. Weekly timing and features

Weekly bins are explicit: **left-closed and right-labelled at Monday**. A row labelled Monday t uses daily dates from the preceding Monday through Sunday; it excludes the daily record labelled t. Rolling weather windows are extracted at the final daily date of that preceding week, without skipping a missing endpoint.

For a concrete example, the row issued on Monday 4 January 2016 uses daily dates 28 December 2015–3 January 2016. Its one-week target is the measured mean over 4–10 January, labelled 11 January. This describes date bins, not a verified provider publication schedule.

Rainfall and ET₀ totals require the stated complete daily window rather than treating absent days as zero. Missing features are handled downstream using frozen training replacement values. Targets are never replaced by feature-imputation values.

Flow-history features include current and lagged weekly flow, within-week variability/extrema, recession and calendar harmonics. Weather features add trailing rainfall totals, dry-day count, temperature, ET₀, water deficit and a normalized exponentially weighted rainfall average. The API feature uses decay 0.85 and divides the weighted sum by its weight sum; it is not an unnormalized accumulated rainfall depth.

The saved tensors contain the measured-day counts, pipeline version and all horizon targets. Their hashes bind the fitted models to the dataset. Actual row and eligible-sample counts are measured from artifacts rather than inferred from the nominal 46-year span.

## 6. Baselines and model choices

**M0:** Lagged persistence at one and two weeks; training-period weekly climatology at four and eight weeks. Persistence tests short-lived flow memory; climatology tests whether the seasonal calendar already explains the apparent skill. Any additional four-week persistence diagnostic is labelled separately from official M0.

**M1:** Flow-history and calendar predictors. Ridge provides a regularized linear comparison; histogram gradient boosting captures nonlinear relationships.

**M2:** M1 predictors plus antecedent weather information. Its gradient-boosting comparison tests predictive value beyond flow history. Raw-target models minimize squared prediction error; log-target variants fit log(1 + specific runoff) to alter emphasis across the flow range. Neither objective guarantees drought-warning skill.

Ridge selects among eight regularization values using chronological validation RMSE. The M1 boosting search evaluates nine depth/regularization combinations at learning rate 0.05. M2 evaluates 27 depth/regularization/learning-rate combinations. Both use the existing 150-iteration configuration. Test scores are not used to choose these settings.

The pooled spatial-transfer model is the existing two-layer HydroLSTM with eight-week sequences, hidden size 64, maximum 50 epochs and validation early stopping. A pooled flow-only model is its comparison. Held-out gauges are excluded from fitting, but local baselines and drought thresholds use their historical records: this is not evidence of performance at an entirely ungauged site.

The fitted feature ordering, training replacement values, scaling, dataset hashes and configuration are saved with models. LSTM validation/test sequences use uninterrupted chronological feature context, including previously observed features from embargoed issue rows, while their future labels remain excluded from fitting. Training and audit inference share the same model class and sequence preprocessing.

## 7. Evaluation design and metrics

The primary split remains **1980–2010 training, 2011–2015 validation, and 2016–2025 testing**. The last H issue rows of training and validation are purged for horizon H so their future targets do not cross split boundaries. Feature-only historical context is not a fitting label.

The model comparison uses a common evaluation mask within each station/horizon: valid observed target and issue flow, required finite predictions, and the station-specific recorded-zero rule. The full forecast export also retains unscored rows; its scored flag must be respected. Reported station medians and pooled weekly scores answer different questions and are labelled separately.

- **NSE:** Squared-error skill relative to the evaluated observations' mean; one is perfect and zero matches that reference. It is not a universal comparison with the project's M0.
- **KGE:** Combines correlation, relative variability and mean bias; one is ideal. Its components help explain failures.
- **Log-NSE:** Applies the NSE calculation to log-transformed discharge to inspect performance across a different flow emphasis.
- **CSI, POD and FAR:** Event detection and false alarms at frozen training thresholds. Q80, Q90 and Q95 denote the training 20th, 10th and 5th percentiles.
- **PR-AUC:** Pooled average precision used by the published precision–recall comparison. A low base rate makes interpretation different from balanced classification.
- **Undefined ratios:** No observed events or no issued alerts can make a ratio undefined. Undefined values must not be replaced by a reported success.

Thresholds, model replacement values and scalers are training-derived. Validation controls model selection and LSTM stopping. Repeated work on an already inspected historical test block is a methodological correction, not a newly collected blind prospective test.

## 8. Interpretation, uncertainty and limitations

The strongest claims concern **comparisons on the same observed targets**. Old-versus-new results describe two pipeline versions: target windows, weather inputs and eligible rows changed. Common issue dates alone do not make them the same evaluation task. A difference between their scores cannot isolate the effect of one methodological change.

Wilcoxon tests are exploratory. For four horizon-specific comparisons of a metric, 0.0125 is a Bonferroni reference; the generated section uses current p-values. Shared drainages and nested gauges limit treating stations as independent replicates. The earlier station bootstrap inference is withdrawn. The replacement preserves weekly calendar gaps and applies explicit coverage, finite-draw, degeneracy and block-sensitivity screens. These are heuristic diagnostics; retained intervals remain exploratory and conditional on fitted models and the observed scoring pattern. Low drought-event counts still limit station-level claims even when aggregate differences appear positive.

Errors should be read alongside the replay, baseline, lead time, number of observed weeks and event counts. Negative scores and cases where the calendar or persistence wins are retained. Forecast accuracy is not evidence of a physical causal mechanism or automatically of warning usefulness.

The primary limitations are incomplete low-flow observation, possible nonrandom missingness, heterogeneous weather products, gauge-point temperature/ET₀, differing PET definitions, a small dependent portfolio and retrospective data availability. A pinned-product sensitivity study and further independent periods are useful future work, not implied completed experiments.

The supplemental historical-period sensitivity uses training issue dates in 1980–2000, validation in 2001–2004 and testing in 2005–2010. Training and validation are purged by H weeks; the test issue-date block is not purged, so its final targets can extend into early 2011. Historical meteorological features use CAMELS-IND, but discharge retains CWC precedence from 2001 where available; this is not a CAMELS-only source ablation. Its `log_nse` column uses log(Q + 0.01), unlike the primary log(1 + Q) score, and must not be directly compared with primary Log-NSE. The supplemental results are separate from the primary 2016–2025 evaluation.

## 9. Reproduction, integrity and assistance

Use the [README reproduction section](https://github.com/abhitejj/Streamflow_Project#reproduction-and-validation) for the ordered commands and verification criteria. The root requirements file records the environment used; it is not a guaranteed cross-platform lock. Raw-source access details are in [DATA_ACCESS.md](DATA_ACCESS.md).

The reproduction order is: verify sources; build all tensors; train raw and log variants and pooled models; audit saved models; recompute severity, uncertainty, feature importance and historical robustness; export benchmark/explorer/replay assets; render the report; rebuild the static site locally. A complete run manifest must bind these artifacts. Audits reject legacy or mismatched model/tensor combinations rather than silently refitting preprocessing.

The website exposes fixed precomputed outputs. It has no live model training or Python service. Browser tests and captures remain local and excluded from Git. Publication requires a separate review of the rendered site.

Antigravity and Codex assisted with coding, debugging, documentation and visualization. The authors remain responsible for scientific choices, source permissions, verification and their individual defence. AI assistance does not establish that an output is correct.

The numerical sections below are generated from the evaluation artifacts.

<!-- REPORT_RESULTS:START -->
## 10. Results and evaluation

These tables are calculated from the saved forecasts and observed targets. The chronological train/validation/test split is applied at every lead. Daily targets are never filled from future observations; a weekly discharge target requires four original daily measurements. Weekly bins explicitly exclude the labelled Monday's daily record. Source harmonisation fits only 2001–2010.

Later temperature and FAO reference ET0 inputs were supplied through Open-Meteo's historical API with its default Best Match model selection. Their filenames retain an ERA5 label for compatibility, but do not establish a homogeneous ERA5-Land product. GMT calendar-day aggregation is documented; original CWC and CAMELS observation-day timing is not established as operationally available at midnight.

### 10.1 Continuous forecast comparisons

| Lead | M0 NSE | M1 NSE | M2 NSE | Paired median M2-M1 | Raw p | Pairs | p<0.0125 | Climatology beats M2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1w | 0.2129 | 0.3578 | 0.4593 | 0.0498 | 0.1514 | 12 | False | 0 |
| 2w | -0.1906 | 0.2184 | 0.2416 | 0.0221 | 0.2661 | 12 | False | 3 |
| 4w | 0.1725 | 0.2548 | 0.2228 | -0.0137 | 0.1514 | 12 | False | 7 |
| 8w | 0.1431 | 0.1943 | 0.1812 | 0.0037 | 0.5186 | 12 | False | 8 |

M0 uses persistence at 1–2 weeks and weekly climatology at 4–8 weeks. Both additional baselines remain in the audited data at every horizon. All comparisons use the shared scored-week mask. Raw signed-rank p-values are exploratory; four lead comparisons imply a Bonferroni reference of 0.0125. Nested gauges and shared major drainages prevent treating all station results as independent experimental replications.

### 10.2 Severe-drought warning performance

| Lead | Model | Median Q95 CSI | CSI stations | Median FAR | FAR stations |
| --- | --- | --- | --- | --- | --- |
| 1w | M0 | 0.5000 | 9 | 0.3067 | 8 |
| 1w | M1 | 0.0000 | 11 | 1.0000 | 5 |
| 1w | M2 | 0.0000 | 9 | 0.7083 | 4 |
| 1w | M1-log | 0.2083 | 9 | 0.0833 | 8 |
| 1w | M2-log | 0.1538 | 9 | 0.3542 | 8 |
| 2w | M0 | 0.2857 | 9 | 0.5051 | 8 |
| 2w | M1 | 0.0192 | 10 | 0.7857 | 7 |
| 2w | M2 | 0.0000 | 10 | 1.0000 | 3 |
| 2w | M1-log | 0.0278 | 9 | 0.3261 | 6 |
| 2w | M2-log | 0.0435 | 9 | 0.5000 | 8 |
| 4w | M0 | 0.0000 | 9 | undefined | 0 |
| 4w | M1 | 0.0357 | 11 | 0.7222 | 10 |
| 4w | M2 | 0.0000 | 11 | 1.0000 | 9 |
| 4w | M1-log | 0.0278 | 9 | 0.5000 | 5 |
| 4w | M2-log | 0.0000 | 9 | 0.5000 | 3 |
| 8w | M0 | 0.0000 | 9 | undefined | 0 |
| 8w | M1 | 0.0882 | 11 | 0.7857 | 11 |
| 8w | M2 | 0.0000 | 10 | 0.7500 | 7 |
| 8w | M1-log | 0.0606 | 9 | 0.5833 | 6 |
| 8w | M2-log | 0.0526 | 9 | 0.6071 | 6 |

CSI is hits/(hits+misses+false alarms). FAR is false alarms/(hits+false alarms). Zero observed/predicted events can leave a ratio undefined; medians therefore have different station counts. Median station scores and pooled contingency scores answer different questions and must not be interchanged.

### 10.3 Scoring coverage and event counts

| station | horizon | n_eval | n_masked_zero_weeks | q95_threshold | test_q95_event_count | test_q95_base_rate | assessment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Karad | 1w | 138 | 374 | 6.0571 | 1 | 0.0072 | observed_events |
| Karad | 2w | 127 | 382 | 6.0500 | 1 | 0.0079 | observed_events |
| Karad | 4w | 109 | 395 | 6.0357 | 1 | 0.0092 | observed_events |
| Karad | 8w | 77 | 423 | 6.1457 | 1 | 0.0130 | observed_events |
| Warunji | 1w | 131 | 383 | 0.5680 | 0 | 0.0000 | no_observed_events |
| Warunji | 2w | 123 | 389 | 0.5764 | 0 | 0.0000 | no_observed_events |
| Warunji | 4w | 105 | 403 | 0.6183 | 0 | 0.0000 | no_observed_events |
| Warunji | 8w | 72 | 432 | 0.6879 | 0 | 0.0000 | no_observed_events |
| Ashti | 1w | 511 | 0 | 3.5886 | 50 | 0.0978 | observed_events |
| Ashti | 2w | 508 | 0 | 3.5864 | 49 | 0.0965 | observed_events |
| Ashti | 4w | 504 | 0 | 3.5821 | 49 | 0.0972 | observed_events |
| Ashti | 8w | 500 | 0 | 3.5736 | 50 | 0.1000 | observed_events |
| Bamni (Ballarsha) | 1w | 507 | 0 | 1.7113 | 32 | 0.0631 | observed_events |
| Bamni (Ballarsha) | 2w | 504 | 0 | 1.7109 | 32 | 0.0635 | observed_events |
| Bamni (Ballarsha) | 4w | 500 | 0 | 1.7100 | 33 | 0.0660 | observed_events |
| Bamni (Ballarsha) | 8w | 496 | 0 | 1.7068 | 33 | 0.0665 | observed_events |
| Anandpur | 1w | 513 | 0 | 4.3431 | 36 | 0.0702 | observed_events |
| Anandpur | 2w | 510 | 0 | 4.3707 | 36 | 0.0706 | observed_events |
| Anandpur | 4w | 506 | 0 | 4.3636 | 36 | 0.0711 | observed_events |
| Anandpur | 8w | 502 | 0 | 4.3493 | 36 | 0.0717 | observed_events |
| Kashinagar | 1w | 511 | 0 | 1.9511 | 9 | 0.0176 | observed_events |
| Kashinagar | 2w | 508 | 0 | 1.9511 | 9 | 0.0177 | observed_events |
| Kashinagar | 4w | 504 | 0 | 1.9509 | 9 | 0.0179 | observed_events |
| Kashinagar | 8w | 500 | 0 | 1.9519 | 9 | 0.0180 | observed_events |
| Gomlai | 1w | 505 | 0 | 10.2590 | 23 | 0.0455 | observed_events |
| Gomlai | 2w | 503 | 0 | 10.2764 | 23 | 0.0457 | observed_events |
| Gomlai | 4w | 499 | 0 | 10.2750 | 23 | 0.0461 | observed_events |
| Gomlai | 8w | 495 | 0 | 10.2862 | 23 | 0.0465 | observed_events |
| Kantamal | 1w | 496 | 0 | 1.1648 | 0 | 0.0000 | no_observed_events |
| Kantamal | 2w | 492 | 0 | 1.1646 | 0 | 0.0000 | no_observed_events |
| Kantamal | 4w | 485 | 0 | 1.1917 | 0 | 0.0000 | no_observed_events |
| Kantamal | 8w | 477 | 0 | 1.2611 | 0 | 0.0000 | no_observed_events |
| Yadgir | 1w | 71 | 153 | 0.0000 | 0 | 0.0000 | unavailable_zero_threshold |
| Yadgir | 2w | 34 | 131 | 0.0000 | 0 | 0.0000 | unavailable_zero_threshold |
| Yadgir | 4w | 132 | 141 | 0.0000 | 0 | 0.0000 | unavailable_zero_threshold |
| Yadgir | 8w | 93 | 149 | 0.0000 | 0 | 0.0000 | unavailable_zero_threshold |
| Huvinhedigi | 1w | 178 | 29 | 11.4100 | 23 | 0.1292 | observed_events |
| Huvinhedigi | 2w | 114 | 31 | 11.4086 | 26 | 0.2281 | observed_events |
| Huvinhedigi | 4w | 228 | 28 | 11.4057 | 35 | 0.1535 | observed_events |
| Huvinhedigi | 8w | 194 | 28 | 11.4289 | 33 | 0.1701 | observed_events |
| KUDIGE | 1w | 188 | 0 | 3.2697 | 11 | 0.0585 | observed_events |
| KUDIGE | 2w | 152 | 0 | 3.2687 | 11 | 0.0724 | observed_events |
| KUDIGE | 4w | 230 | 0 | 3.2668 | 11 | 0.0478 | observed_events |
| KUDIGE | 8w | 201 | 0 | 3.2629 | 10 | 0.0498 | observed_events |
| Haralahalli | 1w | 118 | 0 | 5.4269 | 23 | 0.1949 | observed_events |
| Haralahalli | 2w | 107 | 0 | 5.4267 | 23 | 0.2150 | observed_events |
| Haralahalli | 4w | 152 | 0 | 5.4264 | 17 | 0.1118 | observed_events |
| Haralahalli | 8w | 122 | 0 | 5.4259 | 16 | 0.1311 | observed_events |

12 of 48 station–lead cases have no scored severe-drought events. Detection cannot be assessed in those cases; false-alarm counts are still observable. At Yadgir the frozen Q95 threshold is zero and recorded-zero targets are excluded, so the evaluation rule makes the event structurally unavailable. These statuses do not change thresholds, masks, raw contingency counts or the reported metric definitions. Raw CSI ratios in zero-event cases are not evidence of detection skill.

Recorded-zero gaps at the four configured stations are excluded using the same rule for all models. That classification remains an inference from the monitoring pattern rather than a complete source flag for every measurement.

### 10.4 KGE components and log-target trade-offs

KGE combines correlation, variability ratio and mean bias ratio; its zero is not a universal no-skill reference. Log-target training changes error emphasis, so both ordinary NSE and Log-NSE are reported.

| horizon | m0_kge_fixed | m1_kge | m2_kge | m0_log_nse | m1_gbt_log_nse | m2_gbt_log_nse | m1_gbt_log_log_nse | m2_gbt_log_log_nse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1w | 0.6077 | 0.4674 | 0.5179 | 0.6647 | 0.2923 | 0.4268 | 0.6851 | 0.6698 |
| 2w | 0.4071 | 0.3626 | 0.3993 | 0.4614 | 0.2393 | 0.2879 | 0.6223 | 0.6356 |
| 4w | 0.3069 | 0.3961 | 0.3895 | 0.4411 | 0.2195 | 0.2165 | 0.5657 | 0.5623 |
| 8w | 0.3002 | 0.2423 | 0.3582 | 0.4513 | 0.1804 | 0.1752 | 0.5556 | 0.4827 |

### 10.5 Severity sensitivity

See drought_severity_sensitivity.csv for all station and pooled contingency counts.

### 10.6 Feature reliance

Permutation importance measures the change in prediction error when a feature or group is shuffled. It does not measure causal importance or establish that spatial aggregation errors are negligible.

| index | H1_share_pct | H8_share_pct |
| --- | --- | --- |
| flow_lag_0 | 4.7454 | 0.2791 |
| flow_lag_1 | 1.6865 | 2.6012 |
| flow_lag_2 | 3.7621 | 0.5594 |
| flow_lag_3 | 6.9064 | 1.4806 |
| flow_lag_4 | 2.3938 | 0.7534 |
| flow_lag_7 | 0.0242 | 1.3681 |
| flow_std_7d | 1.9392 | 0.9992 |
| flow_min_7d | 4.5555 | 3.0122 |
| flow_max_7d | 3.9688 | 0.0867 |
| flow_recession_slope | 2.2423 | 0.9202 |
| rain_sum_1w | 2.2661 | 0.5233 |
| rain_sum_2w | 1.4941 | 0.6308 |
| rain_sum_1m | 1.0631 | 0.6718 |
| rain_sum_2m | 3.0720 | 0.6945 |
| rain_sum_3m | 1.6869 | 0.2584 |
| api_30d | 42.4343 | 1.0989 |
| dry_days_count | 2.8522 | 0.0812 |
| temp_mean_1w | 0.8703 | 0.6682 |
| temp_max_1w | 2.0157 | 0.4687 |
| temp_min_1w | 1.8656 | 2.3565 |
| pet_sum_1w | 0.6780 | 0.1506 |
| pet_sum_1m | 0.7664 | 0.6862 |
| water_deficit_1w | 3.5283 | 0.0790 |
| water_deficit_1m | 1.9140 | 1.2436 |
| sin_week | 0.1106 | 1.5182 |
| cos_week | 1.1584 | 76.8100 |

| index | H1_dMSE_over_MSE | H8_dMSE_over_MSE |
| --- | --- | --- |
| Streamflow Memory | 0.2384 | -0.0055 |
| Precipitation & Recharge | 0.7249 | 0.0041 |
| Atmospheric Demand | 0.0091 | 0.0105 |
| Climatic Water Deficit | 0.0362 | 0.0007 |
| Calendar Seasonality | 0.0063 | 1.0608 |

### 10.7 Cross-basin transfer

| station | horizon | n_eval | m0_nse_fixed | p2_gbt_nse | p2_lstm_nse |
| --- | --- | --- | --- | --- | --- |
| Bamni (Ballarsha) | 1w | 507 | 0.1261 | 0.1291 | 0.4049 |
| Bamni (Ballarsha) | 2w | 504 | -0.2938 | -0.5863 | 0.1317 |
| Bamni (Ballarsha) | 4w | 500 | 0.2233 | -0.3139 | -0.0795 |
| Bamni (Ballarsha) | 8w | 496 | 0.2161 | -0.1928 | -0.7337 |
| Kantamal | 1w | 496 | 0.3568 | 0.5095 | 0.5912 |
| Kantamal | 2w | 492 | -0.0096 | 0.4279 | 0.4469 |
| Kantamal | 4w | 485 | 0.4862 | 0.4617 | 0.4194 |
| Kantamal | 8w | 477 | 0.4863 | 0.4427 | 0.3877 |
| Haralahalli | 1w | 118 | 0.5398 | 0.1390 | 0.0498 |
| Haralahalli | 2w | 107 | 0.1792 | 0.0907 | 0.0555 |
| Haralahalli | 4w | 152 | -0.0601 | 0.0917 | 0.0473 |
| Haralahalli | 8w | 122 | -0.0619 | 0.0269 | -0.0141 |

Each transfer baseline uses the held-out station's own history. This is transfer to an unseen gauged catchment, not a proof of operational forecasting at an ungauged site. A positive NSE alone does not imply that the transferred model beats that baseline.

## 11. Errors, uncertainty and robustness

Previous station bootstrap inference withdrawn: it compressed calendar gaps and can produce degenerate intervals for short series. Point-forecast scores are unaffected. The replacement analysis retains every weekly calendar position, including excluded or missing observations, and resamples observations, both models and their shared mask together. The original CSVs and algorithm are preserved as superseded provenance in data/processed/validation_reports/uncertainty_baseline_2026-11-16/.

The primary analysis uses 52-calendar-week moving blocks, with 26- and 13-week sensitivity checks, 2,000 draws and fixed seeds. Intervals are withheld for sparse informative coverage, too few finite draws, degeneracy, or a direction that changes with block length. The effective-bin screen and 95% finite-draw screen are practical diagnostics, not a proof of confidence-interval coverage. Any retained percentile interval is exploratory and conditional on the fitted models and observed scoring pattern; it does not include model-training or source uncertainty. Portfolio station-bootstrap intervals are withheld because shared drainage and nested gauges invalidate an independent-station interpretation. See docs/UNCERTAINTY_METHOD.md for exact screens and uncertainty_block_sensitivity.csv for all settings.

Representative errors can be inspected in the forecast replay using the saved issue and target dates. Compare missed droughts, false alarms, peak underprediction and excluded measurement gaps rather than only portfolio averages.

### 11.1 Calendar-bootstrap diagnostics

| station | horizon | n_scored | coverage_fraction | B_eff | finite_draw_fraction | status | reason | ci_lo | ci_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anandpur | 1w | 513 | 0.9828 | 10.0282 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0398 | 0.2035 |
| Anandpur | 2w | 510 | 0.9770 | 9.9839 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0197 | 0.0804 |
| Anandpur | 4w | 506 | 0.9693 | 9.9687 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0984 | 0.0866 |
| Anandpur | 8w | 502 | 0.9617 | 9.9480 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0520 | 0.0833 |
| Ashti | 1w | 511 | 0.9789 | 10.0281 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | 0.0707 | 0.2666 |
| Ashti | 2w | 508 | 0.9732 | 9.9839 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0361 | 0.0419 |
| Ashti | 4w | 504 | 0.9655 | 9.9755 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0630 | 0.0582 |
| Ashti | 8w | 500 | 0.9579 | 9.9681 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0343 | 0.0478 |
| Bamni (Ballarsha) | 1w | 507 | 0.9713 | 10.0085 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | 0.0396 | 0.1611 |
| Bamni (Ballarsha) | 2w | 504 | 0.9655 | 9.9598 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | 0.0121 | 0.1583 |
| Bamni (Ballarsha) | 4w | 500 | 0.9579 | 9.9491 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0320 | 0.0528 |
| Bamni (Ballarsha) | 8w | 496 | 0.9502 | 9.9425 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0084 | 0.0637 |
| Gomlai | 1w | 505 | 0.9674 | 9.9600 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | 0.0414 | 0.1487 |
| Gomlai | 2w | 503 | 0.9636 | 9.9496 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0503 | 0.0615 |
| Gomlai | 4w | 499 | 0.9559 | 9.9247 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0745 | 0.0120 |
| Gomlai | 8w | 495 | 0.9483 | 9.8828 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | 0.0155 | 0.0759 |
| Haralahalli | 1w | 118 | 0.2261 | 3.0737 | 0.9995 | inference_withheld | B_eff_below_4_heuristic | undefined | undefined |
| Haralahalli | 2w | 107 | 0.2050 | 2.6570 | 0.9740 | inference_withheld | B_eff_below_4_heuristic | undefined | undefined |
| Haralahalli | 4w | 152 | 0.2912 | 5.2319 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0076 | 0.1473 |
| Haralahalli | 8w | 122 | 0.2337 | 3.9025 | 1.0000 | inference_withheld | B_eff_below_4_heuristic | undefined | undefined |
| Huvinhedigi | 1w | 178 | 0.3410 | 6.0420 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.2599 | 0.3380 |
| Huvinhedigi | 2w | 114 | 0.2184 | 2.9766 | 0.9840 | inference_withheld | B_eff_below_4_heuristic | undefined | undefined |
| Huvinhedigi | 4w | 228 | 0.4368 | 8.1276 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.1849 | -0.0116 |
| Huvinhedigi | 8w | 194 | 0.3716 | 7.1989 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.9990 | -0.2534 |
| KUDIGE | 1w | 188 | 0.3602 | 4.4380 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.4412 | 0.0890 |
| KUDIGE | 2w | 152 | 0.2912 | 3.0328 | 0.9825 | inference_withheld | B_eff_below_4_heuristic | undefined | undefined |
| KUDIGE | 4w | 230 | 0.4406 | 6.2031 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0923 | 0.0710 |
| KUDIGE | 8w | 201 | 0.3851 | 5.2408 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.2601 | 0.1017 |
| Kantamal | 1w | 496 | 0.9502 | 9.9144 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | 0.0611 | 0.1639 |
| Kantamal | 2w | 492 | 0.9425 | 9.8432 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | 0.0111 | 0.1477 |
| Kantamal | 4w | 485 | 0.9291 | 9.7632 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0279 | 0.1593 |
| Kantamal | 8w | 477 | 0.9138 | 9.6407 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.1317 | 0.0109 |
| Karad | 1w | 138 | 0.2644 | 8.2800 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.1851 | 0.2362 |
| Karad | 2w | 127 | 0.2433 | 8.0766 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.2263 | 0.1935 |
| Karad | 4w | 109 | 0.2088 | 7.6701 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.3271 | 0.0124 |
| Karad | 8w | 77 | 0.1475 | 6.6394 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | 0.0709 | 0.6165 |
| Kashinagar | 1w | 511 | 0.9789 | 10.0043 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0908 | 0.1241 |
| Kashinagar | 2w | 508 | 0.9732 | 9.9500 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0802 | 0.0267 |
| Kashinagar | 4w | 504 | 0.9655 | 9.9318 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.1759 | -0.0440 |
| Kashinagar | 8w | 500 | 0.9579 | 9.9238 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.1895 | 0.0018 |
| Warunji | 1w | 131 | 0.2510 | 8.5251 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0550 | 0.2434 |
| Warunji | 2w | 123 | 0.2356 | 8.5717 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.0935 | 0.2417 |
| Warunji | 4w | 105 | 0.2011 | 8.1849 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.1390 | 0.1296 |
| Warunji | 8w | 72 | 0.1379 | 6.8031 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.1491 | 0.3287 |
| Yadgir | 1w | 71 | 0.1360 | 6.1551 | 1.0000 | inference_withheld | block_sensitive | undefined | undefined |
| Yadgir | 2w | 34 | 0.0651 | 2.7524 | 0.9780 | inference_withheld | B_eff_below_4_heuristic | undefined | undefined |
| Yadgir | 4w | 132 | 0.2529 | 9.2681 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -0.2313 | 0.0289 |
| Yadgir | 8w | 93 | 0.1782 | 8.6750 | 1.0000 | exploratory_interval | Passed prespecified heuristic screens; conditional exploratory interval only | -1.3628 | 0.5476 |

Blank/undefined interval bounds mean inference was withheld, not that the point difference was zero. No station winner-count claim is made from these exploratory diagnostics.

Historical-period sensitivity uses its separately documented chronological split; it is not the primary 2016–2025 test.

| horizon | n_eval | delta_NSE_M2_M1 | delta_NSE_M2_M1_pval | M0_median_NSE | M1_median_NSE | M2_median_NSE | M0_Q95_pooled_CSI | M1_Q95_pooled_CSI | M2_Q95_pooled_CSI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1w | 3527 | 0.0771 | 0.0005 | 0.0225 | 0.4050 | 0.4263 | 0.4480 | 0.0294 | 0.0000 |
| 2w | 3499 | 0.0189 | 0.0771 | -0.3705 | 0.2326 | 0.2515 | 0.3200 | 0.0000 | 0.0000 |
| 4w | 3525 | 0.0260 | 0.5693 | 0.2409 | 0.2531 | 0.2988 | 0.0000 | 0.0258 | 0.0686 |
| 8w | 3486 | -0.0463 | 0.0771 | 0.2386 | 0.1843 | 0.1623 | 0.0000 | 0.1094 | 0.0782 |

## 12. Conclusions and limitations

Judge each model against the appropriate baseline, lead, severity and scored rows in the tables above. Weather inputs, log-target fitting and recurrent architecture do not guarantee a gain in both flow accuracy and warning skill.

- This is a retrospective hindcast; publication latency and operational availability have not been reproduced.
- The new source is Open-Meteo default Best Match, so a homogeneous reanalysis-model claim is not supported.
- Reference ET0 scaled to historical PET remains a different physical formulation.
- Missing daily discharge is preserved. The <15% raw missingness rule retains twelve gauges; actual weekly coverage and rare drought-event counts vary.
- Shared drainages and the nested cascade limit station-level statistical independence.
- Source replacement, timing correction, interpolation removal and retraining occurred together. The before/after comparison is not an isolated causal ablation.
- Saved model bundles include fitted preprocessing and hashes of the tensors used. Full retraining can vary, especially for the LSTM; committed-output reproducibility differs from bitwise retraining reproducibility.

<!-- REPORT_RESULTS:END -->

---

## 13. Interactive Visualization & Web Deployment Architecture

The website presents exported forecasts, metrics and plots. It does not run Python, retrain models or query the raw datasets in the visitor's browser.

### 13.1 Standalone Interactive Widgets
The evaluation pipeline generates chart sources and standalone interactive HTML views. Browser rendering still depends on its JavaScript resources. These include:
* **Interactive Hydrographs:** Zoom and scored-week selection for inspecting observed discharge, saved forecasts and warning outcomes.
* **Precision-Recall (PR) Curves:** Featuring hover data mapping the tradeoff dynamics for Severe Drought ($Q_{95}$) thresholds.
* **KGE Component Decomposition:** Visualizing the trajectories of correlation ($r$), variability ($\alpha$), and bias ($\beta$) across the $H$-week horizons.

### 13.2 Orphaned Branch Deployment Isolation
The website is a static page on an orphaned `gh-pages` branch that holds only frontend assets: `index.html`, `assets/` (CSS, JS, `benchmark-data.js`, `figdata.js`) and the chart pages in `visualizations/`. Training code, model weights and data stay on `main`.

Local data flow:
1. `scripts/audit_metrics.py` scores every saved model on the shared mask and writes `audited_metrics_recomputed.csv` and `test_predictions.csv`.
2. `scripts/export_benchmark_data.py` writes `assets/benchmark-data.js` (station metrics, paired medians, Wilcoxon tests).
3. `scripts/generate_interactive_plots.py` writes the Plotly chart sources to `visualizations/source/`.
4. Explorer and replay exporters produce five-model comparisons and weekly forecast chunks from the same saved predictions.
5. In the site worktree, `tools/build_site_data.py` decodes chart sources into `assets/figdata.js`, writes standalone pages and regenerates the benchmark table. The browser uses the generated assets and replay chunks with `assets/charts.js`.

The local build, scientific checks and browser review precede any publication. `deploy.sh` also performs Git commits and a push; it is not used for preview-only work. Never copy new tables beside stale chart or replay data.
