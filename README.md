# Streamflow & Hydrological Drought Forecasting

A retrospective machine-learning benchmark at 12 Indian river gauges, comparing flow history, weather inputs and simple baselines at one-, two-, four- and eight-week leads.

**Corrected run, 16 November 2026:** all twelve weekly datasets, 312 tabular models and twelve HydroLSTMs have been rebuilt/retrained. Point metrics, severity analysis, historical sensitivity, chart sources and exports have been regenerated. The earlier station bootstrap is archived and withdrawn. Its replacement preserves calendar gaps and withholds inference when reliability checks fail; retained intervals are exploratory. Earlier outputs are a different run and must not be mixed with these artifacts.

## Question and scope

Does adding antecedent rainfall, temperature and atmospheric-demand information improve weekly streamflow forecasts over flow history alone, and do those improvements also produce better drought warnings? A row represents one gauge at a Monday-labelled forecast issue time. Targets are weekly mean discharge at the selected future lead, expressed in m³/s; specific runoff in mm/week is used for area-normalised and log-target models.

This is a retrospective hindcast. A Monday-labelled row aggregates the preceding Monday through Sunday, excluding the labelled Monday's daily record. Source measurement-day conventions and publication delays have not been demonstrated to support a midnight real-time forecast. Prediction does not establish causation.

## Data and provenance

- CAMELS-IND historical observed flow and catchment forcings support the historical record.
- CWC daily discharge is combined with CAMELS observations. The code uses CWC where available and CAMELS as the fallback; the precise selection is preserved in the rebuild code and source hashes.
- Rainfall after 2020 uses IMD 0.25° grids and catchment-area weighting.
- The supplied later temperature/reference-ET0 inputs come through the Open-Meteo Historical Weather API using **default Best Match**, GMT daily aggregation, and FAO reference evapotranspiration. The request omitted an explicit model selector; these inputs are **not verified pure ERA5-Land**. Existing ERA-named filenames are retained for compatibility.
- The Karnataka CWC archive was reconstructed from existing merged daily records. The corrected run explicitly uses the derived daily table and records its hash and lineage limitation; its rainfall column is not used.

See [source access and limitations](docs/DATA_ACCESS.md), [weather-source inventory](docs/WEATHER_SOURCE_PROVENANCE.md), the [input manifest](data/processed/validation_reports/weather_input_manifest.json), and each tensor's provenance JSON. Raw data stays local. Rights to redistribute source datasets are not asserted.

The portfolio includes shared Godavari and Krishna drainages and nested Warunji–Karad–Huvinhedigi gauges; twelve station names do not establish twelve independent river systems.

## Corrected preprocessing

**CAMELS-IND citation:** [Mangukiya et al. (2025), CAMELS-IND data description](https://essd.copernicus.org/articles/17/461/2025/), with the [dataset repository](https://doi.org/10.5281/zenodo.14005378). The local bundle's exact release version has not been independently established; the run records the actual input-file hashes rather than assigning it a release identity from this citation.

- No two-sided daily flow or weather interpolation. Missing observations remain missing.
- Weekly flow requires at least four original daily measurements, with the observed count retained. Complete seven-day rain/ET0 sums and complete rolling windows preserve missingness.
- The approved **<15% raw daily missingness** rule retains all twelve gauges. KUDIGE and Haralahalli exceed the earlier <10% rule; their former discard reports are marked historical. Weekly scoring coverage still varies.
- Source harmonisation is fitted only on paired finite daily observations during **2001–2010**, before validation and testing. Scaling reference ET0 does not make it physically equivalent to GLEAM PET.
- Predictors missing at inference use frozen training medians. Scalers and feature order are saved with model bundles; LSTM preprocessing is reused unchanged by the auditor.
- At four configured stations, recorded zeros are treated as unmeasured according to the documented monitoring-pattern inference. Training/validation masking and shared evaluation exclusions are retained. Severe-drought thresholds remain fitted on the training targets under the existing definition.
- `api_30d` is a normalised exponentially weighted precipitation average: sum(0.85^k × rainfall[t−k]) / sum(0.85^k), over thirty days. It is a proxy, not measured soil moisture.

## Models and evaluation

M0 uses lagged persistence at one and two weeks and training-period weekly climatology at four and eight weeks. Both baselines are also retained separately in diagnostic outputs.

M1 uses flow and calendar features, comparing Ridge and gradient boosting. M2 gradient boosting adds rainfall, temperature, reference-ET0 and deficit features. Log-target variants fit log(1 + specific runoff); both ordinary NSE and Log-NSE are reported. Protocol 2 pools three training gauges per state and transfers an eight-week-lookback, two-layer HydroLSTM to the fourth gauge. Its local baseline still uses that held-out gauge's history, so this is not an ungauged operational test.

Chronological splits remain **1980–2010 train**, **2011–2015 validation**, **2016–2025 test**. The final H rows of training and validation are purged for an H-week forecast. Hyperparameters/early stopping use validation only; final test results are not used to select the models. Sequence buffers retain chronological historical features without fitting embargoed targets.

NSE measures squared flow error relative to the test mean. KGE combines correlation, variability and bias. Drought thresholds Q80/Q90/Q95 are the training 20th/10th/5th percentiles. CSI, POD, FAR and average precision assess warning behaviour. Undefined ratios are distinct from zero skill. Paired comparisons use the same scored observations within each station and lead.

<!-- README_RESULTS:START -->
### Corrected retrospective results (2016–2025 test block)

Generated from newly audited models and observed targets. M0 is lagged persistence at one and two weeks, weekly climatology at four and eight weeks. NSE differences below are paired station differences, not differences between portfolio medians. Weather includes catchment-mean IMD rainfall and calibrated Open-Meteo default Best Match temperature/reference ET0 after 2020; it is not verified pure ERA5-Land.

| Lead | M0 NSE | M1 NSE | M2 NSE | Paired median M2-M1 | Raw p | Pairs | p<0.0125 | Climatology beats M2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1w | 0.2129 | 0.3578 | 0.4593 | 0.0498 | 0.1514 | 12 | False | 0 |
| 2w | -0.1906 | 0.2184 | 0.2416 | 0.0221 | 0.2661 | 12 | False | 3 |
| 4w | 0.1725 | 0.2548 | 0.2228 | -0.0137 | 0.1514 | 12 | False | 7 |
| 8w | 0.1431 | 0.1943 | 0.1812 | 0.0037 | 0.5186 | 12 | False | 8 |

The p-values are exploratory and uncorrected. The p<0.0125 column is a four-horizon Bonferroni reference, not evidence of independent gauges. Shared drainages and nested gauges still limit inference. Prediction differences do not establish causation.

**Warning performance:** Compare severity-specific counts and the five-model tables in the report/explorer. Good continuous flow scores do not establish good drought-warning performance. Undefined ratios are not zero skill.

**Scope:** These are retrospective forecasts; real-time data publication delays are not simulated. Sources, preprocessing, missingness, and models changed together in this repair, so old/new score changes cannot be attributed to one cause.

<!-- README_RESULTS:END -->

## Reproduction and validation

The installed Python 3.14.4 environment is recorded in `requirements.txt`. This is an observed environment specification, not a platform-independent lock file. No packages were installed during this repair.

After acquiring the documented inputs and retaining source provenance, run these stages in order:

```sh
python3 -m pytest tests -q
python3 scripts/prepare_weekly_dataset.py --derived-cwc data/processed/merged/cwc_imd_merged_2001_2025.csv
python3 scripts/train_forecast_models.py --protocol all --model all --target raw
python3 scripts/train_forecast_models.py --protocol 1 --model all --target log
python3 scripts/train_m2_models.py --protocol 1 --target raw
python3 scripts/train_m2_models.py --protocol 1 --target log
python3 scripts/train_m2_models.py --protocol 2 --epochs 50 --device cpu
python3 scripts/audit_metrics.py
python3 scripts/audit_drought_sensitivity.py
python3 scripts/generate_interactive_plots.py
python3 scripts/uncertainty_tests.py --write-analysis
python3 scripts/robustness_camels_era.py
python3 scripts/export_benchmark_data.py
python3 scripts/export_explorer_data.py
python3 scripts/export_replay_data.py --assets-dir assets
python3 scripts/render_report_results.py
python3 scripts/validate_corrected_run.py
```

Dataset and model stages overwrite processed artifacts; preserve a previous run before executing them. Retraining, especially the LSTM, can vary. Inspecting saved outputs differs from reproducing every training bit.

The validator binds the input/tensor/model/output hashes and checks all 48 station–lead targets and scoring counts. Regression tests cover week boundaries, missing observation counts, calibration-period isolation, frozen preprocessing, model reload parity, sequence continuity, source opt-in, model/tensor mismatch rejection, calendar-bootstrap screening and drought-assessment availability. Browser tests and screenshots remain local under ignored `browser-tests/`.

The static website is maintained in the separate `gh-pages` worktree at `site/`. Rebuild its data locally from the newly generated sources; do not run `deploy.sh` while preparing an unapproved preview, because that script commits and pushes.

## Limitations and interpretation

Shared drainages, rare drought events, missing observations, changing monitoring regimes and the mixed weather-source selection limit generalisation. Permutation importance describes changes in prediction error, not causal importance or proof that point-versus-catchment aggregation is negligible. Previous station bootstrap inference is withdrawn: it compressed calendar gaps and can produce degenerate intervals for short series. Point-forecast scores are unaffected. The replacement calendar bootstrap reports coverage, finite draws, block-length sensitivity and explicit withholding reasons. Its screens are heuristic, and retained intervals are exploratory rather than a guarantee of 95% coverage. Original outputs and the algorithm are preserved in `data/processed/validation_reports/uncertainty_baseline_2026-11-16/` as superseded provenance. Portfolio bootstrap intervals are withheld because stations share drainage systems. Raw exploratory p-values remain separately reported. See [uncertainty method](docs/UNCERTAINTY_METHOD.md). Severe-drought detection is not assessable where scored event counts are zero; at Yadgir a zero Q95 threshold plus the recorded-zero exclusion makes that event structurally unavailable.

The before/after runs changed weather source, week boundaries, interpolation, preprocessing and scoring membership together. Common issue dates can have different weekly targets. A change in score therefore cannot be attributed solely to the repairs.

## AI assistance disclosure

Antigravity and Codex assisted with coding, debugging, documentation, and visualisation. The authors remain responsible for submitted code, interpretations and results. Important outputs require independent checking; use of these tools is not itself evidence that a scientific claim is verified.


See the [repair plan and completion record](docs/METHODOLOGY_REPAIR_PLAN.md) and [technical report](docs/PROJECT_REPORT_DOCUMENTATION.md). The [project decisions](docs/DECISION_LOG.md) give a short explanation of the main choices and why they were made.
