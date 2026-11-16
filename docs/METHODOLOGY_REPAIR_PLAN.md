# Methodology repair plan and completion record

GPT-6 Astra (high effort) produced the plan; GPT-6.1 Sol (medium effort) implemented the corrections. User authorisation includes the supplied weather files, derived Karnataka discharge, the <15% raw-missingness/no-interpolation option, necessary retraining and local preview. Publication requires explicit user authorisation.

| Step | Exact implementation and intended result | Verified status |
| --- | --- | --- |
| 1 | Preserve the earlier tensors, models, results and asset hashes for comparison. | 358 legacy files saved under ignored `browser-tests/artifacts/methodology-legacy`, with SHA256 manifest. |
| 2 | Inventory all twelve weather inputs and daily discharge sources before writing corrected data. | Complete. Each weather file contains 9,131 finite daily rows, 2001–2025. The request uses Open-Meteo default Best Match, GMT daily temperature and FAO reference ET0; not verified homogeneous ERA5-Land. The Karnataka original archive was reconstructed from existing merged records; this run explicitly records the derived daily source. |
| 3 | Fit source harmonisation only on paired finite training observations in 2001–2010; save coefficients, fit bounds, counts and raw hashes. | Complete. Twelve tensor provenance JSONs and calibration report regenerated. |
| 4 | Remove future daily interpolation. Retain original observed-day counts and require four measurements per weekly flow target. | Complete; no scored target is manufactured by daily interpolation. Approved <15% raw missingness retains twelve gauges; old discard notes are marked historical. |
| 5 | Use Monday bins closed left and labelled right. Preserve missing weather in complete sums/rolling windows. | Complete; Monday/future perturbation tests pass. Retrospective interpretation retained because source observation-day and publication-delay assumptions are not validated as real-time operational availability. |
| 6 | Fit frozen training medians/scalers and save ordered features, architecture, horizon, seed and dataset hashes. Audit loads the same preprocessing and model class. | Complete. 312 tabular models and twelve HydroLSTMs retrained; all bundles verified against current tensors. |
| 7 | Build sequences from original chronological history, including embargoed feature context without fitting those labels; enforce seven-day spacing. | Complete; sequence continuity and reload parity tests pass. |
| 8 | Retrain with unchanged research question, split dates, primary metrics, architectures and tuning grids. | Complete. Five sequential training jobs at two CPU threads; no shortened grids or LSTM epoch limit. |
| 9 | Record environment, source access limits and Antigravity/Codex assistance accurately. | Complete in README, requirements and source-access/provenance notes. |
| 10 | Recompute every metric/export and regenerate current prose from one corrected run, then inspect locally before publication. | Audit, severity, importance, uncertainty, historical sensitivity, benchmark/explorer/replay exports and chart sources complete. Local browser review and user publication approval remain required. |

## Executed training sequence

1. `python3 scripts/prepare_weekly_dataset.py --derived-cwc data/processed/merged/cwc_imd_merged_2001_2025.csv`
2. `python3 scripts/train_forecast_models.py --protocol all --model all --target raw`
3. `python3 scripts/train_forecast_models.py --protocol 1 --model all --target log`
4. `python3 scripts/train_m2_models.py --protocol 1 --target raw`
5. `python3 scripts/train_m2_models.py --protocol 1 --target log`
6. `python3 scripts/train_m2_models.py --protocol 2 --epochs 50 --device cpu`

Training took approximately 40, 35, 103, 114 and 58 seconds for steps 2–6 in this installed environment. All processes completed successfully. Timing is not a performance guarantee on another machine.

Next stages were the full audit; severity analysis; chart/feature-importance generation; block-bootstrap uncertainty; historical sensitivity; benchmark/explorer/replay exports; and generated report/README results. Exact commands are in README. Local logs and timings remain in ignored `browser-tests/artifacts/corrected-training/`.

## Checkable completion criteria

- Ten scientific regression tests pass, including actual tabular training-to-audit bundle reload parity and shared LSTM architecture/preprocessing parity.
- All twelve corrected tensors have version 2 and 2,394 weekly rows; weekly targets with fewer than four original observations are missing.
- All 324 models bind to the corrected tensor hashes; incompatible legacy bundles stop evaluation.
- The run validator checks all 48 station–lead target/date/count contracts and verifies recorded raw-source hashes, then records model/code/output hashes in `data/processed/validation_reports/corrected_run_manifest.json`.
- No change is published until the updated static site is inspected and the user confirms.

## Interpretation boundary

This is a new retrospective run, not evidence that one isolated repair improved accuracy. Weather source, temporal windows, missing-data treatment and scoring membership changed together. Common issue dates can therefore refer to different targets in the two runs. Comparisons are descriptive. Open-Meteo source-model heterogeneity, reference-ET0/PET differences, gauge dependence, few-event uncertainty and unverified operational latency remain explicit limitations.
