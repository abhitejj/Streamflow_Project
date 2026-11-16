# Calendar-bootstrap diagnostics

This analysis describes variation in the saved M2-minus-M1 NSE difference. It does not change fitted models, drought thresholds, scoring masks or point scores. It does not establish that weather causes a gain.

## What is sampled

The 2016–2025 issue-date calendar has 522 weekly positions. Missing and excluded weeks stay in place. Ordinary moving blocks sample the observation, M1 prediction, M2 prediction and shared scoring mask together. The primary block spans 52 calendar weeks; 26- and 13-week blocks check sensitivity. Each setting uses 2,000 draws with fixed, reproducible seeds. A block can cross a gap; the gap remains excluded after sampling.

Within each draw, only jointly scored finite observations count. As in the audit, at least five observations are required. Constant-target draws with total squared deviation below `1e-6` are invalid for interval estimation. NSE clipping follows the audit. The observed point difference uses the audit directly, including its separate constant-series rule.

## When an interval is withheld

The observed paired point estimate must be finite. All three settings must pass these prespecified practical screens:

- At least 95% of requested draws have a finite paired NSE difference.
- Finite draws and their percentile interval have nonzero spread: both maximum minus minimum and interval width must exceed `1e-12`.
- The percentile interval has the same sign category for all three block lengths: positive, negative, or spanning zero. A category change is block sensitivity.
- Scored observations are not too concentrated in a few fixed 52-week calendar bins. If each bin has `n_b` scored observations, the diagnostic is `B_eff = (sum n_b)^2 / sum(n_b^2)`. Values below four fail the screen. Bins are anchored at the evaluation calendar's first week; the final partial bin is retained.

`B_eff` is a concentration diagnostic, **not** the number of independent river-years. The four-bin and 95%-finite thresholds are heuristics, **not** a theorem that intervals have 95% coverage. A series can pass them and still have substantial unmodelled uncertainty.

Retained 2.5th–97.5th percentile intervals are exploratory and conditional on the fitted models and the observed scoring pattern. They exclude retraining, source-selection and future missingness uncertainty. Withheld intervals have blank accepted bounds and an explicit reason; the point score remains available. Diagnostic bounds for each block length remain in the sensitivity CSV and must not be read as accepted inference.

## Portfolio and source files

Portfolio station-bootstrap intervals are withheld because shared drainage and nested gauges do not support independent-station resampling. Paired point differences and raw exploratory Wilcoxon tests remain descriptive, with a four-horizon Bonferroni reference of 0.0125. That correction does not repair station dependence.

- `scripts/uncertainty_tests.py`: method and exact seed construction.
- `data/processed/model_results/uncertainty_station_block_bootstrap.csv`: point scores, accepted bounds, diagnostics and status for 48 station–lead cases.
- `data/processed/model_results/uncertainty_block_sensitivity.csv`: all 144 station–lead–block settings.
- `data/processed/model_results/uncertainty_portfolio.csv`: portfolio points and explicit inference-withheld status.
- `data/processed/validation_reports/uncertainty_baseline_2026-11-16/`: archived earlier algorithm and outputs, superseded because it closed calendar gaps and could produce degenerate intervals.

No uncertainty comparison widget is displayed on the website. The site retains the audited flow scores and event counts, and gives an explicit detection-availability note for zero-event drought cases.
