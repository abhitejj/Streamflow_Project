# Superseded uncertainty baseline — 16 November 2026

This archive preserves the previous uncertainty outputs and algorithm byte for byte. They are retained for provenance, **not authoritative current station-level inference**.

Previous station bootstrap inference was withdrawn because the algorithm compressed calendar gaps and can produce degenerate intervals for short series. Point-forecast scores are unaffected. A revised uncertainty analysis has not been implemented or run; it awaits separate authorisation.

`baseline_manifest.json` identifies archived originals with SHA256 hashes and records the unchanged model, tensor, prediction, threshold/severity, primary-metric and benchmark-data artifacts. `corrected_run_manifest.baseline.json` preserves the earlier full-run receipt as a historical reference; it predates this withdrawal notice.

The original files at `data/processed/model_results/uncertainty_*.csv` and `scripts/uncertainty_tests.py` remain unchanged. Their continued presence does not restore the withdrawn inference. No replacement winner counts or uncertainty verdicts are asserted in this step.
