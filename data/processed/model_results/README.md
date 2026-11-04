# Model results

**Published numbers come only from `audited_metrics_recomputed.csv` and `test_predictions.csv`.** Both are written by `scripts/audit_metrics.py`, which rescores every saved model in `models/` on one shared evaluation mask:

- Every model is scored on the same test weeks (2016–2025).
- At Karad, Warunji, Yadgir and Huvinhedigi, weeks whose issue-time or target flow is recorded as exactly 0 m³/s count as unmeasured and are excluded (`ZERO_AS_MISSING_STATIONS`).
- M0 is persistence at 1–2 weeks and the training-period weekly climatology at 4–8 weeks.

| File | Written by | Use |
|---|---|---|
| `audited_metrics_recomputed.csv` | `scripts/audit_metrics.py` | One row per station × lead: NSE, KGE (r, α, β), RMSE, MAE, Q95 contingency, PR-AUC, zero-shot scores |
| `test_predictions.csv` | `scripts/audit_metrics.py` | Every test-week forecast from every model, with the `scored` / `zero_as_missing` flags; source for all charts |
| `feature_importance_pooled.csv`, `feature_importance_groups.csv` | `scripts/generate_interactive_plots.py` | M2 permutation importance over the 8 stations with complete low-flow records |
| `protocol1_temporal_results_*.csv`, `protocol2_spatial_results_*.csv` | training scripts | Training-time logs only. They predate the zero-flow mask and the corrected climatology, so don't cite them. |
