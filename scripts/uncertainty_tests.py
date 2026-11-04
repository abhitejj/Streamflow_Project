#!/usr/bin/env python3
"""
Uncertainty quantification for M2 (hydro-met GBT) vs M1 (flow-only GBT), Issue K in docs/AUDIT_LOG.md.

Uses only the audited outputs of scripts/audit_metrics.py, i.e. the shared evaluation mask
(recorded zeros at Karad / Warunji / Yadgir / Huvinhedigi excluded):
  data/processed/model_results/audited_metrics_recomputed.csv
  data/processed/model_results/test_predictions.csv

1. Wilcoxon signed-rank test of station NSE, M2 vs M1, n = 12, per lead.
2. Portfolio CI: bootstrap over stations (resample the 12 stations with replacement, 1,000
   times) of the median paired ΔNSE. Captures between-station uncertainty.
3. Station CI: moving-block bootstrap over each station's scored test weeks (52-week blocks,
   1,000 resamples) of ΔNSE = NSE(M2) − NSE(M1). Captures within-station sampling uncertainty
   while keeping seasonal autocorrelation inside each block. Scored weeks are treated as one
   contiguous sequence (gaps from excluded weeks are closed up).

Outputs:
  data/processed/model_results/uncertainty_portfolio.csv
  data/processed/model_results/uncertainty_station_block_bootstrap.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "processed" / "model_results"
HS = ["1w", "2w", "4w", "8w"]
N_BOOT, BLOCK, SEED = 1000, 52, 42


def nse(y, yhat):
    """NSE per row of yhat (2-D: resamples x weeks) against y (same shape); clipped like the audit engine."""
    yhat = np.maximum(yhat, 0.0)
    ss_res = ((y - yhat) ** 2).sum(axis=-1)
    ss_tot = ((y - y.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    return np.clip(1.0 - ss_res / np.where(ss_tot < 1e-6, np.nan, ss_tot), -10.0, 1.0)


def block_indices(n, rng):
    """Moving-block bootstrap indices: N_BOOT x n."""
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, max(n - BLOCK, 0) + 1, size=(N_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(min(BLOCK, n))[None, None, :]).reshape(N_BOOT, -1)
    return idx[:, :n]


def main():
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(RES / "audited_metrics_recomputed.csv")
    pr = pd.read_csv(RES / "test_predictions.csv")

    port = []
    for h in HS:
        s = d[d.horizon == h]
        diff = (s.m2_nse - s.m1_nse).values
        w = wilcoxon(s.m2_nse, s.m1_nse)
        boot = np.median(diff[rng.integers(0, len(diff), size=(N_BOOT, len(diff)))], axis=1)
        port.append({
            "horizon": h, "median_paired_dNSE": np.median(diff),
            "diff_of_medians": s.m2_nse.median() - s.m1_nse.median(),
            "wilcoxon_W": w.statistic, "wilcoxon_p": w.pvalue, "n_m2_better": int((diff > 0).sum()),
            "boot_ci_lo": np.percentile(boot, 2.5), "boot_ci_hi": np.percentile(boot, 97.5),
        })
    port = pd.DataFrame(port)

    stn = []
    for (station, h), g in pr[pr.scored].groupby(["station", "horizon"], sort=False):
        y, m1, m2 = g.y_obs.values, g.m1_gbt.values, g.m2_gbt.values
        point = nse(y, m2) - nse(y, m1)
        idx = block_indices(len(y), rng)
        dboot = nse(y[idx], m2[idx]) - nse(y[idx], m1[idx])
        lo, hi = np.nanpercentile(dboot, [2.5, 97.5])
        stn.append({"station": station, "horizon": h, "n_scored": len(y), "dNSE": float(point),
                    "ci_lo": lo, "ci_hi": hi,
                    "verdict": "M2 better" if lo > 0 else ("M1 better" if hi < 0 else "not distinguishable")})
    stn = pd.DataFrame(stn)

    port.to_csv(RES / "uncertainty_portfolio.csv", index=False, float_format="%.4f")
    stn.to_csv(RES / "uncertainty_station_block_bootstrap.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(port.round(4).to_string(index=False))
    print(stn.round(3).to_string(index=False))
    print("\nVerdict counts by lead:\n", stn.groupby(["horizon", "verdict"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
