#!/usr/bin/env python3
"""Exploratory paired moving-block bootstrap on a complete weekly calendar.

Missing/excluded weeks retain their calendar positions. The common scoring mask,
observations and both model predictions are sampled with the same block indices.
Primary blocks span 52 calendar weeks; 26/13-week blocks are sensitivity settings.
Reliability screens are explicitly heuristic, not a coverage theorem. Intervals
are conditional on fitted models and the observed scoring pattern; they do not
include retraining uncertainty. Default output stays isolated. --write-analysis
explicitly replaces current analysis CSVs with screened results and diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data/processed/model_results"
DEFAULT_OUTPUT = ROOT / "browser-tests/artifacts/uncertainty-step2"
BLOCKS, N_BOOT, SEED = (52, 26, 13), 2000, 42


def evaluation_calendar(predictions):
    """Common issue-date range from the complete prediction table, not scored rows."""
    dates = pd.to_datetime(predictions.issue_date)
    if dates.empty or dates.isna().any():
        raise ValueError("Prediction issue dates must be present")
    if not (dates.dt.normalize() == dates).all() or not dates.dt.dayofweek.eq(0).all():
        raise ValueError("Issue dates must be Monday-labelled calendar dates")
    return pd.date_range(dates.min(), dates.max(), freq="7D")


def calendar_arrays(group, calendar):
    """Reindex one station/lead to the shared calendar without closing gaps."""
    calendar = pd.DatetimeIndex(calendar)
    if len(calendar) < 2 or not (np.diff(calendar.values) == np.timedelta64(7, "D")).all():
        raise ValueError("Calendar must be a complete consecutive seven-day grid")
    frame = group.copy()
    frame["issue_date"] = pd.to_datetime(frame.issue_date)
    if frame.issue_date.duplicated().any():
        raise ValueError("Duplicate station/lead issue dates")
    if not frame.issue_date.isin(calendar).all():
        raise ValueError("Station issue dates fall outside the evaluation calendar")
    if frame.scored.isna().any() or not frame.scored.isin([True, False]).all():
        raise ValueError("The shared scored mask must contain Boolean values")
    frame = frame.set_index("issue_date").reindex(calendar)
    values = frame[["y_obs", "m1_gbt", "m2_gbt"]].to_numpy(dtype=float)
    mask = frame.scored.fillna(False).to_numpy(dtype=bool)
    if not np.isfinite(values[mask]).all():
        raise ValueError("Scored observations and paired predictions must be finite")
    return values[:, 0], values[:, 1], values[:, 2], mask


def block_indices(n_calendar, rng, block=52, n_boot=N_BOOT):
    """Ordinary moving blocks, sampled on calendar positions including missing weeks."""
    if block < 1 or n_boot < 1:
        raise ValueError("Block length and draw count must be positive")
    if n_calendar <= block:
        raise ValueError("Calendar length must exceed block length; supply the complete evaluation calendar, not the scored-week count")
    n_blocks = int(np.ceil(n_calendar / block))
    starts = rng.integers(0, n_calendar - block + 1, size=(n_boot, n_blocks))
    indices = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)
    return indices[:, :n_calendar]


def observed_delta(y, m1, m2, mask):
    """Point estimate uses the unchanged audit metric, including its constant-series rule."""
    from audit_metrics import compute_continuous_metrics
    return float(compute_continuous_metrics(y[mask], m2[mask])["NSE"]
                 - compute_continuous_metrics(y[mask], m1[mask])["NSE"])


def bootstrap_delta(y, m1, m2, mask):
    """Mechanical paired NSE differences; empty, short or constant draws are NaN.

    The five-observation arithmetic minimum is the existing audit engine's n<5
    rule, not a new eligibility decision. This function does not implement an
    inferential eligibility threshold or a usable-draw percentage rule.
    """
    y, m1, m2 = map(lambda value: np.asarray(value, dtype=float), (y, m1, m2))
    mask = np.asarray(mask, dtype=bool)
    if not (y.shape == m1.shape == m2.shape == mask.shape):
        raise ValueError("Observations, predictions and scoring mask must have matching shapes")
    valid = mask & np.isfinite(y) & np.isfinite(m1) & np.isfinite(m2)
    counts = valid.sum(axis=-1)
    selected_y = np.where(valid, y, 0.)
    means = selected_y.sum(axis=-1) / np.maximum(counts, 1)
    total = np.where(valid, (y - means[..., None]) ** 2, 0.).sum(axis=-1)
    residual1 = np.where(valid, (y - np.maximum(m1, 0.)) ** 2, 0.).sum(axis=-1)
    residual2 = np.where(valid, (y - np.maximum(m2, 0.)) ** 2, 0.).sum(axis=-1)
    computable = (counts >= 5) & (total >= 1e-6)
    divisor = np.where(computable, total, np.nan)
    score1 = np.clip(1. - residual1 / divisor, -10., 1.)
    score2 = np.clip(1. - residual2 / divisor, -10., 1.)
    return score2 - score1


def paired_calendar_bootstrap(y, m1, m2, mask, block=52, seed=SEED, n_boot=N_BOOT):
    indices = block_indices(len(y), np.random.default_rng(seed), block, n_boot)
    draws = bootstrap_delta(y[indices], m1[indices], m2[indices], mask[indices])
    return observed_delta(y, m1, m2, mask), draws


def effective_calendar_blocks(mask, block=52):
    """Concentration diagnostic from jointly scored counts in anchored calendar bins.

    This is not an estimate of an independently sampled number of river-years.
    """
    mask = np.asarray(mask, dtype=bool)
    counts = np.array([mask[start:start + block].sum() for start in range(0, len(mask), block)], dtype=float)
    return float(counts.sum() ** 2 / np.square(counts).sum()) if counts.any() else 0.


def interval_category(lo, hi):
    if not np.isfinite([lo, hi]).all():
        return 'undefined'
    return 'positive' if lo > 0 else ('negative' if hi < 0 else 'spans_zero')


def assess_reliability(diagnostics, b_eff):
    """Apply prespecified conservative screens; no nominal coverage guarantee."""
    if set(diagnostics.block_calendar_weeks) != set(BLOCKS) or len(diagnostics) != len(BLOCKS):
        raise ValueError('All three block settings are required for reliability screening')
    reasons = []
    if not np.isfinite(diagnostics.point_dNSE).all():
        reasons.append('observed_point_not_computable')
    if b_eff < 4.:
        reasons.append('B_eff_below_4_heuristic')
    if (diagnostics.finite_draw_fraction < .95).any():
        reasons.append('finite_draw_fraction_below_95_percent')
    if diagnostics.degenerate.any():
        reasons.append('degenerate_draw_distribution')
    categories = diagnostics.interval_category.tolist()
    sensitive = len(set(categories)) != 1
    if sensitive:
        reasons.append('block_sensitive')
    return ('inference_withheld' if reasons else 'exploratory_interval',
            '; '.join(reasons) if reasons else 'Passed prespecified heuristic screens; conditional exploratory interval only', sensitive)


def screened_station_summary(diagnostics):
    rows = []
    for (station, horizon), group in diagnostics.groupby(['station', 'horizon'], sort=True):
        primary = group[group.block_calendar_weeks == 52].iloc[0]
        status, reason, sensitive = assess_reliability(group, float(primary.B_eff))
        accepted = status == 'exploratory_interval'
        verdict = ('exploratory interval above zero' if primary.provisional_p025 > 0 else
                   ('exploratory interval below zero' if primary.provisional_p975 < 0 else 'exploratory interval includes zero')) if accepted else 'inference withheld'
        row = {'station': station, 'horizon': horizon, 'n_scored': int(primary.n_scored),
               'dNSE': primary.point_dNSE, 'ci_lo': primary.provisional_p025 if accepted else np.nan,
               'ci_hi': primary.provisional_p975 if accepted else np.nan, 'verdict': verdict,
               'status': status, 'reason': reason, 'calendar_weeks': int(primary.calendar_weeks),
               'coverage_fraction': primary.coverage_fraction, 'block_calendar_weeks': 52,
               'n_boot': int(primary.draws_requested), 'finite_draws': int(primary.finite_draws),
               'finite_draw_fraction': primary.finite_draw_fraction, 'B_eff': primary.B_eff,
               'degenerate': bool(group.degenerate.any()), 'block_sensitive': sensitive,
               'interval_scope': 'Exploratory percentile; conditional on fitted models/scoring pattern; no 95% coverage guarantee'}
        for item in group.itertuples():
            row[f'diagnostic_p025_{item.block_calendar_weeks}'] = item.provisional_p025
            row[f'diagnostic_p975_{item.block_calendar_weeks}'] = item.provisional_p975
            row[f'finite_fraction_{item.block_calendar_weeks}'] = item.finite_draw_fraction
        rows.append(row)
    return pd.DataFrame(rows)


def run(predictions, seed=SEED, n_boot=N_BOOT):
    calendar = evaluation_calendar(predictions)
    summaries, raw_draws = [], {}
    for group_index, ((station, lead), group) in enumerate(predictions.groupby(["station", "horizon"], sort=True)):
        arrays = calendar_arrays(group, calendar)
        for block in BLOCKS:
            draw_seed = np.random.SeedSequence([seed, group_index, block])
            point, draws = paired_calendar_bootstrap(*arrays, block=block, seed=draw_seed, n_boot=n_boot)
            finite = draws[np.isfinite(draws)]
            lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
            degenerate = len(finite) == 0 or float(np.ptp(finite)) <= 1e-12 or hi - lo <= 1e-12
            key = f"case_{group_index}_block_{block}"
            raw_draws[key] = draws
            summaries.append({"station": station, "horizon": lead, "calendar_weeks": len(calendar),
                              "n_scored": int(arrays[3].sum()), "block_calendar_weeks": block,
                              "point_dNSE": point, "draws_requested": n_boot,
                              "finite_draws": len(finite), "provisional_p025": lo,
                              "provisional_p975": hi, "draw_array_key": key,
                              "finite_draw_fraction": len(finite) / n_boot,
                              "B_eff": effective_calendar_blocks(arrays[3]),
                              "coverage_fraction": float(arrays[3].mean()),
                              "degenerate": degenerate, "interval_category": interval_category(lo, hi)})
    return pd.DataFrame(summaries), raw_draws


def portfolio_point_summary(audit):
    """Retain descriptive points/raw tests, withholding dependent-station bootstrap CI."""
    from scipy.stats import wilcoxon
    rows = []
    for horizon, group in audit.groupby('horizon', sort=True):
        paired = group[['m1_nse', 'm2_nse']].dropna()
        difference = paired.m2_nse - paired.m1_nse
        test = wilcoxon(difference) if (difference != 0).any() else None
        rows.append({'horizon': horizon, 'median_paired_dNSE': float(difference.median()),
                     'diff_of_medians': float(paired.m2_nse.median() - paired.m1_nse.median()),
                     'wilcoxon_W': float(test.statistic) if test else 0.,
                     'wilcoxon_p': float(test.pvalue) if test else 1.,
                     'n_m2_better': int((difference > 0).sum()), 'n_stations': len(paired),
                     'boot_ci_lo': np.nan, 'boot_ci_hi': np.nan, 'status': 'inference_withheld',
                     'reason': 'Shared drainage and nested gauges; independent-station bootstrap inference not justified',
                     'test_scope': 'Raw exploratory Wilcoxon; four-horizon Bonferroni reference 0.0125; station dependence remains'})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--write-analysis', action='store_true', help='Explicitly write screened exploratory CSVs to model_results; no nominal coverage guarantee')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == RES.resolve() or RES.resolve() in output.parents:
        parser.error("Step 2 output must be isolated from production model_results")
    predictions_path = RES / "test_predictions.csv"
    predictions = pd.read_csv(predictions_path, parse_dates=["issue_date"])
    summary, draws = run(predictions)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "PROVISIONAL_calendar_bootstrap.csv", index=False)
    np.savez_compressed(output / "PROVISIONAL_draws.npz", **draws)
    metadata = {"scope": "Exploratory analysis with heuristic screens; not certified coverage",
                "method": "paired moving blocks on complete weekly issue-date calendar, missing positions retained",
                "block_calendar_weeks": list(BLOCKS), "draws": N_BOOT, "seed": SEED,
                "reliability_rules": 'All settings require >=95% finite draws and nondegenerate spread; B_eff>=4; interval sign-category must agree across52/26/13. B_eff is a concentration heuristic, not a coverage theorem.',
                "degeneracy_tolerance": 'No finite draws, finite draw max-min <=1e-12, or percentile width <=1e-12 (including clipping-induced collapse)',
                "interval_scope": 'Conditional on fitted models and observed scoring pattern; does not include retraining uncertainty or guarantee95% coverage',
                "prediction_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
                "sampler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output / "STEP2_PROVISIONAL_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.write_analysis:
        station = screened_station_summary(summary)
        portfolio = portfolio_point_summary(pd.read_csv(RES/'audited_metrics_recomputed.csv'))
        for frame in [station, portfolio, summary]:
            frame['prediction_sha256'] = metadata['prediction_sha256']
            frame['sampler_sha256'] = metadata['sampler_sha256']
        station.to_csv(RES/'uncertainty_station_block_bootstrap.csv', index=False)
        portfolio.to_csv(RES/'uncertainty_portfolio.csv', index=False)
        summary.to_csv(RES/'uncertainty_block_sensitivity.csv', index=False)
        print(f'Wrote screened exploratory station/portfolio/sensitivity CSVs; {len(station)} station–lead cases')
    else:
        print(f"Computed {len(summary)} provisional station/lead/block cases in {output}; production uncertainty outputs unchanged")


if __name__ == "__main__":
    main()
