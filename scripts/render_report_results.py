#!/usr/bin/env python3
"""
Render the results sections of the project report (and the README results summary)
from the audited outputs, so the written numbers cannot drift from the data.

Inputs (written by scripts/audit_metrics.py, generate_interactive_plots.py, robustness_camels_era.py):
  data/processed/model_results/audited_metrics_recomputed.csv
  data/processed/model_results/test_predictions.csv
  data/processed/model_results/feature_importance_pooled.csv / _groups.csv
  data/processed/model_results/drought_severity_sensitivity.csv
  data/processed/model_results/robustness_camels_era.csv

Outputs: replaces the text between the RESULTS markers in
  docs/PROJECT_REPORT_DOCUMENTATION.md   (<!-- REPORT_RESULTS:START/END -->)
  README.md                              (<!-- README_RESULTS:START/END -->)

Run:  python3 scripts/render_report_results.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "processed" / "model_results"
REPORT = ROOT / "docs" / "PROJECT_REPORT_DOCUMENTATION.md"
README = ROOT / "README.md"
HS = ["1w", "2w", "4w", "8w"]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_config import ZERO_AS_MISSING_STATIONS as ZERO_AS_MISSING  # noqa: E402
from export_explorer_data import drought_availability  # noqa: E402


def f3(v, signed=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def fp(p):
    if p is None or np.isnan(p):
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def med_n(s: pd.Series):
    s = s.dropna()
    return f"{s.median():.3f} (n={len(s)})" if len(s) else "undefined"


def load():
    d = pd.read_csv(RES / "audited_metrics_recomputed.csv")
    pr = pd.read_csv(RES / "test_predictions.csv")
    fi = pd.read_csv(RES / "feature_importance_pooled.csv", index_col=0)
    fg = pd.read_csv(RES / "feature_importance_groups.csv", index_col=0)
    sens_path = RES / "drought_severity_sensitivity.csv"
    sens = pd.read_csv(sens_path) if sens_path.is_file() else None
    rob_path = RES / "robustness_camels_era.csv"
    rob = pd.read_csv(rob_path) if rob_path.is_file() else None
    return d, pr, fi, fg, sens, rob


def summary(d, pr):
    rows = {}
    for h in HS:
        s = d[d.horizon == h]
        sc = pr[(pr.horizon == h) & pr.scored]
        y = (sc.y_obs <= sc.q95).astype(int)
        
        # Sub-period 2016-2020 metrics
        m1_16_20 = s.m1_nse_2016_20.median() if 'm1_nse_2016_20' in s.columns else np.nan
        m2_16_20 = s.m2_nse_2016_20.median() if 'm2_nse_2016_20' in s.columns else np.nan
        m0clim_16_20 = s.m0clim_nse_2016_20.median() if 'm0clim_nse_2016_20' in s.columns else np.nan
        m0pers_16_20 = s.m0pers_nse_2016_20.median() if 'm0pers_nse_2016_20' in s.columns else np.nan
        dn_16_20 = (s.m2_nse_2016_20 - s.m1_nse_2016_20).median() if 'm1_nse_2016_20' in s.columns else np.nan
        pn_16_20 = wilcoxon(s.m2_nse_2016_20, s.m1_nse_2016_20).pvalue if 'm1_nse_2016_20' in s.columns else np.nan

        # Sub-period 2021-2025 metrics
        m1_21_25 = s.m1_nse_2021_25.median() if 'm1_nse_2021_25' in s.columns else np.nan
        m2_21_25 = s.m2_nse_2021_25.median() if 'm2_nse_2021_25' in s.columns else np.nan
        m0clim_21_25 = s.m0clim_nse_2021_25.median() if 'm0clim_nse_2021_25' in s.columns else np.nan
        m0pers_21_25 = s.m0pers_nse_2021_25.median() if 'm0pers_nse_2021_25' in s.columns else np.nan
        dn_21_25 = (s.m2_nse_2021_25 - s.m1_nse_2021_25).median() if 'm1_nse_2021_25' in s.columns else np.nan
        pn_21_25 = wilcoxon(s.m2_nse_2021_25, s.m1_nse_2021_25).pvalue if 'm1_nse_2021_25' in s.columns else np.nan

        # Log models
        m1_log_nse = s.m1_gbt_log_nse.median() if 'm1_gbt_log_nse' in s.columns else np.nan
        m2_log_nse = s.m2_gbt_log_nse.median() if 'm2_gbt_log_nse' in s.columns else np.nan
        m0_log_nse = s.m0_log_nse.median() if 'm0_log_nse' in s.columns else np.nan
        m1_log_log_nse = s.m1_gbt_log_log_nse.median() if 'm1_gbt_log_log_nse' in s.columns else np.nan
        m2_log_log_nse = s.m2_gbt_log_log_nse.median() if 'm2_gbt_log_log_nse' in s.columns else np.nan
        c1_log = med_n(s.m1_gbt_log_csi) if 'm1_gbt_log_csi' in s.columns else "–"
        c2_log = med_n(s.m2_gbt_log_csi) if 'm2_gbt_log_csi' in s.columns else "–"

        rows[h] = dict(
            m0=s.m0_nse_fixed.median(), m1=s.m1_nse.median(), m2=s.m2_nse.median(),
            m1_log=m1_log_nse, m2_log=m2_log_nse,
            m0_log_nse=m0_log_nse, m1_log_log_nse=m1_log_log_nse, m2_log_log_nse=m2_log_log_nse,
            c1_log=c1_log, c2_log=c2_log,
            m1_16_20=m1_16_20, m2_16_20=m2_16_20,
            m0clim_16_20=m0clim_16_20, m0pers_16_20=m0pers_16_20,
            dn_16_20=dn_16_20, pn_16_20=pn_16_20,
            m1_21_25=m1_21_25, m2_21_25=m2_21_25,
            m0clim_21_25=m0clim_21_25, m0pers_21_25=m0pers_21_25,
            dn_21_25=dn_21_25, pn_21_25=pn_21_25,
            k0=s.m0_kge_fixed.median(), k1=s.m1_kge.median(), k2=s.m2_kge.median(),
            r0=s.m0_r.median(), r1=s.m1_r.median(), r2=s.m2_r.median(),
            a2=s.m2_alpha.median(), b2=s.m2_beta.median(),
            dn=(s.m2_nse - s.m1_nse).median(), dk=(s.m2_kge - s.m1_kge).median(),
            dn0=(s.m2_nse - s.m0_nse_fixed).median(),
            pn=wilcoxon(s.m2_nse, s.m1_nse).pvalue, pk=wilcoxon(s.m2_kge, s.m1_kge).pvalue,
            pn0=wilcoxon(s.m2_nse, s.m0_nse_fixed).pvalue,
            wn=wilcoxon(s.m2_nse, s.m1_nse).statistic,
            m2win=int((s.m2_nse > s.m1_nse).sum()), climwin=int((s.m0_nse_fixed > s.m2_nse).sum()),
            c0=med_n(s.m0_csi), c1=med_n(s.m1_csi), c2=med_n(s.m2_csi),
            f1=med_n(s.m1_far), f2=med_n(s.m2_far),
            base=float(y.mean()), events=int(y.sum()), weeks=int(len(y)),
            ap0=average_precision_score(y, -sc.m0), ap1=average_precision_score(y, -sc.m1_gbt),
            ap2=average_precision_score(y, -sc.m2_gbt),
            ap1_log=average_precision_score(y, -sc.m1_gbt_log) if 'm1_gbt_log' in sc.columns else np.nan,
            ap2_log=average_precision_score(y, -sc.m2_gbt_log) if 'm2_gbt_log' in sc.columns else np.nan,
        )
        for tag, col in (("0", "m0"), ("1", "m1_gbt"), ("2", "m2_gbt")):
            alert = sc[col] <= sc.q95
            hit, fa, miss = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum()), int((~alert & (y == 1)).sum())
            rows[h][f"pcsi{tag}"] = hit / (hit + fa + miss) if (hit + fa + miss) > 0 else 0.0
            rows[h][f"ppod{tag}"] = hit / (hit + miss) if (hit + miss) > 0 else 0.0
    return rows


def md_table(frame):
    """Markdown table without an additional rendering dependency."""
    def cell(value):
        if pd.isna(value):
            return "undefined"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value).replace("|", "/")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def result_tables(d):
    continuous, drought = [], []
    for h in HS:
        frame = d[d.horizon == h]
        pairs = frame[["m1_nse", "m2_nse"]].dropna()
        delta = pairs.m2_nse - pairs.m1_nse
        pvalue = wilcoxon(delta).pvalue if len(delta) and (delta != 0).any() else 1.
        continuous.append({"Lead": h, "M0 NSE": frame.m0_nse_fixed.median(),
                           "M1 NSE": frame.m1_nse.median(), "M2 NSE": frame.m2_nse.median(),
                           "Paired median M2-M1": delta.median(), "Raw p": pvalue,
                           "Pairs": len(pairs), "p<0.0125": bool(pvalue < .0125),
                           "Climatology beats M2": int((frame.m0clim_nse > frame.m2_nse).sum())})
        for prefix, label in [("m0", "M0"), ("m1", "M1"), ("m2", "M2"),
                              ("m1_gbt_log", "M1-log"), ("m2_gbt_log", "M2-log")]:
            csi, far = frame[prefix + "_csi"], frame[prefix + "_far"]
            drought.append({"Lead": h, "Model": label, "Median Q95 CSI": csi.median(),
                            "CSI stations": int(csi.notna().sum()), "Median FAR": far.median(),
                            "FAR stations": int(far.notna().sum())})
    return pd.DataFrame(continuous), pd.DataFrame(drought)


def readme_md(d, pr):
    continuous, drought = result_tables(d)
    lines = ["### Corrected retrospective results (2016–2025 test block)", "",
             "Generated from newly audited models and observed targets. M0 is lagged persistence at one and two weeks, weekly climatology at four and eight weeks. "
             "NSE differences below are paired station differences, not differences between portfolio medians. "
             "Weather includes catchment-mean IMD rainfall and calibrated Open-Meteo default Best Match temperature/reference ET0 after 2020; it is not verified pure ERA5-Land.", "",
             md_table(continuous), "",
             "The p-values are exploratory and uncorrected. The p<0.0125 column is a four-horizon Bonferroni reference, not evidence of independent gauges. "
             "Shared drainages and nested gauges still limit inference. Prediction differences do not establish causation.", "",
             "**Warning performance:** Compare severity-specific counts and the five-model tables in the report/explorer. "
             "Good continuous flow scores do not establish good drought-warning performance. Undefined ratios are not zero skill.", "",
             "**Scope:** These are retrospective forecasts; real-time data publication delays are not simulated. "
             "Sources, preprocessing, missingness, and models changed together in this repair, so old/new score changes cannot be attributed to one cause.", ""]
    return "\n".join(lines)


def report_md(d, pr, fi, fg, sens, rob):
    continuous, drought = result_tables(d)
    availability = d[["station", "horizon", "n_eval", "n_masked_zero_weeks", "q95_threshold", "test_q95_event_count", "test_q95_base_rate"]].copy()
    availability["assessment"] = [drought_availability(row.station, row.q95_threshold, row.test_q95_event_count)["assessment"]
                                  for row in availability.itertuples()]
    unavailable = int(availability.test_q95_event_count.eq(0).sum())
    lines = ["## 10. Corrected results and evaluation", "",
             "These tables are regenerated from the complete corrected run; old model scores are not reused. "
             "Train/validation/test dates, lead horizons and primary metrics are unchanged. "
             "Daily targets are never filled from future observations; a weekly discharge target requires four original daily measurements. "
             "Weekly bins explicitly exclude the labelled Monday's daily record. Source harmonisation fits only 2001–2010.", "",
             "Later temperature and FAO reference ET0 inputs were supplied through Open-Meteo's historical API with its default Best Match model selection. "
             "Their filenames retain an ERA5 label for compatibility, but do not establish a homogeneous ERA5-Land product. "
             "GMT calendar-day aggregation is documented; original CWC and CAMELS observation-day timing is not established as operationally available at midnight.", "",
             "### 10.1 Continuous forecast comparisons", "", md_table(continuous), "",
             "M0 uses persistence at 1–2 weeks and weekly climatology at 4–8 weeks. Both additional baselines remain in the audited data at every horizon. "
             "All comparisons use the shared scored-week mask. Raw signed-rank p-values are exploratory; four lead comparisons imply a Bonferroni reference of 0.0125. "
             "Nested gauges and shared major drainages prevent treating all station results as independent experimental replications.", "",
             "### 10.2 Severe-drought warning performance", "", md_table(drought), "",
             "CSI is hits/(hits+misses+false alarms). FAR is false alarms/(hits+false alarms). "
             "Zero observed/predicted events can leave a ratio undefined; medians therefore have different station counts. "
             "Median station scores and pooled contingency scores answer different questions and must not be interchanged.", "",
             "### 10.3 Scoring coverage and event counts", "",
             md_table(availability), "",
             f"{unavailable} of 48 station–lead cases have no scored severe-drought events. Detection cannot be assessed in those cases; false-alarm counts are still observable. "
             "At Yadgir the frozen Q95 threshold is zero and recorded-zero targets are excluded, so the approved evaluation rule makes the event structurally unavailable. "
             "These statuses do not change thresholds, masks, raw contingency counts or the reported metric definitions. Raw CSI ratios in zero-event cases are not evidence of detection skill.", "",
             "Recorded-zero gaps at the four configured stations are excluded using the same rule for all models. "
             "That classification remains an inference from the monitoring pattern rather than a complete source flag for every measurement.", "",
             "### 10.4 KGE components and log-target trade-offs", "",
             "KGE combines correlation, variability ratio and mean bias ratio; its zero is not a universal no-skill reference. "
             "Log-target training changes error emphasis, so both ordinary NSE and Log-NSE are reported.", "",
             md_table(d.groupby("horizon", sort=False)[["m0_kge_fixed", "m1_kge", "m2_kge",
                       "m0_log_nse", "m1_gbt_log_nse", "m2_gbt_log_nse",
                       "m1_gbt_log_log_nse", "m2_gbt_log_log_nse"]].median().reset_index()), "",
             "### 10.5 Severity sensitivity", ""]
    if sens is not None:
        pooled = sens[sens.scope == "pooled"] if "scope" in sens.columns else pd.DataFrame()
        lines += [md_table(pooled) if len(pooled) else
                  "See drought_severity_sensitivity.csv for all station and pooled contingency counts."]
    lines += ["", "### 10.6 Feature reliance", "",
              "Permutation importance measures the change in prediction error when a feature or group is shuffled. "
              "It does not measure causal importance or establish that spatial aggregation errors are negligible.", "",
              md_table(fi.reset_index()), "", md_table(fg.reset_index()), "",
              "### 10.7 Cross-basin transfer", "",
              md_table(d[d.is_holdout][["station", "horizon", "n_eval", "m0_nse_fixed", "p2_gbt_nse", "p2_lstm_nse"]]), "",
              "Each transfer baseline uses the held-out station's own history. This is transfer to an unseen gauged catchment, not a proof of operational forecasting at an ungauged site. "
              "A positive NSE alone does not imply that the transferred model beats that baseline.", "",
              "## 11. Errors, uncertainty and robustness", ""]
    lines += ["Previous station bootstrap inference withdrawn: it compressed calendar gaps and can produce degenerate intervals for short series. "
              "Point-forecast scores are unaffected. The replacement analysis retains every weekly calendar position, including excluded or missing observations, and resamples observations, both models and their shared mask together. "
              "The original CSVs and algorithm are preserved as superseded provenance in "
              "data/processed/validation_reports/uncertainty_baseline_2026-11-16/.", "",
              "The primary analysis uses 52-calendar-week moving blocks, with 26- and 13-week sensitivity checks, 2,000 draws and fixed seeds. "
              "Intervals are withheld for sparse informative coverage, too few finite draws, degeneracy, or a direction that changes with block length. "
              "The effective-bin screen and 95% finite-draw screen are practical diagnostics, not a proof of confidence-interval coverage. "
              "Any retained percentile interval is exploratory and conditional on the fitted models and observed scoring pattern; it does not include model-training or source uncertainty. "
              "Portfolio station-bootstrap intervals are withheld because shared drainage and nested gauges invalidate an independent-station interpretation. "
              "See docs/UNCERTAINTY_METHOD.md for exact screens and uncertainty_block_sensitivity.csv for all settings.", "",
              "Representative errors can be inspected in the forecast replay using the saved issue and target dates. "
              "Compare missed droughts, false alarms, peak underprediction and excluded measurement gaps rather than only portfolio averages.", ""]
    uncertainty = pd.read_csv(RES / "uncertainty_station_block_bootstrap.csv")
    if "status" not in uncertainty:
        raise ValueError("Refusing superseded uncertainty outputs without reliability status")
    lines += ["### 11.1 Calendar-bootstrap diagnostics", "",
              md_table(uncertainty[["station", "horizon", "n_scored", "coverage_fraction", "B_eff",
                                    "finite_draw_fraction", "status", "reason", "ci_lo", "ci_hi"]]), "",
              "Blank/undefined interval bounds mean inference was withheld, not that the point difference was zero. "
              "No station winner-count claim is made from these exploratory diagnostics.", ""]
    if rob is not None:
        pooled = rob[rob.scope == "portfolio_summary"] if "scope" in rob.columns else rob
        cols = [c for c in ["horizon", "n_eval", "delta_NSE_M2_M1", "delta_NSE_M2_M1_pval",
                            "M0_median_NSE", "M1_median_NSE", "M2_median_NSE",
                            "M0_Q95_pooled_CSI", "M1_Q95_pooled_CSI", "M2_Q95_pooled_CSI"] if c in pooled]
        lines += ["Historical-period sensitivity uses its separately documented chronological split; it is not the primary 2016–2025 test.", "",
                  md_table(pooled[cols]), ""]
    lines += ["## 12. Conclusions and limitations", "",
              "Judge each model against the appropriate baseline, lead, severity and scored rows in the tables above. "
              "Weather inputs, log-target fitting and recurrent architecture do not guarantee a gain in both flow accuracy and warning skill.", "",
              "- This is a retrospective hindcast; publication latency and operational availability have not been reproduced.",
              "- The new source is Open-Meteo default Best Match, so a homogeneous reanalysis-model claim is not supported.",
              "- Reference ET0 scaled to historical PET remains a different physical formulation.",
              "- Missing daily discharge is preserved. The approved <15% raw missingness rule retains twelve gauges; actual weekly coverage and rare drought-event counts vary.",
              "- Shared drainages and the nested cascade limit station-level statistical independence.",
              "- Source replacement, timing correction, interpolation removal and retraining occurred together. The before/after comparison is not an isolated causal ablation.",
              "- Saved model bundles include fitted preprocessing and hashes of the tensors used. Full retraining can vary, especially for the LSTM; committed-output reproducibility differs from bitwise retraining reproducibility.", ""]
    return "\n".join(lines)


def splice(path: Path, start: str, end: str, body: str) -> None:
    s = path.read_text()
    i, j = s.index(start) + len(start), s.index(end)
    path.write_text(s[:i] + "\n" + body + "\n" + s[j:])
    print(f"[+] Updated {path.relative_to(ROOT)}")


def main():
    d, pr, fi, fg, sens, rob = load()
    splice(REPORT, "<!-- REPORT_RESULTS:START -->", "<!-- REPORT_RESULTS:END -->", report_md(d, pr, fi, fg, sens, rob))
    splice(README, "<!-- README_RESULTS:START -->", "<!-- README_RESULTS:END -->", readme_md(d, pr))


if __name__ == "__main__":
    main()
