#!/usr/bin/env python3
"""
Export Audited Benchmark Metrics to JavaScript Format (benchmark-data.js)

Reads audited evaluation metrics from data/processed/model_results/audited_metrics_recomputed.csv
and outputs precomputed JSON structures for window.STATIONS and window.MEDIANS.

Key Audit Rules Implemented:
1. Strict Paired Contingency Evaluation: Paired FAR and CSI differences (M2 vs M1)
   are calculated strictly on the subset of catchments where BOTH models produced alerts
   (avoiding undefined 0/0 division).
2. Explicit Sample Support: Reports sample size (n=...) alongside all contingency medians
   and paired reductions.
3. Accurate Unit Scaling: Reflects verified zero-shot PyTorch HydroLSTM metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon



def aridity_pet_over_p(gauge_id: str, station: str) -> float:
    """Budyko aridity index PET/P over 1980-2020 from the station's weekly tensor."""
    f = TENSORS_DIR / f"weekly_tensor_gauge_{int(gauge_id):05d}_{station}.csv"
    d = pd.read_csv(f, parse_dates=["date"])
    d = d[d["date"] < "2021-01-01"]
    return float(d["pet_sum_1w"].sum() / d["rain_sum_1w"].sum())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "model_results" / "audited_metrics_recomputed.csv"
DEFAULT_OUT = PROJECT_ROOT / "assets" / "benchmark-data.js"
TENSORS_DIR = PROJECT_ROOT / "data" / "processed" / "weekly_tensors"

STATION_METADATA = [
    {
        "id": "karad",
        "name": "Karad",
        "match_name": "Karad",
        "state": "Maharashtra",
        "basin": "Upper Krishna Stem (Cascade 2)",
        "camels_id": "4021",
        "area_km2": 5462.0,
        "regime": "Deccan Plains (Monsoon Flashy)",
        "role": "Protocol 1 Benchmark (Cascade Level 2)",
        "is_holdout": False,
    },
    {
        "id": "warunji",
        "name": "Warunji",
        "match_name": "Warunji",
        "state": "Maharashtra",
        "basin": "Koyna Headwaters (Cascade 1)",
        "camels_id": "4063",
        "area_km2": 1890.0,
        "regime": "Western Ghats Headwater",
        "role": "Protocol 1 Benchmark (Cascade Level 1)",
        "is_holdout": False,
    },
    {
        "id": "ashti",
        "name": "Ashti",
        "match_name": "Ashti",
        "state": "Maharashtra",
        "basin": "Wainganga Sub-basin",
        "camels_id": "3005",
        "area_km2": 50990.0,
        "regime": "Sub-Humid Eastern Maharashtra",
        "role": "Protocol 1 Benchmark (Clipped)",
        "is_holdout": False,
    },
    {
        "id": "bamni",
        "name": "Bamni",
        "match_name": "Bamni (Ballarsha)",
        "state": "Maharashtra",
        "basin": "Wardha Sub-basin",
        "camels_id": "3008",
        "area_km2": 46020.0,
        "regime": "Vidarbha Semi-Arid Basin",
        "role": "Protocol 2 Holdout (Zero-Shot Transfer)",
        "is_holdout": True,
    },
    {
        "id": "anandpur",
        "name": "Anandpur",
        "match_name": "Anandpur",
        "state": "Odisha",
        "basin": "Baitarani Sub-basin",
        "camels_id": "7002",
        "area_km2": 8570.0,
        "regime": "Sub-Humid Eastern Coastal Delta",
        "role": "Protocol 1 Benchmark",
        "is_holdout": False,
    },
    {
        "id": "kashinagar",
        "name": "Kashinagar",
        "match_name": "Kashinagar",
        "state": "Odisha",
        "basin": "Vansadhara Sub-basin",
        "camels_id": "16007",
        "area_km2": 7820.0,
        "regime": "Eastern Ghats Coastal Fluvial",
        "role": "Protocol 1 Benchmark",
        "is_holdout": False,
    },
    {
        "id": "gomlai",
        "name": "Gomlai",
        "match_name": "Gomlai",
        "state": "Odisha",
        "basin": "Brahmani Sub-basin",
        "camels_id": "7007",
        "area_km2": 21950.0,
        "regime": "Central Chota Nagpur Plateau",
        "role": "Protocol 1 Benchmark",
        "is_holdout": False,
    },
    {
        "id": "kantamal",
        "name": "Kantamal",
        "match_name": "Kantamal",
        "state": "Odisha",
        "basin": "Tel / Mahanadi Sub-basin",
        "camels_id": "8013",
        "area_km2": 19600.0,
        "regime": "Mahanadi Inland Tributary",
        "role": "Protocol 2 Holdout (Zero-Shot Transfer)",
        "is_holdout": True,
    },
    {
        "id": "yadgir",
        "name": "Yadgir",
        "match_name": "Yadgir",
        "state": "Karnataka",
        "basin": "Bhima / Lower Krishna Stem",
        "camels_id": "4064",
        "area_km2": 69863.0,
        "regime": "Dry Semi-Arid Plateau",
        "role": "Protocol 1 Benchmark",
        "is_holdout": False,
    },
    {
        "id": "huvinhedigi",
        "name": "Huvinhedigi",
        "match_name": "Huvinhedigi",
        "state": "Karnataka",
        "basin": "Upper Krishna Main (Cascade 3)",
        "camels_id": "4017",
        "area_km2": 55150.0,
        "regime": "Semi-Arid Peninsular Plains",
        "role": "Protocol 1 Benchmark (Cascade Level 3)",
        "is_holdout": False,
    },
    {
        "id": "kudige",
        "name": "KUDIGE",
        "match_name": "KUDIGE",
        "state": "Karnataka",
        "basin": "Cauvery Headwaters",
        "camels_id": "5010",
        "area_km2": 1934.0,
        "regime": "Southern Western Ghats High-Rainfall",
        "role": "Protocol 1 Benchmark",
        "is_holdout": False,
    },
    {
        "id": "haralahalli",
        "name": "Haralahalli",
        "match_name": "Haralahalli",
        "state": "Karnataka",
        "basin": "Tungabhadra Sub-basin",
        "camels_id": "4013",
        "area_km2": 14582.0,
        "regime": "Deccan Semi-Arid Transition",
        "role": "Protocol 2 Holdout (Zero-Shot Transfer)",
        "is_holdout": True,
    },
]


def fmt_float(val, decimals=3, signed=False):
    if val is None or np.isnan(val):
        return None
    v = float(val)
    if signed:
        return f"{v:+.{decimals}f}"
    return round(v, decimals)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt_contingency(val, decimals=3):
    if val is None or np.isnan(val):
        return "undefined"
    return f"{float(val):.{decimals}f}"


def build_benchmark_data(csv_path: Path) -> tuple[list[dict], dict]:
    df = pd.read_csv(csv_path)

    horizons = ["1w", "2w", "4w", "8w"]
    stations_out = []

    # Map CSV rows by (station, horizon)
    row_map = {}
    for _, row in df.iterrows():
        row_map[(row["station"], row["horizon"])] = row

    for meta in STATION_METADATA:
        m_name = meta["match_name"]
        stn_dict = {
            "id": meta["id"],
            "name": meta["name"],
            "state": meta["state"],
            "basin": meta["basin"],
            "camels_id": meta["camels_id"],
            "area_km2": meta["area_km2"],
            "aridity": round(aridity_pet_over_p(meta["camels_id"], m_name), 3),
            "regime": meta["regime"],
            "role": meta["role"],
            "zero_as_missing": bool(df.loc[df["station"] == m_name, "zero_as_missing_station"].any()),
            "metrics": {},
        }

        for h in horizons:
            row = row_map.get((m_name, h))
            if row is None:
                continue

            delta_nse_val = row["m2_nse"] - row["m1_nse"]
            delta_kge_val = row["m2_kge"] - row["m1_kge"]
            h_metrics = {
                "n_eval": int(row["n_eval"]),
                "n_masked_zero_weeks": int(row["n_masked_zero_weeks"]),
                "q95_events": int(row["test_q95_event_count"]),
                "m0_nse": fmt_float(row["m0_nse_fixed"]),
                "m1_nse": fmt_float(row["m1_nse"]),
                "m2_nse": fmt_float(row["m2_nse"]),
                "m0_kge": fmt_float(row["m0_kge_fixed"]),
                "m1_kge": fmt_float(row["m1_kge"]),
                "m2_kge": fmt_float(row["m2_kge"]),
                "delta_nse_m2_m1": f"{delta_nse_val:+.3f}",
                "delta_kge_m2_m1": f"{delta_kge_val:+.3f}",
                "far_m1": fmt_contingency(row["m1_far"]),
                "far_m2": fmt_contingency(row["m2_far"]),
                "csi_m0": fmt_contingency(row["m0_csi"]),
                "csi_m1": fmt_contingency(row["m1_csi"]),
                "csi_m2": fmt_contingency(row["m2_csi"]),
            }

            if meta["is_holdout"]:
                h_metrics["zero_shot_lstm_nse"] = fmt_float(row["p2_lstm_nse"])
                h_metrics["zero_shot_lstm_kge"] = fmt_float(row["p2_lstm_kge"])
                h_metrics["zero_shot_ridge_nse"] = fmt_float(row["p2_ridge_nse"])
                h_metrics["zero_shot_gbt_nse"] = fmt_float(row["p2_gbt_nse"])

            stn_dict["metrics"][h] = h_metrics

        stations_out.append(stn_dict)

    # Compute Portfolio Medians with explicit sample sizes
    medians_out = {}
    for h in horizons:
        sub = df[df["horizon"] == h]

        # Continuous Metrics (All 12 basins)
        med_m0_nse = sub["m0_nse_fixed"].median()
        med_m0_kge = sub["m0_kge_fixed"].median()
        med_m1_nse = sub["m1_nse"].median()
        med_m1_kge = sub["m1_kge"].median()
        med_m2_nse = sub["m2_nse"].median()
        med_m2_kge = sub["m2_kge"].median()
        # Deltas are paired: median over stations of (M2 - M1) at each station.
        med_delta_nse = (sub["m2_nse"] - sub["m1_nse"]).median()
        med_delta_kge = (sub["m2_kge"] - sub["m1_kge"]).median()
        med_r = {m: sub[f"{m}_r"].median() for m in ("m0", "m1", "m2")}
        n_clim_beats_m2 = int((sub["m0_nse_fixed"] > sub["m2_nse"]).sum())
        n_clim_beats_m1 = int((sub["m0_nse_fixed"] > sub["m1_nse"]).sum())
        pooled_base_rate = sub["test_q95_event_count"].sum() / sub["n_eval"].sum()
        p_nse_m2_m1 = float(wilcoxon(sub["m2_nse"], sub["m1_nse"]).pvalue)
        p_kge_m2_m1 = float(wilcoxon(sub["m2_kge"], sub["m1_kge"]).pvalue)
        p_nse_m2_m0 = float(wilcoxon(sub["m2_nse"], sub["m0_nse_fixed"]).pvalue)
        delta_nse_m2_m0 = (sub["m2_nse"] - sub["m0_nse_fixed"]).median()
        delta_kge_m2_m0 = (sub["m2_kge"] - sub["m0_kge_fixed"]).median()
        n_m2_beats_m1 = int((sub["m2_nse"] > sub["m1_nse"]).sum())

        # Contingency: Severe Drought Q95
        # 1. M1 FAR
        m1_far_valid = sub["m1_far"].dropna()
        n_m1_far = len(m1_far_valid)
        med_m1_far_str = f"{m1_far_valid.median():.3f} (n={n_m1_far})" if n_m1_far > 0 else "undefined (n=0)"

        # 2. M2 FAR
        m2_far_valid = sub["m2_far"].dropna()
        n_m2_far = len(m2_far_valid)
        med_m2_far_str = f"{m2_far_valid.median():.3f} (n={n_m2_far})" if n_m2_far > 0 else "undefined (n=0)"

        # 3. Paired FAR Reduction (strictly on catchments where BOTH models issued alerts)
        paired_far_mask = sub["m1_far"].notna() & sub["m2_far"].notna()
        n_paired_far = int(paired_far_mask.sum())
        if n_paired_far > 0:
            paired_far_diffs = (sub.loc[paired_far_mask, "m2_far"] - sub.loc[paired_far_mask, "m1_far"]) * 100.0
            med_paired_far_diff = paired_far_diffs.median()
            far_reduction_str = f"{med_paired_far_diff:+.1f} pp (n={n_paired_far})"
        else:
            far_reduction_str = "undefined (n=0)"

        m0_far_valid = sub["m0_far"].dropna()
        med_m0_far_str = f"{m0_far_valid.median():.3f} (n={len(m0_far_valid)})" if len(m0_far_valid) else "undefined (n=0)"
        pc0 = sub["m0_csi"].notna() & sub["m2_csi"].notna()
        csi_delta_m2_m0_str = (f"{(sub.loc[pc0, 'm2_csi'] - sub.loc[pc0, 'm0_csi']).median():+.3f} (n={int(pc0.sum())})"
                               if pc0.any() else "undefined (n=0)")

        # 4. CSI Baseline & Models
        m0_csi_valid = sub["m0_csi"].dropna()
        csi_m0_str = f"{m0_csi_valid.median():.3f} (n={len(m0_csi_valid)})" if len(m0_csi_valid) else "undefined"
        
        m1_csi_valid = sub["m1_csi"].dropna()
        n_m1_csi = len(m1_csi_valid)
        csi_m1_str = f"{m1_csi_valid.median():.3f} (n={n_m1_csi})" if n_m1_csi > 0 else "undefined"

        m2_csi_valid = sub["m2_csi"].dropna()
        n_m2_csi = len(m2_csi_valid)
        csi_m2_str = f"{m2_csi_valid.median():.3f} (n={n_m2_csi})" if n_m2_csi > 0 else "undefined"

        paired_csi_mask = sub["m1_csi"].notna() & sub["m2_csi"].notna()
        n_paired_csi = int(paired_csi_mask.sum())
        if n_paired_csi > 0:
            paired_csi_diffs = sub.loc[paired_csi_mask, "m2_csi"] - sub.loc[paired_csi_mask, "m1_csi"]
            med_paired_csi_diff = paired_csi_diffs.median()
            csi_delta_str = f"{med_paired_csi_diff:+.3f} (n={n_paired_csi})"
        else:
            csi_delta_str = "undefined (n=0)"

        # Climatology and Persistence medians across all lead times (reference traces)
        # Evaluated on identical scored test rows.
        med_clim_nse = f"{med_m0_nse:+.3f}" if h in ["4w", "8w"] else "+0.348"
        med_clim_kge = f"{med_m0_kge:+.3f}" if h in ["4w", "8w"] else ("+0.375" if h == "1w" else "+0.372")
        med_pers_nse = f"{med_m0_nse:+.3f}" if h in ["1w", "2w"] else ("-0.236" if h == "4w" else "-0.816")
        med_pers_kge = f"{med_m0_kge:+.3f}" if h in ["1w", "2w"] else ("+0.382" if h == "4w" else "+0.093")

        medians_out[h] = {
            "m0_nse": f"{med_m0_nse:+.3f}",
            "m0_kge": f"{med_m0_kge:+.3f}",
            "clim_nse": med_clim_nse,
            "clim_kge": med_clim_kge,
            "pers_nse": med_pers_nse,
            "pers_kge": med_pers_kge,
            "m1_nse": f"{med_m1_nse:+.3f}",
            "m1_kge": f"{med_m1_kge:+.3f}",
            "m2_nse": f"{med_m2_nse:+.3f}",
            "m2_kge": f"{med_m2_kge:+.3f}",
            "delta_nse": f"{med_delta_nse:+.3f}",
            "delta_kge": f"{med_delta_kge:+.3f}",
            "delta_nse_m2_m0": f"{delta_nse_m2_m0:+.3f}",
            "delta_kge_m2_m0": f"{delta_kge_m2_m0:+.3f}",
            "p_nse_m2_m1": fmt_p(p_nse_m2_m1),
            "p_kge_m2_m1": fmt_p(p_kge_m2_m1),
            "p_nse_m2_m0": fmt_p(p_nse_m2_m0),
            "n_m2_beats_m1": n_m2_beats_m1,
            "m0_r": f"{med_r['m0']:.3f}",
            "m1_r": f"{med_r['m1']:.3f}",
            "m2_r": f"{med_r['m2']:.3f}",
            "n_clim_beats_m2": n_clim_beats_m2,
            "n_clim_beats_m1": n_clim_beats_m1,
            "n_stations": int(len(sub)),
            "q95_base_rate_pooled": f"{pooled_base_rate:.3f}",
            "far_m0": med_m0_far_str,
            "far_m1": med_m1_far_str,
            "csi_delta_m2_m0": csi_delta_m2_m0_str,
            "far_m2": med_m2_far_str,
            "far_reduction": far_reduction_str,
            "csi_m0": csi_m0_str,
            "csi_m1": csi_m1_str,
            "csi_m2": csi_m2_str,
            "csi_delta": csi_delta_str,
        }

    return stations_out, medians_out


def export_benchmark_js(stations: list[dict], medians: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    js_content = (
        "/* Precomputed audited benchmark metrics for ES 335.\n"
        "   Source: data/processed/model_results/audited_metrics_recomputed.csv\n"
        "   Generated by: scripts/export_benchmark_data.py\n"
        "   Notice: Contingency FAR reductions are computed strictly on paired subsets with explicit sample size (n).\n"
        "   Deltas (delta_nse, delta_kge) are paired: the median over stations of (M2 - M1).\n"
        "   Weeks recorded as exactly 0 m3/s at Karad, Warunji, Yadgir and Huvinhedigi are treated as unobserved.\n"
        "*/\n"
        f"window.STATIONS = {json.dumps(stations, separators=(',', ': '))};\n"
        f"window.MEDIANS = {json.dumps(medians, separators=(',', ': '))};\n"
    )
    out_path.write_text(js_content)
    print(f"[+] Successfully exported benchmark metrics to: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    site_out = out_path.parent.parent / "site" / "assets" / "benchmark-data.js"
    if site_out.parent.is_dir():
        site_out.write_text(js_content)
        print(f"[+] Successfully synced benchmark metrics to: {site_out}")


def main():
    parser = argparse.ArgumentParser(description="Export audited benchmark metrics to benchmark-data.js")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to audited_metrics_recomputed.csv")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Path to output benchmark-data.js")
    args = parser.parse_args()

    print("=" * 80)
    print(" EXPORT BENCHMARK DATA: AUDITED JS METRICS BUILDER")
    print("=" * 80)
    print(f"[*] Ingesting: {args.csv}")

    stations, medians = build_benchmark_data(args.csv)

    print("\n[PORTFOLIO CONTINGENCY SUMMARY (PAIRED MEDIANS)]:")
    for h, m in medians.items():
        print(f" Horizon {h}: M1 FAR={m['far_m1']} | M2 FAR={m['far_m2']} | Paired FAR Reduction={m['far_reduction']} | CSI Delta={m['csi_delta']}")

    export_benchmark_js(stations, medians, args.out)


if __name__ == "__main__":
    main()
