#!/usr/bin/env python3
"""Export the fixed station comparison results for the static site."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from eval_config import ZERO_AS_MISSING_STATIONS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/processed/model_results"
FLOW = RESULTS / "audited_metrics_recomputed.csv"
DROUGHT = RESULTS / "drought_severity_sensitivity.csv"
STATION_UNCERTAINTY = RESULTS / "uncertainty_station_block_bootstrap.csv"
PORTFOLIO_UNCERTAINTY = RESULTS / "uncertainty_portfolio.csv"
LEADS = {"1w", "2w", "4w", "8w"}
SEVERITIES = {"Q95", "Q90", "Q80"}
FLOW_COLUMNS = {
    "M0": ("m0_nse_fixed", "m0_log_nse", "m0_kge_fixed"),
    "M1": ("m1_nse", "m1_log_nse", "m1_kge"),
    "M2": ("m2_nse", "m2_log_nse", "m2_kge"),
    "M1-log": ("m1_gbt_log_nse", "m1_gbt_log_log_nse", "m1_gbt_log_kge"),
    "M2-log": ("m2_gbt_log_nse", "m2_gbt_log_log_nse", "m2_gbt_log_kge"),
}


def number(value):
    parsed = float(value) if value else None
    return parsed if parsed is not None and math.isfinite(parsed) else None


def source(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def drought_availability(station, threshold, events):
    """Describe whether scored weeks can assess detection; never change raw metrics."""
    if threshold == 0 and station in ZERO_AS_MISSING_STATIONS:
        return {"assessment": "unavailable_zero_threshold", "assessment_note":
                "Detection cannot be assessed: the frozen threshold is zero and recorded-zero targets are excluded by the shared scoring rule. No scored target can meet this event definition."}
    if events == 0:
        return {"assessment": "no_observed_events", "assessment_note":
                "No drought events occurred in the scored weeks at this threshold. Detection cannot be assessed; false-alarm counts remain observable."}
    return {"assessment": "observed_events", "assessment_note":
            "Detection is assessed on the observed drought weeks shown here; small event counts limit interpretation."}


def load_results():
    stations = {}
    names = {}
    with FLOW.open(newline="") as file:
        for row in csv.DictReader(file):
            gauge, lead = row["gauge_id"], row["horizon"]
            names[gauge] = row["station"]
            horizons = stations.setdefault(gauge, {})
            if lead in horizons:
                raise ValueError(f"Duplicate flow row for {gauge}, {lead}")
            horizons[lead] = {
                "n_eval": int(row["n_eval"]),
                "n_masked_zero_weeks": int(row["n_masked_zero_weeks"]),
                "flow": {
                    model: dict(zip(("nse", "log_nse", "kge"), (number(row[col]) for col in columns)))
                    for model, columns in FLOW_COLUMNS.items()
                },
                "drought": {},
            }

    with DROUGHT.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["scope"] != "station":
                continue
            gauge, lead, severity, model = (row[key] for key in ("gauge_id", "horizon", "severity", "model"))
            entry = stations[gauge][lead]
            threshold, events = float(row["threshold"]), int(row["ObsEvents"])
            group = entry["drought"].setdefault(severity, {"threshold": threshold, "events": events, "models": {}})
            if (int(row["n_eval"]) != entry["n_eval"] or group["threshold"] != threshold or
                    group["events"] != events or model in group["models"]):
                raise ValueError(f"Conflicting drought row for {gauge}, {lead}, {severity}, {model}")
            group.update(drought_availability(names[gauge], threshold, events))
            group["models"][model] = {
                "hits": int(row["Hits"]), "misses": int(row["Miss"]), "false_alarms": int(row["FA"]),
                "csi": number(row["CSI"]), "pod": number(row["POD"]), "far": number(row["FAR"]),
            }

    for gauge, horizons in stations.items():
        if horizons.keys() != LEADS:
            raise ValueError(f"Missing or unexpected leads for {gauge}")
        for lead, entry in horizons.items():
            if entry["drought"].keys() != SEVERITIES:
                raise ValueError(f"Missing or unexpected drought severities for {gauge}, {lead}")
            for severity, group in entry["drought"].items():
                if group["models"].keys() != FLOW_COLUMNS.keys():
                    raise ValueError(f"Missing or unexpected drought models for {gauge}, {lead}, {severity}")
    if len(stations) != 12:
        raise ValueError(f"Expected 12 audited gauges, got {len(stations)}")
    return stations


def load_uncertainty(stations):
    """Export checked diagnostics, never the superseded bootstrap inference."""
    with FLOW.open(newline="") as file:
        audited = {(row["station"], row["horizon"]): row for row in csv.DictReader(file)}
    seen = set()
    with STATION_UNCERTAINTY.open(newline="") as file:
        for row in csv.DictReader(file):
            key = row["station"], row["horizon"]
            audit = audited[key]
            delta = float(audit["m2_nse"]) - float(audit["m1_nse"])
            status = row["status"]
            low, high = number(row["ci_lo"]), number(row["ci_hi"])
            if (key in seen or abs(delta - float(row["dNSE"])) > 1e-10 or
                    int(audit["n_eval"]) != int(row["n_scored"]) or
                    status not in {"exploratory_interval", "inference_withheld"}):
                raise ValueError(f"Uncertainty diagnostics do not match the audit for {key}")
            if status == "inference_withheld" and (low is not None or high is not None):
                raise ValueError(f"Withheld inference has public interval bounds for {key}")
            if status == "exploratory_interval" and (low is None or high is None or low > high):
                raise ValueError(f"Invalid exploratory interval for {key}")
            seen.add(key)
            stations[audit["gauge_id"]][row["horizon"]]["uncertainty"] = {
                "delta": delta, "low": low, "high": high, "status": status, "reason": row["reason"],
                "n_scored": int(row["n_scored"]), "calendar_weeks": int(row["calendar_weeks"]),
                "coverage_fraction": float(row["coverage_fraction"]),
                "finite_draw_fraction": float(row["finite_draw_fraction"]), "B_eff": float(row["B_eff"]),
            }
    if seen != audited.keys():
        raise ValueError("Station uncertainty coverage does not match the audit")
    portfolio = {}
    with PORTFOLIO_UNCERTAINTY.open(newline="") as file:
        for row in csv.DictReader(file):
            lead = row["horizon"]
            deltas = [float(a["m2_nse"]) - float(a["m1_nse"]) for a in audited.values() if a["horizon"] == lead]
            if (lead in portfolio or abs(statistics.median(deltas) - float(row["median_paired_dNSE"])) > 1e-10 or
                    row["status"] != "inference_withheld" or number(row["boot_ci_lo"]) is not None or
                    number(row["boot_ci_hi"]) is not None):
                raise ValueError(f"Portfolio diagnostics do not match the audit for {lead}")
            portfolio[lead] = {"delta": float(row["median_paired_dNSE"]), "n_stations": len(deltas),
                               "status": row["status"], "reason": row["reason"]}
    if portfolio.keys() != LEADS:
        raise ValueError("Portfolio uncertainty coverage does not match the audit")
    return portfolio


def guided_examples(stations):
    weather = stations["8013"]["1w"]
    baseline = stations["8013"]["8w"]
    gaps = stations["4021"]["1w"]
    if weather["flow"]["M2"]["nse"] <= weather["flow"]["M1"]["nse"]:
        raise ValueError("The weather-gain example no longer has higher M2 NSE")
    if baseline["flow"]["M0"]["nse"] <= max(baseline["flow"][m]["nse"] for m in FLOW_COLUMNS if m != "M0"):
        raise ValueError("The baseline example no longer has the highest NSE")
    if gaps["n_masked_zero_weeks"] <= 0:
        raise ValueError("The missing-observation example no longer has excluded zero weeks")
    return [
        {"kind": "weather", "station": "kantamal", "gauge_id": "8013", "lead": "1w", "anchor": "explorer"},
        {"kind": "baseline", "station": "kantamal", "gauge_id": "8013", "lead": "8w", "anchor": "explorer"},
        {"kind": "gaps", "station": "karad", "gauge_id": "4021", "lead": "1w", "anchor": "forecast-replay"},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "assets/explorer-data.js")
    path = parser.parse_args().out
    stations = load_results()
    portfolio = load_uncertainty(stations)
    data = {"schema": 2, "uncertainty_note": "Exploratory percentile intervals are conditional on fitted models and the scoring pattern. Screening is heuristic and does not guarantee 95% coverage. No uncertainty widget is displayed.", "sources": {"flow": source(FLOW), "drought": source(DROUGHT),
            "station_uncertainty": source(STATION_UNCERTAINTY), "portfolio_uncertainty": source(PORTFOLIO_UNCERTAINTY),
            "uncertainty_method": source(ROOT / "scripts/uncertainty_tests.py")},
            "stations": stations, "portfolio_uncertainty": portfolio, "guided_examples": guided_examples(stations)}
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    path.write_text("/* Generated by scripts/export_explorer_data.py. */\nwindow.EXPLORER_DATA = " + payload + ";\n")


if __name__ == "__main__":
    main()
