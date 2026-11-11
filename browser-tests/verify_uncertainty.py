"""Verify all published intervals and reject uncertainty from a different audit."""

import csv
import hashlib
import importlib.util
import json
import statistics
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_uncertainty():
    asset = ROOT / "site/assets/explorer-data.js"
    data = json.loads(asset.read_text().split("window.EXPLORER_DATA = ", 1)[1].removesuffix(";\n"))
    results = ROOT / "data/processed/model_results"
    for key, path in (("station_uncertainty", results / "uncertainty_station_block_bootstrap.csv"),
                      ("portfolio_uncertainty", results / "uncertainty_portfolio.csv"),
                      ("uncertainty_method", ROOT / "scripts/uncertainty_tests.py")):
        assert data["sources"][key] == {"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    audits = {(r["station"], r["horizon"]): r for r in csv.DictReader((results / "audited_metrics_recomputed.csv").open())}
    count = 0
    for row in csv.DictReader((results / "uncertainty_station_block_bootstrap.csv").open()):
        audit = audits[row["station"], row["horizon"]]
        interval = data["stations"][audit["gauge_id"]][row["horizon"]]["uncertainty"]
        assert interval == {"delta": float(row["dNSE"]), "low": float(row["ci_lo"]), "high": float(row["ci_hi"]),
                            "n_scored": int(row["n_scored"]), "verdict": row["verdict"]}
        assert f'{interval["delta"]:.4f}' == f'{float(audit["m2_nse"]) - float(audit["m1_nse"]):.4f}'
        assert interval["n_scored"] == int(audit["n_eval"])
        count += 1
    for row in csv.DictReader((results / "uncertainty_portfolio.csv").open()):
        actual = data["portfolio_uncertainty"][row["horizon"]]
        assert actual == {"delta": float(row["median_paired_dNSE"]), "low": float(row["boot_ci_lo"]),
                          "high": float(row["boot_ci_hi"]), "n_stations": 12}
        values = [float(a["m2_nse"]) - float(a["m1_nse"]) for a in audits.values() if a["horizon"] == row["horizon"]]
        assert f'{statistics.median(values):.4f}' == f'{actual["delta"]:.4f}'
    spec = importlib.util.spec_from_file_location("uncertainty_exporter", ROOT / "scripts/export_explorer_data.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    with tempfile.TemporaryDirectory() as directory:
        rows = list(csv.DictReader(exporter.STATION_UNCERTAINTY.open()))
        rows[0]["dNSE"] = "0.9999"
        path = Path(directory) / "stale.csv"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        exporter.STATION_UNCERTAINTY = path
        try:
            exporter.load_uncertainty(exporter.load_results())
        except ValueError:
            pass
        else:
            raise AssertionError("Stale uncertainty was accepted")
        rows[0]["dNSE"] = audits[rows[0]["station"], rows[0]["horizon"]]["delta_nse_recomputed"]
        rows[0]["n_scored"] = "999"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        try:
            exporter.load_uncertainty(exporter.load_results())
        except ValueError:
            pass
        else:
            raise AssertionError("A different scored-week count was accepted")
    assert count == 48
    return {"station_intervals": count, "portfolio_intervals": 4, "stale_result_rejected": True}


if __name__ == "__main__":
    print(verify_uncertainty())
