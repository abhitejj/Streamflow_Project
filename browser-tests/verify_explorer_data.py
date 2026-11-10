"""Check the published comparison asset against every audited source row."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/processed/model_results"


def number(value):
    return float(value) if value else None


def verify_explorer_data():
    asset = (ROOT / "site/assets/explorer-data.js").read_bytes()
    assert asset == (ROOT / "assets/explorer-data.js").read_bytes()
    payload = json.loads(asset.decode().split("window.EXPLORER_DATA = ", 1)[1].removesuffix(";\n"))
    assert payload["schema"] == 1
    flow_path = RESULTS / "audited_metrics_recomputed.csv"
    drought_path = RESULTS / "drought_severity_sensitivity.csv"
    for mode, path in (("flow", flow_path), ("drought", drought_path)):
        assert payload["sources"][mode] == {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    flow_entries = set()
    drought_entries = set()
    with flow_path.open(newline="") as file:
        for row in csv.DictReader(file):
            gauge, lead = row["gauge_id"], row["horizon"]
            flow_entries.add((gauge, lead))
            actual = payload["stations"][gauge][lead]
            assert actual["n_eval"] == int(row["n_eval"])
            assert actual["flow"]["M0"] == {"nse": number(row["m0_nse_fixed"]),
                                             "log_nse": number(row["m0_log_nse"]),
                                             "kge": number(row["m0_kge_fixed"])}
            for model, prefix in (("M1", "m1"), ("M2", "m2"), ("M1-log", "m1_gbt_log"), ("M2-log", "m2_gbt_log")):
                assert actual["flow"][model] == {field: number(row[f"{prefix}_{field}"]) for field in ("nse", "log_nse", "kge")}

    with drought_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["scope"] != "station":
                continue
            gauge, lead, severity, model = (row[key] for key in ("gauge_id", "horizon", "severity", "model"))
            drought_entries.add((gauge, lead, severity, model))
            actual = payload["stations"][gauge][lead]
            group = actual["drought"][severity]
            assert actual["n_eval"] == int(row["n_eval"])
            assert group["threshold"] == float(row["threshold"])
            assert group["events"] == int(row["ObsEvents"])
            assert group["models"][model] == {
                "hits": int(row["Hits"]), "misses": int(row["Miss"]), "false_alarms": int(row["FA"]),
                "csi": number(row["CSI"]), "pod": number(row["POD"]), "far": number(row["FAR"]),
            }

    assert len(flow_entries) == 48 and len(drought_entries) == 720
    assert sum(len(gauge) for gauge in payload["stations"].values()) == len(flow_entries)
    return {"flow_station_leads": len(flow_entries), "flow_models": 5 * len(flow_entries), "drought_rows": len(drought_entries)}


if __name__ == "__main__":
    print(verify_explorer_data())
