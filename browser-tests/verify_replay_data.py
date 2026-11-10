"""Check every replay row and drought outcome against the audited CSVs."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/processed/model_results"
ASSETS = ROOT / "site/assets"


def verify_replay_data():
    manifest = json.loads((ASSETS / "replay-manifest.js").read_text().split("window.REPLAY_DATA = ", 1)[1].removesuffix(";\n"))
    assert manifest["schema"] == 1
    assert manifest["fields"] == ["issue_date", "target_date", "scored", "zero_as_missing", "observed", "M0", "M1", "M2", "M1-log", "M2-log"]
    predictions = RESULTS / "test_predictions.csv"
    thresholds = RESULTS / "drought_severity_sensitivity.csv"
    portfolio, _ = json.JSONDecoder().raw_decode((ASSETS / "benchmark-data.js").read_text().split("window.STATIONS = ", 1)[1])
    ids = {str(station["camels_id"]): station["id"] for station in portfolio}
    with (RESULTS / "audited_metrics_recomputed.csv").open(newline="") as file:
        csv_station_ids = {row["station"]: ids[row["gauge_id"]] for row in csv.DictReader(file)}
    assert set(manifest["chunks"]) == set(ids.values())
    for key, source in (("predictions", predictions), ("thresholds", thresholds)):
        assert manifest["sources"][key] == {"path": source.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}

    expected = defaultdict(list)
    with predictions.open(newline="") as file:
        for row in csv.DictReader(file):
            key = csv_station_ids[row["station"]], row["horizon"]
            values = [row[field] for field in ("y_obs", "m0", "m1_gbt", "m2_gbt", "m1_gbt_log", "m2_gbt_log")]
            expected[key].append([row["issue_date"], row["target_date"], row["scored"] == "True", row["zero_as_missing"] == "True",
                                  *(float(value) if value else None for value in values)])

    chunks = {}
    tail_rows = 0
    for station, leads in manifest["chunks"].items():
        for lead, entry in leads.items():
            payload = (ASSETS / entry["file"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
            chunk = json.loads(payload)
            assert (chunk["station"], chunk["lead"]) == (station, lead)
            assert chunk["rows"] == sorted(expected[station, lead], key=lambda row: row[1])
            for issue, target, scored, zero, *_ in chunk["rows"]:
                assert date.fromisoformat(target) - date.fromisoformat(issue) == timedelta(weeks=int(lead[:-1]))
                assert not (scored and zero)
                if target > manifest["test_end"]:
                    assert not scored
                    tail_rows += 1
            chunks[station, lead] = chunk
    assert len(chunks) == 48 and chunks.keys() == expected.keys()
    assert set((ASSETS / "replay").glob("*.json")) == {ASSETS / item["file"] for leads in manifest["chunks"].values() for item in leads.values()}

    checks = 0
    with thresholds.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["scope"] != "station":
                continue
            station, lead = ids[row["gauge_id"]], row["horizon"]
            chunk = chunks[station, lead]
            threshold = chunk["thresholds"][row["severity"]]
            assert threshold == float(row["threshold"])
            column = manifest["fields"].index(row["model"])
            outcomes = Counter((week[4] <= threshold, week[column] <= threshold) for week in chunk["rows"] if week[2])
            assert (outcomes[True, True], outcomes[True, False], outcomes[False, True], outcomes[False, False]) == tuple(
                int(row[field]) for field in ("Hits", "Miss", "FA", "CR")
            )
            assert outcomes.total() == int(row["n_eval"])
            checks += 1
    assert checks == 720
    return {"chunks": len(chunks), "saved_rows": sum(len(chunk["rows"]) for chunk in chunks.values()),
            "unscored_rows_after_test": tail_rows, "drought_outcome_groups": checks}


if __name__ == "__main__":
    print(verify_replay_data())
