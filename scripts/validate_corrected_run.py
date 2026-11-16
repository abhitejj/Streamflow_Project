#!/usr/bin/env python3
"""Validate one complete benchmark result set and write a traceable run receipt."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pipeline_integrity import dataset_fingerprint, validate_model_dataset

ROOT = Path(__file__).resolve().parents[1]
TENSORS = ROOT / 'data/processed/weekly_tensors'
RESULTS = ROOT / 'data/processed/model_results'

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def main():
    fingerprints = dataset_fingerprint(TENSORS)
    models = list((ROOT/'models/protocol1').glob('*.joblib')) + list((ROOT/'models/protocol2').glob('*.joblib'))
    assert len(models) == 324, f'Expected 312 tabular bundles and 12 LSTM sidecars, got {len(models)}'
    for path in models:
        validate_model_dataset(joblib.load(path), TENSORS)
    weights = list((ROOT/'models/protocol2').glob('*.pt'))
    assert len(weights) == 12
    predictions = pd.read_csv(RESULTS/'test_predictions.csv', parse_dates=['issue_date', 'target_date'])
    audit = pd.read_csv(RESULTS/'audited_metrics_recomputed.csv')
    predictions['gauge_id'] = predictions.station.map(audit[['station', 'gauge_id']].drop_duplicates().set_index('station').gauge_id)
    assert len(audit) == 48
    assert not predictions.duplicated(['gauge_id', 'horizon', 'issue_date']).any()
    checks = []
    for (gauge, lead), rows in predictions.groupby(['gauge_id', 'horizon']):
        source = next(TENSORS.glob(f'weekly_tensor_gauge_{int(gauge):05d}_*.csv'))
        tensor = pd.read_csv(source, parse_dates=['date']).set_index('date')
        horizon = int(lead[:-1])
        assert (rows.target_date == rows.issue_date + pd.Timedelta(weeks=horizon)).all()
        expected = tensor.loc[rows.issue_date, f'target_H{horizon}'].to_numpy()
        np.testing.assert_allclose(rows.y_obs.to_numpy(), expected, equal_nan=True)
        assert not tensor.loc[tensor.observed_flow_days.lt(4), 'flow_target_m3s'].notna().any()
        assert not rows.loc[rows.target_date.gt('2025-12-31'), 'scored'].any()
        record = audit[(audit.gauge_id == gauge) & (audit.horizon == lead)].iloc[0]
        assert int(rows.scored.sum()) == int(record.n_eval)
        checks.append({'gauge_id': int(gauge), 'lead': lead, 'scored': int(rows.scored.sum())})
    # Verify raw source bytes still match the inputs recorded during the rebuild.
    raw_hashes = {}
    for provenance in TENSORS.glob('*.provenance.json'):
        metadata = json.loads(provenance.read_text())
        for name, expected in metadata['source_sha256'].items():
            raw_hashes[name] = expected
    for name, expected in raw_hashes.items():
        assert sha(ROOT/'data/raw'/name) == expected, f'Raw input changed after rebuild: {name}'
    paths = models + weights + list(TENSORS.glob('*.provenance.json')) + list(RESULTS.glob('*.csv'))
    paths += list((ROOT/'scripts').glob('*.py')) + list((ROOT/'assets').glob('*.js')) + list((ROOT/'assets/replay').glob('*.json'))
    paths += list((ROOT/'visualizations/source').glob('*.html')) + [ROOT/'requirements.txt']
    paths += [ROOT/'README.md', ROOT/'docs/PROJECT_REPORT_DOCUMENTATION.md', ROOT/'docs/DATA_ACCESS.md', ROOT/'docs/DECISION_LOG.md', ROOT/'docs/UNCERTAINTY_METHOD.md']
    paths += list((ROOT/'tests').glob('*.py'))
    paths += list((ROOT/'tools').glob('*.py'))
    paths += [ROOT/'data/processed/validation_reports/weather_input_manifest.json']
    manifest = {'run_kind': 'retrospective streamflow and drought benchmark', 'validated_at_utc': datetime.now(timezone.utc).isoformat(),
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'working_tree_changes': True, 'tensor_sha256': fingerprints, 'raw_source_sha256': raw_hashes,
                'artifact_sha256': {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(paths))},
                'model_counts': {'tabular': 312, 'lstm': 12}, 'station_lead_checks': checks,
                'weather_source': 'User-supplied Open-Meteo Archive default Best Match, GMT daily temperature and FAO reference ET0; not a verified homogeneous ERA5-Land series'}
    target = ROOT/'data/processed/validation_reports/corrected_run_manifest.json'
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Validated {len(checks)} station–lead cases, 324 models and exact target alignment; wrote {target}')

if __name__ == '__main__':
    main()
