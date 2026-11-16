"""Drought assessment labels must preserve raw contingency evidence."""
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_explorer_data import drought_availability, load_results


def test_structurally_unavailable_event_is_distinguished_from_zero_test_events():
    assert drought_availability('Yadgir', 0., 0)['assessment'] == 'unavailable_zero_threshold'
    assert drought_availability('Kantamal', 1., 0)['assessment'] == 'no_observed_events'
    assert drought_availability('Haralahalli', 0., 5)['assessment'] == 'observed_events'


def test_actual_availability_preserves_raw_counts_and_ratios():
    results = load_results()
    source = pd.read_csv(Path(__file__).resolve().parents[1] / 'data/processed/model_results/drought_severity_sensitivity.csv')
    rows = source[(source.scope == 'station') & (source.model == 'M2')]
    assert len(rows) == 144
    for row in rows.itertuples():
        group = results[str(row.gauge_id)][row.horizon]['drought'][row.severity]
        assert group['events'] == row.ObsEvents
        assert group['threshold'] == pytest.approx(row.threshold, rel=1e-14)
        assert group['models']['M2']['hits'] == row.Hits
        assert group['models']['M2']['false_alarms'] == row.FA
        if pd.isna(row.CSI):
            assert group['models']['M2']['csi'] is None
        else:
            assert group['models']['M2']['csi'] == pytest.approx(row.CSI)
        assert group['assessment'] == drought_availability(row.station, row.threshold, row.ObsEvents)['assessment']
    assert all(results['4064'][h]['drought']['Q95']['assessment'] == 'unavailable_zero_threshold' for h in ['1w','2w','4w','8w'])
    q95 = rows[rows.severity == 'Q95']
    assert int(q95.ObsEvents.eq(0).sum()) == 12
