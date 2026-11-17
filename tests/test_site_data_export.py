"""Keep the website's default Anandpur series tied to the audited forecasts."""
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_site_data
from build_site_data import ROOT, hydrographs


def test_anandpur_hero_series_preserves_audited_dates_scores_and_gaps():
    results = ROOT / 'data/processed/model_results'
    predictions = pd.read_csv(results / 'test_predictions.csv')
    audit = pd.read_csv(results / 'audited_metrics_recomputed.csv')
    exported = hydrographs()['anandpur']
    for lead in ('1w', '4w'):
        rows = predictions[predictions.station.eq('Anandpur') & predictions.horizon.eq(lead)].sort_values('target_date')
        actual = exported[lead]
        assert actual['x'] == rows.target_date.tolist()
        assert sum(value is None for value in actual['obs']) == int((~rows.scored).sum())
        for key, column in (('obs', 'y_obs'), ('m1', 'm1_gbt'), ('m2', 'm2_gbt')):
            values = np.array([np.nan if value is None else value for value in actual[key]])
            np.testing.assert_allclose(values[rows.scored], rows.loc[rows.scored, column], rtol=0, atol=0.000500001)
            assert np.isnan(values[~rows.scored]).all()
        record = audit[audit.station.eq('Anandpur') & audit.horizon.eq(lead)].iloc[0]
        assert actual['nse_m1'] == round(record.m1_nse, 3)
        assert actual['nse_m2'] == round(record.m2_nse, 3)
        assert actual['q95'] == round(rows.q95.iloc[0], 3)
        assert actual['q90'] == round(rows.q90.iloc[0], 3)


def test_anandpur_export_uses_shipped_assets_without_private_results(tmp_path, monkeypatch):
    expected = hydrographs()['anandpur']
    for name in ('benchmark-data.js', 'replay-manifest.js', 'replay/anandpur-1w.json', 'replay/anandpur-4w.json'):
        target = tmp_path / 'assets' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'assets' / name, target)
    monkeypatch.setattr(build_site_data, 'ROOT', tmp_path)
    assert not (tmp_path / 'data').exists()
    assert hydrographs()['anandpur'] == expected
    chunk = tmp_path / 'assets/replay/anandpur-1w.json'
    chunk.write_bytes(chunk.read_bytes() + b' ')
    with pytest.raises(ValueError, match='checksum mismatch'):
        hydrographs()
