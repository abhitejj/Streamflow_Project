"""Scientific invariants for corrected preprocessing, independent of missing raw data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import joblib
import numpy as np
import pandas as pd
import pytest
from pipeline_integrity import WEEKLY, fit_sequence_preprocessor, transform_sequence_features, preflight_era
from train_m2_models import extract_station_sequences
from pipeline_integrity import observed_weekly_flow, complete_weekly_sum
from pipeline_integrity import calibration_pairs
from pipeline_integrity import dataset_fingerprint, validate_model_dataset
from pipeline_integrity import resolve_cwc_source

def test_derived_cwc_requires_explicit_opt_in_and_complete_station_schema(tmp_path):
    archive = tmp_path/'river_discharge_manual_daily_cwc_ka_2001_2025.csv'
    with pytest.raises(FileNotFoundError, match='explicit opt-in'):
        resolve_cwc_source(archive)
    records = pd.DataFrame({'Station': ['Yadgir','Huvinhedigi','KUDIGE','Haralahalli'], 'State': ['Karnataka']*4,
                            'Data Acquisition Time': ['01-01-2021']*4,
                            'Manual Daily River Water Discharge (m3/sec)': [1.]*4,
                            'Latitude': [15.]*4, 'Longitude': [75.]*4})
    source = tmp_path/'derived.csv'
    records.to_csv(source, index=False)
    assert resolve_cwc_source(archive, source) == source
    records.iloc[:3].to_csv(source, index=False)
    with pytest.raises(ValueError, match='HARALAHALLI'):
        resolve_cwc_source(archive, source)
    records.drop(columns='Latitude').to_csv(source, index=False)
    with pytest.raises(ValueError, match='columns'):
        resolve_cwc_source(archive, source)

def test_changed_tensor_rejects_checkpoint(tmp_path):
    for i in range(12):
        pd.DataFrame({'pipeline_version': [2], 'x': [i]}).to_csv(tmp_path/f'weekly_tensor_{i}.csv', index=False)
    metadata = {'dataset_fingerprint': dataset_fingerprint(tmp_path)}
    validate_model_dataset(metadata, tmp_path)
    pd.DataFrame({'pipeline_version': [2], 'x': [999]}).to_csv(tmp_path/'weekly_tensor_0.csv', index=False)
    with pytest.raises(ValueError, match='not fitted'):
        validate_model_dataset(metadata, tmp_path)

def test_actual_tabular_training_bundle_audit_prediction_parity(tmp_path, monkeypatch):
    import train_forecast_models as trainer
    import audit_metrics as auditor
    tensors = tmp_path/'tensors'
    tensors.mkdir()
    for i in range(12):
        pd.DataFrame({'pipeline_version': [2]}).to_csv(tensors/f'weekly_tensor_{i}.csv', index=False)
    monkeypatch.setattr(trainer, 'TENSORS_DIR', str(tensors))
    monkeypatch.setattr(auditor, 'TENSORS_DIR', tensors)
    frame = pd.DataFrame({'flow_lag_0': np.arange(100, dtype=float), 'target_H1': np.arange(100, dtype=float)*1.1+2})
    train, val, test = frame.iloc[:60], frame.iloc[60:80], frame.iloc[80:]
    checkpoint = tmp_path/'model.joblib'
    expected = trainer.train_and_predict_m1(train, val, test, ['flow_lag_0'], 'target_H1', model_type='ridge', save_path=str(checkpoint))
    model, scaler, medians = auditor.load_estimator(checkpoint)
    features = test[['flow_lag_0']].fillna(medians).values
    actual = np.maximum(0, model.predict(scaler.transform(features)))
    np.testing.assert_allclose(expected, actual)

def test_calibration_ignores_post_training_changes():
    reference = pd.Series(2., index=pd.date_range('2001-01-01', '2020-12-31'))
    candidate = reference.copy() / 2
    before = calibration_pairs(reference, candidate)
    candidate.loc['2011-01-01':] = 10000
    pd.testing.assert_frame_equal(before, calibration_pairs(reference, candidate))

def test_model_reload_inference_parity(tmp_path):
    import torch
    from train_m2_models import HydroLSTM
    from audit_metrics import HydroLSTM as AuditHydroLSTM
    assert AuditHydroLSTM is HydroLSTM
    frame = pd.DataFrame({'x': [1., 2., 3., 4.], 'target': [2., 3., 4., 5.]}, index=pd.date_range('2020-01-06', periods=4, freq='7D'))
    scaler = fit_sequence_preprocessor(frame, ['x'])
    seq, _, _, _ = extract_station_sequences(frame, 3, ['x'], 'target', scaler=scaler)
    model = HydroLSTM(input_dim=1, hidden_dim=4, num_layers=2).eval()
    x = torch.tensor(seq, dtype=torch.float32)
    with torch.no_grad():
        expected = model(x)
    torch.save(model.state_dict(), tmp_path/'weights.pt')
    joblib.dump(scaler, tmp_path/'prep.joblib')
    loaded = AuditHydroLSTM(input_dim=1, hidden_dim=4, num_layers=2).eval()
    loaded.load_state_dict(torch.load(tmp_path/'weights.pt', weights_only=True))
    restored_seq, _, _, _ = extract_station_sequences(frame, 3, ['x'], 'target', scaler=joblib.load(tmp_path/'prep.joblib'))
    with torch.no_grad():
        torch.testing.assert_close(expected, loaded(torch.tensor(restored_seq, dtype=torch.float32)))

def test_week_excludes_issue_monday_and_future():
    s = pd.Series(1., index=pd.date_range('2020-01-06', periods=15))
    a = complete_weekly_sum(s)
    s.loc['2020-01-13':] = 999
    b = complete_weekly_sum(s)
    assert a.loc['2020-01-13'] == b.loc['2020-01-13'] == 7

def test_raw_counts_and_missing_rain():
    s = pd.Series([1., 2., 3., np.nan, np.nan, np.nan, np.nan], index=pd.date_range('2020-01-06', periods=7))
    assert observed_weekly_flow(s)[0].isna().all()
    s.iloc[3] = 4
    assert observed_weekly_flow(s)[0].iloc[0] == 2.5
    assert complete_weekly_sum(s).isna().all()

def test_frozen_preprocessing_checkpoint_parity(tmp_path):
    train = pd.DataFrame({'x': [1., 2., 3.]})
    scaler = fit_sequence_preprocessor(train, ['x'])
    target = pd.DataFrame({'x': [np.nan, 1000.]})
    before = transform_sequence_features(target, ['x'], scaler)
    joblib.dump(scaler, tmp_path/'preprocessor.joblib')
    restored = joblib.load(tmp_path/'preprocessor.joblib')
    np.testing.assert_array_equal(before, transform_sequence_features(target, ['x'], restored))
    assert restored.training_medians_['x'] == 2
    target.iloc[1] = -9999
    assert transform_sequence_features(target, ['x'], restored)[0,0] == before[0,0]

def test_sequence_gap_rejected():
    frame = pd.DataFrame({'x': [1., 2., 3.], 'target': [1., 2., 3.]}, index=pd.to_datetime(['2020-01-06','2020-01-13','2020-01-27']))
    scaler = fit_sequence_preprocessor(frame, ['x'])
    with pytest.raises(ValueError, match='seven-day'):
        extract_station_sequences(frame, 3, ['x'], 'target', scaler=scaler)

def test_missing_source_fails_before_rebuild(tmp_path):
    with pytest.raises(FileNotFoundError, match='Full rebuild blocked'):
        preflight_era(tmp_path, [{'gauge_id': 1, 'station': 'Example'}])
