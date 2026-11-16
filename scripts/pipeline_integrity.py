"""Shared, testable temporal and preprocessing contracts for corrected reruns."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CALIBRATION_START = '2001-01-01'
CALIBRATION_END = '2010-12-31'
WEEKLY = dict(rule='W-MON', closed='left', label='right')

def observed_weekly_flow(series):
    weeks = series.resample(**WEEKLY)
    counts = weeks.count()
    return weeks.mean().where(counts >= 4), counts

def complete_weekly_sum(series):
    return series.resample(**WEEKLY).sum(min_count=7)

def resolve_cwc_source(original, derived=None):
    """Use an explicitly chosen derived daily CWC table for the Karnataka archive."""
    original = Path(original)
    if derived is not None and original.name == 'river_discharge_manual_daily_cwc_ka_2001_2025.csv':
        derived = Path(derived)
        frame = pd.read_csv(derived, low_memory=False)
        discharge = 'Manual Daily River Water Discharge (m3/sec)'
        required = {'Station', 'State', 'Data Acquisition Time', discharge, 'Latitude', 'Longitude'}
        if not required.issubset(frame.columns):
            raise ValueError('Derived CWC source missing required columns: ' + ', '.join(sorted(required - set(frame.columns))))
        state = frame.State.astype(str).str.strip().str.upper()
        for station in ['YADGIR', 'HUVINHEDIGI', 'KUDIGE', 'HARALAHALLI']:
            selected = frame[(state == 'KARNATAKA') & (frame.Station.astype(str).str.strip().str.upper() == station)]
            dates = pd.to_datetime(selected['Data Acquisition Time'], dayfirst=True, errors='coerce')
            values = pd.to_numeric(selected[discharge], errors='coerce')
            usable = dates.notna() & np.isfinite(values)
            if not usable.any() or not dates[usable].between('2021-01-01', '2025-12-31').any():
                raise ValueError(f'Derived CWC source has no usable 2021–2025 daily records for Karnataka station {station}')
        return derived
    if original.is_file():
        return original
    if derived is None:
        raise FileNotFoundError(f'Missing original CWC archive: {original}; --derived-cwc is an explicit opt-in')
    if original.name != 'river_discharge_manual_daily_cwc_ka_2001_2025.csv':
        raise FileNotFoundError(f'Derived CWC opt-in does not replace this archive: {original}')
    derived = Path(derived)
    frame = pd.read_csv(derived, low_memory=False)
    discharge = 'Manual Daily River Water Discharge (m3/sec)'
    required = {'Station', 'State', 'Data Acquisition Time', discharge, 'Latitude', 'Longitude'}
    if not required.issubset(frame.columns):
        raise ValueError('Derived CWC source missing required columns: ' + ', '.join(sorted(required - set(frame.columns))))
    state = frame.State.astype(str).str.strip().str.upper()
    for station in ['YADGIR', 'HUVINHEDIGI', 'KUDIGE', 'HARALAHALLI']:
        selected = frame[(state == 'KARNATAKA') & (frame.Station.astype(str).str.strip().str.upper() == station)]
        dates = pd.to_datetime(selected['Data Acquisition Time'], dayfirst=True, errors='coerce')
        values = pd.to_numeric(selected[discharge], errors='coerce')
        usable = dates.notna() & np.isfinite(values)
        if not usable.any() or not dates[usable].between('2021-01-01', '2025-12-31').any():
            raise ValueError(f'Derived CWC source has no usable 2021–2025 daily records for Karnataka station {station}')
    return derived

def calibration_pairs(reference, candidate):
    paired = pd.concat({'reference': reference, 'candidate': candidate}, axis=1)
    paired = paired.loc[CALIBRATION_START:CALIBRATION_END].replace([np.inf, -np.inf], np.nan).dropna()
    if len(paired) < 365:
        raise ValueError('Calibration requires at least 365 paired finite training-period days')
    return paired.rename(columns={'reference': 'camels', 'candidate': 'imd'})

def require_corrected_tensors(directory):
    paths = list(Path(directory).glob('weekly_tensor_*.csv'))
    if len(paths) != 12:
        raise ValueError('Corrected rebuild requires all 12 station tensors')
    for path in paths:
        frame = pd.read_csv(path, usecols=lambda col: col == 'pipeline_version')
        if 'pipeline_version' not in frame or not frame.pipeline_version.eq(2).all():
            raise ValueError(f'Legacy tensor cannot enter corrected training/audit: {path.name}; rebuild raw inputs first')

def dataset_fingerprint(directory):
    require_corrected_tensors(directory)
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(directory).glob('weekly_tensor_*.csv'))}

def validate_model_dataset(metadata, directory):
    if metadata.get('dataset_fingerprint') != dataset_fingerprint(directory):
        raise ValueError('Model was not fitted against these corrected tensors; retrain before audit')

def fit_sequence_preprocessor(frame, columns):
    values = frame.loc[:, columns]
    medians = values.median()
    if medians.isna().any():
        raise ValueError('Training features have no observed values: ' + ', '.join(medians.index[medians.isna()]))
    scaler = StandardScaler().fit(values.fillna(medians))
    scaler.training_medians_ = medians
    scaler.feature_columns_ = list(columns)
    return scaler

def transform_sequence_features(frame, columns, scaler):
    if scaler is None or not hasattr(scaler, 'training_medians_'):
        raise ValueError('Missing frozen training preprocessing; retrain legacy checkpoint')
    if list(columns) != scaler.feature_columns_:
        raise ValueError('Checkpoint feature order differs from requested features')
    return scaler.transform(frame.loc[:, columns].fillna(scaler.training_medians_))

def require_saved_medians(medians):
    if medians is None:
        raise ValueError('Legacy estimator lacks frozen medians; retrain before evaluation')
    return medians

def preflight_era(directory, stations):
    missing = []
    for station in stations:
        path = Path(directory) / f"era5_land_{station['gauge_id']:05d}_{station['station']}.csv"
        if not path.is_file():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        required = {'Date', 'tavg_c', 'tmax_c', 'tmin_c', 'pet_mm'}
        if not required.issubset(frame.columns):
            raise ValueError(f'ERA5 input schema incomplete: {path}')
        dates = pd.to_datetime(frame.Date)
        if dates.duplicated().any():
            raise ValueError(f'Duplicate ERA5 dates: {path}')
        for col in required - {'Date'}:
            values = pd.to_numeric(frame[col], errors='raise')
            if not np.isfinite(values).any():
                raise ValueError(f'ERA5 field has no finite observations: {path} {col}')
        if not ((dates >= CALIBRATION_START) & (dates <= CALIBRATION_END)).any():
            raise ValueError(f'ERA5 input has no training-period calibration overlap: {path}')
        if dates.min() > pd.Timestamp('2001-01-01') or dates.max() < pd.Timestamp('2025-12-31'):
            raise ValueError(f'ERA5 source must cover calibration2001–2010 and evaluation2021–2025: {path}')
    if missing:
        raise FileNotFoundError('Full rebuild blocked: original daily ERA5 files required:\n' + '\n'.join(missing))
