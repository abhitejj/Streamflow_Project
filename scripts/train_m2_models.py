#!/usr/bin/env python3
"""
Hydro-Meteorological Model Training & Benchmarking Pipeline (Tier M2)

Evaluates the predictive value of multi-source meteorological forcings
(IMD Rainfall, ERA5-Land Temperature & PET, Climatic Water Deficit, and API)
combined with antecedent streamflow memory.

Architectural Configurations:
1. Protocol 1 (Temporal Multi-Horizon Benchmarking):
   - Algorithm: HistGradientBoostingRegressor with validation hyperparameter tuning.
   - Features: 26 hydro-meteorological predictors (Flow lags + Rain sums + API + Temp + PET + Deficits + Calendar).
   - Target: Volumetric streamflow (m³/s) across lead times H in [1, 2, 4, 8] weeks.
2. Protocol 2 (Cross-Basin Zero-Shot Spatial Generalization):
   - Algorithm: PyTorch Deep Recurrent LSTM (HydroLSTM).
   - Sequential Formulation: Sliding temporal lookback window of L=8 weeks leading up to issuance week t.
   - Features: Area-normalized intensive variables (specific runoff, rain, PET, water deficit, temp, API, calendar).
   - Target: Area-normalized specific runoff (mm/week) across lead times H in [1, 2, 4, 8] weeks.
   - Post-processing: Converted back to volumetric discharge (m³/s) for holdout basin evaluation.

Operational & Leakage Guardrails (Strict Compliance):
1. Terminal H-week purge embargo (Approval Gate AG-TS-02) applied to Training (1980–2010)
   and Validation (2011–2015) sets to completely eliminate future target leakage.
2. Identical-row evaluation mask against M0 Baseline (Persistence / Seasonal Climatology).
3. Physical non-negativity constraint (y >= 0) enforced on all model outputs.
4. Categorical drought thresholds (Q80, Q90, Q95) fit strictly on the Training fold (1980–2010).
"""

import os
import sys
import copy
import argparse
import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. DIRECTORY AND PATH ENFORCEMENT
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

TENSORS_DIR = os.path.join(PROJECT_ROOT, 'data/processed/weekly_tensors')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'data/processed/model_results')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'data/processed/validation_reports')
MODELS_DIR  = os.path.join(PROJECT_ROOT, 'models')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'protocol1'), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, 'protocol2'), exist_ok=True)

# ==============================================================================
# 2. EXPERIMENTAL SPLIT BOUNDARIES & EMBARGO RULES (AG-TS-02)
# ==============================================================================
TRAIN_START = '1980-01-01'
TRAIN_END   = '2010-12-31'
VAL_START   = '2011-01-01'
VAL_END     = '2015-12-31'
TEST_START  = '2016-01-01'
TEST_END    = '2025-12-31'

# 12-Station Matrix with State and Holdout Status
PORTFOLIO_CONFIG = [
    # Maharashtra (MH)
    {'state': 'Maharashtra', 'station': 'Karad', 'gauge_id': 4021, 'is_holdout': False, 'area_km2': 5462.0},
    {'state': 'Maharashtra', 'station': 'Warunji', 'gauge_id': 4063, 'is_holdout': False, 'area_km2': 1890.0},
    {'state': 'Maharashtra', 'station': 'Ashti', 'gauge_id': 3005, 'is_holdout': False, 'area_km2': 50990.0},
    {'state': 'Maharashtra', 'station': 'Bamni (Ballarsha)', 'gauge_id': 3008, 'is_holdout': True, 'area_km2': 46020.0},
    # Odisha (OD)
    {'state': 'Odisha', 'station': 'Anandpur', 'gauge_id': 7002, 'is_holdout': False, 'area_km2': 8570.0},
    {'state': 'Odisha', 'station': 'Kashinagar', 'gauge_id': 16007, 'is_holdout': False, 'area_km2': 7820.0},
    {'state': 'Odisha', 'station': 'Gomlai', 'gauge_id': 7007, 'is_holdout': False, 'area_km2': 21950.0},
    {'state': 'Odisha', 'station': 'Kantamal', 'gauge_id': 8013, 'is_holdout': True, 'area_km2': 19600.0},
    # Karnataka (KA)
    {'state': 'Karnataka', 'station': 'Yadgir', 'gauge_id': 4064, 'is_holdout': False, 'area_km2': 69863.0},
    {'state': 'Karnataka', 'station': 'Huvinhedigi', 'gauge_id': 4017, 'is_holdout': False, 'area_km2': 55150.0},
    {'state': 'Karnataka', 'station': 'KUDIGE', 'gauge_id': 5010, 'is_holdout': False, 'area_km2': 1934.0},
    {'state': 'Karnataka', 'station': 'Haralahalli', 'gauge_id': 4013, 'is_holdout': True, 'area_km2': 14582.0},
]

# Complete 26-Feature M2 Tabular Predictor Set (Protocol 1)
M2_TABULAR_FEATURES = [
    # 1. Flow memory & recession dynamics (10 features)
    'flow_lag_0',
    'flow_lag_1',
    'flow_lag_2',
    'flow_lag_3',
    'flow_lag_4',
    'flow_lag_7',
    'flow_std_7d',
    'flow_min_7d',
    'flow_max_7d',
    'flow_recession_slope',
    # 2. Antecedent precipitation memory & soil recharge (7 features)
    'rain_sum_1w',
    'rain_sum_2w',
    'rain_sum_1m',
    'rain_sum_2m',
    'rain_sum_3m',
    'api_30d',
    'dry_days_count',
    # 3. Thermal drivers & evaporative demand (5 features)
    'temp_mean_1w',
    'temp_max_1w',
    'temp_min_1w',
    'pet_sum_1w',
    'pet_sum_1m',
    # 4. Climatic water deficits / catchment drying stress (2 features)
    'water_deficit_1w',
    'water_deficit_1m',
    # 5. Annual Fourier calendar harmonics (2 features)
    'sin_week',
    'cos_week',
]

# Sequential Intensive Features for PyTorch LSTM (Protocol 2)
# All features are normalized by depth (mm), degrees (°C), or dimensionless harmonics
M2_SEQUENTIAL_FEATURES = [
    'flow_target_specific_mm',  # Specific runoff (mm/week)
    'rain_sum_1w',              # 7-day precipitation (mm)
    'pet_sum_1w',               # 7-day evaporative demand (mm)
    'water_deficit_1w',         # Net atmospheric water balance P - PET (mm)
    'temp_mean_1w',             # Mean weekly temperature (°C)
    'temp_max_1w',              # Maximum weekly temperature (°C)
    'api_30d',                  # Antecedent Precipitation Index (mm)
    'dry_days_count',           # Count of dry days (P < 1mm)
    'sin_week',                 # Calendar Fourier sine
    'cos_week',                 # Calendar Fourier cosine
]

# ==============================================================================
# 3. SPLIT PARTITIONING WITH TERMINAL H-WEEK EMBARGO (AG-TS-02)
# ==============================================================================
def partition_splits_with_embargo(df, horizon_weeks):
    """
    Partitions the 46-year dataset into Train, Val, and Test folds.
    Enforces Approval Gate AG-TS-02:
    - Drops the terminal H weeks of the Training fold (ending 2010-12-31).
    - Drops the terminal H weeks of the Validation fold (ending 2015-12-31).
    - Test fold is un-purged (out-of-sample final evaluation block).
    """
    mask_train = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    mask_val   = (df.index >= VAL_START) & (df.index <= VAL_END)
    mask_test  = (df.index >= TEST_START) & (df.index <= TEST_END)
    
    df_train_raw = df[mask_train].copy()
    df_val_raw   = df[mask_val].copy()
    df_test      = df[mask_test].copy()
    
    # Enforce terminal H-week purge embargo
    if horizon_weeks > 0:
        df_train = df_train_raw.iloc[:-horizon_weeks].copy()
        df_val   = df_val_raw.iloc[:-horizon_weeks].copy()
    else:
        df_train = df_train_raw.copy()
        df_val   = df_val_raw.copy()
        
    return df_train, df_val, df_test

# ==============================================================================
# 4. AUDITED EVALUATION ENGINE (TRACK A & TRACK B)
# ==============================================================================
def compute_track_a_metrics(y_true, y_pred, clip_nonnegative=True, max_clip_dist=10.0, eps_var=1e-6, eps_mean=1e-6):
    """
    Computes Track A continuous hydrograph metrics:
    - NSE with near-zero variance guards and lower-bound clamping.
    - KGE-2009 (r, alpha, beta) with exact flat-regime guards.
    - RMSE and MAE.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    n_valid = int(np.sum(valid))
    if n_valid < 5:
        return {
            'N_obs': n_valid, 'NSE': np.nan, 'NSE_raw': np.nan, 'KGE': np.nan, 'KGE_raw': np.nan,
            'RMSE': np.nan, 'MAE': np.nan, 'KGE_r': np.nan, 'KGE_alpha': np.nan, 'KGE_beta': np.nan,
            'is_degenerate': True, 'degeneracy_reasons': [f'N_valid ({n_valid}) < 5'], 'n_negative_preds': 0
        }
    
    y = y_true[valid]
    y_hat_raw = y_pred[valid]
    n_neg = int(np.sum(y_hat_raw < 0))
    y_hat = np.maximum(y_hat_raw, 0.0) if clip_nonnegative else y_hat_raw
    
    mae = float(np.mean(np.abs(y - y_hat)))
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    
    # 1. Nash-Sutcliffe Efficiency (NSE)
    ss_res = float(np.sum((y - y_hat) ** 2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    
    is_degenerate = False
    degeneracy_reasons = []
    
    if ss_tot < eps_var:
        is_degenerate = True
        degeneracy_reasons.append('Zero-variance observations (ss_tot < 1e-6)')
        if ss_res < eps_var:
            nse_raw = 1.0
            nse = 1.0
        else:
            nse_raw = -float('inf')
            nse = -max_clip_dist
    else:
        nse_raw = float(1.0 - (ss_res / ss_tot))
        nse = float(np.clip(nse_raw, -max_clip_dist, 1.0))
        if nse_raw < -max_clip_dist:
            degeneracy_reasons.append(f'NSE clamped from {nse_raw:.2f} to {-max_clip_dist}')
            
    # 2. Kling-Gupta Efficiency (KGE-2009: r, alpha, beta)
    std_y = float(np.std(y))
    std_hat = float(np.std(y_hat))
    mean_y = y_mean
    mean_hat = float(np.mean(y_hat))
    
    if std_y < eps_var and std_hat < eps_var:
        if np.abs(mean_y - mean_hat) < eps_mean:
            r = 1.0
            alpha = 1.0
        else:
            r = 0.0
            alpha = 1.0
    elif std_y < eps_var:
        r = 0.0
        alpha = max_clip_dist
        is_degenerate = True
        degeneracy_reasons.append('Observation variance near zero (std_y < 1e-6)')
    elif std_hat < eps_var:
        r = 0.0
        alpha = 0.0
    else:
        cov = float(np.mean((y - mean_y) * (y_hat - mean_hat)))
        r = float(np.clip(cov / (std_y * std_hat), -1.0, 1.0))
        alpha = float(std_hat / std_y)
        
    if mean_y < eps_mean:
        if np.abs(mean_hat - mean_y) < eps_mean:
            beta = 1.0
        else:
            beta = float(np.clip((mean_hat + eps_mean) / (mean_y + eps_mean), 0.0, max_clip_dist))
    else:
        beta = float(np.clip(mean_hat / mean_y, 0.0, max_clip_dist))
        
    ed = np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    kge_raw = float(1.0 - ed)
    kge = float(np.clip(kge_raw, -max_clip_dist, 1.0))
    if kge_raw < -max_clip_dist:
        degeneracy_reasons.append(f'KGE clamped from {kge_raw:.2f} to {-max_clip_dist}')
        
    return {
        'N_obs': n_valid,
        'NSE': nse,
        'NSE_raw': nse_raw,
        'KGE': kge,
        'KGE_raw': kge_raw,
        'RMSE': rmse,
        'MAE': mae,
        'KGE_r': r,
        'KGE_alpha': alpha,
        'KGE_beta': beta,
        'is_degenerate': is_degenerate,
        'degeneracy_reasons': degeneracy_reasons,
        'n_negative_preds': n_neg
    }

def fit_training_drought_thresholds(y_train):
    """
    Computes drought severity thresholds strictly from the Training fold (1980–2010):
    - Mild Drought: 20th percentile (Q_80)
    - Moderate Drought: 10th percentile (Q_90)
    - Severe Drought: 5th percentile (Q_95)
    Strict Rule: Never fit thresholds on validation or test sets.
    """
    valid_train = y_train[~np.isnan(y_train)]
    if len(valid_train) == 0:
        return {'Q80_mild': 0.0, 'Q90_moderate': 0.0, 'Q95_severe': 0.0}
    return {
        'Q80_mild': float(np.percentile(valid_train, 20.0)),
        'Q90_moderate': float(np.percentile(valid_train, 10.0)),
        'Q95_severe': float(np.percentile(valid_train, 5.0)),
    }

def calc_drought_contingency(y_true, y_pred, threshold):
    """
    Computes 2x2 contingency table metrics for hydrological drought alerts:
    Drought event defined as: Flow <= Threshold.
    Returns: POD (Recall), FAR (False Alarm Ratio), CSI (Critical Success Index / Threat Score).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if np.sum(valid) == 0:
        return {'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan, 'Hits': 0, 'Misses': 0, 'FalseAlarms': 0, 'CorrectNegatives': 0}
        
    obs_drought  = y_true[valid] <= threshold
    pred_drought = y_pred[valid] <= threshold
    
    hits = int(np.sum(obs_drought & pred_drought))
    misses = int(np.sum(obs_drought & (~pred_drought)))
    false_alarms = int(np.sum((~obs_drought) & pred_drought))
    correct_negs = int(np.sum((~obs_drought) & (~pred_drought)))
    
    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else np.nan
    far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else np.nan
    denom_csi = hits + misses + false_alarms
    csi = float(hits / denom_csi) if denom_csi > 0 else np.nan
    
    return {
        'POD': pod,
        'FAR': far,
        'CSI': csi,
        'Hits': hits,
        'Misses': misses,
        'FalseAlarms': false_alarms,
        'CorrectNegatives': correct_negs
    }

# ==============================================================================
# 5. M0 BASELINE ENGINE
# ==============================================================================
def predict_m0_baseline(df_train, df_eval, horizon_weeks, target_col='target_H1'):
    """
    Enforces M0 Baseline rules:
    - H in [1, 2]: Pure persistence (Q_t, contemporaneous issue-week flow).
    - H in [4, 8]: Training-fold seasonal weekly climatology.
    - Halts execution if pure persistence is attempted for H > 2.
    """
    if horizon_weeks > 2:
        flow_col = 'flow_target_specific_mm' if 'specific' in target_col else ('flow_target_m3s' if 'flow_target_m3s' in df_train.columns else 'flow_lag_0')
        train_valid = df_train.dropna(subset=[flow_col]).copy()
        train_weeks = train_valid.index.isocalendar().week
        climatology = train_valid.groupby(train_weeks)[flow_col].mean()
        
        target_dates = df_eval.index + pd.Timedelta(weeks=horizon_weeks)
        target_weeks = target_dates.isocalendar().week
        
        pred_baseline = target_weeks.map(climatology).fillna(train_valid[flow_col].mean())
        return np.maximum(0.0, pred_baseline.values)
    elif horizon_weeks in [1, 2]:
        # For specific runoff targets, use specific flow; for volumetric, use flow_lag_0
        if 'specific' in target_col:
            return np.maximum(0.0, df_eval['flow_target_specific_mm'].values)
        return np.maximum(0.0, df_eval['flow_lag_0'].values)
    else:
        raise ValueError(f"Unsupported forecast horizon: {horizon_weeks}w")

# ==============================================================================
# 6. PROTOCOL 1: M2 HIST-GRADIENT-BOOSTING ENGINE
# ==============================================================================
def train_and_predict_m2_gbt(df_train, df_val, df_test, feature_cols, target_col, save_path=None):
    """
    Trains HistGradientBoostingRegressor on the 26 hydro-meteorological features.
    Tuning: Evaluates hyperparameter combinations on chronological df_val.
    Models are never refit on Train+Val, preserving parameter estimation integrity.
    """
    tr_clean = df_train.dropna(subset=[target_col])
    X_train = tr_clean[feature_cols].copy()
    y_train = tr_clean[target_col].values
    
    val_clean = df_val.dropna(subset=[target_col])
    X_val = val_clean[feature_cols].copy()
    y_val = val_clean[target_col].values
    
    # Feature imputation: compute medians strictly on Training fold
    feat_medians = X_train.median()
    X_train = X_train.fillna(feat_medians)
    X_val   = X_val.fillna(feat_medians)
    X_test  = df_test[feature_cols].fillna(feat_medians)
    
    # Hyperparameter tuning on validation fold
    best_score = float('inf')
    best_model = None
    
    depth_grid = [3, 5, 7]
    l2_grid = [0.0, 1.0, 10.0]
    lr_grid = [0.03, 0.05, 0.1]
    
    for d in depth_grid:
        for l2 in l2_grid:
            for lr in lr_grid:
                est = HistGradientBoostingRegressor(
                    max_depth=d,
                    l2_regularization=l2,
                    learning_rate=lr,
                    max_iter=150,
                    random_state=42
                )
                est.fit(X_train, y_train)
                val_preds = est.predict(X_val)
                score = float(np.sqrt(np.mean((y_val - val_preds) ** 2)))
                if score < best_score:
                    best_score = score
                    best_model = est
                    
    preds_test = best_model.predict(X_test)
    preds_test = np.maximum(0.0, preds_test)
    
    if save_path:
        joblib.dump(best_model, save_path)
        
    return preds_test

def execute_protocol_1(horizons=[1, 2, 4, 8]):
    """
    Executes Protocol 1 (Temporal Multi-Horizon Benchmarking):
    - Evaluates M2 (HistGradientBoosting) vs M0 Baseline on the 10-year Test block (2016–2025).
    - Features: Complete 26-feature hydro-meteorological matrix.
    - Slices 46-year dataset for each of the 12 catchments with H-week purge embargo.
    - Evaluates on identical test row intersections.
    """
    print("\n" + "=" * 80)
    print(" EXECUTION: PROTOCOL 1 (Station-Specific Multi-Horizon Benchmarking)")
    print(" Model Type: M2 (HistGradientBoosting: Flow Lags + Multi-Source Weather)")
    print(f" Horizons  : {horizons} weeks")
    print("=" * 80)
    
    results = []
    
    for stn_info in PORTFOLIO_CONFIG:
        station = stn_info['station']
        gauge_id = stn_info['gauge_id']
        state = stn_info['state']
        area_km2 = stn_info['area_km2']
        
        tensor_file = os.path.join(TENSORS_DIR, f'weekly_tensor_gauge_{gauge_id:05d}_{station}.csv')
        if not os.path.isfile(tensor_file):
            print(f"[-] WARNING: Feature tensor not found for {station}: {tensor_file}")
            continue
            
        df = pd.read_csv(tensor_file, parse_dates=['date']).set_index('date')
        
        for H in horizons:
            target_col = f'target_H{H}'
            if target_col not in df.columns:
                print(f"[-] Target {target_col} missing in tensor for {station}.")
                continue
                
            # Split with terminal H-week embargo
            df_train, df_val, df_test = partition_splits_with_embargo(df, horizon_weeks=H)
            
            # 1. Fit drought thresholds on Training fold only
            drought_thresholds = fit_training_drought_thresholds(df_train[target_col].values)
            q80 = drought_thresholds['Q80_mild']
            q90 = drought_thresholds['Q90_moderate']
            q95 = drought_thresholds['Q95_severe']
            
            # 2. Generate M0 predictions
            preds_m0 = predict_m0_baseline(df_train, df_test, horizon_weeks=H, target_col=target_col)
            
            # 3. Generate M2 predictions & serialize trained estimator
            clean_name = station.replace(' ', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(MODELS_DIR, 'protocol1', f"m2_gbt_gauge_{gauge_id:05d}_{clean_name}_H{H}.joblib")
            
            preds_m2 = train_and_predict_m2_gbt(
                df_train, df_val, df_test,
                feature_cols=M2_TABULAR_FEATURES,
                target_col=target_col,
                save_path=save_path
            )
            
            y_test = df_test[target_col].values
            flow_0 = df_test['flow_lag_0'].values
            
            # 4. Enforce Identical Row Evaluation Contract
            eval_mask = (~np.isnan(y_test)) & (~np.isnan(preds_m0)) & (~np.isnan(preds_m2)) & (~np.isnan(flow_0))
            
            y_eval        = y_test[eval_mask]
            preds_m0_eval = preds_m0[eval_mask]
            preds_m2_eval = preds_m2[eval_mask]
            
            # Low-flow / zero-variance flagging
            std_y = float(np.std(y_eval)) if len(y_eval) > 0 else 0.0
            mean_y = float(np.mean(y_eval)) if len(y_eval) > 0 else 0.0
            unstable_flag = bool((std_y < 0.05) or (mean_y < 0.1))
            
            # 5. Compute Track A Metrics
            m0_track_a = compute_track_a_metrics(y_eval, preds_m0_eval)
            m2_track_a = compute_track_a_metrics(y_eval, preds_m2_eval)
            
            # 6. Compute Track B Drought Metrics
            m0_drought_q80 = calc_drought_contingency(y_eval, preds_m0_eval, q80)
            m0_drought_q90 = calc_drought_contingency(y_eval, preds_m0_eval, q90)
            m0_drought_q95 = calc_drought_contingency(y_eval, preds_m0_eval, q95)
            
            m2_drought_q80 = calc_drought_contingency(y_eval, preds_m2_eval, q80)
            m2_drought_q90 = calc_drought_contingency(y_eval, preds_m2_eval, q90)
            m2_drought_q95 = calc_drought_contingency(y_eval, preds_m2_eval, q95)
            
            delta_nse = m2_track_a['NSE'] - m0_track_a['NSE']
            delta_kge = m2_track_a['KGE'] - m0_track_a['KGE']
            
            print(f"[{state[:2]}] {station:<18} H={H}w | M0 NSE: {m0_track_a['NSE']:+.3f} -> M2 NSE: {m2_track_a['NSE']:+.3f} (Δ {delta_nse:+.3f}) | M2 KGE: {m2_track_a['KGE']:+.3f}")
            
            results.append({
                'Protocol': 'Protocol 1 (Temporal)',
                'State': state,
                'Station': station,
                'Gauge_ID': gauge_id,
                'Catchment_Area_km2': area_km2,
                'Horizon_Weeks': H,
                'Model_Family': 'M2_GBT',
                'N_Obs': m2_track_a['N_obs'],
                'Unstable_Metrics_Flag': unstable_flag,
                'Obs_Mean_m3s': mean_y,
                'Obs_Std_m3s': std_y,
                # Drought Thresholds
                'Q80_Mild_Thresh': q80,
                'Q90_Moderate_Thresh': q90,
                'Q95_Severe_Thresh': q95,
                # Track A: Continuous Hydrograph
                'M0_NSE': m0_track_a['NSE'],
                'M0_KGE': m0_track_a['KGE'],
                'M0_RMSE': m0_track_a['RMSE'],
                'M0_MAE': m0_track_a['MAE'],
                'M0_KGE_r': m0_track_a['KGE_r'],
                'M0_KGE_alpha': m0_track_a['KGE_alpha'],
                'M0_KGE_beta': m0_track_a['KGE_beta'],
                'M2_NSE': m2_track_a['NSE'],
                'M2_KGE': m2_track_a['KGE'],
                'M2_RMSE': m2_track_a['RMSE'],
                'M2_MAE': m2_track_a['MAE'],
                'M2_KGE_r': m2_track_a['KGE_r'],
                'M2_KGE_alpha': m2_track_a['KGE_alpha'],
                'M2_KGE_beta': m2_track_a['KGE_beta'],
                'Delta_NSE': delta_nse,
                'Delta_KGE': delta_kge,
                # Track B: Mild Drought Q80
                'M0_Mild_CSI': m0_drought_q80['CSI'],
                'M0_Mild_POD': m0_drought_q80['POD'],
                'M0_Mild_FAR': m0_drought_q80['FAR'],
                'M2_Mild_CSI': m2_drought_q80['CSI'],
                'M2_Mild_POD': m2_drought_q80['POD'],
                'M2_Mild_FAR': m2_drought_q80['FAR'],
                # Track B: Moderate Drought Q90
                'M0_Mod_CSI': m0_drought_q90['CSI'],
                'M0_Mod_POD': m0_drought_q90['POD'],
                'M0_Mod_FAR': m0_drought_q90['FAR'],
                'M2_Mod_CSI': m2_drought_q90['CSI'],
                'M2_Mod_POD': m2_drought_q90['POD'],
                'M2_Mod_FAR': m2_drought_q90['FAR'],
                # Track B: Severe Drought Q95
                'M0_Severe_CSI': m0_drought_q95['CSI'],
                'M0_Severe_POD': m0_drought_q95['POD'],
                'M0_Severe_FAR': m0_drought_q95['FAR'],
                'M2_Severe_CSI': m2_drought_q95['CSI'],
                'M2_Severe_POD': m2_drought_q95['POD'],
                'M2_Severe_FAR': m2_drought_q95['FAR'],
            })
            
    df_results = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, 'protocol1_temporal_results_m2_gbt.csv')
    df_results.to_csv(out_csv, index=False)
    print(f"\n[+] Protocol 1 models saved to: {os.path.join(MODELS_DIR, 'protocol1')}")
    print(f"[+] Protocol 1 results saved to: {out_csv}")
    return df_results

# ==============================================================================
# 7. PROTOCOL 2: PYTORCH LSTM ARCHITECTURE (SPATIAL GENERALIZATION)
# ==============================================================================
class HydroLSTM(nn.Module):
    """
    Deep Recurrent Hydrological Model for Multi-Horizon Streamflow Forecasting.
    
    Architecture:
    - Input: Sliding sequence of length L (lookback weeks) x F (intensive hydro-met features).
    - Recurrent Layer: 2-layer stacked LSTM with inter-layer dropout to model non-linear
      catchment storage, delayed groundwater drainage, and soil moisture recharge dynamics.
    - Readout Head: Multi-layer perceptron (Linear -> ReLU -> Dropout -> Linear)
      mapping the sequence's final hidden state to future specific runoff (q_{t+H}).
    - Physics Constraint: ReLU activation on final output enforcing physical non-negativity (q >= 0).
    """
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2, dropout=0.2):
        super(HydroLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Extract representation from the final lookback timestep (week t)
        last_hidden = out[:, -1, :]  # (batch_size, hidden_dim)
        h = self.relu(self.fc1(last_hidden))
        h = self.dropout(h)
        pred = self.fc2(h)  # (batch_size, 1)
        # Enforce physical non-negativity: specific runoff is strictly >= 0
        return torch.relu(pred).squeeze(-1)

class HydroSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping multi-week temporal lookback sequences and targets.
    """
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def extract_station_sequences(df, seq_len, feature_cols, target_col, scaler=None, fit_scaler=False, buffer_df=None):
    """
    Constructs chronological sequences of length seq_len ending on each sample week t.
    If buffer_df is provided (the final seq_len - 1 rows of the preceding fold),
    prepends it to df so that 100% of rows in df have complete, unbroken lookback sequences.
    """
    if buffer_df is not None and len(buffer_df) > 0:
        buffer_slice = buffer_df.iloc[-(seq_len - 1):]
        combined = pd.concat([buffer_slice, df], axis=0)
        start_idx = len(buffer_slice)
    else:
        combined = df.copy()
        start_idx = seq_len - 1
        
    features_raw = combined[feature_cols].copy()
    
    # Impute missing feature values using column medians
    if fit_scaler:
        scaler = StandardScaler()
        impute_vals = features_raw.median()
        features_imputed = features_raw.fillna(impute_vals)
        features_scaled = scaler.fit_transform(features_imputed)
    else:
        impute_vals = features_raw.median()
        features_imputed = features_raw.fillna(impute_vals)
        features_scaled = scaler.transform(features_imputed) if scaler is not None else features_imputed.values
        
    targets_raw = combined[target_col].values
    
    sequences = []
    targets = []
    valid_indices = []
    
    for i in range(start_idx, len(combined)):
        seq = features_scaled[i - seq_len + 1 : i + 1, :]
        tgt = targets_raw[i]
        
        # Check validity: sequence must have no NaNs and target must be observed
        if not np.isnan(tgt) and not np.isnan(seq).any():
            sequences.append(seq)
            targets.append(tgt)
            valid_indices.append(combined.index[i])
            
    if len(sequences) == 0:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,)), [], scaler
        
    return np.array(sequences), np.array(targets), valid_indices, scaler

def train_lstm_regressor(train_loader, val_loader, input_dim, hidden_dim=64, num_layers=2,
                         epochs=50, lr=0.005, patience=10, device='cpu'):
    """
    Trains the HydroLSTM model using early stopping on the validation fold.
    Restores the best weights to ensure optimal generalization without overfitting.
    """
    model = HydroLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5)
    
    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
            n_train += len(batch_y)
            
        train_loss = train_loss / max(1, n_train)
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_y)
                n_val += len(batch_y)
                
        val_loss = val_loss / max(1, n_val)
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    model.load_state_dict(best_weights)
    return model

def execute_protocol_2(horizons=[1, 2, 4, 8], seq_len=8, hidden_dim=64, epochs=50, batch_size=64, device_arg='auto'):
    """
    Executes Protocol 2 (Cross-Basin Zero-Shot Spatial Transfer via PyTorch LSTM):
    - Within each state (MH, OD, KA), trains HydroLSTM on Stations 1, 2, and 3 using specific runoff (mm/week).
    - Evaluates zero-shot on the unseen held-out Station 4 (Bamni, Kantamal, Haralahalli).
    - Converts specific runoff predictions back to volumetric discharge (Q = q * Area / 604.8).
    - Evaluates on identical row mask against M0 persistence.
    """
    print("\n" + "=" * 80)
    print(" EXECUTION: PROTOCOL 2 (Cross-Basin Zero-Shot Spatial Transfer)")
    print(" Model Type: M2 (PyTorch HydroLSTM: Specific Runoff q + Multi-Source Weather)")
    print(f" Horizons  : {horizons} weeks | Sequence Lookback: {seq_len} weeks")
    print("=" * 80)
    
    # Device selection
    if device_arg == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
        
    print(f"[*] PyTorch Backend Device: {device}")
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = []
    
    for state in ['Maharashtra', 'Odisha', 'Karnataka']:
        state_stns = [s for s in PORTFOLIO_CONFIG if s['state'] == state]
        train_stns = [s for s in state_stns if not s['is_holdout']]
        holdout_stn = [s for s in state_stns if s['is_holdout']][0]
        
        # Load training stations data
        dfs_train_list = []
        for s in train_stns:
            tfile = os.path.join(TENSORS_DIR, f"weekly_tensor_gauge_{s['gauge_id']:05d}_{s['station']}.csv")
            if os.path.isfile(tfile):
                d = pd.read_csv(tfile, parse_dates=['date']).set_index('date')
                dfs_train_list.append((s, d))
                
        # Load holdout station data
        h_file = os.path.join(TENSORS_DIR, f"weekly_tensor_gauge_{holdout_stn['gauge_id']:05d}_{holdout_stn['station']}.csv")
        if not os.path.isfile(h_file):
            print(f"[-] Holdout tensor missing for {holdout_stn['station']}.")
            continue
        df_holdout = pd.read_csv(h_file, parse_dates=['date']).set_index('date')
        area_h = holdout_stn['area_km2']
        conv_h = 604.8 / area_h
        
        for H in horizons:
            target_spec_col = f'target_specific_H{H}'
            target_vol_col  = f'target_H{H}'
            
            # Pool training stations with embargo
            pooled_train_dfs = []
            pooled_val_dfs   = []
            for s_info, df_s in dfs_train_list:
                tr, va, _ = partition_splits_with_embargo(df_s, horizon_weeks=H)
                pooled_train_dfs.append(tr)
                pooled_val_dfs.append(va)
                
            # Fit feature scaler strictly on pooled training fold
            concat_pool_train = pd.concat(pooled_train_dfs, axis=0)
            scaler = StandardScaler()
            scaler.fit(concat_pool_train[M2_SEQUENTIAL_FEATURES].fillna(concat_pool_train[M2_SEQUENTIAL_FEATURES].median()))
            
            # Extract sequences from each training station independently
            train_seqs, train_tgts = [], []
            val_seqs, val_tgts = [], []
            
            for tr_df, va_df in zip(pooled_train_dfs, pooled_val_dfs):
                s_tr, y_tr, _, _ = extract_station_sequences(
                    tr_df, seq_len=seq_len, feature_cols=M2_SEQUENTIAL_FEATURES,
                    target_col=target_spec_col, scaler=scaler, fit_scaler=False
                )
                if len(s_tr) > 0:
                    train_seqs.append(s_tr)
                    train_tgts.append(y_tr)
                    
                # For validation, use buffer from end of train fold to preserve initial weeks
                s_va, y_va, _, _ = extract_station_sequences(
                    va_df, seq_len=seq_len, feature_cols=M2_SEQUENTIAL_FEATURES,
                    target_col=target_spec_col, scaler=scaler, fit_scaler=False,
                    buffer_df=tr_df
                )
                if len(s_va) > 0:
                    val_seqs.append(s_va)
                    val_tgts.append(y_va)
                    
            X_train = np.concatenate(train_seqs, axis=0)
            y_train = np.concatenate(train_tgts, axis=0)
            X_val   = np.concatenate(val_seqs, axis=0)
            y_val   = np.concatenate(val_tgts, axis=0)
            
            train_loader = DataLoader(HydroSequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(HydroSequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
            
            # Train PyTorch HydroLSTM
            lstm_model = train_lstm_regressor(
                train_loader, val_loader, input_dim=len(M2_SEQUENTIAL_FEATURES),
                hidden_dim=hidden_dim, num_layers=2, epochs=epochs, lr=0.005,
                patience=10, device=device
            )
            
            # Serialize trained PyTorch weights
            clean_holdout = holdout_stn['station'].replace(' ', '_').replace('(', '').replace(')', '')
            pt_save_path = os.path.join(MODELS_DIR, 'protocol2', f"m2_lstm_pool_{state.lower()}_heldout_{clean_holdout}_H{H}.pt")
            torch.save(lstm_model.state_dict(), pt_save_path)
            
            # Partition holdout station with embargo
            df_h_train, df_h_val, df_h_test = partition_splits_with_embargo(df_holdout, horizon_weeks=H)
            
            # 1. Fit drought thresholds on Holdout Training fold (volumetric m3/s)
            drought_thresholds = fit_training_drought_thresholds(df_h_train[target_vol_col].values)
            q80 = drought_thresholds['Q80_mild']
            q90 = drought_thresholds['Q90_moderate']
            q95 = drought_thresholds['Q95_severe']
            
            # 2. M0 Baseline on holdout
            preds_m0_vol = predict_m0_baseline(df_h_train, df_h_test, horizon_weeks=H, target_col=target_vol_col)
            
            # 3. Extract test sequences from holdout station with buffer from df_h_val
            test_seqs, test_tgts, valid_dates, _ = extract_station_sequences(
                df_h_test, seq_len=seq_len, feature_cols=M2_SEQUENTIAL_FEATURES,
                target_col=target_spec_col, scaler=scaler, fit_scaler=False,
                buffer_df=df_h_val
            )
            
            # Predict zero-shot on holdout sequences
            lstm_model.eval()
            with torch.no_grad():
                tensor_test = torch.tensor(test_seqs, dtype=torch.float32).to(device)
                preds_spec = lstm_model(tensor_test).cpu().numpy()
                
            # Convert specific runoff back to volumetric discharge (m3/s)
            preds_vol_series = pd.Series(preds_spec / conv_h, index=valid_dates)
            preds_m2_vol = preds_vol_series.reindex(df_h_test.index).values
            preds_m2_vol = np.maximum(0.0, preds_m2_vol)
            
            y_test_vol = df_h_test[target_vol_col].values
            flow_0 = df_h_test['flow_lag_0'].values
            
            # 4. Enforce Identical Row Evaluation Contract
            eval_mask = (~np.isnan(y_test_vol)) & (~np.isnan(preds_m0_vol)) & (~np.isnan(preds_m2_vol)) & (~np.isnan(flow_0))
            
            y_eval        = y_test_vol[eval_mask]
            preds_m0_eval = preds_m0_vol[eval_mask]
            preds_m2_eval = preds_m2_vol[eval_mask]
            
            # Low-flow / zero-variance flagging
            std_y = float(np.std(y_eval)) if len(y_eval) > 0 else 0.0
            mean_y = float(np.mean(y_eval)) if len(y_eval) > 0 else 0.0
            unstable_flag = bool((std_y < 0.05) or (mean_y < 0.1))
            
            # 5. Compute Track A Metrics
            m0_track_a = compute_track_a_metrics(y_eval, preds_m0_eval)
            m2_track_a = compute_track_a_metrics(y_eval, preds_m2_eval)
            
            # 6. Compute Track B Drought Metrics
            m0_drought_q80 = calc_drought_contingency(y_eval, preds_m0_eval, q80)
            m0_drought_q90 = calc_drought_contingency(y_eval, preds_m0_eval, q90)
            m0_drought_q95 = calc_drought_contingency(y_eval, preds_m0_eval, q95)
            
            m2_drought_q80 = calc_drought_contingency(y_eval, preds_m2_eval, q80)
            m2_drought_q90 = calc_drought_contingency(y_eval, preds_m2_eval, q90)
            m2_drought_q95 = calc_drought_contingency(y_eval, preds_m2_eval, q95)
            
            delta_nse = m2_track_a['NSE'] - m0_track_a['NSE']
            delta_kge = m2_track_a['KGE'] - m0_track_a['KGE']
            
            print(f"[{state[:2]}] Zero-Shot Holdout: {holdout_stn['station']:<18} H={H}w | M0 NSE: {m0_track_a['NSE']:+.3f} -> M2 LSTM NSE: {m2_track_a['NSE']:+.3f} (Δ {delta_nse:+.3f}) | KGE: {m2_track_a['KGE']:+.3f}")
            
            results.append({
                'Protocol': 'Protocol 2 (Spatial Transfer)',
                'State': state,
                'Holdout_Station': holdout_stn['station'],
                'Holdout_Gauge': holdout_stn['gauge_id'],
                'Catchment_Area_km2': area_h,
                'Horizon_Weeks': H,
                'Model_Family': 'M2_PyTorch_LSTM',
                'N_Obs': m2_track_a['N_obs'],
                'Unstable_Metrics_Flag': unstable_flag,
                'Obs_Mean_m3s': mean_y,
                'Obs_Std_m3s': std_y,
                # Drought Thresholds
                'Q80_Mild_Thresh': q80,
                'Q90_Moderate_Thresh': q90,
                'Q95_Severe_Thresh': q95,
                # Track A: Continuous Hydrograph
                'M0_NSE': m0_track_a['NSE'],
                'M0_KGE': m0_track_a['KGE'],
                'M0_RMSE': m0_track_a['RMSE'],
                'M0_MAE': m0_track_a['MAE'],
                'M0_KGE_r': m0_track_a['KGE_r'],
                'M0_KGE_alpha': m0_track_a['KGE_alpha'],
                'M0_KGE_beta': m0_track_a['KGE_beta'],
                'M2_NSE': m2_track_a['NSE'],
                'M2_KGE': m2_track_a['KGE'],
                'M2_RMSE': m2_track_a['RMSE'],
                'M2_MAE': m2_track_a['MAE'],
                'M2_KGE_r': m2_track_a['KGE_r'],
                'M2_KGE_alpha': m2_track_a['KGE_alpha'],
                'M2_KGE_beta': m2_track_a['KGE_beta'],
                'Delta_NSE': delta_nse,
                'Delta_KGE': delta_kge,
                # Track B: Mild Drought Q80
                'M0_Mild_CSI': m0_drought_q80['CSI'],
                'M0_Mild_POD': m0_drought_q80['POD'],
                'M0_Mild_FAR': m0_drought_q80['FAR'],
                'M2_Mild_CSI': m2_drought_q80['CSI'],
                'M2_Mild_POD': m2_drought_q80['POD'],
                'M2_Mild_FAR': m2_drought_q80['FAR'],
                # Track B: Moderate Drought Q90
                'M0_Mod_CSI': m0_drought_q90['CSI'],
                'M0_Mod_POD': m0_drought_q90['POD'],
                'M0_Mod_FAR': m0_drought_q90['FAR'],
                'M2_Mod_CSI': m2_drought_q90['CSI'],
                'M2_Mod_POD': m2_drought_q90['POD'],
                'M2_Mod_FAR': m2_drought_q90['FAR'],
                # Track B: Severe Drought Q95
                'M0_Severe_CSI': m0_drought_q95['CSI'],
                'M0_Severe_POD': m0_drought_q95['POD'],
                'M0_Severe_FAR': m0_drought_q95['FAR'],
                'M2_Severe_CSI': m2_drought_q95['CSI'],
                'M2_Severe_POD': m2_drought_q95['POD'],
                'M2_Severe_FAR': m2_drought_q95['FAR'],
            })
            
    df_results = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, 'protocol2_spatial_results_m2_lstm.csv')
    df_results.to_csv(out_csv, index=False)
    print(f"\n[+] Protocol 2 models saved to: {os.path.join(MODELS_DIR, 'protocol2')}")
    print(f"[+] Protocol 2 results saved to: {out_csv}")
    return df_results

# ==============================================================================
# 8. MAIN CLI ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tier M2 Hydro-Meteorological Model Training (GBT & PyTorch LSTM)."
    )
    parser.add_argument(
        '--protocol',
        choices=['1', '2', 'all'],
        default='all',
        help="Evaluation protocol: '1' (temporal GBT), '2' (spatial transfer LSTM), or 'all' (default: all)"
    )
    parser.add_argument(
        '--horizons',
        nargs='+',
        type=int,
        default=[1, 2, 4, 8],
        help="Forecast horizons in weeks (default: 1 2 4 8)"
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=8,
        help="Temporal sequence lookback window in weeks for LSTM (default: 8)"
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=64,
        help="Hidden dimension of LSTM layer (default: 64)"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help="Maximum training epochs for PyTorch LSTM (default: 50)"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help="Batch size for PyTorch DataLoader (default: 64)"
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'mps', 'cuda'],
        default='auto',
        help="Compute device for PyTorch: auto, cpu, mps, cuda (default: auto)"
    )
    args = parser.parse_args()
    
    print("\n" + "#" * 80)
    print("      STREAMFLOW & DROUGHT FORECASTING: TIER M2 HYDRO-METEOROLOGY")
    print(f"      Timestamp: {datetime.datetime.now().isoformat()}")
    print(f"      Models Directory: {MODELS_DIR}")
    print("#" * 80)
    
    if args.protocol in ['1', 'all']:
        execute_protocol_1(horizons=args.horizons)
    if args.protocol in ['2', 'all']:
        execute_protocol_2(
            horizons=args.horizons,
            seq_len=args.seq_len,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device_arg=args.device
        )
        
    print("\n[+] Tier M2 execution pipeline completed successfully.")
