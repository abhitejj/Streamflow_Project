#!/usr/bin/env python3
"""
Model Training & Benchmarking Pipeline: M0 (Persistence / Climatology) & M1 (Autoregressive Flow-Only)

Complies with strict operational and evaluation guardrails:
1. Approval Gate AG-TS-02 Enforced:
   Terminal H-week purge embargo applied to Training (1980–2010) and Validation (2011–2015) folds
   to completely eliminate future target leakage across temporal split boundaries.
2. M0 Baseline Engine:
   - H in [1, 2] weeks: Pure persistence (Q_t).
   - H in [4, 8] weeks: Seasonal weekly climatology fit strictly on the Training fold.
   - Halts execution if pure persistence is attempted for H > 2.
3. M1 Autoregressive Flow-Only Engine:
   - Features strictly restricted to flow history and calendar Fourier harmonics.
   - Strictly ZERO meteorological forcings (P, T, PET, Deficit, API excluded).
   - Implements Ridge Regression (validation alpha-tuned) and HistGradientBoostingRegressor.
4. Dual Evaluation Protocols:
   - Protocol 1: Station-specific temporal forecasting across all 12 independent catchments.
   - Protocol 2: Regional cross-basin zero-shot spatial transfer to held-out sub-basins
     (Bamni in MH, Kantamal in OD, Haralahalli in KA) using specific runoff (mm/week).
5. Evaluation Integrity:
   - Identical row evaluation mask ensuring M0 and M1 are evaluated on exact identical timestamps.
   - Track A: Continuous hydrograph accuracy (NSE, KGE-2009 with r, alpha, beta, RMSE, MAE).
   - Track B: Drought early warning reliability (Mild Q80, Moderate Q90, Severe Q95)
     derived strictly from empirical percentiles of the Training fold.
"""

import os
import sys
import argparse
import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. DIRECTORY AND PATH ENFORCEMENT
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

TENSORS_DIR = os.path.join(PROJECT_ROOT, 'data/processed/weekly_tensors')
CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data/processed/station_catalog.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'data/processed/model_results')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'data/processed/validation_reports')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

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

# Feature Set for M1: Flow history and calendar harmonics ONLY (Strictly ZERO weather features)
M1_FLOW_FEATURES = [
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
    'sin_week',
    'cos_week',
]

# ==============================================================================
# 3. SPLIT PARTITIONING WITH TERMINAL H-WEEK EMBARGO
# ==============================================================================
def partition_splits_with_embargo(df, horizon_weeks):
    """
    Partitions the 46-year dataset into Train, Val, and Test folds.
    Enforces Approval Gate AG-TS-02:
    - Drops the terminal H weeks of the Training fold (ending 2010-12-31).
    - Drops the terminal H weeks of the Validation fold (ending 2015-12-31).
    - Test fold is un-purged (out-of-sample final evaluation block).
    """
    # Raw date masks
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
    - NSE (Nash-Sutcliffe Efficiency) with near-zero variance guards and lower-bound clamping.
    - KGE-2009 (Kling-Gupta Efficiency decomposed into r, alpha, beta) with exact flat-regime guards.
    - RMSE (Root Mean Squared Error).
    - MAE (Mean Absolute Error).
    - Diagnostic flags for zero-variance ephemeral basins and negative prediction detection.
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
    
    # Correlation (r) & Variability ratio (alpha)
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
        
    # Bias ratio (beta)
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
    Calculates drought severity thresholds strictly from the Training fold (1980–2010):
    - Mild Drought: 20th percentile (Q_80)
    - Moderate Drought: 10th percentile (Q_90)
    - Severe Drought: 5th percentile (Q_95)
    Strict Rule: Never fit thresholds on validation or test sets.
    """
    y_clean = y_train[~np.isnan(y_train)]
    if len(y_clean) == 0:
        return {'Q80_mild': 0.0, 'Q90_moderate': 0.0, 'Q95_severe': 0.0}
    return {
        'Q80_mild': float(np.percentile(y_clean, 20.0)),
        'Q90_moderate': float(np.percentile(y_clean, 10.0)),
        'Q95_severe': float(np.percentile(y_clean, 5.0))
    }

def compute_track_b_drought_metrics(y_true, y_pred, threshold, clip_nonnegative=True, csi_zero_events_default=1.0):
    """
    Computes Track B drought early warning metrics for a given threshold:
    - POD (Probability of Detection / Hit Rate / Recall) = H / (H + M)
    - FAR (False Alarm Ratio) = F / (H + F)
    - CSI (Critical Success Index / Threat Score) = H / (H + M + F)
    - Full 2x2 contingency table: Hits (H), False Alarms (F), Misses (M), Correct Rejections (CR)
    - Rigorous zero-division handling for unobserved events and unissued alerts.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return {
            'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan,
            'Hits': 0, 'FalseAlarms': 0, 'Misses': 0, 'CorrectRejections': 0,
            'ObsEvents': 0, 'PredAlerts': 0, 'BaseRate': np.nan, 'Threshold': float(threshold)
        }
    
    y = y_true[valid]
    y_hat = np.maximum(y_pred[valid], 0.0) if clip_nonnegative else y_pred[valid]
    
    obs_event = (y <= threshold).astype(int)
    pred_event = (y_hat <= threshold).astype(int)
    
    hits = int(np.sum((obs_event == 1) & (pred_event == 1)))
    fa   = int(np.sum((obs_event == 0) & (pred_event == 1)))
    miss = int(np.sum((obs_event == 1) & (pred_event == 0)))
    cr   = int(np.sum((obs_event == 0) & (pred_event == 0)))
    
    obs_count = hits + miss
    pred_count = hits + fa
    denom_csi = hits + miss + fa
    
    pod = float(hits / obs_count) if obs_count > 0 else np.nan
    far = float(fa / pred_count) if pred_count > 0 else np.nan
    csi = float(hits / denom_csi) if denom_csi > 0 else np.nan
        
    return {
        'POD': pod,
        'FAR': far,
        'CSI': csi,
        'Hits': hits,
        'FalseAlarms': fa,
        'Misses': miss,
        'CorrectRejections': cr,
        'ObsEvents': obs_count,
        'PredAlerts': pred_count,
        'BaseRate': float(obs_count / n_valid) if n_valid > 0 else np.nan,
        'Threshold': float(threshold)
    }

def evaluate_all_drought_tiers(y_true, y_pred, drought_thresholds, prefix=''):
    """
    Evaluates Track B drought metrics across all three standardized severity tiers:
    - Mild (Q_80, 20th percentile)
    - Moderate (Q_90, 10th percentile)
    - Severe (Q_95, 5th percentile)
    """
    metrics = {}
    tier_map = [
        ('Mild_Q80', drought_thresholds['Q80_mild']),
        ('Moderate_Q90', drought_thresholds['Q90_moderate']),
        ('Severe_Q95', drought_thresholds['Q95_severe']),
    ]
    for tier_name, thresh in tier_map:
        res = compute_track_b_drought_metrics(y_true, y_pred, threshold=thresh)
        p = f"{prefix}{tier_name}_" if prefix else f"{tier_name}_"
        metrics[f"{p}Threshold"] = res['Threshold']
        metrics[f"{p}CSI"] = res['CSI']
        metrics[f"{p}POD"] = res['POD']
        metrics[f"{p}FAR"] = res['FAR']
        metrics[f"{p}Hits"] = res['Hits']
        metrics[f"{p}FA"] = res['FalseAlarms']
        metrics[f"{p}Misses"] = res['Misses']
        metrics[f"{p}ObsEvents"] = res['ObsEvents']
        metrics[f"{p}PredAlerts"] = res['PredAlerts']
    return metrics

# ==============================================================================
# 5. M0 BASELINE ENGINE (PERSISTENCE & CLIMATOLOGY)
# ==============================================================================
def predict_m0_baseline(df_train, df_eval, horizon_weeks, target_col):
    """
    Enforces M0 Baseline rules:
    - Horizon 1w, 2w: Pure persistence (Q_t from flow_lag_0).
    - Horizon 4w, 8w: Seasonal weekly climatology fit on Training fold only.
    - Halts execution if pure persistence is attempted for horizon > 2 weeks.
    """
    if horizon_weeks in [1, 2]:
        # Pure persistence allowed for short horizons
        return df_eval['flow_lag_0'].values.copy()
    elif horizon_weeks in [4, 8]:
        # Calculate seasonal weekly climatology from training fold only (ISO weeks 1 to 53)
        # Use unshifted flow so climatology is horizon-agnostic and free from phase shifts
        flow_col = 'flow_target_m3s' if 'flow_target_m3s' in df_train.columns else 'flow_lag_0'
        train_climatology = df_train.groupby(df_train.index.isocalendar().week)[flow_col].mean()
        overall_train_mean = float(df_train[flow_col].mean())
        
        # Target week at forecast issuance t + H
        target_dates = df_eval.index + pd.Timedelta(weeks=horizon_weeks)
        target_weeks = target_dates.isocalendar().week.astype(int)
        
        # Map target week to training climatology lookup
        climatology_preds = target_weeks.map(train_climatology).fillna(overall_train_mean).values
        return np.maximum(0.0, climatology_preds)
    else:
        raise ValueError(
            f"[-] M0 BASELINE VIOLATION: Pure persistence attempted for horizon H={horizon_weeks}w. "
            f"Pure persistence is strictly prohibited beyond 2-week horizons."
        )

# ==============================================================================
# 6. M1 AUTOREGRESSIVE MODEL ENGINE (RIDGE & GRADIENT BOOSTING)
# ==============================================================================
def train_and_predict_m1(df_train, df_val, df_test, feature_cols, target_col, model_type='ridge', save_path=None):
    """
    Trains M1 Autoregressive Flow-Only model:
    - Fits candidate models strictly on df_train.
    - Selects optimal regularization / complexity strictly on chronological df_val.
    - Evaluates the selected model on df_test (no retraining on val fold to preserve split purity).
    - Serializes trained estimator bundle (model, scaler, metadata) to disk via joblib if save_path is provided.
    - Enforces physical non-negativity constraint: streamflow >= 0.
    """
    # Prepare training and validation data (drop NaNs in features or target)
    train_clean = df_train.dropna(subset=feature_cols + [target_col])
    val_clean   = df_val.dropna(subset=feature_cols + [target_col])
    
    if len(train_clean) == 0:
        raise ValueError(f"[-] Training set contains 0 valid rows for {target_col}.")
    if len(val_clean) == 0:
        raise ValueError(f"[-] Validation set contains 0 valid rows for {target_col}.")
    
    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_col].values
    
    X_val = val_clean[feature_cols].values
    y_val = val_clean[target_col].values
    
    # Impute missing feature rows in test set with training fold medians
    X_test_df = df_test[feature_cols].copy()
    train_medians = train_clean[feature_cols].median()
    X_test = X_test_df.fillna(train_medians).values
    
    fitted_scaler = None
    best_params = {}

    if model_type == 'ridge':
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)
        
        best_alpha = 1.0
        best_val_rmse = float('inf')
        best_model = None
        
        # Grid search over regularization parameter alpha on validation RMSE
        for alpha in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_s, y_train)
            val_preds = np.maximum(0.0, model.predict(X_val_s))
            val_rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_alpha = alpha
                best_model = model
                
        test_preds = best_model.predict(X_test_s)
        fitted_scaler = scaler
        best_params = {'alpha': best_alpha}
        
    elif model_type == 'gbt':
        best_val_rmse = float('inf')
        best_model = None
        best_depth = 3
        best_l2 = 0.0
        
        # Chronological validation tuning over tree depth and regularization
        # Early stopping with random shuffle disabled to prevent temporal leakage
        for depth in [3, 5, 7]:
            for l2 in [0.0, 1.0, 10.0]:
                model = HistGradientBoostingRegressor(
                    max_iter=150,
                    learning_rate=0.05,
                    max_depth=depth,
                    l2_regularization=l2,
                    min_samples_leaf=20,
                    early_stopping=False,
                    random_state=42
                )
                model.fit(X_train, y_train)
                val_preds = np.maximum(0.0, model.predict(X_val))
                val_rmse = np.sqrt(np.mean((y_val - val_preds) ** 2))
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_model = model
                    best_depth = depth
                    best_l2 = l2
                    
        test_preds = best_model.predict(X_test)
        best_params = {'max_depth': best_depth, 'l2_regularization': best_l2}
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'ridge' or 'gbt'.")
    
    # Serialize trained estimator bundle to disk if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        artifact = {
            'model': best_model,
            'scaler': fitted_scaler,
            'model_type': model_type,
            'best_params': best_params,
            'best_val_rmse': float(best_val_rmse),
            'feature_cols': feature_cols,
            'target_col': target_col,
            'train_medians': train_medians.to_dict(),
            'saved_at': datetime.datetime.now().isoformat()
        }
        joblib.dump(artifact, save_path)
    
    # Enforce physical non-negativity constraint: river discharge cannot be negative
    return np.maximum(0.0, np.asarray(test_preds))

# ==============================================================================
# 7. PROTOCOL 1: STATION-SPECIFIC TEMPORAL EVALUATION
# ==============================================================================
def execute_protocol_1(horizons=[1, 2, 4, 8], model_type='ridge'):
    """
    Executes Protocol 1:
    - Slices 46-year dataset for each of the 12 catchments.
    - Enforces H-week terminal purge embargo on Train and Val folds.
    - Evaluates M0 and M1 models on identical test row intersections.
    - Computes continuous Track A (NSE, KGE, RMSE, MAE) and Track B drought metrics
      (Mild Q80, Moderate Q90, Severe Q95 derived strictly from Training fold).
    """
    print("\n" + "=" * 80)
    print(f" EXECUTION: PROTOCOL 1 (Station-Specific Multi-Horizon Benchmarking)")
    print(f" Model Type: M1 ({model_type.upper()}) vs M0 Baseline")
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
            
            # 3. Generate M1 predictions & serialize trained estimator
            clean_name = station.replace(' ', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(MODELS_DIR, 'protocol1', f"m1_{model_type}_gauge_{gauge_id:05d}_{clean_name}_H{H}.joblib")
            
            preds_m1 = train_and_predict_m1(
                df_train, df_val, df_test,
                feature_cols=M1_FLOW_FEATURES,
                target_col=target_col,
                model_type=model_type,
                save_path=save_path
            )
            
            y_test = df_test[target_col].values
            flow_0 = df_test['flow_lag_0'].values
            
            # 4. Enforce Identical Row Evaluation Contract:
            # Row is evaluated only if target, M0, M1, and persistence feature (flow_lag_0) are all valid.
            eval_mask = (~np.isnan(y_test)) & (~np.isnan(preds_m0)) & (~np.isnan(preds_m1)) & (~np.isnan(flow_0))
            
            y_eval        = y_test[eval_mask]
            preds_m0_eval = preds_m0[eval_mask]
            preds_m1_eval = preds_m1[eval_mask]
            
            # Low-flow / zero-variance flagging: flag if std < 0.05 or mean < 0.1 or extreme outlier
            std_y = float(np.std(y_eval)) if len(y_eval) > 0 else 0.0
            mean_y = float(np.mean(y_eval)) if len(y_eval) > 0 else 0.0
            max_y = float(np.max(y_eval)) if len(y_eval) > 0 else 0.0
            unstable_flag = (std_y < 0.05) or (mean_y < 0.1) or (max_y > 1e5)
            
            # 5. Compute Track A Continuous Metrics
            m0_track_a = compute_track_a_metrics(y_eval, preds_m0_eval)
            m1_track_a = compute_track_a_metrics(y_eval, preds_m1_eval)
            
            # 6. Compute Track B Drought Metrics (Mild Q80, Moderate Q90, Severe Q95)
            m0_drought_q80 = compute_track_b_drought_metrics(y_eval, preds_m0_eval, q80)
            m1_drought_q80 = compute_track_b_drought_metrics(y_eval, preds_m1_eval, q80)
            
            m0_drought_q90 = compute_track_b_drought_metrics(y_eval, preds_m0_eval, q90)
            m1_drought_q90 = compute_track_b_drought_metrics(y_eval, preds_m1_eval, q90)
            
            m0_drought_q95 = compute_track_b_drought_metrics(y_eval, preds_m0_eval, q95)
            m1_drought_q95 = compute_track_b_drought_metrics(y_eval, preds_m1_eval, q95)
            
            results.append({
                'Protocol': 'Protocol 1 (Temporal)',
                'State': state,
                'Station': station,
                'Gauge_ID': gauge_id,
                'Catchment_Area_km2': area_km2,
                'Horizon_Weeks': H,
                'Model_Family': model_type.upper(),
                'N_Obs': len(y_eval),
                'Unstable_Metrics_Flag': unstable_flag,
                'Obs_Mean_m3s': mean_y,
                'Obs_Std_m3s': std_y,
                'Q80_Mild_Thresh': q80,
                'Q90_Moderate_Thresh': q90,
                'Q95_Severe_Thresh': q95,
                # M0 Track A Metrics
                'M0_NSE': m0_track_a['NSE'],
                'M0_KGE': m0_track_a['KGE'],
                'M0_RMSE': m0_track_a['RMSE'],
                'M0_MAE': m0_track_a['MAE'],
                'M0_KGE_r': m0_track_a['KGE_r'],
                'M0_KGE_alpha': m0_track_a['KGE_alpha'],
                'M0_KGE_beta': m0_track_a['KGE_beta'],
                # M1 Track A Metrics
                'M1_NSE': m1_track_a['NSE'],
                'M1_KGE': m1_track_a['KGE'],
                'M1_RMSE': m1_track_a['RMSE'],
                'M1_MAE': m1_track_a['MAE'],
                'M1_KGE_r': m1_track_a['KGE_r'],
                'M1_KGE_alpha': m1_track_a['KGE_alpha'],
                'M1_KGE_beta': m1_track_a['KGE_beta'],
                # Relative Skill Gains
                'Delta_NSE': m1_track_a['NSE'] - m0_track_a['NSE'] if not np.isnan(m1_track_a['NSE']) and not np.isnan(m0_track_a['NSE']) else np.nan,
                'Delta_KGE': m1_track_a['KGE'] - m0_track_a['KGE'] if not np.isnan(m1_track_a['KGE']) and not np.isnan(m0_track_a['KGE']) else np.nan,
                # Track B: Mild Drought Q80
                'M0_Mild_CSI': m0_drought_q80['CSI'],
                'M0_Mild_POD': m0_drought_q80['POD'],
                'M0_Mild_FAR': m0_drought_q80['FAR'],
                'M1_Mild_CSI': m1_drought_q80['CSI'],
                'M1_Mild_POD': m1_drought_q80['POD'],
                'M1_Mild_FAR': m1_drought_q80['FAR'],
                # Track B: Moderate Drought Q90
                'M0_Mod_CSI': m0_drought_q90['CSI'],
                'M0_Mod_POD': m0_drought_q90['POD'],
                'M0_Mod_FAR': m0_drought_q90['FAR'],
                'M1_Mod_CSI': m1_drought_q90['CSI'],
                'M1_Mod_POD': m1_drought_q90['POD'],
                'M1_Mod_FAR': m1_drought_q90['FAR'],
                # Track B: Severe Drought Q95
                'M0_Severe_CSI': m0_drought_q95['CSI'],
                'M0_Severe_POD': m0_drought_q95['POD'],
                'M0_Severe_FAR': m0_drought_q95['FAR'],
                'M1_Severe_CSI': m1_drought_q95['CSI'],
                'M1_Severe_POD': m1_drought_q95['POD'],
                'M1_Severe_FAR': m1_drought_q95['FAR'],
            })
            
    df_results = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, f'protocol1_temporal_results_{model_type}.csv')
    df_results.to_csv(out_csv, index=False)
    print(f"[+] Protocol 1 models saved to: {os.path.join(MODELS_DIR, 'protocol1')}")
    print(f"[+] Protocol 1 results saved to: {out_csv}")
    return df_results

# ==============================================================================
# 8. PROTOCOL 2: CROSS-BASIN ZERO-SHOT SPATIAL GENERALIZATION
# ==============================================================================
def execute_protocol_2(horizons=[1, 2, 4, 8], model_type='ridge'):
    """
    Executes Protocol 2:
    - In each state, trains M1 on Stations 1, 2, and 3 using specific runoff (mm/week).
    - Evaluates zero-shot on the held-out Station 4 (Bamni, Kantamal, Haralahalli).
    - Converts specific runoff predictions back to volumetric discharge (m3/s).
    - Evaluates on identical row mask against M0 persistence.
    """
    print("\n" + "=" * 80)
    print(f" EXECUTION: PROTOCOL 2 (Cross-Basin Zero-Shot Spatial Transfer)")
    print(f" Model Type: M1 ({model_type.upper()}) vs M0 Baseline (Specific Runoff q)")
    print(f" Horizons  : {horizons} weeks")
    print("=" * 80)
    
    results = []
    
    # Specific runoff feature names (parsimonious low-order transfer formulation)
    M1_SPECIFIC_FEATURES = [
        'flow_target_specific_mm',
        'sin_week',
        'cos_week',
    ]
    
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
            pooled_train = []
            pooled_val   = []
            for s_info, df_s in dfs_train_list:
                tr, va, _ = partition_splits_with_embargo(df_s, horizon_weeks=H)
                pooled_train.append(tr)
                pooled_val.append(va)
            
            df_pool_train = pd.concat(pooled_train, axis=0).dropna(subset=M1_SPECIFIC_FEATURES + [target_spec_col])
            df_pool_val   = pd.concat(pooled_val, axis=0).dropna(subset=M1_SPECIFIC_FEATURES + [target_spec_col])
            
            # Partition holdout station with embargo
            df_h_train, df_h_val, df_h_test = partition_splits_with_embargo(df_holdout, horizon_weeks=H)
            
            # 1. Fit drought thresholds on Holdout Training fold (volumetric m3/s)
            drought_thresholds = fit_training_drought_thresholds(df_h_train[target_vol_col].values)
            q80 = drought_thresholds['Q80_mild']
            q90 = drought_thresholds['Q90_moderate']
            q95 = drought_thresholds['Q95_severe']
            
            # 2. M0 Baseline on holdout
            preds_m0_vol = predict_m0_baseline(df_h_train, df_h_test, horizon_weeks=H, target_col=target_vol_col)
            
            # 3. Train pooled M1 model on specific runoff & serialize trained estimator
            clean_holdout = holdout_stn['station'].replace(' ', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(MODELS_DIR, 'protocol2', f"m1_{model_type}_pool_{state.lower()}_heldout_{clean_holdout}_H{H}.joblib")
            
            preds_m1_spec = train_and_predict_m1(
                df_pool_train, df_pool_val, df_h_test,
                feature_cols=M1_SPECIFIC_FEATURES,
                target_col=target_spec_col,
                model_type=model_type,
                save_path=save_path
            )
            # Convert specific runoff back to volumetric discharge (m3/s)
            preds_m1_vol = np.maximum(0.0, preds_m1_spec / conv_h)
            
            y_h_test_vol = df_h_test[target_vol_col].values
            flow_0_h     = df_h_test['flow_lag_0'].values
            
            # 4. Identical row evaluation mask
            eval_mask = (~np.isnan(y_h_test_vol)) & (~np.isnan(preds_m0_vol)) & (~np.isnan(preds_m1_vol)) & (~np.isnan(flow_0_h))
            
            y_eval        = y_h_test_vol[eval_mask]
            preds_m0_eval = preds_m0_vol[eval_mask]
            preds_m1_eval = preds_m1_vol[eval_mask]
            
            # Low-flow / zero-variance flagging
            std_y = float(np.std(y_eval)) if len(y_eval) > 0 else 0.0
            mean_y = float(np.mean(y_eval)) if len(y_eval) > 0 else 0.0
            max_y = float(np.max(y_eval)) if len(y_eval) > 0 else 0.0
            unstable_flag = (std_y < 0.05) or (mean_y < 0.1) or (max_y > 1e5)
            
            # 5. Continuous Track A Metrics
            m0_track_a = compute_track_a_metrics(y_eval, preds_m0_eval)
            m1_track_a = compute_track_a_metrics(y_eval, preds_m1_eval)
            
            # 6. Categorical Track B Metrics
            m0_drought_q80 = compute_track_b_drought_metrics(y_eval, preds_m0_eval, q80)
            m1_drought_q80 = compute_track_b_drought_metrics(y_eval, preds_m1_eval, q80)
            
            m0_drought_q90 = compute_track_b_drought_metrics(y_eval, preds_m0_eval, q90)
            m1_drought_q90 = compute_track_b_drought_metrics(y_eval, preds_m1_eval, q90)
            
            m0_drought_q95 = compute_track_b_drought_metrics(y_eval, preds_m0_eval, q95)
            m1_drought_q95 = compute_track_b_drought_metrics(y_eval, preds_m1_eval, q95)
            
            results.append({
                'Protocol': 'Protocol 2 (Spatial Transfer)',
                'State': state,
                'Holdout_Station': holdout_stn['station'],
                'Holdout_Gauge': holdout_stn['gauge_id'],
                'Catchment_Area_km2': area_h,
                'Horizon_Weeks': H,
                'Model_Family': model_type.upper(),
                'N_Obs': len(y_eval),
                'Unstable_Metrics_Flag': unstable_flag,
                'Obs_Mean_m3s': mean_y,
                'Obs_Std_m3s': std_y,
                'Q80_Mild_Thresh': q80,
                'Q90_Moderate_Thresh': q90,
                'Q95_Severe_Thresh': q95,
                # M0 Track A Metrics
                'M0_NSE': m0_track_a['NSE'],
                'M0_KGE': m0_track_a['KGE'],
                'M0_RMSE': m0_track_a['RMSE'],
                'M0_MAE': m0_track_a['MAE'],
                'M0_KGE_r': m0_track_a['KGE_r'],
                'M0_KGE_alpha': m0_track_a['KGE_alpha'],
                'M0_KGE_beta': m0_track_a['KGE_beta'],
                # M1 Zero-Shot Metrics
                'M1_NSE': m1_track_a['NSE'],
                'M1_KGE': m1_track_a['KGE'],
                'M1_RMSE': m1_track_a['RMSE'],
                'M1_MAE': m1_track_a['MAE'],
                'M1_KGE_r': m1_track_a['KGE_r'],
                'M1_KGE_alpha': m1_track_a['KGE_alpha'],
                'M1_KGE_beta': m1_track_a['KGE_beta'],
                # Relative Gain
                'Delta_NSE': m1_track_a['NSE'] - m0_track_a['NSE'] if not np.isnan(m1_track_a['NSE']) and not np.isnan(m0_track_a['NSE']) else np.nan,
                'Delta_KGE': m1_track_a['KGE'] - m0_track_a['KGE'] if not np.isnan(m1_track_a['KGE']) and not np.isnan(m0_track_a['KGE']) else np.nan,
                # Track B: Mild Drought Q80
                'M0_Mild_CSI': m0_drought_q80['CSI'],
                'M0_Mild_POD': m0_drought_q80['POD'],
                'M0_Mild_FAR': m0_drought_q80['FAR'],
                'M1_Mild_CSI': m1_drought_q80['CSI'],
                'M1_Mild_POD': m1_drought_q80['POD'],
                'M1_Mild_FAR': m1_drought_q80['FAR'],
                # Track B: Moderate Drought Q90
                'M0_Mod_CSI': m0_drought_q90['CSI'],
                'M0_Mod_POD': m0_drought_q90['POD'],
                'M0_Mod_FAR': m0_drought_q90['FAR'],
                'M1_Mod_CSI': m1_drought_q90['CSI'],
                'M1_Mod_POD': m1_drought_q90['POD'],
                'M1_Mod_FAR': m1_drought_q90['FAR'],
                # Track B: Severe Drought Q95
                'M0_Severe_CSI': m0_drought_q95['CSI'],
                'M0_Severe_POD': m0_drought_q95['POD'],
                'M0_Severe_FAR': m0_drought_q95['FAR'],
                'M1_Severe_CSI': m1_drought_q95['CSI'],
                'M1_Severe_POD': m1_drought_q95['POD'],
                'M1_Severe_FAR': m1_drought_q95['FAR'],
            })
            
    df_results = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, f'protocol2_spatial_results_{model_type}.csv')
    df_results.to_csv(out_csv, index=False)
    print(f"[+] Protocol 2 models saved to: {os.path.join(MODELS_DIR, 'protocol2')}")
    print(f"[+] Protocol 2 results saved to: {out_csv}")
    return df_results

# ==============================================================================
# 9. CLI ENTRY POINT & ORCHESTRATION
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark M0 (Persistence/Climatology) vs M1 (Autoregressive Flow-Only) Models."
    )
    parser.add_argument(
        '--protocol',
        choices=['1', '2', 'all'],
        default='all',
        help="Evaluation protocol: '1' (temporal), '2' (spatial transfer), or 'all' (default: all)"
    )
    parser.add_argument(
        '--horizons',
        nargs='+',
        type=int,
        default=[1, 2, 4, 8],
        help="Forecast horizons in weeks (default: 1 2 4 8)"
    )
    parser.add_argument(
        '--model',
        choices=['ridge', 'gbt', 'all'],
        default='ridge',
        help="Model algorithm for M1: 'ridge', 'gbt', or 'all' (default: ridge)"
    )
    args = parser.parse_args()
    
    print("\n" + "#" * 80)
    print("      STREAMFLOW & DROUGHT FORECASTING BENCHMARK: M0 vs M1")
    print(f"      Timestamp: {datetime.datetime.now().isoformat()}")
    print(f"      Models Directory: {MODELS_DIR}")
    print("#" * 80)
    
    models_to_run = ['ridge', 'gbt'] if args.model == 'all' else [args.model]
    
    for m in models_to_run:
        if args.protocol in ['1', 'all']:
            execute_protocol_1(horizons=args.horizons, model_type=m)
        if args.protocol in ['2', 'all']:
            execute_protocol_2(horizons=args.horizons, model_type=m)
            
    print("\n[+] Benchmark execution pipeline completed successfully.")
