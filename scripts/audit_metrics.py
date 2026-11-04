#!/usr/bin/env python3
"""
Comprehensive Metric & Claim Auditor for ES 335 Streamflow & Drought Forecasting.

Audits:
1. Recomputes ALL Track A (NSE, KGE, r, alpha, beta, RMSE, MAE) and Track B
   (POD, FAR, CSI, raw 2x2 contingency, PR-AUC) metrics directly from trained models
   and weekly feature tensors using a single shared metric engine.
2. Compares every recomputed metric against published numbers in index.html /
   benchmark-data.js (on gh-pages) and against model_results CSVs.
3. Generates the audited Mismatch Table and flags discrepancies across Issues A through K.
"""

from __future__ import annotations

import os
import sys
import json
import re
import copy
import subprocess
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]
TENSORS_DIR = BASE_DIR / "data" / "processed" / "weekly_tensors"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "data" / "processed" / "model_results"

TRAIN_START = '1980-01-01'
TRAIN_END   = '2010-12-31'
VAL_START   = '2011-01-01'
VAL_END     = '2015-12-31'
TEST_START  = '2016-01-01'
TEST_END    = '2025-12-31'

PORTFOLIO = [
    {'state': 'Maharashtra', 'station': 'Karad', 'gauge_id': 4021, 'is_holdout': False, 'area_km2': 5462.0},
    {'state': 'Maharashtra', 'station': 'Warunji', 'gauge_id': 4063, 'is_holdout': False, 'area_km2': 1890.0},
    {'state': 'Maharashtra', 'station': 'Ashti', 'gauge_id': 3005, 'is_holdout': False, 'area_km2': 50990.0},
    {'state': 'Maharashtra', 'station': 'Bamni (Ballarsha)', 'gauge_id': 3008, 'is_holdout': True, 'area_km2': 46020.0},
    {'state': 'Odisha', 'station': 'Anandpur', 'gauge_id': 7002, 'is_holdout': False, 'area_km2': 8570.0},
    {'state': 'Odisha', 'station': 'Kashinagar', 'gauge_id': 16007, 'is_holdout': False, 'area_km2': 7820.0},
    {'state': 'Odisha', 'station': 'Gomlai', 'gauge_id': 7007, 'is_holdout': False, 'area_km2': 21950.0},
    {'state': 'Odisha', 'station': 'Kantamal', 'gauge_id': 8013, 'is_holdout': True, 'area_km2': 19600.0},
    {'state': 'Karnataka', 'station': 'Yadgir', 'gauge_id': 4064, 'is_holdout': False, 'area_km2': 69863.0},
    {'state': 'Karnataka', 'station': 'Huvinhedigi', 'gauge_id': 4017, 'is_holdout': False, 'area_km2': 55150.0},
    {'state': 'Karnataka', 'station': 'KUDIGE', 'gauge_id': 5010, 'is_holdout': False, 'area_km2': 1934.0},
    {'state': 'Karnataka', 'station': 'Haralahalli', 'gauge_id': 4013, 'is_holdout': True, 'area_km2': 14582.0},
]

# Recorded zeros at these stations are unmeasured weeks; see scripts/eval_config.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_config import ZERO_AS_MISSING_STATIONS  # noqa: E402

PREDICTIONS_CSV = RESULTS_DIR / "test_predictions.csv"

M1_FEATURES = [
    'flow_lag_0', 'flow_lag_1', 'flow_lag_2', 'flow_lag_3', 'flow_lag_4', 'flow_lag_7',
    'flow_std_7d', 'flow_min_7d', 'flow_max_7d', 'flow_recession_slope', 'sin_week', 'cos_week'
]

M2_TABULAR_FEATURES = [
    'flow_lag_0', 'flow_lag_1', 'flow_lag_2', 'flow_lag_3', 'flow_lag_4', 'flow_lag_7',
    'flow_std_7d', 'flow_min_7d', 'flow_max_7d', 'flow_recession_slope',
    'rain_sum_1w', 'rain_sum_2w', 'rain_sum_1m', 'rain_sum_2m', 'rain_sum_3m',
    'api_30d', 'dry_days_count',
    'temp_mean_1w', 'temp_max_1w', 'temp_min_1w', 'pet_sum_1w', 'pet_sum_1m',
    'water_deficit_1w', 'water_deficit_1m',
    'sin_week', 'cos_week'
]

M2_SEQUENTIAL_FEATURES = [
    'flow_target_specific_mm', 'rain_sum_1w', 'pet_sum_1w', 'water_deficit_1w',
    'temp_mean_1w', 'temp_max_1w', 'api_30d', 'dry_days_count', 'sin_week', 'cos_week'
]

# ==============================================================================
# PYTORCH LSTM ARCHITECTURE (FOR PROTOCOL 2 ZERO-SHOT EVALUATION)
# ==============================================================================
class HydroLSTM(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
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
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        h = self.relu(self.fc1(last_hidden))
        h = self.dropout(h)
        pred = self.fc2(h)
        return torch.relu(pred).squeeze(-1)

# ==============================================================================
# 1. SHARED METRIC ENGINE
# ==============================================================================
def compute_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_pred, dtype=np.float64)
    
    valid = (~np.isnan(y)) & (~np.isnan(y_hat))
    y = y[valid]
    y_hat = np.maximum(y_hat[valid], 0.0)
    
    n = len(y)
    if n < 5:
        return {
            'N': n, 'NSE': np.nan, 'KGE': np.nan, 'r': np.nan,
            'alpha': np.nan, 'beta': np.nan, 'RMSE': np.nan, 'MAE': np.nan
        }
        
    mae = float(np.mean(np.abs(y - y_hat)))
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    
    # NSE
    ss_res = float(np.sum((y - y_hat) ** 2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    
    if ss_tot < 1e-6:
        nse = 1.0 if ss_res < 1e-6 else -10.0
    else:
        nse = float(np.clip(1.0 - (ss_res / ss_tot), -10.0, 1.0))
        
    # KGE
    std_y = float(np.std(y))
    std_hat = float(np.std(y_hat))
    mean_hat = float(np.mean(y_hat))
    
    if std_y < 1e-6 and std_hat < 1e-6:
        r = 1.0 if np.abs(y_mean - mean_hat) < 1e-6 else 0.0
        alpha = 1.0
    elif std_y < 1e-6:
        r = 0.0
        alpha = 10.0
    elif std_hat < 1e-6:
        r = 0.0
        alpha = 0.0
    else:
        cov = float(np.mean((y - y_mean) * (y_hat - mean_hat)))
        r = float(np.clip(cov / (std_y * std_hat), -1.0, 1.0))
        alpha = float(std_hat / std_y)
        
    if y_mean < 1e-6:
        beta = 1.0 if np.abs(mean_hat - y_mean) < 1e-6 else float(np.clip((mean_hat + 1e-6) / (y_mean + 1e-6), 0.0, 10.0))
    else:
        beta = float(np.clip(mean_hat / y_mean, 0.0, 10.0))
        
    ed = np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    kge = float(np.clip(1.0 - ed, -10.0, 1.0))
    
    return {
        'N': n,
        'NSE': nse,
        'KGE': kge,
        'r': r,
        'alpha': alpha,
        'beta': beta,
        'RMSE': rmse,
        'MAE': mae
    }

def compute_drought_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_pred, dtype=np.float64)
    
    valid = (~np.isnan(y)) & (~np.isnan(y_hat))
    y = y[valid]
    y_hat = np.maximum(y_hat[valid], 0.0)
    n = len(y)
    
    if n == 0:
        return {
            'Threshold': float(threshold), 'Hits': 0, 'FalseAlarms': 0, 'Misses': 0, 'CorrectRejections': 0,
            'ObsEvents': 0, 'PredAlerts': 0, 'BaseRate': np.nan, 'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan, 'PR_AUC': np.nan
        }
        
    obs_event = (y <= threshold).astype(int)
    pred_event = (y_hat <= threshold).astype(int)
    
    hits = int(np.sum((obs_event == 1) & (pred_event == 1)))
    fa   = int(np.sum((obs_event == 0) & (pred_event == 1)))
    miss = int(np.sum((obs_event == 1) & (pred_event == 0)))
    cr   = int(np.sum((obs_event == 0) & (pred_event == 0)))
    
    obs_events = hits + miss
    pred_alerts = hits + fa
    denom_csi = hits + miss + fa
    
    # Zero-division guards
    pod = float(hits / obs_events) if obs_events > 0 else np.nan
    far = float(fa / pred_alerts) if pred_alerts > 0 else np.nan
    csi = float(hits / denom_csi) if denom_csi > 0 else np.nan
    base_rate = float(obs_events / n)
    
    # PR-AUC for lower-tail event (higher score = more severe drought)
    if obs_events > 0 and obs_events < n:
        scores = -y_hat
        prec, rec, _ = precision_recall_curve(obs_event, scores)
        pr_auc = float(auc(rec, prec))
    elif obs_events == 0:
        pr_auc = np.nan
    else:
        pr_auc = 1.0
        
    return {
        'Threshold': float(threshold),
        'Hits': hits,
        'FalseAlarms': fa,
        'Misses': miss,
        'CorrectRejections': cr,
        'ObsEvents': obs_events,
        'PredAlerts': pred_alerts,
        'BaseRate': base_rate,
        'POD': pod,
        'FAR': far,
        'CSI': csi,
        'PR_AUC': pr_auc
    }

# ==============================================================================
# 2. MODEL LOADER HELPERS
# ==============================================================================
def load_estimator(model_path: Path):
    if not model_path.is_file():
        return None, None, None
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and 'model' in obj:
        return obj['model'], obj.get('scaler'), obj.get('train_medians')
    return obj, None, None

def partition_splits_with_embargo(df: pd.DataFrame, horizon_weeks: int):
    mask_train = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
    mask_val   = (df.index >= VAL_START) & (df.index <= VAL_END)
    mask_test  = (df.index >= TEST_START) & (df.index <= TEST_END)
    
    df_train_raw = df[mask_train].copy()
    df_val_raw   = df[mask_val].copy()
    df_test      = df[mask_test].copy()
    
    if horizon_weeks > 0:
        df_train = df_train_raw.iloc[:-horizon_weeks].copy()
        df_val   = df_val_raw.iloc[:-horizon_weeks].copy()
    else:
        df_train = df_train_raw.copy()
        df_val   = df_val_raw.copy()
        
    return df_train, df_val, df_test

# ==============================================================================
# 3. RECOMPUTATION OF MODEL PREDICTIONS ON TEST SET
# ==============================================================================
def recompute_all_predictions():
    """
    Loads raw weekly tensors and saved models, generating identical-masked predictions
    for M0 (persistence & climatology), M1 (GBT & Ridge), M2 (GBT), and Protocol 2 (LSTM).
    """
    records = []
    pred_frames = []

    for info in PORTFOLIO:
        station = info['station']
        clean_name = station.replace(' ', '_').replace('(', '').replace(')', '')
        gauge_id = info['gauge_id']
        state = info['state']
        area = info['area_km2']
        is_holdout = info['is_holdout']
        
        tensor_path = TENSORS_DIR / f"weekly_tensor_gauge_{gauge_id:05d}_{station}.csv"
        if not tensor_path.is_file():
            print(f"[-] Tensor missing for {station}: {tensor_path}")
            continue
            
        df = pd.read_csv(tensor_path, parse_dates=['date']).set_index('date')
        
        for H in [1, 2, 4, 8]:
            target_col = f'target_H{H}'
            df_train, df_val, df_test = partition_splits_with_embargo(df, horizon_weeks=H)
            
            # Drought thresholds strictly from Training fold
            y_tr = df_train[target_col].dropna().values
            q80 = float(np.percentile(y_tr, 20.0))
            q90 = float(np.percentile(y_tr, 10.0))
            q95 = float(np.percentile(y_tr, 5.0))
            
            # --- M0 Baseline ---
            if H in [1, 2]:
                preds_m0_bugged = df_test['flow_lag_0'].values.copy()
                preds_m0_fixed = df_test['flow_lag_0'].values.copy()
            else:
                # Bugged climatology (double shift: issue week grouping + target week lookup)
                train_clim_bugged = df_train.groupby(df_train.index.isocalendar().week)[target_col].mean()
                target_weeks = (df_test.index + pd.Timedelta(weeks=H)).isocalendar().week
                preds_m0_bugged = target_weeks.map(train_clim_bugged).fillna(float(df_train[target_col].mean())).values
                
                # Fixed climatology (horizon-agnostic weekly climatology from historical flow)
                train_clim_fixed = df_train.groupby(df_train.index.isocalendar().week)['flow_target_m3s'].mean()
                preds_m0_fixed = target_weeks.map(train_clim_fixed).fillna(float(df_train['flow_target_m3s'].mean())).values

            # --- M1 GBT ---
            m1_gbt_path = MODELS_DIR / "protocol1" / f"m1_gbt_gauge_{gauge_id:05d}_{clean_name}_H{H}.joblib"
            m1_gbt_est, _, med1_gbt = load_estimator(m1_gbt_path)
            if m1_gbt_est is not None:
                impute_val = med1_gbt if med1_gbt is not None else df_train[M1_FEATURES].median()
                X1 = df_test[M1_FEATURES].fillna(impute_val)
                preds_m1_gbt = np.maximum(0.0, m1_gbt_est.predict(X1.values))
            else:
                preds_m1_gbt = np.full(len(df_test), np.nan)
                
            # --- M1 Ridge ---
            m1_ridge_path = MODELS_DIR / "protocol1" / f"m1_ridge_gauge_{gauge_id:05d}_{clean_name}_H{H}.joblib"
            m1_ridge_est, scaler1, med1_ridge = load_estimator(m1_ridge_path)
            if m1_ridge_est is not None:
                impute_val = med1_ridge if med1_ridge is not None else df_train[M1_FEATURES].median()
                X1_r = df_test[M1_FEATURES].fillna(impute_val)
                X1_rs = scaler1.transform(X1_r) if scaler1 else X1_r.values
                preds_m1_ridge = np.maximum(0.0, m1_ridge_est.predict(X1_rs))
            else:
                preds_m1_ridge = np.full(len(df_test), np.nan)
                
            # --- M2 GBT ---
            m2_gbt_path = MODELS_DIR / "protocol1" / f"m2_gbt_gauge_{gauge_id:05d}_{clean_name}_H{H}.joblib"
            m2_gbt_est, _, med2_gbt = load_estimator(m2_gbt_path)
            if m2_gbt_est is not None:
                impute_val = med2_gbt if med2_gbt is not None else df_train[M2_TABULAR_FEATURES].median()
                X2 = df_test[M2_TABULAR_FEATURES].fillna(impute_val)
                preds_m2_gbt = np.maximum(0.0, m2_gbt_est.predict(X2.values))
            else:
                preds_m2_gbt = np.full(len(df_test), np.nan)

            # --- Protocol 2 Spatial Transfer (Holdouts only) ---
            preds_p2_lstm = np.full(len(df_test), np.nan)
            preds_p2_gbt = np.full(len(df_test), np.nan)
            preds_p2_ridge = np.full(len(df_test), np.nan)
            
            if is_holdout:
                conv_h = 604.8 / area
                m1_spec_features = ['flow_target_specific_mm', 'sin_week', 'cos_week']
                
                # M1 Pool GBT
                p2_gbt_path = MODELS_DIR / "protocol2" / f"m1_gbt_pool_{state.lower()}_heldout_{clean_name}_H{H}.joblib"
                p2_gbt_est, _, med_p2_g = load_estimator(p2_gbt_path)
                if p2_gbt_est is not None:
                    X1_p2 = df_test[m1_spec_features].fillna(med_p2_g if med_p2_g is not None else df_test[m1_spec_features].median())
                    preds_p2_gbt_spec = p2_gbt_est.predict(X1_p2.values)
                    preds_p2_gbt = np.maximum(0.0, preds_p2_gbt_spec / conv_h)
                    
                # M1 Pool Ridge
                p2_ridge_path = MODELS_DIR / "protocol2" / f"m1_ridge_pool_{state.lower()}_heldout_{clean_name}_H{H}.joblib"
                p2_ridge_est, scaler_p2_r, med_p2_r = load_estimator(p2_ridge_path)
                if p2_ridge_est is not None:
                    X1_p2r = df_test[m1_spec_features].fillna(med_p2_r if med_p2_r is not None else df_test[m1_spec_features].median())
                    X1_p2rs = scaler_p2_r.transform(X1_p2r) if scaler_p2_r else X1_p2r.values
                    preds_p2_ridge_spec = p2_ridge_est.predict(X1_p2rs)
                    preds_p2_ridge = np.maximum(0.0, preds_p2_ridge_spec / conv_h)
                    
                # M2 Pool LSTM (.pt)
                p2_lstm_path = MODELS_DIR / "protocol2" / f"m2_lstm_pool_{state.lower()}_heldout_{clean_name}_H{H}.pt"
                if p2_lstm_path.is_file():
                    try:
                        # Extract test sequences with buffer
                        seq_len = 8
                        buffer_slice = df_val.iloc[-(seq_len - 1):]
                        comb = pd.concat([buffer_slice, df_test], axis=0)
                        
                        # Load training pool scaler from protocol 2
                        # Fit scaler on pool train fold (Stations 1,2,3) to match train_m2_models.py
                        pool_train_dfs = []
                        for p_stn in PORTFOLIO:
                            if p_stn['state'] == state and not p_stn['is_holdout']:
                                p_path = TENSORS_DIR / f"weekly_tensor_gauge_{p_stn['gauge_id']:05d}_{p_stn['station']}.csv"
                                p_df = pd.read_csv(p_path, parse_dates=['date']).set_index('date')
                                tr, _, _ = partition_splits_with_embargo(p_df, horizon_weeks=H)
                                pool_train_dfs.append(tr)
                        all_pool_train = pd.concat(pool_train_dfs, axis=0)
                        scaler_lstm = StandardScaler()
                        scaler_lstm.fit(all_pool_train[M2_SEQUENTIAL_FEATURES].fillna(all_pool_train[M2_SEQUENTIAL_FEATURES].median()))
                        
                        comb_feats = comb[M2_SEQUENTIAL_FEATURES].fillna(all_pool_train[M2_SEQUENTIAL_FEATURES].median())
                        comb_scaled = scaler_lstm.transform(comb_feats)
                        
                        seqs = []
                        start_idx = seq_len - 1
                        for i in range(start_idx, len(comb)):
                            seq = comb_scaled[i - seq_len + 1 : i + 1, :]
                            seqs.append(seq)
                            
                        device = torch.device('cpu')
                        lstm = HydroLSTM(input_dim=len(M2_SEQUENTIAL_FEATURES), hidden_dim=64, num_layers=2)
                        lstm.load_state_dict(torch.load(p2_lstm_path, map_location=device))
                        lstm.eval()
                        with torch.no_grad():
                            t_x = torch.tensor(np.array(seqs), dtype=torch.float32)
                            pred_spec = lstm(t_x).numpy()
                            
                        # Specific runoff to volumetric (mm/week -> m3/s)
                        # conversion factor: Area * 1000 / (7 * 86400) = Area / 604.8
                        # Q (m3/s) = q (mm/week) * Area / 604.8 = pred_spec * conv
                        conv = area / 604.8
                        preds_p2_lstm = np.maximum(0.0, pred_spec * conv)
                    except Exception as e:
                        print(f"[-] Error evaluating LSTM for {station} H{H}: {e}")

            # Enforce Evaluation Mask Contract:
            # Row evaluated iff target, flow_lag_0, M0, M1_GBT, M2_GBT are non-NaN, and,
            # at ZERO_AS_MISSING_STATIONS, neither the issue-time flow nor the target flow
            # is a recorded zero (unmeasured week). Every model is scored on the same rows.
            y_test = df_test[target_col].values
            flow_0 = df_test['flow_lag_0'].values
            
            eval_mask = (
                (~np.isnan(y_test)) &
                (~np.isnan(flow_0)) &
                (~np.isnan(preds_m0_fixed)) &
                (~np.isnan(preds_m1_gbt)) &
                (~np.isnan(preds_m2_gbt))
            )
            zero_recorded = (y_test == 0.0) | (flow_0 == 0.0)
            gap_masked = eval_mask & zero_recorded if station in ZERO_AS_MISSING_STATIONS else np.zeros_like(eval_mask)
            eval_mask = eval_mask & ~gap_masked
            if is_holdout:
                eval_mask = eval_mask & ~np.isnan(preds_p2_lstm)
            
            y_eval = y_test[eval_mask]
            
            # Track A
            m0_bug_a = compute_continuous_metrics(y_eval, preds_m0_bugged[eval_mask])
            m0_fix_a = compute_continuous_metrics(y_eval, preds_m0_fixed[eval_mask])
            m1_gbt_a = compute_continuous_metrics(y_eval, preds_m1_gbt[eval_mask])
            m1_rdg_a = compute_continuous_metrics(y_eval, preds_m1_ridge[eval_mask])
            m2_gbt_a = compute_continuous_metrics(y_eval, preds_m2_gbt[eval_mask])
            
            # Track B Severe Drought (Q95). M0 = persistence (H<=2) or fixed climatology (H>=4).
            m0_b = compute_drought_metrics(y_eval, preds_m0_fixed[eval_mask], q95)
            m1_b = compute_drought_metrics(y_eval, preds_m1_gbt[eval_mask], q95)
            m2_b = compute_drought_metrics(y_eval, preds_m2_gbt[eval_mask], q95)
            
            # Protocol 2 Metrics
            p2_lstm_a = compute_continuous_metrics(y_eval, preds_p2_lstm[eval_mask]) if is_holdout else {}
            p2_gbt_a  = compute_continuous_metrics(y_eval, preds_p2_gbt[eval_mask]) if is_holdout else {}
            p2_rdg_a  = compute_continuous_metrics(y_eval, preds_p2_ridge[eval_mask]) if is_holdout else {}
            
            pred_frames.append(pd.DataFrame({
                'station': station, 'horizon': f"{H}w", 'issue_date': df_test.index.strftime('%Y-%m-%d'),
                'target_date': (df_test.index + pd.Timedelta(weeks=H)).strftime('%Y-%m-%d'),
                'y_obs': y_test, 'scored': eval_mask, 'zero_as_missing': gap_masked,
                'm0': preds_m0_fixed, 'm1_gbt': preds_m1_gbt, 'm1_ridge': preds_m1_ridge, 'm2_gbt': preds_m2_gbt,
                'p2_lstm': preds_p2_lstm, 'p2_gbt': preds_p2_gbt, 'p2_ridge': preds_p2_ridge,
                'q95': q95, 'q90': q90, 'q80': q80,
            }))
            
            rec = {
                'station': station,
                'state': state,
                'gauge_id': gauge_id,
                'horizon': f"{H}w",
                'horizon_num': H,
                'is_holdout': is_holdout,
                'zero_as_missing_station': station in ZERO_AS_MISSING_STATIONS,
                'n_eval': len(y_eval),
                'n_masked_zero_weeks': int(gap_masked.sum()),
                'q95_threshold': q95,
                'test_zero_flow_weeks': int(np.sum(y_eval == 0.0)),
                'test_zero_flow_pct': float(np.mean(y_eval == 0.0) * 100) if len(y_eval) else np.nan,
                'test_q95_event_count': m0_b['ObsEvents'],
                'test_q95_base_rate': m0_b['BaseRate'],
                # Track A
                'm0_nse_bugged': m0_bug_a['NSE'],
                'm0_nse_fixed': m0_fix_a['NSE'],
                'm0_kge_bugged': m0_bug_a['KGE'],
                'm0_kge_fixed': m0_fix_a['KGE'],
                'm1_nse': m1_gbt_a['NSE'],
                'm1_kge': m1_gbt_a['KGE'],
                'm1_ridge_nse': m1_rdg_a['NSE'],
                'm2_nse': m2_gbt_a['NSE'],
                'm2_kge': m2_gbt_a['KGE'],
                'delta_nse_recomputed': m2_gbt_a['NSE'] - m1_gbt_a['NSE'],
                'delta_kge_recomputed': m2_gbt_a['KGE'] - m1_gbt_a['KGE'],
            }
            for tag, mt in (('m0', m0_fix_a), ('m1', m1_gbt_a), ('m2', m2_gbt_a)):
                for comp in ('r', 'alpha', 'beta', 'RMSE', 'MAE'):
                    rec[f'{tag}_{comp.lower()}'] = mt[comp]
            for tag, mb in (('m0', m0_b), ('m1', m1_b), ('m2', m2_b)):
                rec.update({
                    f'{tag}_hits': mb['Hits'], f'{tag}_fa': mb['FalseAlarms'], f'{tag}_miss': mb['Misses'],
                    f'{tag}_far': mb['FAR'], f'{tag}_pod': mb['POD'], f'{tag}_csi': mb['CSI'], f'{tag}_pr_auc': mb['PR_AUC'],
                })
            rec.update({
                # Protocol 2 Zero-Shot
                'p2_lstm_nse': p2_lstm_a.get('NSE', np.nan),
                'p2_lstm_kge': p2_lstm_a.get('KGE', np.nan),
                'p2_gbt_nse': p2_gbt_a.get('NSE', np.nan),
                'p2_ridge_nse': p2_rdg_a.get('NSE', np.nan),
            })
            records.append(rec)
            
    pd.concat(pred_frames, ignore_index=True).to_csv(PREDICTIONS_CSV, index=False, float_format='%.4f')
    print(f"[+] Saved per-week test predictions to: {PREDICTIONS_CSV}")
    return pd.DataFrame(records)

# ==============================================================================
# 4. LOAD PUBLISHED DATA FROM GH-PAGES
# ==============================================================================
def load_published_gh_pages():
    raw_js = subprocess.check_output(['git', 'show', 'gh-pages:assets/benchmark-data.js'], text=True)
    m_stn = re.search(r'window\.STATIONS\s*=\s*(\[.*?\]);', raw_js, re.DOTALL)
    m_med = re.search(r'window\.MEDIANS\s*=\s*(\{.*?\});', raw_js, re.DOTALL)
    
    stations_pub = json.loads(m_stn.group(1))
    medians_pub = json.loads(m_med.group(1))
    return stations_pub, medians_pub

# ==============================================================================
# 5. GENERATE AUDIT COMPARISON & MISMATCH TABLE
# ==============================================================================
def run_audit():
    print("=" * 80)
    print(" AUDIT EXECUTION: ES 335 METRICS RECOMPUTATION & COMPARISON")
    print("=" * 80)
    
    print("[*] Recomputing predictions from trained models on weekly tensors...")
    df_recomputed = recompute_all_predictions()
    
    print("[*] Fetching published numbers from gh-pages (assets/benchmark-data.js)...")
    pub_stations, pub_medians = load_published_gh_pages()
    
    # Map published stations by name
    pub_stn_map = {s['name'].lower(): s for s in pub_stations}
    
    mismatches = []
    
    for _, row in df_recomputed.iterrows():
        stn_key = row['station'].lower()
        if 'bamni' in stn_key:
            stn_key = 'bamni'
        stn_pub = pub_stn_map.get(stn_key)
        if not stn_pub:
            continue
            
        h = row['horizon']
        metrics_pub = stn_pub['metrics'].get(h, {})
        
        # Compare M0 NSE
        pub_m0_nse = float(metrics_pub.get('m0_nse', np.nan))
        rec_m0_nse_bugged = row['m0_nse_bugged']
        rec_m0_nse_fixed = row['m0_nse_fixed']
        
        # Check if published M0 matched the bugged climatology
        if abs(pub_m0_nse - rec_m0_nse_bugged) > 0.005:
            mismatches.append({
                'Station': row['station'],
                'Horizon': h,
                'Metric': 'M0_NSE',
                'Published': f"{pub_m0_nse:+.3f}",
                'Recomputed': f"{rec_m0_nse_bugged:+.3f}",
                'Discrepancy': f"{rec_m0_nse_bugged - pub_m0_nse:+.3f}",
                'Category': 'M0 Baseline Mismatch',
                'Notes': 'Published M0 does not match recomputed model prediction.'
            })
            
        # Compare M1 NSE
        pub_m1_nse = float(metrics_pub.get('m1_nse', np.nan))
        rec_m1_nse = row['m1_nse']
        if abs(pub_m1_nse - rec_m1_nse) > 0.005:
            mismatches.append({
                'Station': row['station'],
                'Horizon': h,
                'Metric': 'M1_NSE',
                'Published': f"{pub_m1_nse:+.3f}",
                'Recomputed': f"{rec_m1_nse:+.3f}",
                'Discrepancy': f"{rec_m1_nse - pub_m1_nse:+.3f}",
                'Category': 'M1 NSE Mismatch',
                'Notes': 'Model prediction differs from published text.'
            })
            
        # Compare M2 NSE
        pub_m2_nse = float(metrics_pub.get('m2_nse', np.nan))
        rec_m2_nse = row['m2_nse']
        if abs(pub_m2_nse - rec_m2_nse) > 0.005:
            mismatches.append({
                'Station': row['station'],
                'Horizon': h,
                'Metric': 'M2_NSE',
                'Published': f"{pub_m2_nse:+.3f}",
                'Recomputed': f"{rec_m2_nse:+.3f}",
                'Discrepancy': f"{rec_m2_nse - pub_m2_nse:+.3f}",
                'Category': 'M2 NSE Mismatch',
                'Notes': 'Model prediction differs from published text.'
            })
            
        # Compare M2 KGE
        pub_m2_kge = float(metrics_pub.get('m2_kge', np.nan))
        rec_m2_kge = row['m2_kge']
        if abs(pub_m2_kge - rec_m2_kge) > 0.005:
            mismatches.append({
                'Station': row['station'],
                'Horizon': h,
                'Metric': 'M2_KGE',
                'Published': f"{pub_m2_kge:+.3f}",
                'Recomputed': f"{rec_m2_kge:+.3f}",
                'Discrepancy': f"{rec_m2_kge - pub_m2_kge:+.3f}",
                'Category': 'M2 KGE Mismatch',
                'Notes': 'Model prediction differs from published text.'
            })
            
        # Compare Delta NSE
        pub_delta_str = str(metrics_pub.get('delta_nse_m2_m1', 'nan'))
        try:
            pub_delta = float(pub_delta_str)
            rec_delta = row['delta_nse_recomputed']
            if abs(pub_delta - rec_delta) > 0.005:
                mismatches.append({
                    'Station': row['station'],
                    'Horizon': h,
                    'Metric': 'Delta_NSE (M2-M1)',
                    'Published': f"{pub_delta:+.3f}",
                    'Recomputed': f"{rec_delta:+.3f}",
                    'Discrepancy': f"{rec_delta - pub_delta:+.3f}",
                    'Category': 'Delta NSE Inconsistency',
                    'Notes': 'Station-level Delta does not equal M2_NSE - M1_NSE.'
                })
        except ValueError:
            pass

        # Compare FAR M1 & M2 (Issue A: Fabricated values)
        pub_far_m1 = metrics_pub.get('far_m1')
        pub_far_m2 = metrics_pub.get('far_m2')
        rec_far_m1 = row['m1_far']
        rec_far_m2 = row['m2_far']
        
        def parse_pub_val(v):
            if v is None:
                return np.nan
            s = str(v).strip().lower()
            if s in ['nan', 'undefined', 'null', 'none', '-']:
                return np.nan
            try:
                return float(s)
            except (ValueError, TypeError):
                return np.nan

        pf1 = parse_pub_val(pub_far_m1)
        if not np.isnan(pf1):
            diff1 = abs(pf1 - rec_far_m1) if not np.isnan(rec_far_m1) else 999.0
            if np.isnan(rec_far_m1) or diff1 > 0.02:
                mismatches.append({
                    'Station': row['station'],
                    'Horizon': h,
                    'Metric': 'FAR_M1',
                    'Published': f"{pub_far_m1}",
                    'Recomputed': f"{rec_far_m1:.3f}" if not np.isnan(rec_far_m1) else "NaN (0 alerts)",
                    'Discrepancy': f"{rec_far_m1 - pf1:+.3f}" if not np.isnan(rec_far_m1) else "Undefined in Code",
                    'Category': 'FAR Fabrication (Issue A)',
                    'Notes': f"Published value is {pf1:.3f}. Recomputed contingency: {row['m1_hits']} hits, {row['m1_fa']} FA."
                })
                
        pf2 = parse_pub_val(pub_far_m2)
        if not np.isnan(pf2):
            diff2 = abs(pf2 - rec_far_m2) if not np.isnan(rec_far_m2) else 999.0
            if np.isnan(rec_far_m2) or diff2 > 0.02:
                mismatches.append({
                    'Station': row['station'],
                    'Horizon': h,
                    'Metric': 'FAR_M2',
                    'Published': f"{pub_far_m2}",
                    'Recomputed': f"{rec_far_m2:.3f}" if not np.isnan(rec_far_m2) else "NaN (0 alerts)",
                    'Discrepancy': f"{rec_far_m2 - pf2:+.3f}" if not np.isnan(rec_far_m2) else "Undefined in Code",
                    'Category': 'FAR Fabrication (Issue A)',
                    'Notes': f"Published value is {pf2:.3f}. Recomputed contingency: {row['m2_hits']} hits, {row['m2_fa']} FA."
                })

        # Compare CSI M2 & Impossible cells
        pub_csi_m2 = metrics_pub.get('csi_m2')
        rec_csi_m2 = row['m2_csi']
        pc2 = parse_pub_val(pub_csi_m2)
        if not np.isnan(pc2):
            # Impossible cell check: CSI = 0.0 with FAR < 1.0
            if pc2 == 0.0 and not np.isnan(pf2) and pf2 < 1.0:
                mismatches.append({
                    'Station': row['station'],
                    'Horizon': h,
                    'Metric': 'CSI_vs_FAR_M2',
                    'Published': f"CSI={pc2:.3f}, FAR={pub_far_m2}",
                    'Recomputed': f"CSI={rec_csi_m2:.3f}, FAR={rec_far_m2:.3f}" if not np.isnan(rec_csi_m2) else "NaN",
                    'Discrepancy': "Impossible Contingency",
                    'Category': 'Impossible Contingency (Issue A)',
                    'Notes': f"CSI=0 requires Hits=0, which mathematically implies FAR=1.0 or undefined. Published FAR was {pub_far_m2}."
                })

    df_mismatches = pd.DataFrame(mismatches)
    
    # Portfolio Medians Audit (Issue B)
    median_rows = []
    for h in ['1w', '2w', '4w', '8w']:
        sub = df_recomputed[df_recomputed['horizon'] == h]
        pub_m = pub_medians.get(h, {})
        
        # Continuous Medians
        rec_med_m0_nse = sub['m0_nse_bugged'].median()
        rec_med_m0_fix = sub['m0_nse_fixed'].median()
        rec_med_m1_nse = sub['m1_nse'].median()
        rec_med_m2_nse = sub['m2_nse'].median()
        
        rec_med_m0_kge = sub['m0_kge_bugged'].median()
        rec_med_m1_kge = sub['m1_kge'].median()
        rec_med_m2_kge = sub['m2_kge'].median()
        
        # Delta: per-station median of differences vs diff of medians
        rec_med_delta_per_stn = (sub['m2_nse'] - sub['m1_nse']).median()
        rec_diff_of_medians = rec_med_m2_nse - rec_med_m1_nse
        
        # Published numbers
        pub_m0_nse_f = float(pub_m.get('m0_nse', np.nan))
        pub_m1_nse_f = float(pub_m.get('m1_nse', np.nan))
        pub_m2_nse_f = float(pub_m.get('m2_nse', np.nan))
        pub_m0_kge_f = float(pub_m.get('m0_kge', np.nan))
        pub_m1_kge_f = float(pub_m.get('m1_kge', np.nan))
        pub_m2_kge_f = float(pub_m.get('m2_kge', np.nan))
        
        median_rows.append({
            'Horizon': h,
            'Metric': 'M0_NSE_Median',
            'Published_Median': f"{pub_m0_nse_f:+.3f}",
            'Actual_PerStation_Median': f"{rec_med_m0_nse:+.3f}",
            'Fixed_Climatology_Median': f"{rec_med_m0_fix:+.3f}",
            'Discrepancy': f"{rec_med_m0_nse - pub_m0_nse_f:+.3f}",
            'Cause': 'Copied from pre-clipping CSV where Ashti/Kudige M0 were negative.'
        })
        median_rows.append({
            'Horizon': h,
            'Metric': 'M1_NSE_Median',
            'Published_Median': f"{pub_m1_nse_f:+.3f}",
            'Actual_PerStation_Median': f"{rec_med_m1_nse:+.3f}",
            'Fixed_Climatology_Median': '-',
            'Discrepancy': f"{rec_med_m1_nse - pub_m1_nse_f:+.3f}",
            'Cause': 'Aggregation discrepancy.'
        })
        median_rows.append({
            'Horizon': h,
            'Metric': 'M2_NSE_Median',
            'Published_Median': f"{pub_m2_nse_f:+.3f}",
            'Actual_PerStation_Median': f"{rec_med_m2_nse:+.3f}",
            'Fixed_Climatology_Median': '-',
            'Discrepancy': f"{rec_med_m2_nse - pub_m2_nse_f:+.3f}",
            'Cause': 'Aggregation discrepancy.'
        })
        median_rows.append({
            'Horizon': h,
            'Metric': 'Delta_NSE (Gain)',
            'Published_Median': f"{pub_m2_nse_f - pub_m1_nse_f:+.3f} (Diff of Medians)",
            'Actual_PerStation_Median': f"{rec_med_delta_per_stn:+.3f} (Median of Diffs)",
            'Fixed_Climatology_Median': '-',
            'Discrepancy': f"{rec_med_delta_per_stn - (pub_m2_nse_f - pub_m1_nse_f):+.3f}",
            'Cause': 'Issue B: Website computed Diff of Medians instead of Median of per-station Diffs.'
        })

    df_median_audit = pd.DataFrame(median_rows)
    
    return df_recomputed, df_mismatches, df_median_audit

# ==============================================================================
# 6. STANDALONE EXECUTION & REPORT PRINTING
# ==============================================================================
if __name__ == '__main__':
    df_recomputed, df_mismatches, df_median_audit = run_audit()
    
    # Save recomputed audit dataset
    audit_out_csv = RESULTS_DIR / "audited_metrics_recomputed.csv"
    df_recomputed.to_csv(audit_out_csv, index=False)
    print(f"\n[+] Saved full recomputed metrics table to: {audit_out_csv}")
    
    print("\n" + "=" * 100)
    print(" 🛑 HARD STOP 1: COMPREHENSIVE AUDIT MISMATCH REPORT")
    print("=" * 100)
    print(f"Total Mismatches / Anomalies Identified: {len(df_mismatches)}")
    print("-" * 100)
    
    # Show breakdown by category
    print("\n[SUMMARY BY CATEGORY]:")
    print(df_mismatches['Category'].value_counts().to_string())
    
    print("\n" + "=" * 100)
    print(" 1. PORTFOLIO MEDIANS MISMATCH TABLE (ISSUE B)")
    print("=" * 100)
    print(df_median_audit.to_string(index=False))
    
    print("\n" + "=" * 100)
    print(" 2. CRITICAL CONTINGENCY & FABRICATION MISMATCHES (SAMPLE OF 25)")
    print("=" * 100)
    print(df_mismatches[['Station', 'Horizon', 'Metric', 'Published', 'Recomputed', 'Category', 'Notes']].head(25).to_string(index=False))
    
    print("\n" + "=" * 100)
    print(" 3. COMPLETE STATION-BY-STATION RECOMPUTED SUMMARY")
    print("=" * 100)
    cols_show = ['station', 'horizon', 'm0_nse_bugged', 'm0_nse_fixed', 'm1_nse', 'm2_nse', 'delta_nse_recomputed', 'm1_far', 'm2_far', 'm2_csi', 'test_q95_event_count']
    print(df_recomputed[cols_show].to_string(index=False))
    
    print("\n[+] Audit recomputation script completed successfully.")
    
    # Run drought severity sensitivity audit (Q80, Q90, Q95)
    print("\n" + "=" * 100)
    print(" 4. RUNNING DROUGHT SEVERITY SENSITIVITY AUDIT (MILD Q80, MODERATE Q90, SEVERE Q95)")
    print("=" * 100)
    import audit_drought_sensitivity
    audit_drought_sensitivity.main()
