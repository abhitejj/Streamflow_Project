#!/usr/bin/env python3
"""
CAMELS-Era Robustness Experiment (1980–2010)

Evaluates whether the empirical findings from the primary 2016–2025 test block
transfer to the historical CAMELS-IND record (1980–2010), where streamflow and
all meteorological forcings are 100% sourced from CAMELS-IND.

Experimental Protocol:
- Train fold: 1980-01-01 to 2000-12-31 (purged terminal H weeks)
- Val fold:   2001-01-01 to 2004-12-31 (purged terminal H weeks)
- Test fold:  2005-01-01 to 2010-12-31 (unpurged out-of-sample block, 6 years)
- Zero-as-missing masking at Karad, Warunji, Yadgir, Huvinhedigi enforced.
- Models: M0 (Baseline), M1 GBT (Flow-only), M2 GBT (Coupled),
          M1-log GBT (Log-target), M2-log GBT (Log-target Coupled).
- Output: data/processed/model_results/robustness_camels_era.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# Add scripts directory to path for eval_config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_config import ZERO_AS_MISSING_STATIONS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TENSORS_DIR = PROJECT_ROOT / "data" / "processed" / "weekly_tensors"
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "model_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "robustness_camels_era.csv"

# Date partitions
TRAIN_START = "1980-01-01"
TRAIN_END   = "2000-12-31"
VAL_START   = "2001-01-01"
VAL_END     = "2004-12-31"
TEST_START  = "2005-01-01"
TEST_END    = "2010-12-31"

HORIZONS = [1, 2, 4, 8]

PORTFOLIO_CONFIG = [
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

M1_FEATURES = [
    'flow_lag_0', 'flow_lag_1', 'flow_lag_2', 'flow_lag_3', 'flow_lag_4', 'flow_lag_7',
    'flow_std_7d', 'flow_min_7d', 'flow_max_7d', 'flow_recession_slope',
    'sin_week', 'cos_week'
]

M2_FEATURES = M1_FEATURES + [
    'rain_sum_1w', 'rain_sum_2w', 'rain_sum_1m', 'rain_sum_2m', 'rain_sum_3m', 'api_30d', 'dry_days_count',
    'temp_mean_1w', 'temp_max_1w', 'temp_min_1w', 'pet_sum_1w', 'pet_sum_1m',
    'water_deficit_1w', 'water_deficit_1m'
]

NON_NEG_FLOW_FEATS = [
    'flow_lag_0', 'flow_lag_1', 'flow_lag_2', 'flow_lag_3', 'flow_lag_4', 'flow_lag_7',
    'flow_min_7d', 'flow_max_7d', 'flow_std_7d'
]

def prepare_log_features(df: pd.DataFrame, feature_cols: list, area_km2: float) -> pd.DataFrame:
    df_out = df[feature_cols].copy()
    conv = 604.8 / area_km2
    for col in feature_cols:
        if col in NON_NEG_FLOW_FEATS and col in df_out.columns:
            df_out[col] = np.log1p(np.maximum(0.0, df_out[col].values * conv))
        elif col == 'flow_recession_slope' and col in df_out.columns:
            df_out[col] = df_out[col].values * conv
    return df_out

def compute_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y = y_true[valid]
    y_hat = np.maximum(y_pred[valid], 0.0)
    n = len(y)
    if n == 0:
        return {'N': 0, 'NSE': np.nan, 'log_nse': np.nan, 'KGE': np.nan, 'RMSE': np.nan, 'MAE': np.nan}
    
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    nse = float(np.clip(1.0 - (ss_res / ss_tot), -10.0, 1.0)) if ss_tot > 1e-9 else np.nan
    
    # Log-NSE
    eps = 0.01
    z_obs = np.log(y + eps)
    z_hat = np.log(y_hat + eps)
    ss_res_log = float(np.sum((z_obs - z_hat) ** 2))
    ss_tot_log = float(np.sum((z_obs - np.mean(z_obs)) ** 2))
    log_nse = float(np.clip(1.0 - (ss_res_log / ss_tot_log), -10.0, 1.0)) if ss_tot_log > 1e-9 else np.nan
    
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    mae = float(np.mean(np.abs(y - y_hat)))
    
    # KGE
    std_y = float(np.std(y))
    std_hat = float(np.std(y_hat))
    mean_y = float(np.mean(y))
    mean_hat = float(np.mean(y_hat))
    
    if std_y < 1e-6 or std_hat < 1e-6:
        r = 0.0
        alpha = 0.0
    else:
        cov = float(np.mean((y - mean_y) * (y_hat - mean_hat)))
        r = float(np.clip(cov / (std_y * std_hat), -1.0, 1.0))
        alpha = float(std_hat / std_y)
        
    beta = float(mean_hat / mean_y) if mean_y > 1e-6 else 1.0
    ed = np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    kge = float(np.clip(1.0 - ed, -10.0, 1.0))
    
    return {'N': n, 'NSE': nse, 'log_nse': log_nse, 'KGE': kge, 'RMSE': rmse, 'MAE': mae}

def compute_contingency(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y = y_true[valid]
    y_hat = np.maximum(y_pred[valid], 0.0)
    n = len(y)
    if n == 0:
        return {'Hits': 0, 'FA': 0, 'Miss': 0, 'CR': 0, 'ObsEvents': 0, 'PredAlerts': 0, 'BaseRate': np.nan, 'CSI': np.nan, 'POD': np.nan, 'FAR': np.nan, 'PR_AUC': np.nan}
    
    obs = (y <= threshold).astype(int)
    alert = (y_hat <= threshold).astype(int)
    
    h = int(np.sum((obs == 1) & (alert == 1)))
    f = int(np.sum((obs == 0) & (alert == 1)))
    m = int(np.sum((obs == 1) & (alert == 0)))
    cr = int(np.sum((obs == 0) & (alert == 0)))
    
    obs_ev = h + m
    pred_al = h + f
    csi_denom = h + m + f
    
    csi = float(h / csi_denom) if csi_denom > 0 else np.nan
    pod = float(h / obs_ev) if obs_ev > 0 else np.nan
    far = float(f / pred_al) if pred_al > 0 else np.nan
    base_rate = float(obs_ev / n)
    
    if obs_ev > 0 and obs_ev < n:
        prec, rec, _ = precision_recall_curve(obs, -y_hat)
        pr_auc = float(auc(rec, prec))
    elif obs_ev == 0:
        pr_auc = np.nan
    else:
        pr_auc = 1.0
        
    return {
        'Hits': h, 'FA': f, 'Miss': m, 'CR': cr,
        'ObsEvents': obs_ev, 'PredAlerts': pred_al, 'BaseRate': base_rate,
        'CSI': csi, 'POD': pod, 'FAR': far, 'PR_AUC': pr_auc
    }

def fit_tune_gbt(X_tr, y_tr, X_val, y_val_vol, target_mode='raw', area_km2=1000.0):
    best_rmse = float('inf')
    best_model = None
    for depth in [3, 5, 7]:
        for l2 in [0.0, 1.0, 10.0]:
            m = HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.05, max_depth=depth, l2_regularization=l2,
                min_samples_leaf=20, early_stopping=False, random_state=42
            )
            m.fit(X_tr, y_tr)
            if target_mode == 'log':
                preds_log = m.predict(X_val)
                val_preds = np.maximum(0.0, np.expm1(preds_log)) * (area_km2 / 604.8)
            else:
                val_preds = np.maximum(0.0, m.predict(X_val))
            rmse = np.sqrt(np.mean((y_val_vol - val_preds) ** 2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = m
    return best_model

def run_experiment():
    print("=" * 80)
    print("CAMELS-Era Robustness Experiment: Train 1980-2000, Val 2001-2004, Test 2005-2010")
    print("=" * 80)
    
    all_station_results = []
    pooled_data = {h: {m: {'y_obs': [], 'y_pred': [], 'stn_mean': []} for m in ['M0', 'M1', 'M2', 'M1-log', 'M2-log']} for h in HORIZONS}
    pooled_drought = {h: {sev: {m: {'Hits': 0, 'FA': 0, 'Miss': 0, 'CR': 0, 'ObsEvents': 0, 'PredAlerts': 0, 'y_obs_bin': [], 'y_pred': []} for m in ['M0', 'M1', 'M2', 'M1-log', 'M2-log']} for sev in ['Q80', 'Q90', 'Q95']} for h in HORIZONS}

    for cfg in PORTFOLIO_CONFIG:
        station = cfg['station']
        gauge_id = cfg['gauge_id']
        area = cfg['area_km2']
        state = cfg['state']
        is_holdout = cfg['is_holdout']
        is_zero_missing = station in ZERO_AS_MISSING_STATIONS
        print(f"[*] Processing {station:20s} (ID: {gauge_id:05d}, State: {state}) ...")
        
        tensor_file = TENSORS_DIR / f"weekly_tensor_gauge_{gauge_id:05d}_{station}.csv"
        df = pd.read_csv(tensor_file, parse_dates=['date']).set_index('date')
        
        for H in HORIZONS:
            target_col = f'target_H{H}'
            target_spec_col = f'target_specific_H{H}'
            
            # 1. Splits with terminal H-week purge embargo
            train_raw = df[TRAIN_START:TRAIN_END].iloc[:-H].copy()
            val_raw   = df[VAL_START:VAL_END].iloc[:-H].copy()
            test_raw  = df[TEST_START:TEST_END].copy()
            
            # Drop rows with NaN target or NaN flow_lag_0
            train = train_raw.dropna(subset=[target_col, target_spec_col, 'flow_lag_0']).copy()
            val   = val_raw.dropna(subset=[target_col, target_spec_col, 'flow_lag_0']).copy()
            test  = test_raw.dropna(subset=[target_col, target_spec_col, 'flow_lag_0']).copy()
            
            # 2. Filter zero-flow unmeasured rows for zero-as-missing stations
            if is_zero_missing:
                train = train[(train['flow_lag_0'] > 0.0) & (train[target_col] > 0.0)]
                val   = val[(val['flow_lag_0'] > 0.0) & (val[target_col] > 0.0)]
                test  = test[(test['flow_lag_0'] > 0.0) & (test[target_col] > 0.0)]
                
            n_eval = len(test)
            
            # 3. Drought thresholds strictly from Training fold
            y_tr_vol = train[target_col].values
            q80 = float(np.percentile(y_tr_vol, 20.0))
            q90 = float(np.percentile(y_tr_vol, 10.0))
            q95 = float(np.percentile(y_tr_vol, 5.0))
            
            # 4. M0 Baseline: Persistence at 1-2w, Weekly Climatology at 4-8w
            if H in [1, 2]:
                preds_m0 = test['flow_lag_0'].values.copy()
            else:
                train_clim = train.groupby(train.index.isocalendar().week)['flow_target_m3s'].mean()
                target_weeks = (test.index + pd.Timedelta(weeks=H)).isocalendar().week
                preds_m0 = target_weeks.map(train_clim).fillna(float(train['flow_target_m3s'].mean())).values
            preds_m0 = np.maximum(0.0, preds_m0)
            
            # 5. Features & Imputation
            med_m1 = train[M1_FEATURES].median()
            X_tr_m1 = train[M1_FEATURES].fillna(med_m1).values
            X_val_m1 = val[M1_FEATURES].fillna(med_m1).values
            X_test_m1 = test[M1_FEATURES].fillna(med_m1).values
            
            med_m2 = train[M2_FEATURES].median()
            X_tr_m2 = train[M2_FEATURES].fillna(med_m2).values
            X_val_m2 = val[M2_FEATURES].fillna(med_m2).values
            X_test_m2 = test[M2_FEATURES].fillna(med_m2).values
            
            y_tr_vol_vals = train[target_col].values
            y_val_vol_vals = val[target_col].values
            y_test_vol = test[target_col].values
            
            # --- M1 GBT ---
            model_m1 = fit_tune_gbt(X_tr_m1, y_tr_vol_vals, X_val_m1, y_val_vol_vals, target_mode='raw', area_km2=area)
            preds_m1 = np.maximum(0.0, model_m1.predict(X_test_m1))
            
            # --- M2 GBT ---
            model_m2 = fit_tune_gbt(X_tr_m2, y_tr_vol_vals, X_val_m2, y_val_vol_vals, target_mode='raw', area_km2=area)
            preds_m2 = np.maximum(0.0, model_m2.predict(X_test_m2))
            
            # --- Log Features & Targets ---
            tr_log_m1 = prepare_log_features(train, M1_FEATURES, area)
            val_log_m1 = prepare_log_features(val, M1_FEATURES, area)
            test_log_m1 = prepare_log_features(test, M1_FEATURES, area)
            med_m1_log = tr_log_m1.median()
            X_tr_m1_log = tr_log_m1.fillna(med_m1_log).values
            X_val_m1_log = val_log_m1.fillna(med_m1_log).values
            X_test_m1_log = test_log_m1.fillna(med_m1_log).values
            
            tr_log_m2 = prepare_log_features(train, M2_FEATURES, area)
            val_log_m2 = prepare_log_features(val, M2_FEATURES, area)
            test_log_m2 = prepare_log_features(test, M2_FEATURES, area)
            med_m2_log = tr_log_m2.median()
            X_tr_m2_log = tr_log_m2.fillna(med_m2_log).values
            X_val_m2_log = val_log_m2.fillna(med_m2_log).values
            X_test_m2_log = test_log_m2.fillna(med_m2_log).values
            
            y_tr_log = np.log1p(np.maximum(0.0, train[target_spec_col].values))
            
            # --- M1-log GBT ---
            model_m1_log = fit_tune_gbt(X_tr_m1_log, y_tr_log, X_val_m1_log, y_val_vol_vals, target_mode='log', area_km2=area)
            preds_m1_log = np.maximum(0.0, np.expm1(model_m1_log.predict(X_test_m1_log))) * (area / 604.8)
            
            # --- M2-log GBT ---
            model_m2_log = fit_tune_gbt(X_tr_m2_log, y_tr_log, X_val_m2_log, y_val_vol_vals, target_mode='log', area_km2=area)
            preds_m2_log = np.maximum(0.0, np.expm1(model_m2_log.predict(X_test_m2_log))) * (area / 604.8)
            
            predictions = {
                'M0': preds_m0,
                'M1': preds_m1,
                'M2': preds_m2,
                'M1-log': preds_m1_log,
                'M2-log': preds_m2_log,
            }
            
            stn_rec = {
                'scope': 'station',
                'station': station,
                'state': state,
                'gauge_id': gauge_id,
                'horizon': f"{H}w",
                'is_holdout': is_holdout,
                'zero_as_missing': is_zero_missing,
                'n_eval': n_eval,
                'q80_threshold': q80,
                'q90_threshold': q90,
                'q95_threshold': q95,
            }
            
            stn_mean_obs = float(np.mean(y_test_vol))
            
            for m_name, preds in predictions.items():
                c_mets = compute_continuous_metrics(y_test_vol, preds)
                stn_rec[f'{m_name}_NSE'] = c_mets['NSE']
                stn_rec[f'{m_name}_log_NSE'] = c_mets['log_nse']
                stn_rec[f'{m_name}_KGE'] = c_mets['KGE']
                stn_rec[f'{m_name}_RMSE'] = c_mets['RMSE']
                stn_rec[f'{m_name}_MAE'] = c_mets['MAE']
                
                # Drought metrics across Q80, Q90, Q95
                for sev, th in [('Q80', q80), ('Q90', q90), ('Q95', q95)]:
                    d_mets = compute_contingency(y_test_vol, preds, th)
                    stn_rec[f'{m_name}_{sev}_CSI'] = d_mets['CSI']
                    stn_rec[f'{m_name}_{sev}_POD'] = d_mets['POD']
                    stn_rec[f'{m_name}_{sev}_FAR'] = d_mets['FAR']
                    stn_rec[f'{m_name}_{sev}_PR_AUC'] = d_mets['PR_AUC']
                    
                    # Accumulate for pooled contingency
                    pooled_drought[H][sev][m_name]['Hits'] += d_mets['Hits']
                    pooled_drought[H][sev][m_name]['FA'] += d_mets['FA']
                    pooled_drought[H][sev][m_name]['Miss'] += d_mets['Miss']
                    pooled_drought[H][sev][m_name]['CR'] += d_mets['CR']
                    pooled_drought[H][sev][m_name]['ObsEvents'] += d_mets['ObsEvents']
                    pooled_drought[H][sev][m_name]['PredAlerts'] += d_mets['PredAlerts']
                    pooled_drought[H][sev][m_name]['y_obs_bin'].extend((y_test_vol <= th).astype(int).tolist())
                    pooled_drought[H][sev][m_name]['y_pred'].extend(preds.tolist())
                    
                # Accumulate for pooled continuous metrics
                pooled_data[H][m_name]['y_obs'].extend(y_test_vol.tolist())
                pooled_data[H][m_name]['y_pred'].extend(preds.tolist())
                pooled_data[H][m_name]['stn_mean'].extend([stn_mean_obs] * len(y_test_vol))
                
            stn_rec['delta_NSE_M2_M1'] = stn_rec['M2_NSE'] - stn_rec['M1_NSE']
            stn_rec['delta_NSE_log'] = stn_rec['M2-log_NSE'] - stn_rec['M1-log_NSE']
            all_station_results.append(stn_rec)
            
    df_station = pd.DataFrame(all_station_results)
    
    # --- Compute Portfolio Summary & Pooled Metrics ---
    summary_rows = []
    print("\n" + "=" * 80)
    print("CAMELS-Era Robustness Summary Results:")
    print("=" * 80)
    
    for H in HORIZONS:
        sub = df_station[df_station['horizon'] == f"{H}w"]
        
        # Paired Wilcoxon tests
        d_nse = sub['delta_NSE_M2_M1'].dropna().values
        w_res = wilcoxon(d_nse) if len(d_nse) > 0 else None
        p_val = float(w_res.pvalue) if w_res else np.nan
        
        d_nse_log = sub['delta_NSE_log'].dropna().values
        w_res_log = wilcoxon(d_nse_log) if len(d_nse_log) > 0 else None
        p_val_log = float(w_res_log.pvalue) if w_res_log else np.nan
        
        # Total weeks & Q95 event check
        q95_pooled_events = pooled_drought[H]['Q95']['M0']['ObsEvents']
        total_eval_weeks = len(pooled_data[H]['M0']['y_obs'])
        print(f"\n[Horizon {H}w] Total Test Weeks = {total_eval_weeks}, Pooled Q95 Events = {q95_pooled_events}")
        assert q95_pooled_events >= 100, f"Error: Pooled Q95 events ({q95_pooled_events}) < 100!"
        
        row = {
            'scope': 'portfolio_summary',
            'station': 'PORTFOLIO_12_STATIONS',
            'state': 'ALL',
            'gauge_id': 0,
            'horizon': f"{H}w",
            'is_holdout': False,
            'zero_as_missing': False,
            'n_eval': total_eval_weeks,
            'q80_threshold': np.nan,
            'q90_threshold': np.nan,
            'q95_threshold': np.nan,
            'delta_NSE_M2_M1': float(np.median(d_nse)),
            'delta_NSE_M2_M1_pval': p_val,
            'delta_NSE_log': float(np.median(d_nse_log)),
            'delta_NSE_log_pval': p_val_log,
        }
        
        for m_name in ['M0', 'M1', 'M2', 'M1-log', 'M2-log']:
            # Median metrics across 12 stations
            row[f'{m_name}_median_NSE'] = float(sub[f'{m_name}_NSE'].median())
            row[f'{m_name}_median_log_NSE'] = float(sub[f'{m_name}_log_NSE'].median())
            row[f'{m_name}_median_KGE'] = float(sub[f'{m_name}_KGE'].median())
            
            # Pooled continuous NSE: 1 - sum((y - y_hat)^2) / sum((y - y_bar_stn)^2)
            y_arr = np.array(pooled_data[H][m_name]['y_obs'])
            y_hat_arr = np.maximum(0.0, np.array(pooled_data[H][m_name]['y_pred']))
            stn_mean_arr = np.array(pooled_data[H][m_name]['stn_mean'])
            
            ss_res = np.sum((y_arr - y_hat_arr) ** 2)
            ss_tot = np.sum((y_arr - stn_mean_arr) ** 2)
            row[f'{m_name}_pooled_NSE'] = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan
            
            # Pooled drought metrics
            for sev in ['Q80', 'Q90', 'Q95']:
                d_dict = pooled_drought[H][sev][m_name]
                h = d_dict['Hits']
                f = d_dict['FA']
                m = d_dict['Miss']
                cr = d_dict['CR']
                obs_ev = d_dict['ObsEvents']
                denom = h + m + f
                csi = float(h / denom) if denom > 0 else np.nan
                pod = float(h / obs_ev) if obs_ev > 0 else np.nan
                far = float(f / (h + f)) if (h + f) > 0 else np.nan
                
                # Micro-pooled PR-AUC
                y_bin = np.array(d_dict['y_obs_bin'])
                scores = -np.array(d_dict['y_pred'])
                pr_auc = float(average_precision_score(y_bin, scores)) if obs_ev > 0 else np.nan
                
                row[f'{m_name}_{sev}_pooled_CSI'] = csi
                row[f'{m_name}_{sev}_pooled_POD'] = pod
                row[f'{m_name}_{sev}_pooled_FAR'] = far
                row[f'{m_name}_{sev}_pooled_PR_AUC'] = pr_auc
                row[f'{m_name}_{sev}_median_CSI'] = float(sub[f'{m_name}_{sev}_CSI'].median())
                row[f'{m_name}_{sev}_median_PR_AUC'] = float(sub[f'{m_name}_{sev}_PR_AUC'].median())
                
        summary_rows.append(row)
        print(f"H={H}w: Median NSE M0={row['M0_median_NSE']:.3f}, M1={row['M1_median_NSE']:.3f}, M2={row['M2_median_NSE']:.3f}, M1-log={row['M1-log_median_NSE']:.3f}, M2-log={row['M2-log_median_NSE']:.3f}")
        print(f"       Paired dNSE (M2 - M1): {row['delta_NSE_M2_M1']:+.3f} (Wilcoxon p = {p_val:.4f})")
        print(f"       Log-NSE M0={row['M0_median_log_NSE']:.3f}, M1={row['M1_median_log_NSE']:.3f}, M2={row['M2_median_log_NSE']:.3f}, M1-log={row['M1-log_median_log_NSE']:.3f}, M2-log={row['M2-log_median_log_NSE']:.3f}")
        print(f"       Q95 CSI: M0={row['M0_Q95_pooled_CSI']:.3f}, M1={row['M1_Q95_pooled_CSI']:.3f}, M2={row['M2_Q95_pooled_CSI']:.3f}, M1-log={row['M1-log_Q95_pooled_CSI']:.3f}, M2-log={row['M2-log_Q95_pooled_CSI']:.3f}")

    df_summary = pd.DataFrame(summary_rows)
    df_combined = pd.concat([df_station, df_summary], ignore_index=True)
    df_combined.to_csv(OUT_CSV, index=False)
    print("\n" + "=" * 80)
    print(f"[+] Saved CAMELS-era robustness results to {OUT_CSV}")
    print(f"    Total rows: {len(df_combined)} ({len(df_station)} station rows + {len(df_summary)} summary rows)")
    print("=" * 80)

if __name__ == '__main__':
    run_experiment()
