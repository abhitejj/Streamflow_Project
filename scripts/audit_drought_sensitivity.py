#!/usr/bin/env python3
"""
Audit & Sensitivity Analysis for Drought Early Warning Severity (Mild Q80, Moderate Q90, Severe Q95).

Scope:
- Evaluates Track B drought warning performance across three empirical threshold levels:
  * Mild Drought: 20th percentile (Q80)
  * Moderate Drought: 10th percentile (Q90)
  * Severe Drought: 5th percentile (Q95)
- All thresholds fit strictly on the purged Training fold (1980-01-01 to 2010-12-31 with H-week embargo).
- Models evaluated: M0 (persistence for H<=2w, fixed climatology for H>=4w), M1 (GBT), M2 (GBT).
- Scored strictly on the identical test set evaluated rows (where scored == True).
- Outputs comprehensive per-station and portfolio pooled metrics to:
  data/processed/model_results/drought_severity_sensitivity.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

# Ensure scripts dir is on sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts"))
from eval_config import ZERO_AS_MISSING_STATIONS  # noqa: E402

from audit_metrics import PORTFOLIO, TENSORS_DIR, TRAIN_START, TRAIN_END

RESULTS_DIR = BASE_DIR / "data" / "processed" / "model_results"
PREDICTIONS_CSV = RESULTS_DIR / "test_predictions.csv"
OUTPUT_CSV = RESULTS_DIR / "drought_severity_sensitivity.csv"

HORIZONS = ['1w', '2w', '4w', '8w']
SEVERITIES = [
    ('Q80', 20.0, 'Mild Drought (20th percentile)'),
    ('Q90', 10.0, 'Moderate Drought (10th percentile)'),
    ('Q95', 5.0,  'Severe Drought (5th percentile)')
]
MODELS = [
    ('M0', 'm0', 'Persistence (1-2w) / Fixed Climatology (4-8w)'),
    ('M1', 'm1_gbt', 'M1 GBT (Autoregressive Flow History)'),
    ('M2', 'm2_gbt', 'M2 GBT (Autoregressive Flow + Meteorology)'),
    ('M1-log', 'm1_gbt_log', 'M1-log GBT (Log Specific Runoff)'),
    ('M2-log', 'm2_gbt_log', 'M2-log GBT (Log Specific Runoff + Meteorology)')
]


def fit_training_thresholds() -> dict[tuple[str, str], dict[str, float]]:
    """
    Fits Q80, Q90, and Q95 empirical thresholds strictly on the purged Training fold (1980–2010).
    Applies H-week embargo at end of training fold to prevent forward leakage.
    """
    thresholds = {}
    for p in PORTFOLIO:
        stn = p['station']
        gid = p['gauge_id']
        t_path = TENSORS_DIR / f"weekly_tensor_gauge_{gid:05d}_{stn}.csv"
        df = pd.read_csv(t_path, parse_dates=['date']).set_index('date')
        
        for h_str in HORIZONS:
            H = int(h_str.replace('w', ''))
            target_col = f'target_H{H}'
            
            mask_train = (df.index >= TRAIN_START) & (df.index <= TRAIN_END)
            df_train = df[mask_train].iloc[:-H]
            y_train = df_train[target_col].dropna().values
            
            q80 = float(np.percentile(y_train, 20.0))
            q90 = float(np.percentile(y_train, 10.0))
            q95 = float(np.percentile(y_train, 5.0))
            
            thresholds[(stn, h_str)] = {
                'Q80': q80,
                'Q90': q90,
                'Q95': q95
            }
    return thresholds


def compute_contingency(y_true: np.ndarray, y_pred: np.ndarray, threshold: float | np.ndarray) -> dict:
    """
    Standard contingency metric engine adhering to drought warning definitions:
    Hits, FA, Miss, CR, ObsEvents, PredAlerts, BaseRate, POD, FAR, CSI, PR_AUC (trapezoid), PR_AUC_AP (avg precision).
    """
    y = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_pred, dtype=np.float64)
    
    valid = (~np.isnan(y)) & (~np.isnan(y_hat))
    y = y[valid]
    y_hat = np.maximum(y_hat[valid], 0.0)
    
    thresh = threshold[valid] if isinstance(threshold, np.ndarray) else threshold
    n = len(y)
    
    if n == 0:
        return {
            'n_eval': 0, 'Hits': 0, 'FA': 0, 'Miss': 0, 'CR': 0,
            'ObsEvents': 0, 'PredAlerts': 0, 'BaseRate': np.nan,
            'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan,
            'PR_AUC': np.nan, 'PR_AUC_AP': np.nan
        }
        
    obs_event = (y <= thresh).astype(int)
    pred_event = (y_hat <= thresh).astype(int)
    
    hits = int(np.sum((obs_event == 1) & (pred_event == 1)))
    fa   = int(np.sum((obs_event == 0) & (pred_event == 1)))
    miss = int(np.sum((obs_event == 1) & (pred_event == 0)))
    cr   = int(np.sum((obs_event == 0) & (pred_event == 0)))
    
    obs_events = hits + miss
    pred_alerts = hits + fa
    denom_csi = hits + miss + fa
    
    pod = float(hits / obs_events) if obs_events > 0 else np.nan
    far = float(fa / pred_alerts) if pred_alerts > 0 else np.nan
    csi = float(hits / denom_csi) if denom_csi > 0 else np.nan
    base_rate = float(obs_events / n)
    
    # Lower-tail event ranking: smaller discharge = stronger predicted drought
    if obs_events > 0 and obs_events < n:
        scores = -y_hat
        prec, rec, _ = precision_recall_curve(obs_event, scores)
        pr_auc = float(auc(rec, prec))
        ap = float(average_precision_score(obs_event, scores))
    elif obs_events == 0:
        pr_auc = np.nan
        ap = np.nan
    else:
        pr_auc = 1.0
        ap = 1.0
        
    return {
        'n_eval': n,
        'Hits': hits,
        'FA': fa,
        'Miss': miss,
        'CR': cr,
        'ObsEvents': obs_events,
        'PredAlerts': pred_alerts,
        'BaseRate': base_rate,
        'POD': pod,
        'FAR': far,
        'CSI': csi,
        'PR_AUC': pr_auc,
        'PR_AUC_AP': ap
    }


def main():
    print("=" * 80)
    print(" DROUGHT SEVERITY SENSITIVITY AUDIT (MILD Q80, MODERATE Q90, SEVERE Q95)")
    print("=" * 80)
    
    if not PREDICTIONS_CSV.is_file():
        raise FileNotFoundError(f"Missing predictions file: {PREDICTIONS_CSV}")
        
    df_preds = pd.read_csv(PREDICTIONS_CSV)
    print(f"Loaded {len(df_preds)} prediction rows from {PREDICTIONS_CSV}")
    
    # 1. Fit Training Fold Thresholds with embargo
    print("[*] Computing Training Fold (1980–2010) thresholds with embargo...")
    train_thresh = fit_training_thresholds()
    
    # 2. Evaluate Station-Level and Portfolio-Pooled Metrics
    records = []
    
    # --- Per-Station Evaluation ---
    for p in PORTFOLIO:
        stn = p['station']
        state = p['state']
        gid = p['gauge_id']
        is_holdout = p['is_holdout']
        area = p['area_km2']
        
        for h in HORIZONS:
            sub = df_preds[(df_preds.station == stn) & (df_preds.horizon == h) & df_preds.scored]
            y_obs = sub.y_obs.values
            
            for sev, pct, desc in SEVERITIES:
                thresh_val = train_thresh[(stn, h)][sev]
                for m_name, m_col, _ in MODELS:
                    y_pred = sub[m_col].values
                    metrics = compute_contingency(y_obs, y_pred, thresh_val)
                    records.append({
                        'scope': 'station',
                        'station': stn,
                        'state': state,
                        'gauge_id': gid,
                        'area_km2': area,
                        'is_holdout': is_holdout,
                        'horizon': h,
                        'severity': sev,
                        'severity_percentile': pct,
                        'threshold': thresh_val,
                        'model': m_name,
                        **metrics
                    })

    # --- Portfolio-Pooled Evaluation ---
    for h in HORIZONS:
        sub = df_preds[(df_preds.horizon == h) & df_preds.scored].copy()
        y_obs = sub.y_obs.values
        
        for sev, pct, desc in SEVERITIES:
            # Station-specific threshold for each test week
            sub['station_thresh'] = sub.apply(lambda r: train_thresh[(r['station'], h)][sev], axis=1)
            thresh_arr = sub['station_thresh'].values
            
            for m_name, m_col, _ in MODELS:
                y_pred = sub[m_col].values
                metrics = compute_contingency(y_obs, y_pred, thresh_arr)
                records.append({
                    'scope': 'portfolio_pooled',
                    'station': 'PORTFOLIO_POOLED',
                    'state': 'ALL',
                    'gauge_id': 0,
                    'area_km2': np.nan,
                    'is_holdout': False,
                    'horizon': h,
                    'severity': sev,
                    'severity_percentile': pct,
                    'threshold': np.nan,
                    'model': m_name,
                    **metrics
                })

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Saved sensitivity metrics dataset ({len(df_out)} rows) to: {OUTPUT_CSV}")
    
    # 3. Print Portfolio Summary Tables
    print("\n" + "=" * 100)
    print(" PORTFOLIO SUMMARY: MEDIAN CSI, MEDIAN FAR, AND POOLED PR-AUC ACROSS SEVERITY LEVELS")
    print("=" * 100)
    
    summary_rows = []
    for h in HORIZONS:
        for sev, pct, desc in SEVERITIES:
            pooled_m0 = df_out[(df_out.scope == 'portfolio_pooled') & (df_out.horizon == h) & (df_out.severity == sev) & (df_out.model == 'M0')].iloc[0]
            pooled_m1 = df_out[(df_out.scope == 'portfolio_pooled') & (df_out.horizon == h) & (df_out.severity == sev) & (df_out.model == 'M1')].iloc[0]
            pooled_m2 = df_out[(df_out.scope == 'portfolio_pooled') & (df_out.horizon == h) & (df_out.severity == sev) & (df_out.model == 'M2')].iloc[0]
            
            stn_m0 = df_out[(df_out.scope == 'station') & (df_out.horizon == h) & (df_out.severity == sev) & (df_out.model == 'M0')]
            stn_m1 = df_out[(df_out.scope == 'station') & (df_out.horizon == h) & (df_out.severity == sev) & (df_out.model == 'M1')]
            stn_m2 = df_out[(df_out.scope == 'station') & (df_out.horizon == h) & (df_out.severity == sev) & (df_out.model == 'M2')]
            
            def med_str(s):
                v = s.dropna()
                return f"{v.median():.3f} (n={len(v)})" if len(v) else "undefined"

            summary_rows.append({
                'Horizon': h,
                'Severity': sev,
                'Pooled_BaseRate': f"{pooled_m0['BaseRate']:.3f} ({pooled_m0['ObsEvents']}/{pooled_m0['n_eval']})",
                'CSI_M0': med_str(stn_m0['CSI']),
                'CSI_M1': med_str(stn_m1['CSI']),
                'CSI_M2': med_str(stn_m2['CSI']),
                'FAR_M0': med_str(stn_m0['FAR']),
                'FAR_M1': med_str(stn_m1['FAR']),
                'FAR_M2': med_str(stn_m2['FAR']),
                'Pooled_PR_AUC_AP (M0/M1/M2)': f"{pooled_m0['PR_AUC_AP']:.3f} / {pooled_m1['PR_AUC_AP']:.3f} / {pooled_m2['PR_AUC_AP']:.3f}",
                'Pooled_PR_AUC_Trapezoid (M0/M1/M2)': f"{pooled_m0['PR_AUC']:.3f} / {pooled_m1['PR_AUC']:.3f} / {pooled_m2['PR_AUC']:.3f}",
                'Pooled_CSI (M0/M1/M2)': f"{pooled_m0['CSI']:.3f} / {pooled_m1['CSI']:.3f} / {pooled_m2['CSI']:.3f}"
            })

    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False))


if __name__ == '__main__':
    main()
