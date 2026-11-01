#!/usr/bin/env python3
"""
Scientific Visualization Suite for Streamflow & Drought Forecasting (ES 335)

Generates publication-quality, 300-DPI visual figures for the project report
and the interactive GitHub Pages website, based on audited model outputs:
1. Plot A: Hydrographs (Time Series) — Observed vs M1 vs M2 at H=1w & H=4w (2018–2019 window).
2. Plot B: Scatter Plots (Actual vs Predicted) — Sqrt-scaled scatter with 1:1 line at H=4w.
3. Plot C: Confusion Matrices — Severe Drought (Q95) detection comparing M1 vs M2 at H=2w.
4. Plot D: Precision-Recall Curves — PR curves with PR-AUC for Q90 & Q95 droughts at H=2w & H=4w.
5. Plot E: Skill Decay Curves — Median NSE and KGE degradation from H=1w to H=8w (M0 vs M1 vs M2).
6. Plot F: Feature Importance Shift — Permutation importance comparing H=1w (flow memory) vs H=8w (weather memory).
7. Plot G: Zero-Shot Spatial Transfer — Grouped bar charts comparing H=4w & H=8w NSE on holdouts (M0 vs M1 GBT vs M2 PyTorch LSTM).

Compliance:
- High-resolution (300 DPI) academic styling (Nature/AMS hydrology aesthetics).
- Strict non-negative physical flow bounds.
- Automatic output directory creation at 'data/processed/visualizations/'.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance

# ==============================================================================
# 1. DIRECTORY CONFIGURATION & STYLING
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

TENSORS_DIR = os.path.join(PROJECT_ROOT, 'data/processed/weekly_tensors')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'data/processed/model_results')
MODELS_DIR  = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'data/processed/visualizations')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication aesthetic settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#374151',
    'axes.linewidth': 1.0,
    'grid.color': '#E5E7EB',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'figure.autolayout': False,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color Palette definitions
COLOR_OBSERVED = '#1F2937'    # Dark Slate Gray
COLOR_M0       = '#9CA3AF'    # Neutral Gray
COLOR_M1       = '#2563EB'    # Royal Blue
COLOR_M2       = '#DC2626'    # Crimson Red
COLOR_LSTM     = '#7C3AED'    # Deep Violet
COLOR_DROUGHT  = '#D97706'    # Amber / Warning

# Feature categories for Plot F
M2_TABULAR_FEATURES = [
    'flow_lag_0', 'flow_lag_1', 'flow_lag_2', 'flow_lag_3', 'flow_lag_4', 'flow_lag_7',
    'flow_std_7d', 'flow_min_7d', 'flow_max_7d', 'flow_recession_slope',
    'rain_sum_1w', 'rain_sum_2w', 'rain_sum_1m', 'rain_sum_2m', 'rain_sum_3m',
    'api_30d', 'dry_days_count',
    'temp_mean_1w', 'temp_max_1w', 'temp_min_1w',
    'pet_sum_1w', 'pet_sum_1m',
    'water_deficit_1w', 'water_deficit_1m',
    'sin_week', 'cos_week',
]

M1_FLOW_FEATURES = [
    'flow_lag_0', 'flow_lag_1', 'flow_lag_2', 'flow_lag_3', 'flow_lag_4', 'flow_lag_7',
    'flow_std_7d', 'flow_min_7d', 'flow_max_7d', 'flow_recession_slope',
    'sin_week', 'cos_week',
]

def categorize_feature(feat_name):
    """Categorizes predictor names for thematic plotting."""
    if 'flow_' in feat_name:
        return 'Streamflow Memory'
    elif 'rain_' in feat_name or 'api_' in feat_name or 'dry_days' in feat_name:
        return 'Precipitation & Recharge'
    elif 'temp_' in feat_name or 'pet_' in feat_name:
        return 'Atmospheric Demand'
    elif 'water_deficit' in feat_name:
        return 'Climatic Water Deficit'
    elif 'sin_' in feat_name or 'cos_' in feat_name or 'week' in feat_name:
        return 'Calendar Seasonality'
    return 'Other'

# ==============================================================================
# 2. DATA LOADING HELPERS
# ==============================================================================
def load_test_predictions(station_name, gauge_id, horizon, model_m1_path, model_m2_path):
    """
    Loads test fold (2016–2025) for a given station and computes predictions
    from both trained M1 and M2 estimators.
    """
    tensor_path = os.path.join(TENSORS_DIR, f"weekly_tensor_gauge_{gauge_id:05d}_{station_name}.csv")
    if not os.path.isfile(tensor_path):
        raise FileNotFoundError(f"Missing tensor: {tensor_path}")
        
    df = pd.read_csv(tensor_path, parse_dates=['date']).set_index('date')
    df_test = df.loc['2016-01-01':'2025-12-31'].copy()
    
    target_col = f'target_H{horizon}'
    valid_mask = (~df_test[target_col].isna()) & (~df_test['flow_lag_0'].isna())
    df_eval = df_test[valid_mask].copy()
    
    # Load M1 model
    m1_obj = joblib.load(model_m1_path)
    m1_model = m1_obj['model'] if isinstance(m1_obj, dict) and 'model' in m1_obj else m1_obj
    med1 = m1_obj.get('train_medians') if isinstance(m1_obj, dict) else None
    X_m1 = df_eval[M1_FLOW_FEATURES].fillna(med1 if med1 is not None else df_eval[M1_FLOW_FEATURES].median())
    preds_m1 = np.maximum(0.0, m1_model.predict(X_m1.values if hasattr(m1_model, 'n_features_in_') else X_m1))
    
    # Load M2 model
    m2_obj = joblib.load(model_m2_path)
    m2_model = m2_obj['model'] if isinstance(m2_obj, dict) and 'model' in m2_obj else m2_obj
    med2 = m2_obj.get('train_medians') if isinstance(m2_obj, dict) else None
    X_m2 = df_eval[M2_TABULAR_FEATURES].fillna(med2 if med2 is not None else df_eval[M2_TABULAR_FEATURES].median())
    preds_m2 = np.maximum(0.0, m2_model.predict(X_m2.values if hasattr(m2_model, 'n_features_in_') else X_m2))
    
    y_actual = df_eval[target_col].values
    
    # Training fold Q95 threshold for reference
    df_train = df.loc['1980-01-01':'2010-12-31'].dropna(subset=[target_col])
    q95_thresh = float(np.percentile(df_train[target_col], 5.0))
    q90_thresh = float(np.percentile(df_train[target_col], 10.0))
    q80_thresh = float(np.percentile(df_train[target_col], 20.0))
    
    return {
        'dates': df_eval.index,
        'y_actual': y_actual,
        'preds_m1': preds_m1,
        'preds_m2': preds_m2,
        'q95': q95_thresh,
        'q90': q90_thresh,
        'q80': q80_thresh,
    }

# ==============================================================================
# 3. PLOT A: HYDROGRAPHS (TIME SERIES COMPARISON)
# ==============================================================================
def generate_plot_a_hydrographs(out_dir):
    """
    Generates 4-panel hydrographs comparing Observed Streamflow vs M1 vs M2
    for Kantamal (Odisha) and Karad (Maharashtra) at H=1w and H=4w.
    Zoomed in on the 2018–2019 interval containing the 2018 drought and 2019 super-monsoon.
    """
    print("[*] Generating Plot A: Hydrographs (Kantamal & Karad, 2018–2019)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex='col')
    
    stations = [
        ('Kantamal', 8013, 'Odisha (Tel/Mahanadi Basin)'),
        ('Karad', 4021, 'Maharashtra (Upper Krishna Basin)'),
    ]
    horizons = [1, 4]
    
    zoom_start = '2018-01-01'
    zoom_end   = '2019-12-31'
    
    for col_idx, (stn_name, gid, basin_desc) in enumerate(stations):
        for row_idx, H in enumerate(horizons):
            ax = axes[row_idx, col_idx]
            
            clean_name = stn_name.replace(' ', '_')
            m1_path = os.path.join(MODELS_DIR, 'protocol1', f"m1_gbt_gauge_{gid:05d}_{clean_name}_H{H}.joblib")
            m2_path = os.path.join(MODELS_DIR, 'protocol1', f"m2_gbt_gauge_{gid:05d}_{clean_name}_H{H}.joblib")
            
            data = load_test_predictions(stn_name, gid, H, m1_path, m2_path)
            s_df = pd.DataFrame({
                'Actual': data['y_actual'],
                'M1': data['preds_m1'],
                'M2': data['preds_m2']
            }, index=data['dates']).loc[zoom_start:zoom_end]
            
            # Plot lines
            ax.plot(s_df.index, s_df['Actual'], color=COLOR_OBSERVED, label='Observed Streamflow ($Q$)', linewidth=2.0, zorder=4)
            ax.plot(s_df.index, s_df['M1'], color=COLOR_M1, linestyle='--', label='M1: Flow-Only GBT', linewidth=1.5, alpha=0.9, zorder=3)
            ax.plot(s_df.index, s_df['M2'], color=COLOR_M2, linestyle='-', label='M2: Hydro-Meteorological GBT', linewidth=1.7, alpha=0.9, zorder=5)
            
            # Drought threshold line
            ax.axhline(data['q95'], color=COLOR_DROUGHT, linestyle=':', linewidth=1.3, label='Severe Drought ($Q_{95}$)', zorder=2)
            
            # Formatting
            ax.set_title(f"{stn_name} ({basin_desc}) — {H}-Week Ahead Forecast", fontsize=12, fontweight='bold', pad=8)
            ax.set_ylabel(r"Discharge ($m^3/s$)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Calculate Zoom Window NSE
            ss_tot = np.sum((s_df['Actual'] - s_df['Actual'].mean()) ** 2)
            nse_m1 = 1 - (np.sum((s_df['Actual'] - s_df['M1']) ** 2) / ss_tot)
            nse_m2 = 1 - (np.sum((s_df['Actual'] - s_df['M2']) ** 2) / ss_tot)
            
            ax.text(0.02, 0.92, f"Zoom NSE: M1={nse_m1:.2f} | M2={nse_m2:.2f}",
                    transform=ax.transAxes, fontsize=9.5, fontweight='semibold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#D1D5DB', alpha=0.9))
            
            if row_idx == 1:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)
                
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8.5)
                
    fig.suptitle("Observed vs Predicted Streamflow Hydrographs (2018 Drought & 2019 Monsoonal Peak Window)",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig_a_hydrographs_temporal.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")

# ==============================================================================
# 4. PLOT B: SCATTER PLOTS (ACTUAL VS PREDICTED AT H=4w)
# ==============================================================================
def generate_plot_b_scatter(out_dir):
    """
    Generates side-by-side sqrt-scaled scatter plots with 1:1 identity line
    for M1 vs M2 at H=4w across all test samples of representative basins.
    """
    print("[*] Generating Plot B: Actual vs Predicted Scatter Plots (H=4w)...")
    
    # Collect H=4w predictions across key basins
    test_stations = [
        ('Kantamal', 8013),
        ('Karad', 4021),
        ('Gomlai', 7007),
        ('Anandpur', 7002),
        ('Huvinhedigi', 4017),
    ]
    
    all_actual, all_m1, all_m2 = [], [], []
    for stn, gid in test_stations:
        clean_name = stn.replace(' ', '_')
        m1_path = os.path.join(MODELS_DIR, 'protocol1', f"m1_gbt_gauge_{gid:05d}_{clean_name}_H4.joblib")
        m2_path = os.path.join(MODELS_DIR, 'protocol1', f"m2_gbt_gauge_{gid:05d}_{clean_name}_H4.joblib")
        data = load_test_predictions(stn, gid, 4, m1_path, m2_path)
        all_actual.extend(data['y_actual'])
        all_m1.extend(data['preds_m1'])
        all_m2.extend(data['preds_m2'])
        
    y_true = np.array(all_actual)
    y_m1   = np.array(all_m1)
    y_m2   = np.array(all_m2)
    
    # Sqrt transformation for visual dispersion balance
    sqrt_true = np.sqrt(np.maximum(0, y_true))
    sqrt_m1   = np.sqrt(np.maximum(0, y_m1))
    sqrt_m2   = np.sqrt(np.maximum(0, y_m2))
    
    max_val = max(np.max(sqrt_true), np.max(sqrt_m1), np.max(sqrt_m2)) * 1.05
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    
    # Metric calculations
    nse_m1 = 1 - (np.sum((y_true - y_m1) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    nse_m2 = 1 - (np.sum((y_true - y_m2) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    rmse_m1 = np.sqrt(np.mean((y_true - y_m1) ** 2))
    rmse_m2 = np.sqrt(np.mean((y_true - y_m2) ** 2))
    r_m1 = np.corrcoef(y_true, y_m1)[0, 1]
    r_m2 = np.corrcoef(y_true, y_m2)[0, 1]
    
    # Subplot 1: M1
    ax1.scatter(sqrt_true, sqrt_m1, color=COLOR_M1, alpha=0.35, edgecolors='none', s=24)
    ax1.plot([0, max_val], [0, max_val], color='#4B5563', linestyle='--', linewidth=1.5, label='1:1 Ideal Line')
    ax1.set_title("M1: Flow-Only GBT (4-Week Ahead)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel(r"Observed $\sqrt{Q}$ ($\sqrt{m^3/s}$)", fontsize=11)
    ax1.set_ylabel(r"Predicted $\sqrt{Q}$ ($\sqrt{m^3/s}$)", fontsize=11)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.text(0.05, 0.86, f"NSE: {nse_m1:.3f}\nRMSE: {rmse_m1:.1f} m³/s\nr: {r_m1:.3f}",
             transform=ax1.transAxes, fontsize=10, fontweight='semibold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#D1D5DB', alpha=0.9))
    ax1.legend(loc='lower right', frameon=True)
    
    # Subplot 2: M2
    ax2.scatter(sqrt_true, sqrt_m2, color=COLOR_M2, alpha=0.35, edgecolors='none', s=24)
    ax2.plot([0, max_val], [0, max_val], color='#4B5563', linestyle='--', linewidth=1.5, label='1:1 Ideal Line')
    ax2.set_title("M2: Hydro-Meteorological GBT (4-Week Ahead)", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel(r"Observed $\sqrt{Q}$ ($\sqrt{m^3/s}$)", fontsize=11)
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.text(0.05, 0.86, f"NSE: {nse_m2:.3f}\nRMSE: {rmse_m2:.1f} m³/s\nr: {r_m2:.3f}",
             transform=ax2.transAxes, fontsize=10, fontweight='semibold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#D1D5DB', alpha=0.9))
    ax2.legend(loc='lower right', frameon=True)
    
    fig.suptitle("Actual vs Predicted Scatter Plots across Test Folds (H=4w, Sqrt Scale)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig_b_scatter_actual_vs_predicted_h4.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")

# ==============================================================================
# 5. PLOT C: CONFUSION MATRICES (SEVERE DROUGHT Q95 AT H=2w)
# ==============================================================================
def generate_plot_c_confusion_matrices(out_dir):
    """
    Plots 2x2 confusion matrices comparing M1 vs M2 for Severe Drought (Q95)
    detection at H=2w ahead across all test basins.
    """
    print("[*] Generating Plot C: Confusion Matrices (Severe Drought Q95, H=2w)...")
    
    # Aggregate binary event matrices across all test basins
    obs_all, m1_pred_all, m2_pred_all = [], [], []
    
    test_stations = [
        ('Karad', 4021), ('Warunji', 4063), ('Ashti', 3005), ('Bamni (Ballarsha)', 3008),
        ('Anandpur', 7002), ('Kashinagar', 16007), ('Gomlai', 7007), ('Kantamal', 8013),
        ('Yadgir', 4064), ('Huvinhedigi', 4017), ('KUDIGE', 5010), ('Haralahalli', 4013),
    ]
    
    for stn, gid in test_stations:
        clean_name = stn.replace(' ', '_').replace('(', '').replace(')', '')
        m1_path = os.path.join(MODELS_DIR, 'protocol1', f"m1_gbt_gauge_{gid:05d}_{clean_name}_H2.joblib")
        m2_path = os.path.join(MODELS_DIR, 'protocol1', f"m2_gbt_gauge_{gid:05d}_{clean_name}_H2.joblib")
        
        if os.path.isfile(m1_path) and os.path.isfile(m2_path):
            data = load_test_predictions(stn, gid, 2, m1_path, m2_path)
            q95 = data['q95']
            
            obs_binary = (data['y_actual'] <= q95).astype(int)
            m1_binary  = (data['preds_m1'] <= q95).astype(int)
            m2_binary  = (data['preds_m2'] <= q95).astype(int)
            
            obs_all.extend(obs_binary)
            m1_pred_all.extend(m1_binary)
            m2_pred_all.extend(m2_binary)
            
    cm_m1 = confusion_matrix(obs_all, m1_pred_all)
    cm_m2 = confusion_matrix(obs_all, m2_pred_all)
    
    # Calculate derived alert scores
    def get_scores(cm):
        tn, fp, fn, tp = cm.ravel()
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
        return pod, far, csi
        
    pod_m1, far_m1, csi_m1 = get_scores(cm_m1)
    pod_m2, far_m2, csi_m2 = get_scores(cm_m2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = ['Normal / Wet', 'Severe Drought']
    
    # Heatmap M1
    sns.heatmap(cm_m1, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=labels, yticklabels=labels, annot_kws={'size': 14, 'weight': 'bold'})
    ax1.set_title("M1: Flow-Only GBT (H=2w)\nSevere Drought ($Q_{95}$)", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Predicted Event", fontsize=11)
    ax1.set_ylabel("Observed Event", fontsize=11)
    ax1.text(0.5, -0.22, f"POD (Recall): {pod_m1:.3f} | FAR: {far_m1:.3f} | CSI: {csi_m1:.3f}",
             ha='center', transform=ax1.transAxes, fontsize=10.5, fontweight='bold', color='#1E40AF')
             
    # Heatmap M2
    sns.heatmap(cm_m2, annot=True, fmt='d', cmap='Reds', cbar=False, ax=ax2,
                xticklabels=labels, yticklabels=labels, annot_kws={'size': 14, 'weight': 'bold'})
    ax2.set_title("M2: Hydro-Meteorological GBT (H=2w)\nSevere Drought ($Q_{95}$)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Predicted Event", fontsize=11)
    ax2.set_ylabel("Observed Event", fontsize=11)
    ax2.text(0.5, -0.22, f"POD (Recall): {pod_m2:.3f} | FAR: {far_m2:.3f} | CSI: {csi_m2:.3f}",
             ha='center', transform=ax2.transAxes, fontsize=10.5, fontweight='bold', color='#991B1B')
             
    fig.suptitle("Contingency Matrices for Severe Hydrological Drought Early Warning (H=2 Weeks Ahead)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig_c_confusion_matrices_severe_drought_h2.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")

# ==============================================================================
# 6. PLOT D: PRECISION-RECALL CURVES (DROUGHT EARLY WARNING AT H=2w & H=4w)
# ==============================================================================
def generate_plot_d_pr_curves(out_dir):
    """
    Generates Precision-Recall curves comparing M1 vs M2 for Moderate (Q90)
    and Severe (Q95) droughts across 2w and 4w forecast horizons.
    """
    print("[*] Generating Plot D: Precision-Recall Curves (Q90 & Q95, H=2w & H=4w)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    test_stations = [
        ('Karad', 4021), ('Warunji', 4063), ('Ashti', 3005), ('Bamni (Ballarsha)', 3008),
        ('Anandpur', 7002), ('Kashinagar', 16007), ('Gomlai', 7007), ('Kantamal', 8013),
        ('Yadgir', 4064), ('Huvinhedigi', 4017), ('KUDIGE', 5010), ('Haralahalli', 4013),
    ]
    
    tiers = [
        ('Moderate Drought ($Q_{90}$)', 'q90'),
        ('Severe Drought ($Q_{95}$)', 'q95'),
    ]
    horizons = [2, 4]
    
    for row_idx, (tier_label, q_key) in enumerate(tiers):
        for col_idx, H in enumerate(horizons):
            ax = axes[row_idx, col_idx]
            
            y_true_binary = []
            m1_drought_scores = []
            m2_drought_scores = []
            
            for stn, gid in test_stations:
                clean_name = stn.replace(' ', '_').replace('(', '').replace(')', '')
                m1_p = os.path.join(MODELS_DIR, 'protocol1', f"m1_gbt_gauge_{gid:05d}_{clean_name}_H{H}.joblib")
                m2_p = os.path.join(MODELS_DIR, 'protocol1', f"m2_gbt_gauge_{gid:05d}_{clean_name}_H{H}.joblib")
                
                if os.path.isfile(m1_p) and os.path.isfile(m2_p):
                    data = load_test_predictions(stn, gid, H, m1_p, m2_p)
                    thresh = data[q_key]
                    
                    # Target is 1 if in drought
                    y_bin = (data['y_actual'] <= thresh).astype(int)
                    
                    # Inversion score: smaller predicted flow => higher probability of drought
                    # Score = -pred
                    s_m1 = -data['preds_m1']
                    s_m2 = -data['preds_m2']
                    
                    y_true_binary.extend(y_bin)
                    m1_drought_scores.extend(s_m1)
                    m2_drought_scores.extend(s_m2)
                    
            y_true_arr = np.array(y_true_binary)
            m1_scores  = np.array(m1_drought_scores)
            m2_scores  = np.array(m2_drought_scores)
            
            # Precision Recall
            prec_m1, rec_m1, _ = precision_recall_curve(y_true_arr, m1_scores)
            prec_m2, rec_m2, _ = precision_recall_curve(y_true_arr, m2_scores)
            
            auc_m1 = average_precision_score(y_true_arr, m1_scores)
            auc_m2 = average_precision_score(y_true_arr, m2_scores)
            
            prevalence = np.mean(y_true_arr)
            
            # Plot
            ax.plot(rec_m1, prec_m1, color=COLOR_M1, linewidth=2.0, label=f'M1: Flow-Only (PR-AUC = {auc_m1:.3f})')
            ax.plot(rec_m2, prec_m2, color=COLOR_M2, linewidth=2.2, label=f'M2: Hydro-Met (PR-AUC = {auc_m2:.3f})')
            ax.axhline(prevalence, color='#6B7280', linestyle='--', label=f'No-Skill Baseline ({prevalence:.3f})')
            
            ax.set_title(f"{tier_label} — {H} Weeks Ahead", fontsize=12, fontweight='bold', pad=8)
            ax.set_xlabel("Recall (Probability of Detection)", fontsize=10.5)
            ax.set_ylabel("Precision (1 - False Alarm Ratio)", fontsize=10.5)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', frameon=True, fontsize=9.5)
            
    fig.suptitle("Precision-Recall Curves for Hydrological Drought Warnings Across Lead Times",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig_d_precision_recall_curves.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")

# ==============================================================================
# 7. PLOT E: SKILL DECAY CURVES (HORIZON DECAY H=1w TO H=8w)
# ==============================================================================
def generate_plot_e_skill_decay(out_dir):
    """
    Plots median continuous skill degradation (NSE and KGE) from H=1w to H=8w
    comparing M0 Baseline, M1 Flow-Only GBT, and M2 Hydro-Meteorological GBT.
    """
    print("[*] Generating Plot E: Multi-Horizon Skill Decay Curves (H=1w to H=8w)...")
    
    p1_m1_path = os.path.join(RESULTS_DIR, 'protocol1_temporal_results_gbt.csv')
    p1_m2_path = os.path.join(RESULTS_DIR, 'protocol1_temporal_results_m2_gbt.csv')
    
    if not os.path.isfile(p1_m1_path) or not os.path.isfile(p1_m2_path):
        raise FileNotFoundError("Protocol 1 results CSVs missing.")
        
    df_m1 = pd.read_csv(p1_m1_path)
    df_m2 = pd.read_csv(p1_m2_path)
    
    horizons = [1, 2, 4, 8]
    
    m0_nse_med, m1_nse_med, m2_nse_med = [], [], []
    m0_kge_med, m1_kge_med, m2_kge_med = [], [], []
    
    m2_nse_25, m2_nse_75 = [], []
    m2_kge_25, m2_kge_75 = [], []
    
    for h in horizons:
        sub_m1 = df_m1[df_m1['Horizon_Weeks'] == h]
        sub_m2 = df_m2[df_m2['Horizon_Weeks'] == h]
        
        m0_nse_med.append(sub_m1['M0_NSE'].median())
        m1_nse_med.append(sub_m1['M1_NSE'].median())
        m2_nse_med.append(sub_m2['M2_NSE'].median())
        
        m0_kge_med.append(sub_m1['M0_KGE'].median())
        m1_kge_med.append(sub_m1['M1_KGE'].median())
        m2_kge_med.append(sub_m2['M2_KGE'].median())
        
        m2_nse_25.append(np.percentile(sub_m2['M2_NSE'], 25))
        m2_nse_75.append(np.percentile(sub_m2['M2_NSE'], 75))
        m2_kge_25.append(np.percentile(sub_m2['M2_KGE'], 25))
        m2_kge_75.append(np.percentile(sub_m2['M2_KGE'], 75))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    x = np.arange(len(horizons))
    
    # 1. NSE Decay
    ax1.plot(x, m0_nse_med, marker='s', color=COLOR_M0, linewidth=2.0, linestyle='--', label='M0: Persistence / Climatology')
    ax1.plot(x, m1_nse_med, marker='^', color=COLOR_M1, linewidth=2.2, label='M1: Autoregressive Flow-Only')
    ax1.plot(x, m2_nse_med, marker='o', color=COLOR_M2, linewidth=2.4, label='M2: Hydro-Meteorological')
    ax1.fill_between(x, m2_nse_25, m2_nse_75, color=COLOR_M2, alpha=0.15, label='M2 Interquartile Range (IQR)')
    
    ax1.axhline(0.0, color='#4B5563', linestyle=':', label='Zero Predictive Skill Line')
    ax1.set_title("Nash-Sutcliffe Efficiency (NSE) Decay", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Forecast Horizon Ahead", fontsize=11)
    ax1.set_ylabel("Median NSE Across 12 Basins", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['1 Week', '2 Weeks', '4 Weeks', '8 Weeks'], fontsize=10.5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', frameon=True, fontsize=9.5)
    
    # Annotate persistence collapse
    ax1.annotate("Persistence Collapse\n(NSE: -0.339)", xy=(1, m0_nse_med[1]), xytext=(0.8, -0.6),
                 arrowprops=dict(facecolor='#4B5563', shrink=0.08, width=1.0, headwidth=5),
                 fontsize=9, fontweight='semibold')
                 
    # 2. KGE Decay
    ax2.plot(x, m0_kge_med, marker='s', color=COLOR_M0, linewidth=2.0, linestyle='--', label='M0: Persistence / Climatology')
    ax2.plot(x, m1_kge_med, marker='^', color=COLOR_M1, linewidth=2.2, label='M1: Autoregressive Flow-Only')
    ax2.plot(x, m2_kge_med, marker='o', color=COLOR_M2, linewidth=2.4, label='M2: Hydro-Meteorological')
    ax2.fill_between(x, m2_kge_25, m2_kge_75, color=COLOR_M2, alpha=0.15, label='M2 Interquartile Range (IQR)')
    
    ax2.axhline(0.0, color='#4B5563', linestyle=':', label='Zero Predictive Skill Line')
    ax2.set_title("Kling-Gupta Efficiency (KGE) Decay", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Forecast Horizon Ahead", fontsize=11)
    ax2.set_ylabel("Median KGE Across 12 Basins", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['1 Week', '2 Weeks', '4 Weeks', '8 Weeks'], fontsize=10.5)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', frameon=True, fontsize=9.5)
    
    # Annotate M2 KGE advantage
    ax2.annotate("M2 Volume & Deficit Gain\n(+0.116 KGE at H=4w)", xy=(2, m2_kge_med[2]), xytext=(1.8, 0.55),
                 arrowprops=dict(facecolor=COLOR_M2, shrink=0.08, width=1.0, headwidth=5),
                 fontsize=9, fontweight='semibold', color=COLOR_M2)
                 
    fig.suptitle("Predictive Skill Decay Curves Across Multi-Week Operational Horizons",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig_e_skill_decay_curves.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")

# ==============================================================================
# 6. PLOT F: FEATURE IMPORTANCE SHIFT (H=1w VS H=8w)
# ==============================================================================
def generate_plot_f_importance_shift(out_dir):
    """
    Computes permutation importance for M2 HistGradientBoosting at H=1w vs H=8w,
    plotting the top 10 features side-by-side to visualize the transition from
    antecedent hydrograph memory to meteorological driving forcings.
    """
    print("[*] Generating Plot F: Feature Importance Transition (H=1w vs H=8w)...")
    
    # Representative benchmark station: Karad (Deccan headwaters)
    stn_name = 'Karad'
    gid = 4021
    
    tensor_path = os.path.join(TENSORS_DIR, f"weekly_tensor_gauge_{gid:05d}_{stn_name}.csv")
    df = pd.read_csv(tensor_path, parse_dates=['date']).set_index('date')
    df_test = df.loc['2016-01-01':'2025-12-31'].copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    
    palette_map = {
        'Streamflow Memory': '#2563EB',
        'Precipitation & Recharge': '#059669',
        'Atmospheric Demand': '#D97706',
        'Climatic Water Deficit': '#7C3AED',
        'Calendar Seasonality': '#6B7280',
    }
    
    for ax, H in [(ax1, 1), (ax2, 8)]:
        model_path = os.path.join(MODELS_DIR, 'protocol1', f"m2_gbt_gauge_{gid:05d}_{stn_name}_H{H}.joblib")
        model = joblib.load(model_path)
        
        target_col = f'target_H{H}'
        eval_df = df_test.dropna(subset=[target_col] + M2_TABULAR_FEATURES)
        X = eval_df[M2_TABULAR_FEATURES]
        y = eval_df[target_col]
        
        r = permutation_importance(model, X, y, n_repeats=5, random_state=42, scoring='neg_mean_squared_error')
        
        importances = r.importances_mean
        importances = np.maximum(0, importances)
        sum_imp = np.sum(importances)
        norm_imp = (importances / sum_imp) * 100 if sum_imp > 0 else importances
        
        df_imp = pd.DataFrame({
            'Feature': M2_TABULAR_FEATURES,
            'Importance': norm_imp,
            'Category': [categorize_feature(f) for f in M2_TABULAR_FEATURES]
        }).sort_values('Importance', ascending=True).tail(10)
        
        colors = [palette_map.get(cat, '#374151') for cat in df_imp['Category']]
        
        bars = ax.barh(df_imp['Feature'], df_imp['Importance'], color=colors, edgecolor='#374151', height=0.65)
        ax.set_title(f"Top 10 Predictors at Horizon H={H} Week" + ("s" if H > 1 else ""),
                     fontsize=12.5, fontweight='bold', pad=10)
        ax.set_xlabel("Relative Permutation Importance (%)", fontsize=10.5)
        ax.grid(True, linestyle='--', alpha=0.5, axis='x')
        
        # Add value labels
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
                    va='center', ha='left', fontsize=9, fontweight='semibold')
                    
        ax.set_xlim(0, max(df_imp['Importance']) * 1.25)
        
    # Legend for categories
    handles = [plt.Rectangle((0,0),1,1, color=palette_map[cat]) for cat in palette_map]
    fig.legend(handles, palette_map.keys(), loc='lower center', ncol=5, frameon=True,
               bbox_to_anchor=(0.5, -0.05), fontsize=10.5)
               
    fig.suptitle("Predictive Driver Transition: Hydrograph Memory (H=1w) vs Meteorological Memory (H=8w)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig_f_feature_importance_shift.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")

# ==============================================================================
# 8. PLOT G: ZERO-SHOT SPATIAL GENERALIZATION (PROTOCOL 2 BARS)
# ==============================================================================
def generate_plot_g_spatial_transfer(out_dir):
    """
    Generates grouped bar charts comparing H=4w and H=8w Zero-Shot Nash-Sutcliffe Efficiency
    on held-out river basins across M0 Baseline, M1 Flow-Only GBT, and M2 PyTorch HydroLSTM.
    """
    print("[*] Generating Plot G: Zero-Shot Spatial Generalization Grouped Bars...")
    
    p2_m1_path = os.path.join(RESULTS_DIR, 'protocol2_spatial_results_gbt.csv')
    p2_m2_path = os.path.join(RESULTS_DIR, 'protocol2_spatial_results_m2_lstm.csv')
    
    if not os.path.isfile(p2_m1_path) or not os.path.isfile(p2_m2_path):
        raise FileNotFoundError("Protocol 2 result CSVs missing.")
        
    df_p2_m1 = pd.read_csv(p2_m1_path)
    df_p2_m2 = pd.read_csv(p2_m2_path)
    
    holdouts = [
        ('Kantamal', 'Odisha (Tel/Mahanadi)'),
        ('Bamni (Ballarsha)', 'Maharashtra (Wardha)'),
        ('Haralahalli', 'Karnataka (Tungabhadra)'),
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
    
    for ax, H in [(ax1, 4), (ax2, 8)]:
        sub_m1 = df_p2_m1[df_p2_m1['Horizon_Weeks'] == H].set_index('Holdout_Station')
        sub_m2 = df_p2_m2[df_p2_m2['Horizon_Weeks'] == H].set_index('Holdout_Station')
        
        stn_labels = [h[0].split()[0] for h in holdouts]
        x = np.arange(len(stn_labels))
        width = 0.25
        
        m0_vals   = [sub_m1.loc[h[0], 'M0_NSE'] for h in holdouts]
        m1_vals   = [sub_m1.loc[h[0], 'M1_NSE'] for h in holdouts]
        lstm_vals = [sub_m2.loc[h[0], 'M2_NSE'] for h in holdouts]
        
        b1 = ax.bar(x - width, m0_vals, width, label='M0: Baseline', color=COLOR_M0, edgecolor='#374151')
        b2 = ax.bar(x, m1_vals, width, label='M1: Flow-Only GBT', color=COLOR_M1, edgecolor='#374151')
        b3 = ax.bar(x + width, lstm_vals, width, label='M2: PyTorch HydroLSTM', color=COLOR_LSTM, edgecolor='#374151')
        
        ax.axhline(0, color='#374151', linestyle='--', linewidth=1.2)
        ax.set_title(f"Zero-Shot Spatial Transfer — Horizon H={H} Weeks Ahead", fontsize=12.5, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h[0].split()[0]}\n({h[1]})" for h in holdouts], fontsize=10.5)
        ax.set_ylabel("Nash-Sutcliffe Efficiency (NSE)", fontsize=11)
        ax.set_ylim(-1.5, 0.75)
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        
        # Add value labels
        for bar in list(b1) + list(b2) + list(b3):
            h_val = bar.get_height()
            va_align = 'bottom' if h_val >= 0 else 'top'
            offset = 0.03 if h_val >= 0 else -0.06
            ax.text(bar.get_x() + bar.get_width()/2, h_val + offset, f"{h_val:+.2f}",
                    ha='center', va=va_align, fontsize=8.5, fontweight='bold')
                    
        ax.legend(loc='lower left', frameon=True, fontsize=9.5)
        
    fig.suptitle("Protocol 2: Cross-Basin Zero-Shot Transfer to Held-Out Sub-Basins (M0 vs M1 vs M2 LSTM)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig_g_spatial_transfer_grouped_bars.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[+] Saved: {out_file}")

# ==============================================================================
# 9. MAIN PIPELINE ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate Publication-Quality Visualizations for ES 335 Hydro-ML Study."
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=OUTPUT_DIR,
        help=f"Target output directory for PNG figures (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        '--plots',
        nargs='+',
        choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'all'],
        default=['all'],
        help="Specify which plots to generate: A, B, C, D, E, F, G, or all (default: all)"
    )
    args = parser.parse_args()
    
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("      ES 335 PUBLICATION & WEBSITE VISUALIZATION GENERATOR")
    print(f"      Output Directory: {out_dir}")
    print("=" * 80)
    
    plots_to_run = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] if 'all' in args.plots else args.plots
    
    if 'A' in plots_to_run:
        generate_plot_a_hydrographs(out_dir)
    if 'B' in plots_to_run:
        generate_plot_b_scatter(out_dir)
    if 'C' in plots_to_run:
        generate_plot_c_confusion_matrices(out_dir)
    if 'D' in plots_to_run:
        generate_plot_d_pr_curves(out_dir)
    if 'E' in plots_to_run:
        generate_plot_e_skill_decay(out_dir)
    if 'F' in plots_to_run:
        generate_plot_f_importance_shift(out_dir)
    if 'G' in plots_to_run:
        generate_plot_g_spatial_transfer(out_dir)
        
    print("\n" + "=" * 80)
    print(f"[+] Visualization generation finished successfully.")
    print(f"[+] High-resolution PNG figures (300 DPI) saved in: {out_dir}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
