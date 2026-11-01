#!/usr/bin/env python3
"""
Interactive Plotly Visualization Suite for Streamflow & Drought Forecasting (ES 335)

Generates standalone, publication-grade interactive HTML visualizations for
the project web interface and deliverables:
1. Interactive Hydrographs: Observed vs M1 vs M2 with Plotly Range Slider (Karad & Kantamal, H=1w & H=4w)
2. Precision-Recall Curves: Severe Drought (Q95) Early Warning at H=2w & H=4w with rich hover data (Threshold, Precision, Recall, FAR)
3. KGE Component Decomposition: Interactive trajectory of r, alpha, beta across H=1,2,4,8 weeks with 1.0 benchmark line
4. Feature Importance Shift: Grouped horizontal bar chart of M2 GBT predictor importances (H=1w vs H=8w)
5. Zero-Shot Spatial Transfer: Grouped bar chart of H=4w and H=8w NSE on holdout basins (Kantamal, Bamni, Haralahalli) for M0, M1, M2 LSTM

All plots are exported as standalone, self-contained HTML files using fig.write_html().
"""

import os
import sys
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance

# Cleanly suppress scikit-learn unpickling / feature name warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

# ==============================================================================
# 1. DIRECTORY CONFIGURATION & CONSTANTS
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

TENSORS_DIR = os.path.join(PROJECT_ROOT, 'data/processed/weekly_tensors')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'data/processed/model_results')
MODELS_DIR  = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'data/processed/visualizations')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aesthetic Palette (Consistent with ES 335 Report)
COLOR_OBSERVED = '#111827'    # Charcoal / Dark Slate
COLOR_M0       = '#9CA3AF'    # Neutral Gray
COLOR_M1       = '#2563EB'    # Royal Blue
COLOR_M2       = '#DC2626'    # Crimson Red
COLOR_LSTM     = '#7C3AED'    # Deep Violet
COLOR_Q95      = '#D97706'    # Amber / Severe Alert
COLOR_Q90      = '#F59E0B'    # Yellow-Orange / Moderate Alert
COLOR_IDEAL    = '#059669'    # Emerald Green / Ideal 1.0

# Feature definitions
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

FEATURE_LABELS = {
    'flow_lag_0': 'Issue-Time Flow Q(t)',
    'flow_lag_1': 'Lag-1 Week Flow Q(t-1)',
    'flow_lag_2': 'Lag-2 Weeks Flow Q(t-2)',
    'flow_lag_3': 'Lag-3 Weeks Flow Q(t-3)',
    'flow_lag_4': 'Lag-4 Weeks Flow Q(t-4)',
    'flow_lag_7': 'Lag-7 Weeks Flow Q(t-7)',
    'flow_std_7d': '7-Day Flow Volatility',
    'flow_min_7d': '7-Day Flow Minimum',
    'flow_max_7d': '7-Day Flow Maximum',
    'flow_recession_slope': 'Recession Slope ΔQ',
    'rain_sum_1w': 'Precipitation (1-Week)',
    'rain_sum_2w': 'Precipitation (2-Weeks)',
    'rain_sum_1m': 'Precipitation (30-Day)',
    'rain_sum_2m': 'Precipitation (60-Day)',
    'rain_sum_3m': 'Precipitation (90-Day)',
    'api_30d': 'Antecedent Precip Index (30d)',
    'dry_days_count': 'Weekly Dry Days (<1mm)',
    'temp_mean_1w': 'Mean Temperature (1w)',
    'temp_max_1w': 'Max Temperature (1w)',
    'temp_min_1w': 'Min Temperature (1w)',
    'pet_sum_1w': 'PET Atmospheric Demand (1w)',
    'pet_sum_1m': 'PET Atmospheric Demand (1m)',
    'water_deficit_1w': 'Water Deficit (1w P - PET)',
    'water_deficit_1m': 'Water Deficit (1m P - PET)',
    'sin_week': 'Calendar Harmonic (sin)',
    'cos_week': 'Calendar Harmonic (cos)',
}

def categorize_feature(feat_name):
    """Categorizes predictors into hydrological classes."""
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
# 2. MODEL & DATA LOADING HELPERS
# ==============================================================================
def load_estimator(model_path):
    """
    Safely loads trained estimator objects, transparently unpacking metadata dicts
    or standalone estimators.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint missing: {model_path}")
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and 'model' in obj:
        return obj['model'], obj.get('scaler'), obj.get('train_medians')
    return obj, None, None

def load_test_predictions(station_name, gauge_id, horizon, m1_path, m2_path):
    """
    Loads test fold (2016–2025) and generates out-of-sample predictions for M1 & M2.
    """
    tensor_path = os.path.join(TENSORS_DIR, f"weekly_tensor_gauge_{gauge_id:05d}_{station_name}.csv")
    if not os.path.isfile(tensor_path):
        raise FileNotFoundError(f"Weekly tensor missing: {tensor_path}")
        
    df = pd.read_csv(tensor_path, parse_dates=['date']).set_index('date')
    df_test = df.loc['2016-01-01':'2025-12-31'].copy()
    
    target_col = f'target_H{horizon}'
    valid_mask = (~df_test[target_col].isna()) & (~df_test['flow_lag_0'].isna())
    df_eval = df_test[valid_mask].copy()
    
    # Load M1
    m1_est, s1, med1 = load_estimator(m1_path)
    X_m1 = df_eval[M1_FLOW_FEATURES].fillna(med1 if med1 is not None else df_eval[M1_FLOW_FEATURES].median())
    preds_m1 = np.maximum(0.0, m1_est.predict(X_m1.values if hasattr(m1_est, 'n_features_in_') else X_m1))
    
    # Load M2
    m2_est, s2, med2 = load_estimator(m2_path)
    X_m2 = df_eval[M2_TABULAR_FEATURES].fillna(med2 if med2 is not None else df_eval[M2_TABULAR_FEATURES].median())
    preds_m2 = np.maximum(0.0, m2_est.predict(X_m2.values if hasattr(m2_est, 'n_features_in_') else X_m2))
    
    y_actual = df_eval[target_col].values
    
    # Historical training thresholds (1980–2010)
    df_train = df.loc['1980-01-01':'2010-12-31'].dropna(subset=[target_col])
    q95_thresh = float(np.percentile(df_train[target_col], 5.0))
    q90_thresh = float(np.percentile(df_train[target_col], 10.0))
    
    return {
        'dates': df_eval.index,
        'y_actual': y_actual,
        'preds_m1': preds_m1,
        'preds_m2': preds_m2,
        'q95': q95_thresh,
        'q90': q90_thresh,
    }

# ==============================================================================
# 3. PLOT 1: INTERACTIVE HYDROGRAPHS WITH RANGESLIDER
# ==============================================================================
def generate_interactive_hydrographs(out_dir):
    """
    Generates interactive Plotly hydrographs for Karad and Kantamal at H=1w and H=4w.
    Equipped with a Plotly rangeslider and quick-zoom selectors for drought analysis.
    """
    print("[*] Generating Plot 1: Interactive Hydrographs with Range Slider...")
    
    stations = [
        ('Karad', 4021, 'Maharashtra (Upper Krishna Basin)'),
        ('Kantamal', 8013, 'Odisha (Tel / Mahanadi Basin)'),
    ]
    horizons = [1, 4]
    
    saved_files = []
    
    # 1. Individual standalone interactive hydrographs per station/horizon
    for stn_name, gid, basin_desc in stations:
        clean_name = stn_name.replace(' ', '_').replace('(', '').replace(')', '')
        for H in horizons:
            m1_path = os.path.join(MODELS_DIR, 'protocol1', f"m1_gbt_gauge_{gid:05d}_{clean_name}_H{H}.joblib")
            m2_path = os.path.join(MODELS_DIR, 'protocol1', f"m2_gbt_gauge_{gid:05d}_{clean_name}_H{H}.joblib")
            
            data = load_test_predictions(stn_name, gid, H, m1_path, m2_path)
            dates = data['dates']
            y_obs = data['y_actual']
            p_m1  = data['preds_m1']
            p_m2  = data['preds_m2']
            q95   = data['q95']
            q90   = data['q90']
            
            # Metrics
            ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
            nse_m1 = 1 - (np.sum((y_obs - p_m1) ** 2) / ss_tot)
            nse_m2 = 1 - (np.sum((y_obs - p_m2) ** 2) / ss_tot)
            
            # Plotly Figure
            fig = go.Figure()
            
            # Actual Streamflow
            fig.add_trace(go.Scatter(
                x=dates, y=y_obs,
                name='Observed Streamflow (Q)',
                mode='lines',
                line=dict(color=COLOR_OBSERVED, width=2.2),
                hovertemplate='<b>Observed Q</b>: %{y:.2f} m³/s<br>Date: %{x|%Y-%m-%d}<extra></extra>'
            ))
            
            # M1 Flow-Only GBT
            fig.add_trace(go.Scatter(
                x=dates, y=p_m1,
                name=f'M1: Flow-Only GBT (NSE: {nse_m1:.3f})',
                mode='lines',
                line=dict(color=COLOR_M1, width=1.8, dash='dash'),
                hovertemplate='<b>M1 Forecast</b>: %{y:.2f} m³/s<br>Error: %{customdata:.2f} m³/s<extra></extra>',
                customdata=p_m1 - y_obs
            ))
            
            # M2 Hydro-Meteorological GBT
            fig.add_trace(go.Scatter(
                x=dates, y=p_m2,
                name=f'M2: Hydro-Met GBT (NSE: {nse_m2:.3f})',
                mode='lines',
                line=dict(color=COLOR_M2, width=2.0),
                hovertemplate='<b>M2 Forecast</b>: %{y:.2f} m³/s<br>Error: %{customdata:.2f} m³/s<extra></extra>',
                customdata=p_m2 - y_obs
            ))
            
            # Drought Thresholds
            fig.add_hline(
                y=q95, line_dash='dot', line_color=COLOR_Q95, line_width=1.8,
                annotation_text=f"Severe Drought (Q95: {q95:.1f} m³/s)",
                annotation_position="top left",
                annotation_font=dict(color=COLOR_Q95, size=11, family="sans-serif")
            )
            fig.add_hline(
                y=q90, line_dash='dashdot', line_color=COLOR_Q90, line_width=1.4,
                annotation_text=f"Moderate Drought (Q90: {q90:.1f} m³/s)",
                annotation_position="bottom left",
                annotation_font=dict(color=COLOR_Q90, size=10, family="sans-serif")
            )
            
            # Layout & Rangeslider
            fig.update_layout(
                title=dict(
                    text=f"<b>Interactive Hydrograph: {stn_name} ({basin_desc})</b><br>"
                         f"<span style='font-size: 13px; color: #4B5563;'>"
                         f"Lead Horizon H = {H} Week{'s' if H > 1 else ''} | Out-of-Sample Test Record (2016–2025)"
                         f"</span>",
                    x=0.03, y=0.96
                ),
                xaxis=dict(
                    title="Forecast Timeline",
                    rangeslider=dict(visible=True, thickness=0.08, bgcolor='#F3F4F6'),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=6, label="6 Months", step="month", stepmode="backward"),
                            dict(count=1, label="1 Year", step="year", stepmode="backward"),
                            dict(count=2, label="2 Years (2018–19 Window)", step="year", stepmode="backward"),
                            dict(step="all", label="All 10 Years (2016–2025)")
                        ]),
                        x=0.0, y=1.12, bgcolor='#E5E7EB', font=dict(size=11)
                    ),
                    range=['2018-01-01', '2019-12-31'],  # Default focus on the 2018 drought & 2019 monsoon
                    showgrid=True, gridcolor='#E5E7EB'
                ),
                yaxis=dict(
                    title="Discharge Rate (m³/s)",
                    showgrid=True, gridcolor='#E5E7EB',
                    zeroline=True, zerolinecolor='#9CA3AF'
                ),
                template='plotly_white',
                hovermode='x unified',
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#D1D5DB", borderwidth=1
                ),
                margin=dict(l=60, r=40, t=110, b=60),
                height=560
            )
            
            out_filename = f"interactive_hydrograph_{stn_name.lower()}_h{H}w.html"
            out_file = os.path.join(out_dir, out_filename)
            fig.write_html(out_file, include_plotlyjs='cdn')
            saved_files.append(out_file)
            print(f"   [+] Saved standalone hydrograph: {out_file}")
            
    # 2. Comprehensive 2x2 Subplot Dashboard (Karad & Kantamal x H=1w & H=4w)
    fig_dash = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "<b>Karad (Krishna Basin) — 1-Week Ahead</b>",
            "<b>Kantamal (Mahanadi Basin) — 1-Week Ahead</b>",
            "<b>Karad (Krishna Basin) — 4-Weeks Ahead</b>",
            "<b>Kantamal (Mahanadi Basin) — 4-Weeks Ahead</b>"
        ],
        vertical_spacing=0.12, horizontal_spacing=0.08
    )
    
    stn_configs = [
        ('Karad', 4021, 1, 1),
        ('Kantamal', 8013, 1, 2),
        ('Karad', 4021, 2, 1),
        ('Kantamal', 8013, 2, 2),
    ]
    
    for stn_name, gid, row, col in stn_configs:
        H = 1 if row == 1 else 4
        clean_name = stn_name.replace(' ', '_').replace('(', '').replace(')', '')
        m1_path = os.path.join(MODELS_DIR, 'protocol1', f"m1_gbt_gauge_{gid:05d}_{clean_name}_H{H}.joblib")
        m2_path = os.path.join(MODELS_DIR, 'protocol1', f"m2_gbt_gauge_{gid:05d}_{clean_name}_H{H}.joblib")
        
        data = load_test_predictions(stn_name, gid, H, m1_path, m2_path)
        dates = data['dates']
        y_obs = data['y_actual']
        p_m1  = data['preds_m1']
        p_m2  = data['preds_m2']
        q95   = data['q95']
        
        show_leg = (row == 1 and col == 1)
        
        fig_dash.add_trace(go.Scatter(
            x=dates, y=y_obs, name='Observed (Q)', mode='lines',
            line=dict(color=COLOR_OBSERVED, width=1.8), showlegend=show_leg
        ), row=row, col=col)
        
        fig_dash.add_trace(go.Scatter(
            x=dates, y=p_m1, name='M1: Flow-Only GBT', mode='lines',
            line=dict(color=COLOR_M1, width=1.5, dash='dash'), showlegend=show_leg
        ), row=row, col=col)
        
        fig_dash.add_trace(go.Scatter(
            x=dates, y=p_m2, name='M2: Hydro-Met GBT', mode='lines',
            line=dict(color=COLOR_M2, width=1.8), showlegend=show_leg
        ), row=row, col=col)
        
        fig_dash.add_hline(y=q95, line_dash='dot', line_color=COLOR_Q95, line_width=1.2, row=row, col=col)
        
    fig_dash.update_layout(
        title=dict(
            text="<b>Multi-Basin Hydrograph Dashboard (Observed vs M1 vs M2 Across Horizons)</b><br>"
                 "<span style='font-size: 13px; color: #4B5563;'>"
                 "Zoom into individual panels or double-click to reset axis. Dotted amber line represents Q95 Severe Drought."
                 "</span>",
            x=0.03, y=0.98
        ),
        template='plotly_white',
        height=850,
        margin=dict(l=60, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0)
    )
    # Default range to the 2018-2019 drought window across all panels
    for r in [1, 2]:
        for c in [1, 2]:
            fig_dash.update_xaxes(range=['2018-01-01', '2019-12-31'], row=r, col=c, gridcolor='#E5E7EB')
            fig_dash.update_yaxes(title_text="Discharge (m³/s)", row=r, col=c, gridcolor='#E5E7EB')
            
    dash_file = os.path.join(out_dir, "interactive_hydrographs_dashboard.html")
    fig_dash.write_html(dash_file, include_plotlyjs='cdn')
    print(f"   [+] Saved integrated hydrographs dashboard: {dash_file}")
    return saved_files

# ==============================================================================
# 4. PLOT 2: PRECISION-RECALL CURVES WITH DETAILED THRESHOLD HOVER
# ==============================================================================
def generate_interactive_pr_curves(out_dir):
    """
    Generates interactive Precision-Recall curves for Severe Hydrological Drought (Q95)
    at H=2w and H=4w across all 12 test basins, with full threshold hover data.
    """
    print("[*] Generating Plot 2: Interactive Precision-Recall Curves (Q95, H=2w & H=4w)...")
    
    test_stations = [
        ('Karad', 4021), ('Warunji', 4063), ('Ashti', 3005), ('Bamni (Ballarsha)', 3008),
        ('Anandpur', 7002), ('Kashinagar', 16007), ('Gomlai', 7007), ('Kantamal', 8013),
        ('Yadgir', 4064), ('Huvinhedigi', 4017), ('KUDIGE', 5010), ('Haralahalli', 4013),
    ]
    
    horizons = [2, 4]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>Horizon H = 2 Weeks Ahead (Severe Drought Q95)</b>",
            "<b>Horizon H = 4 Weeks Ahead (Severe Drought Q95)</b>"
        ],
        horizontal_spacing=0.10
    )
    
    for col_idx, H in enumerate(horizons, start=1):
        y_true_all, m1_all, m2_all = [], [], []
        
        for stn, gid in test_stations:
            clean_name = stn.replace(' ', '_').replace('(', '').replace(')', '')
            m1_path = os.path.join(MODELS_DIR, 'protocol1', f"m1_gbt_gauge_{gid:05d}_{clean_name}_H{H}.joblib")
            m2_path = os.path.join(MODELS_DIR, 'protocol1', f"m2_gbt_gauge_{gid:05d}_{clean_name}_H{H}.joblib")
            
            if os.path.isfile(m1_path) and os.path.isfile(m2_path):
                data = load_test_predictions(stn, gid, H, m1_path, m2_path)
                thresh = data['q95']
                
                # Binary drought event (1 if flow <= Q95)
                y_bin = (data['y_actual'] <= thresh).astype(int)
                
                # Deficit alert scores: inverted flow predictions
                s_m1 = -data['preds_m1']
                s_m2 = -data['preds_m2']
                
                y_true_all.extend(y_bin)
                m1_all.extend(s_m1)
                m2_all.extend(s_m2)
                
        y_true_arr = np.array(y_true_all)
        m1_arr     = np.array(m1_all)
        m2_arr     = np.array(m2_all)
        
        # PR curves
        p_m1, r_m1, t_m1 = precision_recall_curve(y_true_arr, m1_arr)
        p_m2, r_m2, t_m2 = precision_recall_curve(y_true_arr, m2_arr)
        
        auc_m1 = average_precision_score(y_true_arr, m1_arr)
        auc_m2 = average_precision_score(y_true_arr, m2_arr)
        prevalence = np.mean(y_true_arr)
        
        # Format aligned thresholds for hover info
        t1_aligned = np.append(-t_m1, -t_m1[-1])
        t2_aligned = np.append(-t_m2, -t_m2[-1])
        
        far_m1 = np.maximum(0.0, 1.0 - p_m1)
        far_m2 = np.maximum(0.0, 1.0 - p_m2)
        
        cdata_m1 = np.stack((t1_aligned, far_m1), axis=-1)
        cdata_m2 = np.stack((t2_aligned, far_m2), axis=-1)
        
        # M1 Trace
        fig.add_trace(go.Scatter(
            x=r_m1, y=p_m1,
            name=f"M1: Flow-Only (PR-AUC: {auc_m1:.3f})",
            mode='lines',
            line=dict(color=COLOR_M1, width=2.4),
            customdata=cdata_m1,
            hovertemplate=(
                "<b>M1 (Flow-Only GBT)</b><br>"
                "Recall (POD): %{x:.3f}<br>"
                "Precision (1-FAR): %{y:.3f}<br>"
                "Discharge Cutoff: %{customdata[0]:.2f} m³/s<br>"
                "False Alarm Ratio (FAR): %{customdata[1]:.3f}<extra></extra>"
            ),
            legendgroup=f"m1_h{H}"
        ), row=1, col=col_idx)
        
        # M2 Trace
        fig.add_trace(go.Scatter(
            x=r_m2, y=p_m2,
            name=f"M2: Hydro-Met (PR-AUC: {auc_m2:.3f})",
            mode='lines',
            line=dict(color=COLOR_M2, width=2.6),
            customdata=cdata_m2,
            hovertemplate=(
                "<b>M2 (Hydro-Meteorological GBT)</b><br>"
                "Recall (POD): %{x:.3f}<br>"
                "Precision (1-FAR): %{y:.3f}<br>"
                "Discharge Cutoff: %{customdata[0]:.2f} m³/s<br>"
                "False Alarm Ratio (FAR): %{customdata[1]:.3f}<extra></extra>"
            ),
            legendgroup=f"m2_h{H}"
        ), row=1, col=col_idx)
        
        # Prevalence Baseline
        fig.add_trace(go.Scatter(
            x=[0, 1.0], y=[prevalence, prevalence],
            name=f"No-Skill Baseline ({prevalence:.3f})",
            mode='lines',
            line=dict(color='#6B7280', width=1.5, dash='dash'),
            hoverinfo='none',
            showlegend=(col_idx == 1)
        ), row=1, col=col_idx)
        
        fig.update_xaxes(title_text="Recall / Probability of Detection (POD)", range=[0, 1.02], row=1, col=col_idx, gridcolor='#E5E7EB')
        fig.update_yaxes(title_text="Precision (1 - False Alarm Ratio)", range=[0, 1.02], row=1, col=col_idx, gridcolor='#E5E7EB')
        
    fig.update_layout(
        title=dict(
            text="<b>Interactive Precision-Recall Curves for Severe Hydrological Drought (Q95) Early Warning</b><br>"
                 "<span style='font-size: 13px; color: #4B5563;'>"
                 "Hover over curve points to inspect operational discharge thresholds, detection recall, and false alarm ratios."
                 "</span>",
            x=0.03, y=0.96
        ),
        template='plotly_white',
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=540,
        margin=dict(l=60, r=40, t=100, b=60)
    )
    
    out_file = os.path.join(out_dir, "interactive_pr_curves_severe_drought.html")
    fig.write_html(out_file, include_plotlyjs='cdn')
    print(f"   [+] Saved interactive PR curves: {out_file}")
    return out_file

# ==============================================================================
# 5. PLOT 3: KGE COMPONENT DECOMPOSITION (r, alpha, beta ACROSS HORIZONS)
# ==============================================================================
def generate_interactive_kge_decomposition(out_dir):
    """
    Generates an interactive multi-panel line plot tracking median KGE components
    (Pearson r, variability ratio alpha, and bias ratio beta) across H=1,2,4,8 weeks.
    """
    print("[*] Generating Plot 3: Interactive KGE Component Decomposition (r, α, β)...")
    
    p1_m1_path = os.path.join(RESULTS_DIR, 'protocol1_temporal_results_gbt.csv')
    p1_m2_path = os.path.join(RESULTS_DIR, 'protocol1_temporal_results_m2_gbt.csv')
    
    df1 = pd.read_csv(p1_m1_path)
    df2 = pd.read_csv(p1_m2_path)
    
    horizons = [1, 2, 4, 8]
    
    # Extract median components across the 12 catchments
    m0_r, m1_r, m2_r = [], [], []
    m0_a, m1_a, m2_a = [], [], []
    m0_b, m1_b, m2_b = [], [], []
    
    for h in horizons:
        s1 = df1[df1['Horizon_Weeks'] == h]
        s2 = df2[df2['Horizon_Weeks'] == h]
        
        m0_r.append(float(s2['M0_KGE_r'].median()))
        m1_r.append(float(s1['M1_KGE_r'].median()))
        m2_r.append(float(s2['M2_KGE_r'].median()))
        
        m0_a.append(float(s2['M0_KGE_alpha'].median()))
        m1_a.append(float(s1['M1_KGE_alpha'].median()))
        m2_a.append(float(s2['M2_KGE_alpha'].median()))
        
        m0_b.append(float(s2['M0_KGE_beta'].median()))
        m1_b.append(float(s1['M1_KGE_beta'].median()))
        m2_b.append(float(s2['M2_KGE_beta'].median()))
        
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "<b>(a) Hydrograph Correlation (r)</b><br><span style='font-size:11px;color:#6B7280;'>Timing & Shape Synchronization</span>",
            "<b>(b) Variability Ratio (α = σ<sub>pred</sub> / σ<sub>obs</sub>)</b><br><span style='font-size:11px;color:#6B7280;'>Peak Flow & Discharge Dispersion</span>",
            "<b>(c) Volume Bias Ratio (β = μ<sub>pred</sub> / μ<sub>obs</sub>)</b><br><span style='font-size:11px;color:#6B7280;'>Long-Term Cumulative Water Balance</span>"
        ],
        horizontal_spacing=0.08
    )
    
    components = [
        ('Correlation (r)', m0_r, m1_r, m2_r, 1),
        ('Variability (α)', m0_a, m1_a, m2_a, 2),
        ('Bias Ratio (β)',  m0_b, m1_b, m2_b, 3),
    ]
    
    hz_labels = ['1w', '2w', '4w', '8w']
    
    for comp_name, m0_vals, m1_vals, m2_vals, col_idx in components:
        show_leg = (col_idx == 1)
        
        # Ideal benchmark line (1.0)
        fig.add_trace(go.Scatter(
            x=hz_labels, y=[1.0, 1.0, 1.0, 1.0],
            name='Ideal Benchmark (1.0)',
            mode='lines',
            line=dict(color=COLOR_IDEAL, width=1.6, dash='dot'),
            hovertemplate='<b>Ideal Benchmark</b>: 1.000<extra></extra>',
            showlegend=show_leg
        ), row=1, col=col_idx)
        
        # M0 Persistence / Climatology
        dev_m0 = np.array(m0_vals) - 1.0
        fig.add_trace(go.Scatter(
            x=hz_labels, y=m0_vals,
            name='M0: Baseline',
            mode='lines+markers',
            line=dict(color=COLOR_M0, width=2.0, dash='dash'),
            marker=dict(size=8, symbol='square'),
            customdata=dev_m0,
            hovertemplate=(
                f"<b>M0 Baseline — {comp_name}</b><br>"
                "Lead: H = %{x}<br>"
                "Value: %{y:.3f}<br>"
                "Deviation from Ideal: %{customdata:+.3f}<extra></extra>"
            ),
            showlegend=show_leg
        ), row=1, col=col_idx)
        
        # M1 Flow-Only GBT
        dev_m1 = np.array(m1_vals) - 1.0
        fig.add_trace(go.Scatter(
            x=hz_labels, y=m1_vals,
            name='M1: Flow-Only GBT',
            mode='lines+markers',
            line=dict(color=COLOR_M1, width=2.2),
            marker=dict(size=8, symbol='diamond'),
            customdata=dev_m1,
            hovertemplate=(
                f"<b>M1 (Flow-Only) — {comp_name}</b><br>"
                "Lead: H = %{x}<br>"
                "Value: %{y:.3f}<br>"
                "Deviation from Ideal: %{customdata:+.3f}<extra></extra>"
            ),
            showlegend=show_leg
        ), row=1, col=col_idx)
        
        # M2 Hydro-Meteorological GBT
        dev_m2 = np.array(m2_vals) - 1.0
        fig.add_trace(go.Scatter(
            x=hz_labels, y=m2_vals,
            name='M2: Hydro-Met GBT',
            mode='lines+markers',
            line=dict(color=COLOR_M2, width=2.5),
            marker=dict(size=9, symbol='circle'),
            customdata=dev_m2,
            hovertemplate=(
                f"<b>M2 (Hydro-Met) — {comp_name}</b><br>"
                "Lead: H = %{x}<br>"
                "Value: %{y:.3f}<br>"
                "Deviation from Ideal: %{customdata:+.3f}<extra></extra>"
            ),
            showlegend=show_leg
        ), row=1, col=col_idx)
        
        fig.update_xaxes(title_text="Forecast Horizon Ahead", row=1, col=col_idx, gridcolor='#E5E7EB')
        fig.update_yaxes(title_text="Component Value", range=[-0.05, 1.25], row=1, col=col_idx, gridcolor='#E5E7EB')
        
    fig.update_layout(
        title=dict(
            text="<b>Kling-Gupta Efficiency (KGE) Component Decomposition Across Lead Horizons</b><br>"
                 "<span style='font-size: 13px; color: #4B5563;'>"
                 "Decomposition reveals why M2 excels at H=4w & H=8w: meteorological forcings preserve volume balance (β ≈ 1.0) and correlation (r > 0.55)."
                 "</span>",
            x=0.03, y=0.96
        ),
        template='plotly_white',
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=540,
        margin=dict(l=60, r=40, t=110, b=60)
    )
    
    out_file = os.path.join(out_dir, "interactive_kge_component_decomposition.html")
    fig.write_html(out_file, include_plotlyjs='cdn')
    print(f"   [+] Saved KGE decomposition plot: {out_file}")
    return out_file

# ==============================================================================
# 6. PLOT 4: FEATURE IMPORTANCE TRANSITION (GROUPED HORIZONTAL BARS)
# ==============================================================================
def generate_interactive_feature_importance(out_dir):
    """
    Generates an interactive grouped horizontal bar chart comparing M2 GBT feature
    importances at H=1w vs H=8w for representative Deccan headwater gauge Karad.
    """
    print("[*] Generating Plot 4: Interactive Feature Importance Transition (H=1w vs H=8w)...")
    
    stn_name = 'Karad'
    gid = 4021
    
    tensor_path = os.path.join(TENSORS_DIR, f"weekly_tensor_gauge_{gid:05d}_{stn_name}.csv")
    df = pd.read_csv(tensor_path, parse_dates=['date']).set_index('date')
    df_test = df.loc['2016-01-01':'2025-12-31'].copy()
    
    imp_results = {}
    
    for H in [1, 8]:
        model_path = os.path.join(MODELS_DIR, 'protocol1', f"m2_gbt_gauge_{gid:05d}_{stn_name}_H{H}.joblib")
        model, scaler, medians = load_estimator(model_path)
        
        target_col = f'target_H{H}'
        eval_df = df_test.dropna(subset=[target_col] + M2_TABULAR_FEATURES)
        X = eval_df[M2_TABULAR_FEATURES]
        y = eval_df[target_col]
        
        r = permutation_importance(model, X.values if hasattr(model, 'n_features_in_') else X, y.values,
                                   n_repeats=5, random_state=42, scoring='neg_mean_squared_error')
        importances = np.maximum(0, r.importances_mean)
        sum_imp = np.sum(importances)
        norm_imp = (importances / sum_imp) * 100 if sum_imp > 0 else importances
        imp_results[H] = pd.Series(norm_imp, index=M2_TABULAR_FEATURES)
        
    df_imp = pd.DataFrame({'H1': imp_results[1], 'H8': imp_results[8]})
    df_imp['Max_Imp'] = df_imp.max(axis=1)
    df_top = df_imp.sort_values('Max_Imp', ascending=True).tail(12)  # Top 12 predictors
    
    feat_display_names = [FEATURE_LABELS.get(f, f) for f in df_top.index]
    categories = [categorize_feature(f) for f in df_top.index]
    
    fig = go.Figure()
    
    # H=1w Trace (Short-term flow memory dominant)
    cdata_h1 = np.stack((categories, df_top['H1'].values), axis=-1)
    fig.add_trace(go.Bar(
        y=feat_display_names,
        x=df_top['H1'],
        name='Horizon H = 1 Week (Short-Term Memory)',
        orientation='h',
        marker=dict(color='#2563EB', line=dict(color='#1E40AF', width=1)),
        customdata=cdata_h1,
        hovertemplate=(
            "<b>Predictor: %{y}</b><br>"
            "Hydrological Category: %{customdata[0]}<br>"
            "Lead Time: H = 1 Week Ahead<br>"
            "Relative Importance: <b>%{x:.1f}%</b><extra></extra>"
        )
    ))
    
    # H=8w Trace (Meteorological / calendar memory dominant)
    cdata_h8 = np.stack((categories, df_top['H8'].values), axis=-1)
    fig.add_trace(go.Bar(
        y=feat_display_names,
        x=df_top['H8'],
        name='Horizon H = 8 Weeks (Long-Term Weather & Seasonality)',
        orientation='h',
        marker=dict(color='#DC2626', line=dict(color='#991B1B', width=1)),
        customdata=cdata_h8,
        hovertemplate=(
            "<b>Predictor: %{y}</b><br>"
            "Hydrological Category: %{customdata[0]}<br>"
            "Lead Time: H = 8 Weeks Ahead<br>"
            "Relative Importance: <b>%{x:.1f}%</b><extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Hydrological Driver Shift: Flow Memory (H=1w) vs Meteorological Memory (H=8w)</b><br>"
                 "<span style='font-size: 13px; color: #4B5563;'>"
                 "Permutation feature importance on Karad test fold reveals collapse of autoregressive flow memory and dominance of climate forcings."
                 "</span>",
            x=0.03, y=0.96
        ),
        barmode='group',
        template='plotly_white',
        xaxis=dict(
            title="Relative Permutation Importance (%)",
            showgrid=True, gridcolor='#E5E7EB'
        ),
        yaxis=dict(
            title="Predictor Feature",
            showgrid=False
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#D1D5DB", borderwidth=1
        ),
        height=580,
        margin=dict(l=220, r=40, t=100, b=60)
    )
    
    out_file = os.path.join(out_dir, "interactive_feature_importance_shift.html")
    fig.write_html(out_file, include_plotlyjs='cdn')
    print(f"   [+] Saved interactive feature importance shift: {out_file}")
    return out_file

# ==============================================================================
# 7. PLOT 5: ZERO-SHOT SPATIAL TRANSFER GROUPED BARS
# ==============================================================================
def generate_interactive_spatial_transfer(out_dir):
    """
    Generates interactive grouped bar charts comparing H=4w and H=8w Zero-Shot Nash-Sutcliffe
    Efficiency on held-out test basins across M0, M1 Flow-Only GBT, and M2 PyTorch HydroLSTM.
    """
    print("[*] Generating Plot 5: Interactive Zero-Shot Spatial Generalization...")
    
    p2_m1_path = os.path.join(RESULTS_DIR, 'protocol2_spatial_results_gbt.csv')
    p2_m2_path = os.path.join(RESULTS_DIR, 'protocol2_spatial_results_m2_lstm.csv')
    
    df_p2_m1 = pd.read_csv(p2_m1_path)
    df_p2_m2 = pd.read_csv(p2_m2_path)
    
    holdouts = [
        ('Kantamal', 'Odisha', 'Tel / Mahanadi Basin'),
        ('Bamni (Ballarsha)', 'Maharashtra', 'Wardha / Godavari Basin'),
        ('Haralahalli', 'Karnataka', 'Tungabhadra / Krishna Basin'),
    ]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>Horizon H = 4 Weeks Ahead (Zero-Shot Cross-Basin)</b>",
            "<b>Horizon H = 8 Weeks Ahead (Zero-Shot Cross-Basin)</b>"
        ],
        horizontal_spacing=0.10
    )
    
    horizons = [4, 8]
    
    for col_idx, H in enumerate(horizons, start=1):
        sub_m1 = df_p2_m1[df_p2_m1['Horizon_Weeks'] == H].set_index('Holdout_Station')
        sub_m2 = df_p2_m2[df_p2_m2['Horizon_Weeks'] == H].set_index('Holdout_Station')
        
        stn_names = [h[0] for h in holdouts]
        display_names = [f"<b>{h[0].split()[0]}</b><br><span style='font-size:10px;color:#6B7280;'>{h[1]}</span>" for h in holdouts]
        
        m0_vals   = [float(sub_m1.loc[s, 'M0_NSE']) for s in stn_names]
        m1_vals   = [float(sub_m1.loc[s, 'M1_NSE']) for s in stn_names]
        lstm_vals = [float(sub_m2.loc[s, 'M2_NSE']) for s in stn_names]
        
        show_leg = (col_idx == 1)
        
        # M0 Baseline
        fig.add_trace(go.Bar(
            x=display_names, y=m0_vals,
            name='M0: Baseline (Climatology)',
            marker=dict(color=COLOR_M0, line=dict(color='#4B5563', width=1)),
            text=[f"{v:+.2f}" for v in m0_vals],
            textposition='auto',
            hovertemplate="<b>M0 Baseline</b><br>Station: %{x}<br>NSE: <b>%{y:+.3f}</b><extra></extra>",
            showlegend=show_leg
        ), row=1, col=col_idx)
        
        # M1 Flow-Only GBT
        fig.add_trace(go.Bar(
            x=display_names, y=m1_vals,
            name='M1: Flow-Only GBT',
            marker=dict(color=COLOR_M1, line=dict(color='#1E40AF', width=1)),
            text=[f"{v:+.2f}" for v in m1_vals],
            textposition='auto',
            hovertemplate="<b>M1 Flow-Only GBT</b><br>Station: %{x}<br>NSE: <b>%{y:+.3f}</b><extra></extra>",
            showlegend=show_leg
        ), row=1, col=col_idx)
        
        # M2 PyTorch HydroLSTM
        fig.add_trace(go.Bar(
            x=display_names, y=lstm_vals,
            name='M2: PyTorch HydroLSTM',
            marker=dict(color=COLOR_LSTM, line=dict(color='#5B21B6', width=1)),
            text=[f"{v:+.2f}" for v in lstm_vals],
            textposition='auto',
            hovertemplate="<b>M2 PyTorch HydroLSTM</b><br>Station: %{x}<br>NSE: <b>%{y:+.3f}</b><extra></extra>",
            showlegend=show_leg
        ), row=1, col=col_idx)
        
        # Zero Predictive Skill Boundary Line
        fig.add_hline(y=0.0, line_dash='dash', line_color='#374151', line_width=1.5, row=1, col=col_idx)
        
        fig.update_xaxes(title_text="Held-Out Sub-Basin", row=1, col=col_idx)
        fig.update_yaxes(title_text="Zero-Shot NSE", range=[-1.45, 0.75], row=1, col=col_idx, gridcolor='#E5E7EB')
        
    fig.update_layout(
        title=dict(
            text="<b>Protocol 2: Cross-Basin Zero-Shot Spatial Generalization (M0 vs M1 GBT vs M2 PyTorch LSTM)</b><br>"
                 "<span style='font-size: 13px; color: #4B5563;'>"
                 "Held-out basins were strictly excluded from model training. PyTorch HydroLSTM achieves positive zero-shot transfer on Bamni and Kantamal."
                 "</span>",
            x=0.03, y=0.96
        ),
        barmode='group',
        template='plotly_white',
        hovermode='closest',
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0,
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#D1D5DB", borderwidth=1
        ),
        height=540,
        margin=dict(l=60, r=40, t=105, b=60)
    )
    
    out_file = os.path.join(out_dir, "interactive_spatial_transfer_grouped_bars.html")
    fig.write_html(out_file, include_plotlyjs='cdn')
    print(f"   [+] Saved interactive zero-shot transfer grouped bars: {out_file}")
    return out_file

# ==============================================================================
# 8. MASTER INDEX / SHOWCASE HTML GENERATOR
# ==============================================================================
def generate_interactive_index(out_dir):
    """
    Generates a clean index.html gallery file to navigate all generated interactive
    visualizations for ES 335 Deliverable 4.
    """
    print("[*] Generating Interactive Visualization Index Gallery (index.html)...")
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ES 335 Hydro-ML Study — Interactive Visualizations Gallery</title>
    <style>
        :root {
            --primary: #1E40AF;
            --accent: #DC2626;
            --bg: #F9FAFB;
            --card-bg: #FFFFFF;
            --text-main: #111827;
            --text-muted: #4B5563;
            --border: #E5E7EB;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 30px 20px;
            line-height: 1.5;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
        }
        header {
            margin-bottom: 30px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 20px;
        }
        h1 {
            font-size: 26px;
            font-weight: 700;
            margin: 0 0 8px 0;
            color: var(--text-main);
        }
        p.subtitle {
            font-size: 15px;
            color: var(--text-muted);
            margin: 0;
        }
        .badge {
            display: inline-block;
            background: #EEF2FF;
            color: var(--primary);
            font-size: 12px;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 4px;
            margin-top: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-top: 25px;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.08);
            border-color: #CBD5E1;
        }
        .card h2 {
            font-size: 17px;
            margin: 0 0 8px 0;
            color: var(--primary);
        }
        .card p {
            font-size: 13.5px;
            color: var(--text-muted);
            margin: 0 0 16px 0;
            flex-grow: 1;
        }
        .btn {
            display: inline-block;
            background: var(--primary);
            color: #FFFFFF;
            text-decoration: none;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            text-align: center;
        }
        .btn:hover {
            background: #1D4ED8;
        }
        footer {
            margin-top: 40px;
            font-size: 12.5px;
            color: var(--text-muted);
            text-align: center;
            border-top: 1px solid var(--border);
            padding-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Streamflow & Hydrological Drought Early Warning (ES 335)</h1>
            <p class="subtitle">Interactive Plotly Suite: Multi-Horizon Forecasts, Precision-Recall Alert Dynamics, and Spatial Generalization</p>
            <div class="badge">Audited Out-of-Sample Test Folds (2016–2025)</div>
        </header>

        <div class="grid">
            <div class="card">
                <div>
                    <h2>1. Interactive Hydrographs Dashboard</h2>
                    <p>Zoomable 4-panel dashboard with Plotly rangeslider comparing Observed Q vs M1 Flow-Only vs M2 Hydro-Met across Karad and Kantamal at H=1w & H=4w.</p>
                </div>
                <a href="interactive_hydrographs_dashboard.html" class="btn" target="_blank">Launch Dashboard &rarr;</a>
            </div>

            <div class="card">
                <div>
                    <h2>2. Karad Hydrograph (H=1w)</h2>
                    <p>High-resolution zoomable streamflow hydrograph for Karad (Upper Krishna Basin) at 1-week lead time with Q95 severe drought warning threshold.</p>
                </div>
                <a href="interactive_hydrograph_karad_h1w.html" class="btn" target="_blank">View Karad H=1w &rarr;</a>
            </div>

            <div class="card">
                <div>
                    <h2>3. Kantamal Hydrograph (H=4w)</h2>
                    <p>4-week ahead streamflow predictions and drought tracking for Kantamal (Mahanadi Basin) showing volume deficit capture under M2.</p>
                </div>
                <a href="interactive_hydrograph_kantamal_h4w.html" class="btn" target="_blank">View Kantamal H=4w &rarr;</a>
            </div>

            <div class="card">
                <div>
                    <h2>4. Precision-Recall Curves (Q95 Drought)</h2>
                    <p>Interactive PR curves with hover inspection of operational discharge cutoffs, probability of detection (recall), and false alarm ratios at H=2w & H=4w.</p>
                </div>
                <a href="interactive_pr_curves_severe_drought.html" class="btn" target="_blank">Inspect PR Curves &rarr;</a>
            </div>

            <div class="card">
                <div>
                    <h2>5. KGE Component Decomposition</h2>
                    <p>Tracks correlation (r), variability ratio (α), and volume bias ratio (β) degradation against the 1.0 benchmark from H=1w to H=8w.</p>
                </div>
                <a href="interactive_kge_component_decomposition.html" class="btn" target="_blank">Explore KGE Dynamics &rarr;</a>
            </div>

            <div class="card">
                <div>
                    <h2>6. Feature Importance Transition</h2>
                    <p>Grouped horizontal bar chart showing the fundamental physical transition from short-term flow memory (H=1w) to meteorological memory (H=8w).</p>
                </div>
                <a href="interactive_feature_importance_shift.html" class="btn" target="_blank">Inspect Feature Shift &rarr;</a>
            </div>

            <div class="card">
                <div>
                    <h2>7. Zero-Shot Spatial Transfer</h2>
                    <p>Grouped bar charts comparing out-of-basin transfer to held-out sub-basins (Kantamal, Bamni, Haralahalli) across M0, M1 GBT, and M2 PyTorch HydroLSTM.</p>
                </div>
                <a href="interactive_spatial_transfer_grouped_bars.html" class="btn" target="_blank">View Spatial Generalization &rarr;</a>
            </div>
        </div>

        <footer>
            ES 335 Machine Learning Course Project &bull; Verified under Gate AG-TS-04 & AG-TS-05 &bull; Standalone HTML Visualizations
        </footer>
    </div>
</body>
</html>
"""
    index_file = os.path.join(out_dir, "index.html")
    with open(index_file, 'w') as f:
        f.write(html_content)
    print(f"   [+] Saved interactive visualization gallery: {index_file}")
    return index_file

# ==============================================================================
# 9. MAIN PIPELINE EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate Interactive Plotly Visualizations for ES 335 Hydro-ML Study."
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=OUTPUT_DIR,
        help=f"Target output directory for standalone HTML plots (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        '--plots',
        nargs='+',
        choices=['hydrographs', 'pr_curves', 'kge', 'importance', 'spatial', 'all'],
        default=['all'],
        help="Specify which plots to generate: hydrographs, pr_curves, kge, importance, spatial, or all (default: all)"
    )
    args = parser.parse_args()
    
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("      ES 335 INTERACTIVE PLOTLY VISUALIZATION PIPELINE")
    print(f"      Target Directory: {out_dir}")
    print("=" * 80)
    
    plots_to_run = ['hydrographs', 'pr_curves', 'kge', 'importance', 'spatial'] if 'all' in args.plots else args.plots
    
    if 'hydrographs' in plots_to_run:
        generate_interactive_hydrographs(out_dir)
    if 'pr_curves' in plots_to_run:
        generate_interactive_pr_curves(out_dir)
    if 'kge' in plots_to_run:
        generate_interactive_kge_decomposition(out_dir)
    if 'importance' in plots_to_run:
        generate_interactive_feature_importance(out_dir)
    if 'spatial' in plots_to_run:
        generate_interactive_spatial_transfer(out_dir)
        
    generate_interactive_index(out_dir)
    
    print("\n" + "=" * 80)
    print("[+] Interactive HTML visualization suite generated successfully.")
    print(f"[+] All standalone files are located in: {out_dir}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
