#!/usr/bin/env python3
"""
Interactive Plotly Visualization Suite for Streamflow & Drought Forecasting (ES 335)

Generates standalone, publication-grade interactive HTML visualizations for
the project web interface and deliverables:
1. Interactive Hydrographs: Observed vs M1 vs M2 with Plotly Range Slider (Karad & Kantamal, H=1w & H=4w)
2. Precision-Recall Curves: Severe Drought (Q95) Early Warning at H=2w & H=4w with rich hover data (Threshold, Precision, Recall, FAR)
3. KGE Component Decomposition: Interactive trajectory of r, alpha, beta across H=1,2,4,8 weeks with 1.0 benchmark line
4. Feature Importance Shift: Grouped horizontal bar chart of M2 GBT predictor importances (H=1w vs H=8w)
5. Zero-Shot Spatial Transfer: Grouped bar chart of NSE at every lead on holdout basins (Kantamal, Bamni, Haralahalli) for M0, M1, M2 LSTM

All numbers come from data/processed/model_results/test_predictions.csv and audited_metrics_recomputed.csv,
written by scripts/audit_metrics.py. Run that script first.

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
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, 'visualizations', 'source')

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

PREDICTIONS_CSV = os.path.join(RESULTS_DIR, 'test_predictions.csv')
AUDITED_CSV = os.path.join(RESULTS_DIR, 'audited_metrics_recomputed.csv')

# Stations whose recorded zeros are unmeasured weeks (single definition in scripts/eval_config.py).
sys.path.insert(0, SCRIPT_DIR)
from eval_config import ZERO_AS_MISSING_STATIONS  # noqa: E402
ALL_STATIONS = [
    ('Karad', 4021), ('Warunji', 4063), ('Ashti', 3005), ('Bamni (Ballarsha)', 3008),
    ('Anandpur', 7002), ('Kashinagar', 16007), ('Gomlai', 7007), ('Kantamal', 8013),
    ('Yadgir', 4064), ('Huvinhedigi', 4017), ('KUDIGE', 5010), ('Haralahalli', 4013),
]
FEATURE_GROUPS = ['Streamflow Memory', 'Precipitation & Recharge', 'Atmospheric Demand',
                  'Climatic Water Deficit', 'Calendar Seasonality']


def _load_predictions():
    """Per-week test predictions written by scripts/audit_metrics.py (single source of truth)."""
    if not os.path.isfile(PREDICTIONS_CSV):
        raise FileNotFoundError(f"{PREDICTIONS_CSV} missing: run scripts/audit_metrics.py first")
    return pd.read_csv(PREDICTIONS_CSV, parse_dates=['issue_date', 'target_date'])


def _load_audited():
    return pd.read_csv(AUDITED_CSV)


def load_test_predictions(station_name, horizon, preds=None, audited=None):
    """
    Test-block (2016–2025) series for one station and lead, taken from the audited
    prediction table. Weeks not scored by the audit (missing data, or recorded zeros at
    ZERO_AS_MISSING_STATIONS) are returned as NaN so they plot as gaps.
    Dates are TARGET dates: the week the forecast refers to.
    """
    preds = _load_predictions() if preds is None else preds
    audited = _load_audited() if audited is None else audited
    d = preds[(preds['station'] == station_name) & (preds['horizon'] == f'{horizon}w')]
    scored = d['scored'].values.astype(bool)
    row = audited[(audited['station'] == station_name) & (audited['horizon'] == f'{horizon}w')].iloc[0]
    blank = lambda col: np.where(scored, d[col].values, np.nan)
    return {
        'dates': d['target_date'],
        'y_actual': blank('y_obs'),
        'preds_m1': blank('m1_gbt'),
        'preds_m2': blank('m2_gbt'),
        'scored': scored,
        'n_masked': int(d['zero_as_missing'].sum()),
        'q95': float(d['q95'].iloc[0]),
        'q90': float(d['q90'].iloc[0]),
        'nse_m1': float(row['m1_nse']),
        'nse_m2': float(row['m2_nse']),
    }

# ==============================================================================
# 3. PLOT 1: INTERACTIVE HYDROGRAPHS WITH RANGESLIDER
# ==============================================================================
def generate_interactive_hydrographs(out_dir):
    """
    Interactive hydrographs for Karad and Kantamal at H=1w and H=4w, drawn from the
    audited prediction table. NSE labels equal the audited CSV values.
    """
    print("[*] Generating Plot 1: Interactive Hydrographs with Range Slider...")
    preds, audited = _load_predictions(), _load_audited()
    stations = [
        ('Karad', 'Maharashtra (Upper Krishna Basin)'),
        ('Kantamal', 'Odisha (Tel / Mahanadi Basin)'),
    ]
    saved_files = []
    panels = {}
    for stn_name, basin_desc in stations:
        for H in [1, 4]:
            data = load_test_predictions(stn_name, H, preds, audited)
            panels[(stn_name, H)] = data
            gap_note = (f" | {data['n_masked']} weeks recorded as 0 m³/s treated as unobserved (gaps)"
                        if stn_name in ZERO_AS_MISSING_STATIONS else "")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['dates'], y=data['y_actual'], name='Observed Streamflow (Q)', mode='lines',
                line=dict(color=COLOR_OBSERVED, width=2.2), connectgaps=False,
                hovertemplate='<b>Observed Q</b>: %{y:.2f} m³/s<br>Week: %{x|%Y-%m-%d}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=data['dates'], y=data['preds_m1'], name=f"M1: Flow-Only GBT (NSE: {data['nse_m1']:.3f})",
                mode='lines', line=dict(color=COLOR_M1, width=1.8, dash='dash'), connectgaps=False,
                customdata=data['preds_m1'] - data['y_actual'],
                hovertemplate='<b>M1 Forecast</b>: %{y:.2f} m³/s<br>Error: %{customdata:.2f} m³/s<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=data['dates'], y=data['preds_m2'], name=f"M2: Hydro-Met GBT (NSE: {data['nse_m2']:.3f})",
                mode='lines', line=dict(color=COLOR_M2, width=2.0), connectgaps=False,
                customdata=data['preds_m2'] - data['y_actual'],
                hovertemplate='<b>M2 Forecast</b>: %{y:.2f} m³/s<br>Error: %{customdata:.2f} m³/s<extra></extra>'
            ))
            fig.add_hline(y=data['q95'], line_dash='dot', line_color=COLOR_Q95, line_width=1.8,
                          annotation_text=f"Severe Drought (Q95: {data['q95']:.2f} m³/s)", annotation_position="top left",
                          annotation_font=dict(color=COLOR_Q95, size=11))
            fig.add_hline(y=data['q90'], line_dash='dashdot', line_color=COLOR_Q90, line_width=1.4,
                          annotation_text=f"Moderate Drought (Q90: {data['q90']:.2f} m³/s)", annotation_position="bottom left",
                          annotation_font=dict(color=COLOR_Q90, size=10))
            fig.update_layout(
                title=dict(
                    text=f"<b>Interactive Hydrograph: {stn_name} ({basin_desc})</b><br>"
                         f"<span style='font-size: 13px; color: #4B5563;'>"
                         f"Lead H = {H} week{'s' if H > 1 else ''} | Test block 2016–2025, plotted at the forecast target week{gap_note}"
                         f"</span>", x=0.03, y=0.96),
                xaxis=dict(title="Target week", rangeslider=dict(visible=True, thickness=0.08, bgcolor='#F3F4F6'),
                           range=['2018-01-01', '2019-12-31'], showgrid=True, gridcolor='#E5E7EB'),
                yaxis=dict(title="Discharge Rate (m³/s)", showgrid=True, gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#9CA3AF'),
                template='plotly_white', hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                margin=dict(l=60, r=40, t=110, b=60), height=560
            )
            out_file = os.path.join(out_dir, f"interactive_hydrograph_{stn_name.lower()}_h{H}w.html")
            fig.write_html(out_file, include_plotlyjs='cdn')
            saved_files.append(out_file)
            print(f"   [+] Saved standalone hydrograph: {out_file}")

    fig_dash = make_subplots(rows=2, cols=2, vertical_spacing=0.12, horizontal_spacing=0.08, subplot_titles=[
        "<b>Karad — 1-Week Ahead</b>", "<b>Kantamal — 1-Week Ahead</b>",
        "<b>Karad — 4-Weeks Ahead</b>", "<b>Kantamal — 4-Weeks Ahead</b>"])
    for (stn_name, H), (row, col) in zip([('Karad', 1), ('Kantamal', 1), ('Karad', 4), ('Kantamal', 4)],
                                         [(1, 1), (1, 2), (2, 1), (2, 2)]):
        data = panels[(stn_name, H)]
        show_leg = (row == 1 and col == 1)
        for y, nm, ln in ((data['y_actual'], 'Observed (Q)', dict(color=COLOR_OBSERVED, width=1.8)),
                          (data['preds_m1'], 'M1: Flow-Only GBT', dict(color=COLOR_M1, width=1.5, dash='dash')),
                          (data['preds_m2'], 'M2: Hydro-Met GBT', dict(color=COLOR_M2, width=1.8))):
            fig_dash.add_trace(go.Scatter(x=data['dates'], y=y, name=nm, mode='lines', line=ln,
                                          connectgaps=False, showlegend=show_leg), row=row, col=col)
        fig_dash.add_hline(y=data['q95'], line_dash='dot', line_color=COLOR_Q95, line_width=1.2, row=row, col=col)
    fig_dash.update_layout(
        title=dict(text="<b>Hydrograph Dashboard (Observed vs M1 vs M2)</b><br>"
                        "<span style='font-size: 13px; color: #4B5563;'>Dotted amber line: Q95 severe-drought threshold. "
                        "Gaps at Karad are weeks recorded as 0 m³/s (unmeasured), excluded from scoring.</span>", x=0.03, y=0.98),
        template='plotly_white', height=850, margin=dict(l=60, r=40, t=90, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0))
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
    Pooled precision-recall curves for severe drought (Q95) at H=2w and H=4w over all
    scored test weeks of the 12 stations. Score = negated forecast flow.
    PR-AUC is reported as average precision.
    """
    print("[*] Generating Plot 2: Interactive Precision-Recall Curves (Q95, H=2w & H=4w)...")
    preds = _load_predictions()
    horizons = [2, 4]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10, subplot_titles=[
        "<b>H = 2 Weeks Ahead (Severe Drought Q95)</b>", "<b>H = 4 Weeks Ahead (Severe Drought Q95)</b>"])
    summary = []
    for col_idx, H in enumerate(horizons, start=1):
        d = preds[(preds['horizon'] == f'{H}w') & preds['scored']]
        y_true = (d['y_obs'] <= d['q95']).astype(int).values
        prevalence = float(y_true.mean())
        for key, color, width, label in (('m1_gbt', COLOR_M1, 2.4, 'M1: Flow-Only'),
                                         ('m2_gbt', COLOR_M2, 2.6, 'M2: Hydro-Met')):
            score = -d[key].values
            p, r, t = precision_recall_curve(y_true, score)
            ap = average_precision_score(y_true, score)
            summary.append(f"H={H}w {label} AP={ap:.3f}")
            cdata = np.stack((np.append(-t, -t[-1]), np.maximum(0.0, 1.0 - p)), axis=-1)
            fig.add_trace(go.Scatter(
                x=r, y=p, name=f"{label} (PR-AUC: {ap:.3f})", mode='lines', line=dict(color=color, width=width),
                customdata=cdata, legendgroup=f"{key}_h{H}",
                hovertemplate=(f"<b>{label}</b><br>Recall (POD): %{{x:.3f}}<br>Precision (1-FAR): %{{y:.3f}}<br>"
                               "Score cutoff (forecast flow): %{customdata[0]:.2f} m³/s<br>FAR: %{customdata[1]:.3f}<extra></extra>")
            ), row=1, col=col_idx)
        fig.add_trace(go.Scatter(x=[0, 1.0], y=[prevalence, prevalence], name=f"No-Skill Baseline ({prevalence:.3f})",
                                 mode='lines', line=dict(color='#6B7280', width=1.5, dash='dash'), hoverinfo='none',
                                 showlegend=(col_idx == 1)), row=1, col=col_idx)
        fig.update_xaxes(title_text="Recall / Probability of Detection (POD)", range=[0, 1.02], row=1, col=col_idx, gridcolor='#E5E7EB')
        fig.update_yaxes(title_text="Precision (1 - False Alarm Ratio)", range=[0, 1.02], row=1, col=col_idx, gridcolor='#E5E7EB')
    print("   " + " | ".join(summary))
    fig.update_layout(
        title=dict(text="<b>Precision-Recall Curves for Severe Hydrological Drought (Q95)</b><br>"
                        "<span style='font-size: 13px; color: #4B5563;'>Pooled over all scored test weeks of the 12 stations. "
                        "PR-AUC = average precision. No-skill line = pooled event base rate.</span>", x=0.03, y=0.96),
        template='plotly_white', hovermode='closest', height=540, margin=dict(l=60, r=40, t=100, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))
    out_file = os.path.join(out_dir, "interactive_pr_curves_severe_drought.html")
    fig.write_html(out_file, include_plotlyjs='cdn')
    print(f"   [+] Saved interactive PR curves: {out_file}")
    return out_file

# ==============================================================================
# 5. PLOT 3: KGE COMPONENT DECOMPOSITION (r, alpha, beta ACROSS HORIZONS)
# ==============================================================================
def generate_interactive_kge_decomposition(out_dir):
    """Median KGE components (r, alpha, beta) over the 12 stations, from the audited CSV."""
    print("[*] Generating Plot 3: Interactive KGE Component Decomposition (r, α, β)...")
    df = _load_audited()
    hz_labels = ['1w', '2w', '4w', '8w']
    med = df.groupby('horizon')[[f'{m}_{c}' for m in ('m0', 'm1', 'm2') for c in ('r', 'alpha', 'beta')]].median().loc[hz_labels]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08, subplot_titles=[
        "<b>(a) Correlation (r)</b>", "<b>(b) Variability Ratio (α = σ<sub>pred</sub> / σ<sub>obs</sub>)</b>",
        "<b>(c) Bias Ratio (β = μ<sub>pred</sub> / μ<sub>obs</sub>)</b>"])
    styles = {'m0': ('M0: Baseline', COLOR_M0, 'dash', 'square'), 'm1': ('M1: Flow-Only GBT', COLOR_M1, 'solid', 'diamond'),
              'm2': ('M2: Hydro-Met GBT', COLOR_M2, 'solid', 'circle')}
    for col_idx, comp in enumerate(('r', 'alpha', 'beta'), start=1):
        show_leg = col_idx == 1
        fig.add_trace(go.Scatter(x=hz_labels, y=[1.0] * 4, name='Ideal Benchmark (1.0)', mode='lines',
                                 line=dict(color=COLOR_IDEAL, width=1.6, dash='dot'), showlegend=show_leg), row=1, col=col_idx)
        for m, (nm, color, dash, sym) in styles.items():
            vals = med[f'{m}_{comp}'].values
            fig.add_trace(go.Scatter(x=hz_labels, y=vals, name=nm, mode='lines+markers', line=dict(color=color, width=2.2, dash=dash),
                                     marker=dict(size=8, symbol=sym), showlegend=show_leg,
                                     hovertemplate=f"<b>{nm}</b><br>Lead %{{x}}<br>Median {comp}: %{{y:.3f}}<extra></extra>"),
                          row=1, col=col_idx)
        fig.update_xaxes(title_text="Forecast lead", row=1, col=col_idx, gridcolor='#E5E7EB')
        fig.update_yaxes(title_text="Component value", range=[-0.05, 1.25], row=1, col=col_idx, gridcolor='#E5E7EB')
    r8 = med.loc['8w']
    fig.update_layout(
        title=dict(text="<b>KGE Component Decomposition Across Lead Times (12-station medians)</b><br>"
                        f"<span style='font-size: 13px; color: #4B5563;'>At 8 weeks the median correlation is "
                        f"{r8['m0_r']:.3f} for M0 climatology, {r8['m1_r']:.3f} for M1 and {r8['m2_r']:.3f} for M2.</span>",
                   x=0.03, y=0.96),
        template='plotly_white', hovermode='closest', height=540, margin=dict(l=60, r=40, t=110, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))
    out_file = os.path.join(out_dir, "interactive_kge_component_decomposition.html")
    fig.write_html(out_file, include_plotlyjs='cdn')
    print(f"   [+] Saved KGE decomposition plot: {out_file}")
    return out_file

# ==============================================================================
# 6. PLOT 4: FEATURE IMPORTANCE (POOLED OVER STATIONS WITH COMPLETE LOW-FLOW RECORDS)
# ==============================================================================
def _station_importance(stn_name, gid, H, preds, n_repeats=10, seed=42):
    """
    Permutation importance of the M2 GBT on the audited scored test weeks of one station.
    Returns (per-feature share in %, per-group ΔMSE as a fraction of baseline MSE).
    Group importance permutes all features of a family together, so correlated features
    (e.g. flow lags) cannot hide each other's contribution.
    """
    clean_name = stn_name.replace(' ', '_').replace('(', '').replace(')', '')
    model, _, medians = load_estimator(os.path.join(MODELS_DIR, 'protocol1', f"m2_gbt_gauge_{gid:05d}_{clean_name}_H{H}.joblib"))
    df = pd.read_csv(os.path.join(TENSORS_DIR, f"weekly_tensor_gauge_{gid:05d}_{stn_name}.csv"), parse_dates=['date']).set_index('date')
    d = preds[(preds['station'] == stn_name) & (preds['horizon'] == f'{H}w') & preds['scored']]
    X = df.loc[pd.to_datetime(d['issue_date']).values, M2_TABULAR_FEATURES]
    X = X.fillna(medians if medians is not None else X.median()).values
    y = d['y_obs'].values
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed, scoring='neg_mean_squared_error')
    imp = np.maximum(0.0, r.importances_mean)
    share = pd.Series(100.0 * imp / imp.sum() if imp.sum() > 0 else imp, index=M2_TABULAR_FEATURES)
    base_mse = float(np.mean((y - model.predict(X)) ** 2))
    rng = np.random.default_rng(seed)
    group_gain = {}
    for g in FEATURE_GROUPS:
        cols = [i for i, f in enumerate(M2_TABULAR_FEATURES) if categorize_feature(f) == g]
        incs = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            incs.append(np.mean((y - model.predict(Xp)) ** 2) - base_mse)
        group_gain[g] = float(np.mean(incs)) / base_mse
    return share, pd.Series(group_gain)


def generate_interactive_feature_importance(out_dir):
    """
    M2 GBT permutation importance at H=1w vs H=8w, averaged over the 8 stations with
    complete low-flow records (Karad, Warunji, Yadgir and Huvinhedigi excluded because
    their test weeks are dominated by unmeasured zeros).
    """
    print("[*] Generating Plot 4: Feature Importance (pooled, H=1w vs H=8w)...")
    preds = _load_predictions()
    stations = [(s, g) for s, g in ALL_STATIONS if s not in ZERO_AS_MISSING_STATIONS]
    shares, groups = {}, {}
    for H in [1, 8]:
        per_stn = [_station_importance(s, g, H, preds) for s, g in stations]
        shares[H] = pd.concat([p[0] for p in per_stn], axis=1).mean(axis=1)
        groups[H] = pd.concat([p[1] for p in per_stn], axis=1).median(axis=1)
    df_imp = pd.DataFrame({'H1': shares[1], 'H8': shares[8]})
    df_top = df_imp.assign(Max_Imp=df_imp.max(axis=1)).sort_values('Max_Imp', ascending=True)
    pd.DataFrame({'H1_share_pct': shares[1], 'H8_share_pct': shares[8]}).to_csv(os.path.join(RESULTS_DIR, 'feature_importance_pooled.csv'))
    pd.DataFrame({'H1_dMSE_over_MSE': groups[1], 'H8_dMSE_over_MSE': groups[8]}).to_csv(os.path.join(RESULTS_DIR, 'feature_importance_groups.csv'))
    print("   Group importance (median ΔMSE / MSE when the whole family is permuted):")
    print(pd.DataFrame({'1w': groups[1], '8w': groups[8]}).round(3).to_string())

    names = [FEATURE_LABELS.get(f, f) for f in df_top.index]
    cats = [categorize_feature(f) for f in df_top.index]
    fig = go.Figure()
    for col, nm, color, line in (('H1', 'Horizon H = 1 Week', '#2563EB', '#1E40AF'), ('H8', 'Horizon H = 8 Weeks', '#DC2626', '#991B1B')):
        fig.add_trace(go.Bar(y=names, x=df_top[col], name=nm, orientation='h',
                             marker=dict(color=color, line=dict(color=line, width=1)),
                             customdata=np.stack((cats, df_top[col].values), axis=-1),
                             hovertemplate="<b>%{y}</b><br>Group: %{customdata[0]}<br>Mean share: <b>%{x:.1f}%</b><extra></extra>"))
    g1, g8 = groups[1], groups[8]
    fig.update_layout(
        title=dict(text="<b>M2 Permutation Importance: 1-Week vs 8-Week Lead (mean over 8 stations)</b><br>"
                        "<span style='font-size: 13px; color: #4B5563;'>Excludes Karad, Warunji, Yadgir, Huvinhedigi (unmeasured zero weeks). "
                        f"Permuting all flow features together raises MSE by {100*g1['Streamflow Memory']:.0f}% at 1 week and "
                        f"{100*g8['Streamflow Memory']:.0f}% at 8 weeks; all precipitation features: "
                        f"{100*g1['Precipitation & Recharge']:.0f}% and {100*g8['Precipitation & Recharge']:.0f}% (station medians).</span>",
                   x=0.03, y=0.96),
        barmode='group', template='plotly_white',
        xaxis=dict(title="Share of permutation importance (%)", showgrid=True, gridcolor='#E5E7EB'),
        yaxis=dict(title="Predictor", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        height=850, margin=dict(l=220, r=40, t=110, b=60))
    out_file = os.path.join(out_dir, "interactive_feature_importance_shift.html")
    fig.write_html(out_file, include_plotlyjs='cdn')
    print(f"   [+] Saved feature importance: {out_file}")
    return out_file

# ==============================================================================
# 7. PLOT 5: ZERO-SHOT SPATIAL TRANSFER (ALL FOUR LEADS)
# ==============================================================================
def generate_interactive_spatial_transfer(out_dir):
    """
    Zero-shot NSE on the three held-out basins at every lead, from the audited CSV:
    M0 climatology/persistence (station's own history), M1 pooled GBT, M2 pooled HydroLSTM.
    """
    print("[*] Generating Plot 5: Interactive Zero-Shot Spatial Generalization...")
    df = _load_audited()
    holdouts = [('Kantamal', 'Odisha'), ('Bamni (Ballarsha)', 'Maharashtra'), ('Haralahalli', 'Karnataka')]
    horizons = [1, 2, 4, 8]
    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.05,
                        subplot_titles=[f"<b>H = {H} Week{'s' if H > 1 else ''}</b>" for H in horizons])
    for col_idx, H in enumerate(horizons, start=1):
        sub = df[df['horizon'] == f'{H}w'].set_index('station')
        x = [f"<b>{s.split()[0]}</b><br><span style='font-size:10px;color:#6B7280;'>{st}</span>" for s, st in holdouts]
        for col, nm, color in (('m0_nse_fixed', 'M0: Baseline (Persistence / Climatology)', COLOR_M0),
                               ('p2_gbt_nse', 'M1: Flow-Only GBT (pooled)', COLOR_M1),
                               ('p2_lstm_nse', 'M2: PyTorch HydroLSTM (pooled)', COLOR_LSTM)):
            vals = [float(sub.loc[s, col]) for s, _ in holdouts]
            fig.add_trace(go.Bar(x=x, y=vals, name=nm, marker=dict(color=color), text=[f"{v:+.2f}" for v in vals],
                                 textposition='auto', showlegend=(col_idx == 1),
                                 hovertemplate=f"<b>{nm}</b><br>%{{x}}<br>NSE: <b>%{{y:+.3f}}</b><extra></extra>"),
                          row=1, col=col_idx)
        fig.add_hline(y=0.0, line_dash='dash', line_color='#374151', line_width=1.5, row=1, col=col_idx)
        fig.update_yaxes(range=[-1.45, 0.75], row=1, col=col_idx, gridcolor='#E5E7EB')
    fig.update_yaxes(title_text="Zero-shot NSE", row=1, col=1)
    fig.update_layout(
        title=dict(text="<b>Protocol 2: Zero-Shot Transfer to Held-Out Basins</b><br>"
                        "<span style='font-size: 13px; color: #4B5563;'>Held-out basins were excluded from training. "
                        "M0 uses the held-out station's own flow history (persistence at 1–2 weeks, climatology at 4–8 weeks).</span>",
                   x=0.03, y=0.96),
        barmode='group', template='plotly_white', hovermode='closest', height=540, margin=dict(l=60, r=40, t=105, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))
    out_file = os.path.join(out_dir, "interactive_spatial_transfer_grouped_bars.html")
    fig.write_html(out_file, include_plotlyjs='cdn')
    print(f"   [+] Saved interactive zero-shot transfer: {out_file}")
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
