#!/usr/bin/env python3
"""
Audit CAMELS-IND vs CWC 2011–2020 Daily Overlap Lineage.

Verifies whether post-2010 zero-flow inflation at Karad, Warunji, Yadgir, and Huvinhedigi
originated from the pipeline merge (s_cwc.combine_first(s_camels)) or was already
present in upstream published CAMELS-IND.

Outputs:
  data/processed/validation_reports/overlap_lineage_audit.csv
"""
from pathlib import Path
import os
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "data" / "processed" / "validation_reports"
OUTPUT_CSV = REPORT_DIR / "overlap_lineage_audit.csv"
CWC_MERGED = ROOT / "data" / "processed" / "merged" / "cwc_imd_merged_2001_2025.csv"
RAW_CAMELS = ROOT / "data" / "raw" / "camels_ind" / "streamflow_timeseries" / "streamflow_observed.csv"
ALT_CAMELS = Path("/Users/abhitej/Downloads/ML Project 2/data/raw/camels_ind/streamflow_timeseries/streamflow_observed.csv")
TRASH_ZIP = Path(os.path.expanduser("~/.Trash/ML Project.zip"))

STATIONS = [
    ('Karad', '4021'),
    ('Warunji', '4063'),
    ('Yadgir', '4064'),
    ('Huvinhedigi', '4017'),
]

def load_camels():
    if RAW_CAMELS.is_file():
        return pd.read_csv(RAW_CAMELS)
    if ALT_CAMELS.is_file():
        return pd.read_csv(ALT_CAMELS)
    if TRASH_ZIP.is_file():
        try:
            with zipfile.ZipFile(TRASH_ZIP, 'r') as z:
                name = 'ML Project/data/raw/camels_ind/streamflow_timeseries/streamflow_observed.csv'
                if name in z.namelist():
                    with z.open(name) as f:
                        return pd.read_csv(f)
        except Exception:
            pass
    return None

def main():
    print("=" * 80)
    print(" AUDIT: CAMELS-IND vs CWC 2011–2020 DAILY OVERLAP LINEAGE")
    print("=" * 80)

    if not CWC_MERGED.is_file():
        print(f"[-] CWC merged dataset not found: {CWC_MERGED}")
        return

    df_camels = load_camels()
    if df_camels is None:
        print("[!] data/raw/camels_ind not found in repository.")
        print("[!] Overlap lineage figures are reproducible when data/raw/camels_ind is provided.")
        if OUTPUT_CSV.is_file():
            print(f"[+] Existing audit artifact preserved at: {OUTPUT_CSV}")
        return

    print("[*] Processing CAMELS-IND and CWC records over 2011-01-01 to 2020-12-31...")
    df_camels['dt'] = pd.to_datetime(df_camels[['year', 'month', 'day']]).dt.normalize()
    df_camels = df_camels.set_index('dt')

    df_cwc = pd.read_csv(CWC_MERGED)
    df_cwc['dt'] = pd.to_datetime(df_cwc['Date']).dt.normalize()

    timeline = pd.date_range('2011-01-01', '2020-12-31', freq='D')
    rows = []

    for stn, gid in STATIONS:
        sub = df_cwc[df_cwc['Station'].str.strip().str.upper() == stn.upper()].drop_duplicates('dt').set_index('dt')
        q_cwc = sub['Manual Daily River Water Discharge (m3/sec)'].reindex(timeline)
        q_cam = df_camels[gid].reindex(timeline)

        is_zero_cwc = (q_cwc == 0.0)
        total_zero_cwc = int(is_zero_cwc.sum())

        cam_on_zero = q_cam[is_zero_cwc]
        cam_pos = int((cam_on_zero > 0.0).sum())
        cam_zero = int((cam_on_zero == 0.0).sum())
        cam_nan = int(cam_on_zero.isna().sum())

        rows.append({
            'station': stn,
            'gauge_id': gid,
            'cwc_zero_days': total_zero_cwc,
            'camels_zero_days': cam_zero,
            'camels_nan_days': cam_nan,
            'camels_positive_days': cam_pos,
            'camels_positive_pct': (cam_pos / total_zero_cwc * 100) if total_zero_cwc > 0 else 0.0,
            'camels_zero_or_nan_pct': ((cam_zero + cam_nan) / total_zero_cwc * 100) if total_zero_cwc > 0 else 0.0
        })

    df_out = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Saved overlap lineage audit to: {OUTPUT_CSV}")
    print(df_out.to_string(index=False))

    tot_cwc = df_out['cwc_zero_days'].sum()
    tot_zero = df_out['camels_zero_days'].sum()
    tot_nan = df_out['camels_nan_days'].sum()
    tot_pos = df_out['camels_positive_days'].sum()
    print("-" * 80)
    print(f"Total CWC-zero days: {tot_cwc}")
    print(f"CAMELS exact 0.0:    {tot_zero} ({tot_zero/tot_cwc*100:.1f}%)")
    print(f"CAMELS missing NaN:  {tot_nan} ({tot_nan/tot_cwc*100:.1f}%)")
    print(f"CAMELS positive:     {tot_pos} ({tot_pos/tot_cwc*100:.2f}%)")
    print("=" * 80)

if __name__ == '__main__':
    main()
