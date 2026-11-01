#!/usr/bin/env python3
"""
IMD & CWC Alignment Test CLI Suite
Allows running the meteorological alignment test for ANY set of stations in CWC MH.
Default stations fallback to Warunji, Karad, Ashti if none are specified.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import imdlib

# Determine base project root relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

CATALOG_FILE = os.path.join(PROJECT_ROOT, 'data/processed/station_catalog.csv')
CWC_RAW_DIR = os.path.join(PROJECT_ROOT, 'data/raw/cwc')
DEFAULT_CWC_FILE = os.path.join(CWC_RAW_DIR, 'river_discharge_manual_daily_cwc_mh_2001_2025.csv')
DEFAULT_IMD_DIR = os.path.join(PROJECT_ROOT, 'data/raw/imd')
DEFAULT_PROCESSED_STATION_DIR = os.path.join(PROJECT_ROOT, 'data/processed/station_rainfall')
DEFAULT_MERGED_DIR = os.path.join(PROJECT_ROOT, 'data/processed/merged')

START_YEAR = 2001
END_YEAR = 2025
MAX_MISSING_RATE_THRESHOLD = 0.15
MAX_CONTIGUOUS_GAP_THRESHOLD = 30

def load_cwc_data(stations, cwc_file):
    """
    Loads CWC discharge observations for target stations.
    If cwc_file is explicitly provided, loads from that file.
    Otherwise, automatically resolves and merges across state files using station_catalog.csv.
    """
    if cwc_file and os.path.isfile(cwc_file):
        df_raw = pd.read_csv(cwc_file, low_memory=False)
        available = df_raw['Station'].dropna().unique().tolist()
        # Check if all stations are present
        if all(s in available for s in stations):
            selected = [s for s in stations if s in available]
            return df_raw[df_raw['Station'].isin(selected)].copy(), cwc_file

    # Auto-resolve using station_catalog.csv
    if os.path.isfile(CATALOG_FILE):
        cat = pd.read_csv(CATALOG_FILE)
        stn_map = {}
        for s in stations:
            match = cat[cat['Station'].str.strip().str.upper() == s.strip().upper()]
            if len(match) > 0:
                canonical_name = match.iloc[0]['Station']
                src_file = match.iloc[0]['Source_File']
                stn_map[canonical_name] = src_file
        
        if stn_map:
            dfs = []
            files_used = set()
            for src_file in set(stn_map.values()):
                full_path = os.path.join(CWC_RAW_DIR, src_file)
                if os.path.isfile(full_path):
                    raw = pd.read_csv(full_path, low_memory=False)
                    needed = [stn for stn, sf in stn_map.items() if sf == src_file]
                    sub = raw[raw['Station'].isin(needed)]
                    dfs.append(sub)
                    files_used.add(src_file)
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                desc = ", ".join(sorted(files_used))
                return combined, f"Auto-resolved from catalog ({desc})"

    # Fallback to default cwc_file if catalog resolution was not needed
    if cwc_file and os.path.isfile(cwc_file):
        df_raw = pd.read_csv(cwc_file, low_memory=False)
        selected = [s for s in stations if s in df_raw['Station'].dropna().unique().tolist()]
        return df_raw[df_raw['Station'].isin(selected)].copy(), cwc_file

    print(f"[-] ERROR: CWC discharge file not found at {cwc_file}")
    sys.exit(1)

def run_alignment_test(stations, cwc_file, imd_dir, processed_station_dir, merged_dir):
    print("=" * 78)
    print("      DATA VALIDATION SUITE: IMD METEOROLOGICAL ALIGNMENT TEST")
    print("=" * 78)
    print(f"Target Stations : {', '.join(stations)}")
    print(f"Temporal Window : {START_YEAR}-01-01 to {END_YEAR}-12-31 (25 Years)")
    print(f"IMD Grids Dir   : {imd_dir}")
    print("-" * 78)

    # 1. Load and filter CWC
    print("[1/5] Loading and filtering CWC discharge observations...")
    df_cwc, resolved_src = load_cwc_data(stations, cwc_file)
    print(f"      Source: {resolved_src}")
    available_stns = df_cwc['Station'].dropna().unique().tolist()

    selected = [s for s in stations if s in available_stns]
    missing = [s for s in stations if s not in available_stns]

    if missing:
        print(f"[-] WARNING: Stations not found in CWC dataset: {missing}")
    if not selected:
        print("[-] ERROR: None of the target stations found in the CWC file.")
        sys.exit(1)

    # 2. Extract coordinates
    print("[2/5] Extracting station coordinates...")
    coords = {}
    for stn in selected:
        stn_sub = df_cwc[df_cwc['Station'] == stn]
        lat = float(stn_sub['Latitude'].iloc[0])
        lon = float(stn_sub['Longitude'].iloc[0])
        coords[stn] = {'Latitude': lat, 'Longitude': lon}
        print(f"      - {stn:<15}: Latitude = {lat:.6f}, Longitude = {lon:.6f}")

    # 3. Load IMD
    print(f"\n[3/5] Loading IMD daily gridded rainfall data ({START_YEAR}-{END_YEAR})...")
    data = imdlib.open_data('rain', START_YEAR, END_YEAR, file_dir=imd_dir)

    dates = pd.date_range(start=f"{START_YEAR}-01-01", end=f"{END_YEAR}-12-31", freq='D')
    imd_records = []
    os.makedirs(processed_station_dir, exist_ok=True)

    for stn, coord in coords.items():
        lat, lon = coord['Latitude'], coord['Longitude']
        lat_idx, lon_idx = imdlib.util.get_lat_lon(lat, lon, data.lat_array, data.lon_array)
        grid_lat = float(data.lat_array[lat_idx])
        grid_lon = float(data.lon_array[lon_idx])
        rf_series = data.data[:, lon_idx, lat_idx]

        stn_df = pd.DataFrame({
            'Date': dates,
            'Station': stn,
            'Latitude': lat,
            'Longitude': lon,
            'Grid_Latitude': grid_lat,
            'Grid_Longitude': grid_lon,
            'IMD_Rainfall_mm': rf_series
        })
        imd_records.append(stn_df)

        stn_csv = os.path.join(processed_station_dir, f"imd_rainfall_{stn}_{START_YEAR}_{END_YEAR}.csv")
        stn_df.to_csv(stn_csv, index=False)
        print(f"      - {stn:<15}: Nearest grid ({grid_lat:.2f}, {grid_lon:.2f}) -> {len(stn_df)} daily records saved")

    df_imd = pd.concat(imd_records, ignore_index=True)

    # 4. Parse CWC Dates
    print("\n[4/5] Parsing CWC 'Data Acquisition Time' (dayfirst=True)...")
    df_cwc['Data Acquisition Time'] = pd.to_datetime(df_cwc['Data Acquisition Time'], dayfirst=True)
    df_cwc['Date'] = df_cwc['Data Acquisition Time'].dt.normalize()

    # 5. Merge
    print("[5/5] Merging IMD daily rainfall with CWC daily discharge...")
    merged_df = pd.merge(
        df_cwc,
        df_imd,
        on=['Date', 'Station', 'Latitude', 'Longitude'],
        how='inner'
    )

    os.makedirs(merged_dir, exist_ok=True)
    merged_output_path = os.path.join(merged_dir, f"cwc_imd_merged_{START_YEAR}_{END_YEAR}.csv")
    merged_df.to_csv(merged_output_path, index=False)
    print(f"      -> Merged dataset saved to: {merged_output_path}")

    # 6. Evaluation
    print("\n" + "=" * 78)
    print("                    IMD METEOROLOGICAL VALIDATION REPORT")
    print("=" * 78)

    imd_min_date = df_imd['Date'].min()
    imd_max_date = df_imd['Date'].max()
    cwc_min_date = df_cwc['Date'].min()
    cwc_max_date = df_cwc['Date'].max()

    temporal_overlap_pass = (
        imd_min_date <= cwc_min_date and 
        imd_max_date >= cwc_max_date and
        imd_min_date == pd.Timestamp(f"{START_YEAR}-01-01") and
        imd_max_date == pd.Timestamp(f"{END_YEAR}-12-31")
    )

    total_merged_rows = len(merged_df)
    total_missing_rf = merged_df['IMD_Rainfall_mm'].isna().sum() + (merged_df['IMD_Rainfall_mm'] < 0).sum()
    missing_rf_rate = total_missing_rf / total_merged_rows if total_merged_rows > 0 else 1.0
    missingness_pass = missing_rf_rate < MAX_MISSING_RATE_THRESHOLD

    max_weather_gap_per_stn = {}
    contiguous_gap_pass = True
    for stn in selected:
        stn_imd = df_imd[df_imd['Station'] == stn].sort_values('Date').reset_index(drop=True)
        is_missing = stn_imd['IMD_Rainfall_mm'].isna() | (stn_imd['IMD_Rainfall_mm'] < 0)
        gap_sizes = is_missing.groupby((~is_missing).cumsum()).sum() if is_missing.any() else pd.Series([0])
        max_gap = int(gap_sizes.max())
        max_weather_gap_per_stn[stn] = max_gap
        if max_gap > MAX_CONTIGUOUS_GAP_THRESHOLD:
            contiguous_gap_pass = False

    station_match_counts = {}
    station_total_cwc = {}
    merge_success_pass = True
    for stn in selected:
        n_cwc = len(df_cwc[df_cwc['Station'] == stn])
        n_merged = len(merged_df[merged_df['Station'] == stn])
        station_total_cwc[stn] = n_cwc
        station_match_counts[stn] = n_merged
        if n_merged != n_cwc:
            merge_success_pass = False

    overall_pass = (
        temporal_overlap_pass and
        missingness_pass and
        contiguous_gap_pass and
        merge_success_pass
    )

    def status_tag(p):
        return "[PASS]" if p else "[FAIL]"

    print(f"\n1. TEMPORAL OVERLAP {status_tag(temporal_overlap_pass):>57}")
    print(f"   - Target Range            : {START_YEAR}-01-01 to {END_YEAR}-12-31")
    print(f"   - IMD Coverage Window     : {imd_min_date.strftime('%Y-%m-%d')} to {imd_max_date.strftime('%Y-%m-%d')} ({(imd_max_date - imd_min_date).days + 1} days)")
    print(f"   - CWC Discharge Window    : {cwc_min_date.strftime('%Y-%m-%d')} to {cwc_max_date.strftime('%Y-%m-%d')}")
    print(f"   - Full Temporal Match     : {'YES' if temporal_overlap_pass else 'NO'}")

    print(f"\n2. WEATHER MISSINGNESS RATE {status_tag(missingness_pass):>50}")
    print(f"   - Missingness Threshold   : < {MAX_MISSING_RATE_THRESHOLD:.1%}")
    print(f"   - Merged Weather Missing  : {total_missing_rf} / {total_merged_rows} rows ({missing_rf_rate:.2%})")
    for stn in selected:
        stn_m = merged_df[merged_df['Station'] == stn]
        stn_miss = stn_m['IMD_Rainfall_mm'].isna().sum() + (stn_m['IMD_Rainfall_mm'] < 0).sum()
        r = stn_miss / len(stn_m) if len(stn_m) > 0 else 0
        print(f"     * {stn:<15}: {stn_miss} missing ({r:.2%})")

    print(f"\n3. CONTIGUOUS WEATHER GAPS {status_tag(contiguous_gap_pass):>51}")
    print(f"   - Gap Limit Threshold     : <= {MAX_CONTIGUOUS_GAP_THRESHOLD} consecutive days")
    for stn in selected:
        gap = max_weather_gap_per_stn[stn]
        st = "PASS" if gap <= MAX_CONTIGUOUS_GAP_THRESHOLD else "FAIL"
        print(f"     * {stn:<15}: Max missing gap = {gap} days [{st}]")

    print(f"\n4. MERGE SUCCESS BY STATION {status_tag(merge_success_pass):>50}")
    print(f"   {'Station':<15} | {'CWC Rows':<12} | {'Matched Rows':<14} | {'Match Rate':<12} | {'Status':<8}")
    print("   " + "-" * 69)
    for stn in selected:
        c_count = station_total_cwc[stn]
        m_count = station_match_counts[stn]
        pct = (m_count / c_count * 100) if c_count > 0 else 0.0
        st = "PASS" if m_count == c_count else "FAIL"
        print(f"   {stn:<15} | {c_count:<12} | {m_count:<14} | {pct:>9.2f}% | {st:<8}")
    print("   " + "-" * 69)
    print(f"   {'TOTAL':<15} | {len(df_cwc):<12} | {len(merged_df):<14} | {len(merged_df)/len(df_cwc)*100:>9.2f}% | {status_tag(merge_success_pass)}")

    print("\n" + "=" * 78)
    print(f" OVERALL ALIGNMENT TEST RESULT: {'>>> PASS <<<' if overall_pass else '>>> FAIL <<<'}")
    print("=" * 78 + "\n")
    return 0 if overall_pass else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test meteorological alignment between CWC discharge and IMD rainfall.")
    parser.add_argument('--stations', nargs='+', default=['Warunji', 'Karad', 'Ashti'],
                        help="List of station names to test (default: Warunji Karad Ashti)")
    parser.add_argument('--cwc', default=None, help="Path to CWC discharge CSV (optional; auto-resolves across states if omitted)")
    parser.add_argument('--imd', default=DEFAULT_IMD_DIR, help="Path to IMD gridded data directory")
    args = parser.parse_args()

    sys.exit(run_alignment_test(args.stations, args.cwc, args.imd, DEFAULT_PROCESSED_STATION_DIR, DEFAULT_MERGED_DIR))
