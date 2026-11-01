#!/usr/bin/env python3
"""
Station Explorer & Alignment Selector
Enables browsing, filtering, and running alignments for CWC and CAMELS-IND stations across India
(Supports Maharashtra, Punjab, West Bengal, and CAMELS-IND Peninsular catchments).
"""

import os
import sys
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CATALOG_PATH = os.path.join(PROJECT_ROOT, 'data/processed/station_catalog.csv')

def load_data():
    if not os.path.isfile(CATALOG_PATH):
        print(f"Catalog not found at {CATALOG_PATH}")
        sys.exit(1)
    return pd.read_csv(CATALOG_PATH)

def list_stations(state=None, basin=None, min_records=0, matched_only=False, search=None):
    df = load_data()
    
    if state:
        df = df[df['State'].str.contains(state, case=False, na=False)]
    if basin:
        df = df[df['Basin'].str.contains(basin, case=False, na=False)]
    if search:
        df = df[df['Station'].str.contains(search, case=False, na=False) | 
                df['River'].str.contains(search, case=False, na=False) |
                df['District'].str.contains(search, case=False, na=False)]
    if matched_only:
        df = df[df['CAMELS_Gauge_ID'].notna()]
    if min_records > 0:
        df = df[df['Record_Count'] >= min_records]

    df = df.sort_values(by=['State', 'Record_Count'], ascending=[True, False]).reset_index(drop=True)

    print("=" * 105)
    print(f"                       MULTI-STATE STATION CATALOG ({len(df)} stations)")
    print("=" * 105)
    print(f"{'#':<3} | {'State':<12} | {'Station Name':<32} | {'District':<12} | {'Records':<8} | {'CAMELS ID':<10}")
    print("-" * 105)
    for idx, row in df.iterrows():
        camels_id = f"{int(row['CAMELS_Gauge_ID'])}" if pd.notna(row['CAMELS_Gauge_ID']) else "-"
        print(f"{idx+1:<3} | {str(row['State'])[:12]:<12} | {str(row['Station'])[:32]:<32} | {str(row['District'])[:12]:<12} | {row['Record_Count']:<8} | {camels_id:<10}")
    print("-" * 105)
    print("Use --run-test <Station1> <Station2> ... to execute IMD alignment for chosen stations.\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Explore and select CWC streamflow stations across India.")
    parser.add_argument('--list', action='store_true', help="List all available stations")
    parser.add_argument('--state', type=str, help="Filter stations by state (e.g. Punjab, Maharashtra, West Bengal)")
    parser.add_argument('--basin', type=str, help="Filter stations by river basin (e.g. Krishna, Godavari, Indus, Subernarekha)")
    parser.add_argument('--search', type=str, help="Search stations by name, river, or district")
    parser.add_argument('--matched-only', action='store_true', help="Show only stations present in both CWC and CAMELS-IND")
    parser.add_argument('--min-records', type=int, default=0, help="Minimum number of discharge observations")
    parser.add_argument('--run-test', nargs='+', help="Run alignment test for the specified stations")
    parser.add_argument('--cwc-file', type=str, help="Explicit CWC CSV file to use with --run-test")

    args = parser.parse_args()

    if args.run_test:
        test_script = os.path.join(SCRIPT_DIR, 'imd_cwc_alignment_test.py')
        stations_arg = " ".join([f'"{s}"' for s in args.run_test])
        cmd = f'python3 "{test_script}" --stations {stations_arg}'
        if args.cwc_file:
            cmd += f' --cwc "{args.cwc_file}"'
        os.system(cmd)
    else:
        list_stations(state=args.state, basin=args.basin, min_records=args.min_records, matched_only=args.matched_only, search=args.search)
