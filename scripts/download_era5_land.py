#!/usr/bin/env python3
"""Download daily ERA5-Land reanalysis temperature and PET data (2001-2025) for all 12 gauging stations."""

import os
import sys
import time
import json
import urllib.request
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_ERA5_DIR = PROJECT_ROOT / 'data' / 'raw' / 'era5_land'
CATALOG_PATH = PROJECT_ROOT / 'data' / 'processed' / 'station_catalog.csv'

sys.path.insert(0, str(SCRIPT_DIR))
from train_m2_models import PORTFOLIO_CONFIG
from pipeline_integrity import preflight_era

RAW_ERA5_DIR.mkdir(parents=True, exist_ok=True)
cat = pd.read_csv(CATALOG_PATH)

print(f'Starting ERA5-Land download for {len(PORTFOLIO_CONFIG)} stations...')
for idx, stn in enumerate(PORTFOLIO_CONFIG, 1):
    gid = stn['gauge_id']
    name = stn['station']
    out_file = RAW_ERA5_DIR / f'era5_land_{gid:05d}_{name}.csv'

    if out_file.is_file():
        try:
            existing = pd.read_csv(out_file)
            if len(existing) == 9131 and not existing.isna().any().any():
                print(f'[{idx}/{len(PORTFOLIO_CONFIG)}] {name} ({gid:05d}) already downloaded and verified. Skipping.')
                continue
        except Exception:
            pass

    # Coordinates from catalog
    match = cat[cat['CAMELS_Gauge_ID'] == gid]
    if len(match) == 0:
        match = cat[cat['Station'].str.contains(name, case=False, na=False)]
    if len(match) == 0:
        raise ValueError(f'Could not find coordinates for gauge {gid:05d} ({name})')

    lat = float(match.iloc[0]['Latitude'])
    lon = float(match.iloc[0]['Longitude'])
    print(f'[{idx}/{len(PORTFOLIO_CONFIG)}] Fetching {name} ({gid:05d}) at ({lat:.4f}, {lon:.4f})...')

    url = (
        f'https://archive-api.open-meteo.com/v1/archive?'
        f'latitude={lat:.4f}&longitude={lon:.4f}&'
        f'start_date=2001-01-01&end_date=2025-12-31&'
        f'daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,et0_fao_evapotranspiration&'
        f'timezone=GMT'
    )

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Research Pipeline)'})
    data = None
    for attempt in range(1, 8):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_sec = 15 * attempt
                print(f'       [Rate limit 429] Backing off for {wait_sec}s (attempt {attempt}/7)...')
                time.sleep(wait_sec)
            else:
                if attempt == 7:
                    raise e
                time.sleep(5)
        except Exception as e:
            if attempt == 7:
                raise e
            time.sleep(5)

    if data is None:
        raise RuntimeError(f'Failed to fetch data for {name} ({gid:05d})')

    daily = data['daily']
    df = pd.DataFrame({
        'Date': daily['time'],
        'tavg_c': daily['temperature_2m_mean'],
        'tmax_c': daily['temperature_2m_max'],
        'tmin_c': daily['temperature_2m_min'],
        'pet_mm': daily['et0_fao_evapotranspiration'],
    })

    # Assert integrity
    assert len(df) == 9131, f'Expected 9131 days, got {len(df)}'
    assert not df.isna().any().any(), f'Found missing values in {name}'

    df.to_csv(out_file, index=False)
    print(f'       Saved {out_file.name} ({len(df)} rows)')
    time.sleep(2.5)

print('\nAll 12 files downloaded. Running pipeline integrity preflight check...')
preflight_era(RAW_ERA5_DIR, PORTFOLIO_CONFIG)
print('SUCCESS: All 12 ERA5-Land station files passed preflight_era validation!')
