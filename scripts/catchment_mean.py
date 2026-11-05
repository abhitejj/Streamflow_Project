#!/usr/bin/env python3
"""
Catchment-Averaged Zonal Extraction for IMD 0.25° Gridded Rainfall.

Computes exact area-weighted catchment means for each gauge by:
1. Intersecting the CAMELS-IND catchment boundary polygon from catchments.shp
   with 0.25° x 0.25° IMD grid cell bounding boxes.
2. Weighting each cell by its fractional intersection area times cos(radians(lat)).
3. Normalizing weights to sum to 1.0.
4. Extracting the weighted daily timeseries across the grid points.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get_raw_dir():
    raw_env = os.environ.get("RAW_DIR")
    if raw_env and os.path.isdir(raw_env):
        return Path(raw_env)
    local_raw = PROJECT_ROOT / "data" / "raw"
    if local_raw.is_dir() and (local_raw / "camels_ind").is_dir():
        return local_raw
    ml2_raw = Path("/Users/abhitej/Downloads/ML Project 2/data/raw")
    if ml2_raw.is_dir():
        return ml2_raw
    return local_raw

def get_shapefile_path():
    raw_dir = get_raw_dir()
    candidates = [
        raw_dir / "camels_ind" / "shapefiles_catchment" / "catchments.shp",
        PROJECT_ROOT / "data" / "raw" / "camels_ind" / "shapefiles_catchment" / "catchments.shp",
        Path("/Users/abhitej/Downloads/ML Project 2/data/raw/camels_ind/shapefiles_catchment/catchments.shp"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"catchments.shp not found in candidate paths: {candidates}")

_WEIGHTS_CACHE = {}
_GDF_CACHE = None

def _load_gdf():
    global _GDF_CACHE
    if _GDF_CACHE is None:
        shp_path = get_shapefile_path()
        gdf = gpd.read_file(shp_path)
        gdf["gid_int"] = gdf["gauge_id"].astype(int)
        _GDF_CACHE = gdf
    return _GDF_CACHE

def compute_weights_for_gauge(gauge_id, lats, lons):
    """
    Computes normalized area weights for grid cells intersecting gauge_id's catchment polygon.
    Returns: (lats_list, lons_list, norm_weights_array, n_cells)
    """
    gid_int = int(gauge_id)
    if gid_int in _WEIGHTS_CACHE:
        return _WEIGHTS_CACHE[gid_int]

    gdf = _load_gdf()
    match = gdf[gdf["gid_int"] == gid_int]
    if len(match) == 0:
        raise ValueError(f"Gauge ID {gauge_id} not found in catchments.shp")

    poly = match.geometry.iloc[0]
    minx, miny, maxx, maxy = poly.bounds

    # Grid step is 0.25 degree, cell radius is 0.125
    cand_lats = lats[(lats >= miny - 0.25) & (lats <= maxy + 0.25)]
    cand_lons = lons[(lons >= minx - 0.25) & (lons <= maxx + 0.25)]

    weights = []
    lt_list = []
    ln_list = []

    for lt in cand_lats:
        cos_lat = np.cos(np.radians(lt))
        for ln in cand_lons:
            c_box = box(ln - 0.125, lt - 0.125, ln + 0.125, lt + 0.125)
            if poly.intersects(c_box):
                inter = poly.intersection(c_box)
                if inter.area > 0:
                    w = inter.area * cos_lat
                    weights.append(w)
                    lt_list.append(float(lt))
                    ln_list.append(float(ln))

    weights = np.array(weights, dtype=np.float64)
    total_w = weights.sum()
    if total_w <= 0:
        raise ValueError(f"Zero total intersection weight for gauge {gauge_id}")

    norm_w = weights / total_w
    res = (lt_list, ln_list, norm_w, len(lt_list))
    _WEIGHTS_CACHE[gid_int] = res
    return res

def get_catchment_cell_count(gauge_id, lats=None, lons=None):
    gid_int = int(gauge_id)
    if gid_int in _WEIGHTS_CACHE:
        return _WEIGHTS_CACHE[gid_int][3]
    if lats is None or lons is None:
        raw_dir = get_raw_dir()
        nc_candidates = [
            raw_dir / "imd" / "imd_rainfall_2001_2025.nc",
            PROJECT_ROOT / "data" / "raw" / "imd" / "imd_rainfall_2001_2025.nc",
            Path("/Users/abhitej/Downloads/ML Project 2/data/raw/imd/imd_rainfall_2001_2025.nc")
        ]
        nc_path = next(p for p in nc_candidates if p.is_file())
        ds = xr.open_dataset(nc_path)
        lats, lons = ds.lat.values, ds.lon.values
    return compute_weights_for_gauge(gid_int, lats, lons)[3]

def catchment_mean(ds_var, gauge_id) -> pd.Series:
    """
    Extracts the area-weighted catchment mean timeseries from an xarray DataArray.
    ds_var must have dimensions (time, lat, lon) or coordinates lat, lon.
    Returns: pd.Series indexed by normalized datetime.
    """
    gid_int = int(gauge_id)
    lats = ds_var.lat.values
    lons = ds_var.lon.values

    lt_list, ln_list, norm_w, _ = compute_weights_for_gauge(gid_int, lats, lons)

    da_lats = xr.DataArray(lt_list, dims="points")
    da_lons = xr.DataArray(ln_list, dims="points")
    da_weights = xr.DataArray(norm_w, dims="points")

    pts_data = ds_var.sel(lat=da_lats, lon=da_lons)
    weighted_sum = (pts_data * da_weights).sum(dim="points")
    s = weighted_sum.to_series()
    s.index = pd.to_datetime(s.index).normalize()
    return s

if __name__ == "__main__":
    print(f"[*] Raw data directory: {get_raw_dir()}")
    print(f"[*] Shapefile path: {get_shapefile_path()}")
    print(f"[*] Precomputing weights for all gauges in catchments.shp...")
    gdf = _load_gdf()
    print(f"[*] Loaded {len(gdf)} catchment polygons.")
