#!/usr/bin/env python3
"""Export a display-only subset of local CAMELS-IND catchments and gauge points.

This does not change the model inputs or rainfall extraction geometries. All
twelve polygons and gauge points were compared with the open GHI 2023 dataset
(Goteti, https://doi.org/10.5281/zenodo.7563600, CC BY 4.0): catchments are
topologically identical and gauge points match exactly. The local CAMELS-IND
bundle supplies the mapped subset; boundaries are simplified only for display.
Rebuilding requires that bundle plus the GHI source record and comparison
receipt in data/raw/map_context/ghi/; raw files remain outside Git.
"""
import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import mapping

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/camels_ind/shapefiles_catchment'
GHI = ROOT / 'data/raw/map_context/ghi'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'site/assets/station-map-data.js')
    args = parser.parse_args()
    metadata_text = (ROOT / 'site/assets/benchmark-data.js').read_text()
    stations, _ = json.JSONDecoder().raw_decode(metadata_text.split('window.STATIONS =', 1)[1].lstrip())
    original_record = json.loads((GHI / 'original_2023_record.json').read_text())
    comparisons = json.loads((GHI / 'geometry_comparison.json').read_text())
    if (original_record['metadata']['license']['id'] != 'cc-by-4.0' or
            original_record['metadata']['access_right'] != 'open' or
            {row['gauge_id'] for row in comparisons} != {str(s['camels_id']).zfill(5) for s in stations} or
            len(comparisons) != 12 or not all(
                row['catchment_topologically_equal'] and row['symmetric_difference_square_degrees'] == 0 and
                row['hausdorff_degrees'] == 0 and row['point_distance_degrees'] == 0 for row in comparisons)):
        raise ValueError('Expected licensed GHI source and exact twelve-feature geometry comparison')
    catchments = gpd.read_file(SOURCE / 'catchments.shp')
    gauges = gpd.read_file(SOURCE / 'gauge_stations.shp')
    if catchments.crs.to_epsg() != 4326 or gauges.crs.to_epsg() != 4326:
        raise ValueError('Expected geographic WGS84 source coordinates')
    features = []
    for station in stations:
        gauge_id = str(station['camels_id']).zfill(5)
        polygon_rows = catchments[catchments.gauge_id.astype(str).str.zfill(5).eq(gauge_id)]
        point_rows = gauges[gauges.gauge_id.astype(str).str.zfill(5).eq(gauge_id)]
        if len(polygon_rows) != 1 or len(point_rows) != 1:
            raise ValueError(f'Expected one catchment and gauge for {gauge_id}')
        polygon = polygon_rows.iloc[0].geometry
        point = point_rows.iloc[0].geometry
        if polygon.is_empty or not polygon.is_valid or point.geom_type != 'Point':
            raise ValueError(f'Invalid source geometry for {gauge_id}')
        geometry = polygon.simplify(.003, preserve_topology=True)
        features.append({
            'id': station['id'], 'gauge_id': gauge_id, 'name': station['name'],
            'state': station['state'], 'subbasin': station['basin'],
            'area_km2': station['area_km2'], 'role': station['role'],
            'point': [point.x, point.y], 'geometry': mapping(geometry),
            'source_station_name': str(point_rows.iloc[0].CWC_SITE_N),
        })
    if len(features) != 12 or len({f['id'] for f in features}) != 12:
        raise ValueError('Expected twelve unique station catchments')
    source_hashes = {}
    for stem in ('catchments', 'gauge_stations'):
        for suffix in ('.shp', '.shx', '.dbf', '.prj'):
            path = SOURCE / (stem + suffix)
            source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = {
        'schema': 1, 'crs': 'EPSG:4326', 'features': features,
        'source': {'name': 'GHI 2023 catchments and gauge locations, via CAMELS-IND',
                   'citation': 'Goteti (2023), Geospatial dataset for Hydrologic analyses in India (GHI)',
                   'url': 'https://doi.org/10.5281/zenodo.7563600',
                   'license': 'CC BY 4.0',
                   'license_url': 'https://creativecommons.org/licenses/by/4.0/',
                   'via': {'citation': 'Mangukiya et al. (2025), CAMELS-IND',
                           'url': 'https://essd.copernicus.org/articles/17/461/2025/'},
                   'sha256': source_hashes,
                   'verification': {'matching_catchments': 12, 'matching_gauge_points': 12,
                                    'method': 'Topological polygon equality and zero gauge-point distance in EPSG:4326',
                                    'evidence_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                                       for path in [GHI / 'original_2023_record.json',
                                                                    GHI / 'geometry_comparison.json',
                                                                    GHI / 'extracted_geometry_manifest.json']}}},
        'display': {'simplification_tolerance_degrees': .003,
                    'note': 'Upstream gauge drainage areas; display simplification only, not analysis geometry'},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('/* Goteti (2023), GHI, CC BY 4.0, via CAMELS-IND. Simplified for display. */\n'
                        + 'window.STATION_MAP_DATA = ' + json.dumps(data, separators=(',', ':'), allow_nan=False) + ';\n')
    print(f'Exported {len(features)} catchments and gauge points to {args.out}')


if __name__ == '__main__':
    main()
