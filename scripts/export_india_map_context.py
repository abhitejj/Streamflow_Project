#!/usr/bin/env python3
"""Build simplified local map context from attributed DataMeet source geometries."""
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import mapping, shape

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/map_context'


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    return round(value, 4) if isinstance(value, float) else value


def main():
    source_manifest = json.loads((SOURCE / 'source_manifest.json').read_text())
    country = json.loads((SOURCE / 'india-composite.geojson').read_text())
    geometry = shape(country['features'][0]['geometry']).simplify(.015, preserve_topology=True)
    states = gpd.read_file(SOURCE / 'datameet_states/Admin2.shp')
    if states.crs.to_epsg() != 4326 or len(states) != 36:
        raise ValueError('Expected thirty-six WGS84 state/union-territory geometries')
    data = {
        'schema': 1, 'crs': 'EPSG:4326', 'country': mapping(geometry),
        'states': [{'name': row.ST_NM, 'geometry': mapping(row.geometry.simplify(.015, preserve_topology=True))}
                   for _, row in states.iterrows()],
        'source': source_manifest,
        'display': {'simplification_tolerance_degrees': .015, 'coordinate_rounding_decimals': 4,
                    'note': 'Generalized source boundaries for geographic context; not an authoritative boundary map'},
    }
    output = ROOT / 'site/assets/india-map-context.js'
    output.write_text('/* DataMeet map context: country CC0; states CC BY 4.0. Simplified for display. */\n'
                      + 'window.STATION_MAP_CONTEXT = ' + json.dumps(rounded(data), separators=(',', ':'), allow_nan=False) + ';\n')
    print(f'Exported country boundary and {len(states)} state/union-territory geometries: {output}')


if __name__ == '__main__':
    main()
