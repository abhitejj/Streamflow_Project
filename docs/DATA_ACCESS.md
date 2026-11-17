# Data access and reproduction inputs

Raw datasets remain local and are excluded from Git. Their presence does not establish permission to redistribute them. Check each source's terms; this project does not grant rights to share raw inputs.

## Inputs used by the benchmark

- `data/raw/camels_ind/`: historical streamflow and catchment forcings through 2020, station attributes and catchment shapefiles. The bundle includes a disclaimer/data description; consult them before sharing.
- `data/raw/imd/imd_rainfall_2001_2025.nc`: IMD 0.25° rainfall, used for catchment-area extraction. Paired 2001–2010 observations support training-period source calibration.
- `data/raw/cwc/`: original Maharashtra/Odisha daily manual-discharge archives. The supplied Karnataka archive was reconstructed from `data/processed/merged/cwc_imd_merged_2001_2025.csv`; it is not an independently recovered original archive.
- `data/raw/era5_land/`: twelve supplied weather CSVs named `era5_land_<gauge>_<station>.csv` for compatibility. Each has 9,131 daily rows from 2001-01-01 to 2025-12-31 and the fields `Date,tavg_c,tmax_c,tmin_c,pet_mm`. The actual acquisition script uses the **Open-Meteo Historical Weather API, default Best Match**, not an explicitly selected ERA5-Land model. `pet_mm` contains FAO reference ET0.

The `data/processed/validation_reports/weather_input_manifest.json` file records filenames, hashes, reconstructed request coordinates and integrity checks. The downloader joins station coordinates by exact CAMELS gauge ID and rounds them to four decimal places. Request parameters were reconstructed from the acquisition script and catalog; original HTTP response metadata was not retained. All twelve CSVs have unique daily dates, finite numeric fields, ordered temperature extrema and nonnegative ET0. These checks establish usable file structure, not a homogeneous source model.

## Weather-source qualifications

The local CAMELS-IND disclaimer requires citation of its data description: [Mangukiya et al. (2025)](https://essd.copernicus.org/articles/17/461/2025/). The paper links the [CAMELS-IND dataset repository](https://doi.org/10.5281/zenodo.14005378). The bundle's exact release version is not independently verified; actual file hashes identify this run's inputs. This citation does not grant redistribution rights beyond the source terms.

The [official Open-Meteo Historical Weather API documentation](https://open-meteo.com/en/docs/historical-weather-api) describes its default Best Match selection and consistent-model guidance. The supplied request omits `models`, requests mean/max/min 2 m temperature plus `et0_fao_evapotranspiration`, and sets timezone GMT. Best Match can combine products across the record. Neither the directory name nor a successful schema check proves a homogeneous ERA5-Land series.

The benchmark uses these supplied files under their actual source description. Source-model heterogeneity remains a limitation. Scaling reference ET0 to historical PET does not make the physical formulations equivalent. Source publication delays and CWC/CAMELS measurement-day timing are not established as Monday-midnight operational availability.

## Display map sources and rebuilding

The website includes precomputed map assets; viewing them requires no raw geographic files. Its twelve upstream catchment polygons and gauge points come through the local CAMELS-IND bundle and were checked against the original [GHI 2023 dataset, Goteti (2023)](https://doi.org/10.5281/zenodo.7563600), licensed CC BY 4.0. All twelve polygons were topologically equal and all twelve gauge points matched exactly. The display boundaries are simplified; model inputs and rainfall-extraction polygons are unchanged.

The India outline and 36 state/territory areas come from [DataMeet Maps](https://github.com/datameet/maps): `Country/india-composite.geojson` is CC0, and the `States` dataset uses the repository's CC BY 4.0 licence. The exact administrative geometry vintage is not established; this is geographic context, not an authoritative boundary map.

Optional map rebuilding uses `python3 scripts/export_station_map.py` and `python3 scripts/export_india_map_context.py`. These require the local CAMELS shapefiles plus `data/raw/map_context/`: the DataMeet source files/manifest and `ghi/` source-record, exact-geometry-comparison and extraction receipts. The published map assets retain source URLs, licence attribution, display simplification and input/evidence hashes. Raw geography and verification receipts remain local.

## Explicit Karnataka derived-discharge option

The full build was run with:

```sh
python3 scripts/prepare_weekly_dataset.py --derived-cwc data/processed/merged/cwc_imd_merged_2001_2025.csv
```

The option validates the existing daily CWC fields, Karnataka state/station coverage, numeric discharge and usable dates. It does not use the merged rainfall column. Tensor provenance records the actual derived-file hash and archive-lineage limitation. The reconstructed archive contains 34,007 records covering all four Karnataka stations; the independent inventory found matching original-field row multisets in the merged data, excluding the serial number. This does not establish the original external archive's identity or permission status.

## Rebuilding and environment

The raw input preflight runs before tensor mutation. It checks the weather files, all station forcings, CWC paths, streamflow, station catalog, topography and shapefile components. `RAW_DIR` can explicitly select a different raw-input root; an invalid setting fails rather than silently falling back.

The benchmark includes saved datasets, fitted models, evaluation results and website exports. Source paths, hashes, finite calibration fit counts and weekly-window rules are saved in each tensor's provenance JSON. The full-run validator links the tensors, fitted models, code and generated outputs. Reproduction commands and overwrite warnings are in README.

The installed Python 3.14.4 package versions are in `requirements.txt`. This is an observed environment specification, not a cross-platform lock file; PyTorch support depends on the platform. Raw input acquisition/licensing still requires the individual source providers; a processed tensor is not a substitute for documenting source provenance.
