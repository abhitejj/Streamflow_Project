# Validation Report: 2021–2025 Rainfall Catchment-Mean Rebuild

## Overview
This report documents the reconstruction of 2021–2025 precipitation forcings using exact area-weighted catchment zonal averages across IMD 0.25° grid cells, replacing single-cell nearest-gauge point extractions. Each cell was weighted by its fractional intersection area with the catchment polygon times `cos(lat)`. Boundary scaling factors (`alpha_P`) and Pearson correlations against CAMELS-IND catchment-averaged daily precipitation were evaluated over the 2015–2020 boundary calibration period (2,192 daily steps).

## Comparison: Old Point Extraction vs. New Catchment Mean (2015–2020 Calibration)

| Station | Gauge ID | Area (km²) | Intersecting Grid Cells | Old α_P | New α_P | Old Daily r | New Daily r | Old Weekly r | New Weekly r |
| :--- | :---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Karad** | `04021` | 5,462.0 | 15 | 1.5616 | **1.0009** | 0.8227 | **0.9997** | 0.8966 | **0.9999** |
| **Warunji** | `04063` | 1,890.0 | 10 | 2.1797 | **1.0023** | 0.7671 | **0.9995** | 0.8413 | **0.9999** |
| **Ashti** | `03005` | 50,990.0 | 97 | 0.7824 | **1.0029** | 0.6450 | **0.9994** | 0.8567 | **0.9999** |
| **Bamni (Ballarsha)** | `03008` | 46,020.0 | 95 | 0.8548 | **1.0021** | 0.6911 | **0.9996** | 0.8933 | **0.9998** |
| **Anandpur** | `07002` | 8,570.0 | 22 | 0.8995 | **1.0029** | 0.7788 | **0.9994** | 0.8962 | **0.9997** |
| **Kashinagar** | `16007` | 7,820.0 | 21 | 1.1447 | **1.0036** | 0.6778 | **0.9972** | 0.8065 | **0.9988** |
| **Gomlai** | `07007` | 21,950.0 | 47 | 0.8509 | **1.0006** | 0.7448 | **0.9984** | 0.8500 | **0.9997** |
| **Kantamal** | `08013` | 19,600.0 | 46 | 1.0135 | **1.0007** | 0.6383 | **0.9997** | 0.8207 | **0.9999** |
| **Yadgir** | `04064` | 69,863.0 | 129 | 0.8560 | **1.0032** | 0.5712 | **0.9996** | 0.8081 | **0.9998** |
| **Huvinhedigi** | `04017` | 55,150.0 | 105 | 0.9987 | **1.0044** | 0.3493 | **0.9994** | 0.5494 | **0.9998** |
| **KUDIGE** | `05010` | 1,934.0 | 8 | 2.1191 | **1.0045** | 0.6778 | **0.9432** | 0.8010 | **0.9946** |
| **Haralahalli** | `04013` | 14,582.0 | 32 | 1.6131 | **0.9833** | 0.5533 | **0.9827** | 0.6759 | **0.9938** |

## Summary Findings
1. **Alpha Scaling Factor Stabilization**: Across all 12 stations, the boundary scaling factor `alpha_P` collapsed from extreme point distortions (0.782 to 2.180) to virtually exact physical parity (`0.9833` to `1.0045`, portfolio mean `1.0009`).
2. **Correlation Alignment**: Daily correlation against CAMELS-IND precipitation surged from a median of `0.6778` to `0.9994` (weekly correlation from `0.8310` to `0.9998`).
3. **Preservation of Pre-2021 Data**: All weekly tensor rows prior to 2021 and all non-rainfall columns remain 100% bit-identical to the original audited benchmark tensors.
