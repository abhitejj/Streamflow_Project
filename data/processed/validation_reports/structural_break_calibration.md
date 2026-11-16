# 📐 Structural Break Mitigation: 2001–2010 Boundary Calibration

Calibrates the spatial and instrument mismatch between CAMELS-IND catchment-averaged historical forcings (1980–2020) and later-source products (2021–2025: IMD 0.25° Precipitation, Open-Meteo default Best Match temperature and FAO reference ET0) using overlapping boundary observations (2001–2010: paired finite training daily observations).

### 1. Data Lineage & Extraction Rules Contract

- **Discharge ($Q$):** Stitched from CAMELS-IND observed streamflow (1980–2020) and CWC manual daily discharge (2001–2025). Coordinates: WGS84 (EPSG:4326). Units: $\text{m}^3/\text{s}$. Specific Runoff ($q = Q \times 604.8 / \text{Area}_{\text{km}^2}$, mm/week).
- **Precipitation ($P$):** 1980–2020 from CAMELS-IND (**Catchment-Averaged** zonal mean over polygon shapefile). 2021–2025 from IMD 0.25° NetCDF (**Catchment-area weighted extraction**). Scaled by $\alpha_P = \sum P_{\text{CAMELS}} / \sum P_{\text{IMD}}$.
- **Temperature ($T_{\text{avg}}, T_{\text{max}}, T_{\text{min}}$):** 1980–2020 from CAMELS-IND (**Catchment-Averaged** from IMD 1.0° grids). 2021–2025 from Open-Meteo default Best Match daily CSVs (**Gauge-point request; provider grid metadata not retained**). Bias corrected via $\text{Bias}_T = \mu(T_{\text{CAMELS}}) - \mu(T_{\text{Best Match}})$.
- **Evaporative Demand (PET):** 1980–2020 from CAMELS-IND GLEAM v3.6a (**Priestley-Taylor** formulation, catchment-averaged). 2021–2025 from Open-Meteo default Best Match (**FAO-56 Penman-Monteith** reference crop evapotranspiration, point extraction). Scaled by $\alpha_{\text{PET}} = \sum \text{PET}_{\text{GLEAM}} / \sum \text{PET}_{\text{Best Match}}$.

### 2. Comprehensive Boundary Calibration Audit Table (2001–2010: training-period paired days)

| Station | Gauge | Area (km²) | P Mean (CAM/IMD) | α_P | P Daily r | P Wkly r | Tavg Mean (CAM/Best Match) | Tavg Bias (°C) | Tavg Daily r | Tavg Wkly r | PET Mean (GLM/Best Match) | α_PET | PET Daily r | PET Wkly r | Daily Gate | Wkly Gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Karad** | `4021` | 5462 | 3.16/3.66 | `0.8636` | 0.912 | **0.942** | 25.8/24.8 | +1.04 | 0.914 | **0.942** | 2.90/4.42 | `0.6562` | 0.873 | **0.892** | `PASS` | `PASS` |
| **Warunji** | `4063` | 1890 | 4.11/4.32 | `0.9499` | 0.945 | **0.974** | 25.9/24.7 | +1.24 | 0.907 | **0.937** | 2.98/4.41 | `0.6768` | 0.867 | **0.890** | `PASS` | `PASS` |
| **Ashti** | `3005` | 50990 | 3.32/3.35 | `0.9906` | 0.990 | **0.997** | 26.3/27.8 | -1.47 | 0.947 | **0.964** | 2.85/4.78 | `0.5965` | 0.845 | **0.883** | `PASS` | `PASS` |
| **Bamni (Ballarsha)** | `3008` | 46020 | 2.66/2.66 | `1.0021` | 0.988 | **0.996** | 27.2/27.9 | -0.70 | 0.970 | **0.984** | 2.93/4.80 | `0.6113` | 0.881 | **0.915** | `PASS` | `PASS` |
| **Anandpur** | `7002` | 8570 | 4.19/4.17 | `1.0044` | 0.996 | **0.998** | 26.7/26.5 | +0.23 | 0.974 | **0.985** | 2.79/3.92 | `0.7110` | 0.890 | **0.896** | `PASS` | `PASS` |
| **Kashinagar** | `16007` | 7820 | 3.62/3.58 | `1.0110` | 0.987 | **0.993** | 26.9/27.1 | -0.21 | 0.957 | **0.973** | 2.83/4.00 | `0.7083` | 0.896 | **0.918** | `PASS` | `PASS` |
| **Gomlai** | `7007` | 21950 | 3.43/3.42 | `1.0012` | 0.996 | **0.999** | 25.6/26.4 | -0.80 | 0.946 | **0.957** | 2.74/4.21 | `0.6497` | 0.874 | **0.895** | `PASS` | `PASS` |
| **Kantamal** | `8013` | 19600 | 4.14/4.15 | `0.9989` | 0.986 | **0.994** | 26.6/27.7 | -1.16 | 0.951 | **0.961** | 2.79/4.59 | `0.6069` | 0.882 | **0.912** | `PASS` | `PASS` |
| **Yadgir** | `4064` | 69863 | 2.07/2.34 | `0.8879` | 0.892 | **0.914** | 26.5/27.5 | -1.00 | 0.932 | **0.953** | 2.77/5.24 | `0.5279` | 0.791 | **0.832** | `PASS` | `PASS` |
| **Huvinhedigi** | `4017` | 55150 | 2.15/2.14 | `1.0057` | 0.990 | **0.996** | 26.3/27.7 | -1.35 | 0.928 | **0.954** | 2.85/5.33 | `0.5341` | 0.824 | **0.875** | `PASS` | `PASS` |
| **KUDIGE** | `5010` | 1934 | 4.87/4.88 | `0.9974` | 0.950 | **0.973** | 24.6/22.7 | +1.92 | 0.886 | **0.925** | 3.50/3.93 | `0.8910` | 0.100 | **-0.005** | `FAIL (T+PET)` | `FAIL (PET)` |
| **Haralahalli** | `4013` | 14582 | 3.51/3.44 | `1.0205` | 0.994 | **0.996** | 24.8/25.5 | -0.70 | 0.912 | **0.940** | 3.16/4.78 | `0.6623` | 0.689 | **0.720** | `PASS` | `PASS` |

### 3. Portfolio Summary Statistics (12 Catchments):
- **Precipitation Correlation:** Daily Mean $r = 0.9688$ | Weekly Mean $r = 0.9810$ (Target: >0.6)
- **Temperature ($T_{avg}$) Correlation:** Daily Mean $r = 0.9353$ | Weekly Mean $r = 0.9562$ (Target: >0.9), Mean Bias: `-0.2467 ^\circ C`
- **Evaporative Demand (PET) Correlation:** Daily Mean $r = 0.7844$ | Weekly Mean $r = 0.8018$ (Target: >0.6), Mean Scaling Ratio ($\alpha_{PET}$): `0.6527`
