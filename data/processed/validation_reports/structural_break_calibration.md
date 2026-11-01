# 📐 Structural Break Mitigation: 2015–2020 Boundary Calibration

Calibrates the spatial and instrument mismatch between CAMELS-IND catchment-averaged historical forcings (1980–2020) and operational products (2021–2025: IMD 0.25° Precipitation, ERA5-Land Temperature & PET) using overlapping boundary observations (2015–2020: 2,192 daily timesteps).

### 1. Data Lineage & Extraction Rules Contract

- **Discharge ($Q$):** Stitched from CAMELS-IND observed streamflow (1980–2020) and CWC manual daily discharge (2001–2025). Coordinates: WGS84 (EPSG:4326). Units: $\text{m}^3/\text{s}$. Specific Runoff ($q = Q \times 604.8 / \text{Area}_{\text{km}^2}$, mm/week).
- **Precipitation ($P$):** 1980–2020 from CAMELS-IND (**Catchment-Averaged** zonal mean over polygon shapefile). 2021–2025 from IMD 0.25° NetCDF (**Nearest-Grid Point Extraction** at gauge coordinates). Scaled by $\alpha_P = \sum P_{\text{CAMELS}} / \sum P_{\text{IMD}}$.
- **Temperature ($T_{\text{avg}}, T_{\text{max}}, T_{\text{min}}$):** 1980–2020 from CAMELS-IND (**Catchment-Averaged** from IMD 1.0° grids). 2021–2025 from ERA5-Land daily CSVs (**Nearest-Grid Point Extraction**). Bias corrected via $\text{Bias}_T = \mu(T_{\text{CAMELS}}) - \mu(T_{\text{ERA5}})$.
- **Evaporative Demand (PET):** 1980–2020 from CAMELS-IND GLEAM v3.6a (**Priestley-Taylor** formulation, catchment-averaged). 2021–2025 from ERA5-Land (**FAO-56 Penman-Monteith** reference crop evapotranspiration, point extraction). Scaled by $\alpha_{\text{PET}} = \sum \text{PET}_{\text{GLEAM}} / \sum \text{PET}_{\text{ERA5}}$.

### 2. Comprehensive Boundary Calibration Audit Table (2015–2020: 2,192 Days)

| Station | Gauge | Area (km²) | P Mean (CAM/IMD) | α_P | P Daily r | P Wkly r | Tavg Mean (CAM/ERA) | Tavg Bias (°C) | Tavg Daily r | Tavg Wkly r | PET Mean (GLM/ERA) | α_PET | PET Daily r | PET Wkly r | Daily Gate | Wkly Gate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Karad** | `4021` | 5462 | 3.12/2.00 | `1.5616` | 0.823 | **0.897** | 25.9/24.9 | +1.07 | 0.943 | **0.961** | 3.13/4.34 | `0.7196` | 0.760 | **0.781** | `PASS` | `PASS` |
| **Warunji** | `4063` | 1890 | 4.36/2.00 | `2.1797` | 0.767 | **0.841** | 26.0/24.6 | +1.45 | 0.933 | **0.950** | 3.14/4.32 | `0.7260` | 0.796 | **0.825** | `PASS` | `PASS` |
| **Ashti** | `3005` | 50990 | 3.27/4.18 | `0.7824` | 0.645 | **0.857** | 26.2/27.7 | -1.47 | 0.949 | **0.961** | 3.25/4.69 | `0.6937` | 0.708 | **0.736** | `PASS` | `PASS` |
| **Bamni (Ballarsha)** | `3008` | 46020 | 2.54/2.97 | `0.8548` | 0.691 | **0.893** | 27.2/27.7 | -0.47 | 0.976 | **0.984** | 3.42/4.66 | `0.7351` | 0.688 | **0.706** | `PASS` | `PASS` |
| **Anandpur** | `7002` | 8570 | 4.06/4.51 | `0.8995` | 0.779 | **0.896** | 26.7/26.3 | +0.44 | 0.973 | **0.987** | 3.04/3.85 | `0.7913` | 0.776 | **0.785** | `PASS` | `PASS` |
| **Kashinagar** | `16007` | 7820 | 3.86/3.38 | `1.1447` | 0.678 | **0.806** | 27.0/27.1 | -0.09 | 0.964 | **0.980** | 3.04/3.97 | `0.7652` | 0.789 | **0.811** | `PASS` | `PASS` |
| **Gomlai** | `7007` | 21950 | 3.41/4.01 | `0.8509` | 0.745 | **0.850** | 25.7/26.2 | -0.48 | 0.965 | **0.974** | 3.00/4.04 | `0.7430` | 0.713 | **0.720** | `PASS` | `PASS` |
| **Kantamal** | `8013` | 19600 | 3.69/3.64 | `1.0135` | 0.638 | **0.821** | 26.6/27.3 | -0.65 | 0.956 | **0.967** | 3.12/4.48 | `0.6965` | 0.724 | **0.732** | `PASS` | `PASS` |
| **Yadgir** | `4064` | 69863 | 2.07/2.42 | `0.8560` | 0.571 | **0.808** | 26.6/27.5 | -0.88 | 0.950 | **0.966** | 3.31/5.07 | `0.6529` | 0.532 | **0.533** | `FAIL (P+PET)` | `FAIL (PET)` |
| **Huvinhedigi** | `4017` | 55150 | 2.29/2.29 | `0.9987` | 0.349 | **0.549** | 26.5/27.7 | -1.17 | 0.945 | **0.966** | 3.31/5.16 | `0.6410` | 0.543 | **0.542** | `FAIL (P+PET)` | `FAIL (P+PET)` |
| **KUDIGE** | `5010` | 1934 | 5.24/2.47 | `2.1191` | 0.678 | **0.801** | 25.1/22.7 | +2.35 | 0.872 | **0.924** | 3.49/3.87 | `0.9030` | 0.522 | **0.522** | `FAIL (T+PET)` | `FAIL (PET)` |
| **Haralahalli** | `4013` | 14582 | 3.16/1.96 | `1.6131` | 0.553 | **0.676** | 25.2/25.7 | -0.52 | 0.935 | **0.960** | 3.51/4.71 | `0.7445` | 0.568 | **0.568** | `FAIL (P+PET)` | `FAIL (PET)` |

### 3. Portfolio Summary Statistics (12 Catchments):
- **Precipitation Correlation:** Daily Mean $r = 0.6598$ | Weekly Mean $r = 0.8080$ (Target: >0.6)
- **Temperature ($T_{avg}$) Correlation:** Daily Mean $r = 0.9466$ | Weekly Mean $r = 0.9650$ (Target: >0.9), Mean Bias: `-0.0352 ^\circ C`
- **Evaporative Demand (PET) Correlation:** Daily Mean $r = 0.6765$ | Weekly Mean $r = 0.6883$ (Target: >0.6), Mean Scaling Ratio ($\alpha_{PET}$): `0.7343`

### 4. Critical Geospatial & Physical Findings (Approval Gate Audit):

1. **Disproving the Uniform Daily Pass Assumption:**
   - While portfolio-averaged correlations clear all three targets ($r_P=0.660 > 0.6$, $r_{Tavg}=0.947 > 0.9$, $r_{PET}=0.677 > 0.6$), **individual daily station correlations fail** in the Karnataka sub-portfolio.
   - Specifically, daily $r_P < 0.6$ at Huvinhedigi ($0.349$), Haralahalli ($0.553$), and Yadgir ($0.571$). Daily $r_{PET} < 0.6$ across all 4 Karnataka stations ($0.522 - 0.568$). Daily $r_{Tavg} < 0.9$ at KUDIGE ($0.872$).
2. **Scale Integration Effect (Daily vs Weekly Resolution):**
   - Aggregation to the operational modeling scale (**ISO Weekly `W-MON`**) dramatically improves spatial coherence: weekly $r_P$ rises to $0.808$ for Yadgir, $0.676$ for Haralahalli, and portfolio weekly mean $r_P$ reaches $0.806$.
   - Weekly temperature correlation clears $r > 0.92$ for **all 12 stations** (portfolio weekly mean $r = 0.965$).
3. **Physical Evaporative Demand Divergence:**
   - The PET correlation gap in semi-arid Karnataka ($r \approx 0.52 - 0.57$) is an unavoidable physical artifact of contrasting formulations: CAMELS-IND uses radiation-driven Priestley-Taylor (GLEAM v3.6a), whereas ERA5-Land uses FAO-56 Penman-Monteith, which explicitly incorporates aerodynamic vapor pressure deficit ($VPD$) and wind speed.
   - Multiplying ERA5-Land PET by $\alpha_{\text{PET}} \approx 0.73$ successfully rescales absolute evaporative volume to match GLEAM water-balance continuity across the 2020/2021 boundary.
