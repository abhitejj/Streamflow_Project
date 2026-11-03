# 📋 Project Decision & Approval Gates Log (ES 335)

This document tracks all formal methodological, architectural, and data governance decisions made during the project. Every significant modification passed a formal human approval gate with explicit evidence before code execution.

---

### Gate AG-01: Relaxation from Macro-Basin to Sub-Basin Zero-Overlap Constraint
* **Date:** October 31, 2026
* **Status:** **APPROVED BY USER**
* **Context:** The initial proposal mandated zero overlap across major river basins (Krishna, Godavari, Tapi). However, the coastal station *Durvesh* failed data continuity constraints (>15% missingness). Finding a continuous replacement station outside the major basins was physically impossible within Maharashtra.
* **Specialist Evidence:** Geospatial engineering audit showed that major river basins cover continental scales spanning divergent hydro-climatic zones, whereas distinct sub-basins represent hydrologically decoupled systems with independent drainage geometries and elevation gradients.
* **Approved Decision:** Relaxed the macro-basin zero-overlap rule to a strictly enforced **sub-basin zero-overlap rule** across all three states (Maharashtra, Odisha, Karnataka) to prioritize the **46-year continuous timeline (1980–2025: 16,802 daily timesteps)**. Assigned Bamni (Gauge 3008, Wardha sub-basin) as the Maharashtra holdout.

---

### Gate AG-02: Missingness & Contiguous Gap Policy Relaxation
* **Date:** October 31, 2026
* **Status:** **APPROVED BY USER**
* **Context:** Several hydrologically critical stations experienced recording cessations during the dry season or the nationwide May 2020 COVID-19 lockdown, slightly exceeding a strict 30-day contiguous gap threshold.
* **Specialist Evidence:** Hydrology and statistical audits confirmed that discarding these stations would introduce geographical selection bias. Automated synthetic imputation across multi-month gaps would fabricate streamflow.
* **Approved Decision:** Retain stations with overall 46-year missingness $< 10.0\%$. Linearly interpolate minor gaps $\le 35$ days. For gaps $> 35$ days, leave daily observations strictly unobserved (`NaN`) and mark resampled weeks with $< 4$ daily observations as `NaN`, masking them out of loss and metric calculations to evaluate models exclusively on authentic physical observations.

---

### Gate AG-03: Dropping Artificial IMD-Only Constraint for Non-Rainfall Forcings
* **Date:** October 31, 2026
* **Status:** **APPROVED BY USER**
* **Context:** Sourcing temperature and PET exclusively from IMD gridded products was causing a severe structural break at the train/test seam (2020/2021) because CAMELS-IND historical data (1980–2020) incorporated GLEAM v3.6a PET and catchment-averaged temperature, while modern IMD lacks native multi-depth soil moisture and operational GLEAM PET.
* **Specialist Evidence:** Ingesting ERA5-Land reanalysis for 2021–2025 non-rainfall forcings ($T_{\text{avg}}, T_{\text{max}}, T_{\text{min}}$, and FAO-56 Penman-Monteith PET) aligns with CAMELS-IND. Boundary calibration over the 2015–2020 overlap period (2,192 daily timesteps) demonstrated strong correlation ($r = 0.947$ for $T_{\text{avg}}$, $r = 0.677$ for PET).
* **Approved Decision:** Dropped the artificial IMD-only constraint. Retained IMD 0.25° gridded rainfall for precipitation; ingested ERA5-Land reanalysis for temperature and PET with boundary scaling ($\alpha_{\text{PET}} \approx 0.734$, additive $\text{Bias}_T \approx -0.035^\circ\text{C}$). Pruned non-native multi-depth soil moisture.

---

### Gate AG-TS-01: Antecedent Precipitation Index (API) Chronological Weight Reversal
* **Date:** October 31, 2026
* **Status:** **APPROVED & IMPLEMENTED**
* **Context:** An audit of `scripts/prepare_weekly_dataset.py` revealed that rolling dot products applied decay weights forward instead of backward, mistakenly giving day $t-29$ weight 1.0 and day $t$ weight $0.85^{29} \approx 0.007$.
* **Specialist Evidence:** Mathematical proof of rolling array index ordering in pandas.
* **Approved Decision:** Reversed weight vector (`weights[::-1]`). Day $t$ now receives weight 1.0 and past days decay backwards chronologically.

---

### Gate AG-TS-02: Terminal $H$-Week Purge Embargo on Train and Validation Sets
* **Date:** October 31, 2026
* **Status:** **APPROVED BY USER**
* **Context:** Direct multi-step targets ($H \in \{1, 2, 4, 8\}$) for issue weeks in late 2010 (Train end) and late 2015 (Val end) reached across split boundaries into the subsequent out-of-sample evaluation blocks.
* **Specialist Evidence:** Time-Series ML Scientist proved that without an embargo, $H=8$ predictions on week 52 of 2010 use week 8 of 2011 as targets, leaking validation ground truth into training.
* **Approved Decision:** Purged the terminal $H$ weeks of the Training fold (`iloc[:-H]` before 2010-12-31) and the terminal $H$ weeks of the Validation fold (`iloc[:-H]` before 2015-12-31). Test set remains un-purged. Cross-split target leakage is strictly 0 weeks.

---

### Gate AG-TS-03 / AG-DR-01: Training-Only Drought Severity Threshold Fitting
* **Date:** October 31, 2026
* **Status:** **APPROVED & AUDITED**
* **Context:** Setting drought severity thresholds on the full dataset or on the Test fold constitutes lookahead label leakage.
* **Specialist Evidence:** Evaluation Auditor established the strict rule: Mild ($Q_{80}$), Moderate ($Q_{90}$), and Severe ($Q_{95}$) thresholds must be computed exclusively on the purged Training fold (1980–2010).
* **Approved Decision:** All categorical drought thresholds are frozen from the Training fold and applied out-of-sample to Validation and Test predictions.

---

### Gate AG-TS-04: Ashti Raw CWC Data Anomaly (Upper Physical Limit Masking)
* **Date:** November 1, 2026
* **Status:** **APPROVED & IMPLEMENTED**
* **Context:** On `05-10-2025`, raw CWC daily discharge table for Maharashtra contained a clerical typo of **$1,829,351\,\text{m}^3/\text{s}$** at Ashti (and `69,514 m³/s` at Kudige).
* **Specialist Evidence:** Geospatial and Time-Series audits proved these were clerical data entry errors (e.g. missing decimal point in 1829.351 m³/s), distorting pooled training distribution in Maharashtra.
* **Approved Decision:** Updated `scripts/prepare_weekly_dataset.py` to enforce a strict physical upper threshold of $50,000\,\text{m}^3/\text{s}$. Any daily discharge exceeding this limit is masked as `NaN` (unobserved) before weekly resampling. Re-generated all 12 weekly tensors (Ashti peak streamflow cleanly corrected from $263,469\,\text{m}^3/\text{s}$ to $17,522.7\,\text{m}^3/\text{s}$).

---

### Gate AG-TS-05: Advance to Tier M2 Hydro-Meteorological Models (Training & Evaluation)
* **Date:** November 1, 2026
* **Status:** **IMPLEMENTED AND EVALUATED**
* **Context:** Transitioned from flow-only autoregression (M1) to full hydro-meteorological models (M2) incorporating IMD 0.25° rainfall, ERA5-Land PET/Temperature, and climatic water deficits.
* **Specialist Evidence & Results:**
  - Protocol 1 (Temporal Multi-Horizon): `HistGradientBoostingRegressor` evaluated out-of-sample across all 12 basins. Meteorological memory prevented the long-range skill collapse of climatology, with 10 of 12 catchments improving at $H=8\text{w}$ (median $\text{NSE} = 0.290$, $\text{KGE} = 0.308$) and severe drought false alarm ratios dropping by up to 27.4 percentage points.
  - Protocol 2 (Cross-Basin Spatial Transfer): PyTorch `HydroLSTM` trained on sliding $L=8$ week lookbacks of intensive depth-normalized features. Successfully generalized zero-shot to held-out sub-basins (Kantamal $H=8\text{w}$ $\text{NSE} = +0.403$, beating baseline by $+0.822\,\Delta\text{NSE}$; Bamni $H=1\text{w}$ $\text{NSE} = +0.339$).
* **Approved Decision:** Approved the completed empirical evaluation of Tier M2 hydro-meteorological models under strict terminal $H$-week purge embargoes and identical-row evaluation masks. Formally concluded the empirical modeling phase.



---

### Gate AG-04: Isolated Web Deployment & Visualization Architecture
* **Date:** November 2, 2026
* **Status:** **APPROVED & IMPLEMENTED**
* **Context:** Deploying the interactive project website required hosting Plotly widgets without exposing the underlying Python training pipeline, raw datasets, or incurring browser-side latency from parsing 46-year CSVs.
* **Specialist Evidence:** Software engineering audit confirmed that merging the backend main branch into a GitHub Pages deployment branch would leak sensitive data and bloat the repository. Generating dynamic plots client-side from raw CSVs would severely degrade mobile browser performance.
* **Approved Decision:** Created an orphaned gh-pages branch serving a static Tailwind CSS site. Plots are pre-rendered in Python as standalone Plotly .html files and embedded via lazy-loaded <iframe> tags. Implemented deploy.sh to enforce a one-way, automated artifact transition from main to gh-pages, ensuring absolute separation of computation and presentation while forcing automated documentation updates.
