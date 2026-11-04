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

---

### Gate AG-TS-06: Basin Independence Audit & Reframe to 10 Independent Systems + 1 Nested Cascade (Option H1)
* **Date:** November 3, 2026
* **Status:** **APPROVED BY USER & IMPLEMENTED**
* **Context:** Comprehensive geospatial and hydrological network audit uncovered that the previous claim of "12 Independent Sub-Basins with Zero Overlap" was topologically invalid. In the Upper Krishna basin, Warunji ($1,890\,\text{km}^2$) is 100% nested within Karad ($5,462\,\text{km}^2$, comprising 34.6% of its area), and both are nested within Huvinhedigi ($55,150\,\text{km}^2$, comprising 3.4% and 9.9% of its area, respectively).
* **Specialist Evidence:** Geospatial Data Engineer and Hydrology Expert proved that rather than flawed spatial independence, these three stations form a physically compelling multi-scale drainage cascade along a 55,000 km² elevation transect from the high-rainfall Western Ghats headwaters to the semi-arid peninsular plains.
* **Approved Decision (Option H1):** Formally retired the "Zero Sub-Basin Overlap" claim. Reframed the experimental matrix as **10 hydrologically independent river systems and 1 multi-scale nested drainage cascade ($\text{Warunji} \subset \text{Karad} \subset \text{Huvinhedigi}$)**. All documentation, figures, and website text updated to accurately describe multi-scale propagation rather than false spatial independence.

---

> **Note (2026-11-04):** in counts, this is 10 river systems: 9 independent gauges plus 1 three-gauge cascade (Warunji ⊂ Karad ⊂ Huvinhedigi) = 12 gauges.

### Gate AG-TS-07: Karad Anthropogenic Regime Shift & Empirical Drought Baseline (Option C1)
* **Date:** November 3, 2026
* **Status:** **APPROVED BY USER & IMPLEMENTED**
* **Context:** An external audit identified that Karad's severe drought test-set PR-AUC was abnormally high (>0.95) despite low CSI. Investigation revealed a severe anthropogenic/operational regime shift starting around 2010: upstream Koyna trans-basin hydroelectric diversion (diverting dry-season flow westward to the Arabian Sea) combined with CWC non-monsoon gauge shutdown caused 361 out of 522 test weeks (69.2%) to record exactly $0.0000\,\text{m}^3/\text{s}$, compared to only 2.1% zero-flow weeks in the pre-2010 training fold.
* **Specialist Evidence:** Drought Modelling Scientist and Evaluation Auditor confirmed that under training-fitted $Q_{95} = 5.994\,\text{m}^3/\text{s}$, the test drought event base rate is 69.2% (not 5.0%), shifting the empirical no-skill PR-AUC baseline to 0.692 for Karad and 0.238 pooled across 12 basins.
* **Approved Decision (Option C1):** Maintained strict temporal evaluation purity by retaining the frozen training-fitted threshold ($Q_{95} = 5.994\,\text{m}^3/\text{s}$) rather than retroactively tuning thresholds on test data. Fully documented the anthropogenic diversion/shutdown regime shift and updated the true empirical no-skill PR baseline to 0.238 across all text, tables, and PR curves.

---

### Gate AG-TS-08: Strategic Pivot to Objective Statistical Storytelling
* **Date:** November 3, 2026
* **Status:** **APPROVED BY USER & IMPLEMENTED**
* **Context:** Rigorous audit of model metrics revealed two critical scientific realities: (1) Wilcoxon signed-rank tests across the 12 basins showed M2 hydro-met models were NOT statistically significantly superior to M1 flow-only models at any horizon ($p > 0.20$); and (2) after fixing a double-shift bug in climatology indexing, seasonal weekly climatology (M0 = +0.348 at 8w, +0.349 at 4w) actually outperformed both M2 (+0.290, +0.267) and M1 (+0.248, +0.308) at extended lead times.
* **Specialist Evidence:** Statistical Research Analyst and Evaluation Auditor demonstrated that hydraulic streamflow memory collapses completely to 0.0% by 8 weeks lead (proven via permutation feature importance), while local meteorological memory accounts for only 8.5% of variance, leaving annual astronomical calendar seasonality (72.0% of variance) as the dominant physical driver.
* **Approved Decision:** Approved the strategic pivot to complete intellectual honesty and rigorous statistical transparency. Explicitly communicate that M2 does not provide statistically significant continuous accuracy gains over M1 ($p > 0.20$), highlight seasonal climatology as the winning continuous model at 4- and 8-week horizons, and focus M2's true practical value on short-term severe drought false alarm suppression and multi-hazard early warning.
