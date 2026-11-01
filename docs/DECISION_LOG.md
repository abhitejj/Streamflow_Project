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

### Gate AG-TS-04: Ashti Raw CWC Data Anomaly (Pending User Approval)
* **Date:** October 31, 2026
* **Status:** **IDENTIFIED & ESCALATED**
* **Context:** On `05-10-2025`, raw CWC daily discharge table for Maharashtra contains a clerical data entry of **$1,829,351\,\text{m}^3/\text{s}$** at Ashti (Amazon River peak flow is $\sim 300,000\,\text{m}^3/\text{s}$).
* **Specialist Evidence:** This single clerical typo distorts pooled model weights in Maharashtra Protocol 2 cross-basin transfer.
* **Recommended Decision:** Apply an upper physical sanity bound ($Q \le 50,000\,\text{m}^3/\text{s}$ or historical max) to mask this single typo as unobserved `NaN`.
