# ⚠️ CRITICAL ERROR: Protocol 2 Out-Of-Distribution Transfer Violation

- **Target Holdout Station:** `DURVESH` (Coastal WFRN Basin)
- **Hydrological Regime:** Coastal Hyper-Flashy (>3,000 mm/yr precipitation)
- **Violation:** Evaluating models trained on inland / semi-arid Deccan basins (Karad, Ashti, Yerli) against coastal basin `DURVESH` without explicit metric scaling.
- **Mandated Rule:** Halt execution. Do not execute out-of-distribution transfer tasks without explicit metric scaling (`--metric-scaling`).
