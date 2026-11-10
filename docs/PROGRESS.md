# Project Progress & Deployment Log

This log tracks project milestones, pipeline updates, and deployments to `main` and `gh-pages`.

- **2026-11-02 00:47:46 IST**: Configure deployment automation and update web architecture documentation
- **2026-11-03 22:30:00 IST**: Phase 2 Independent Scientific Audit: completed root-cause diagnosis of 11 issues (A–K), including Karad zero-flows (Issue C), nested basin overlap (Issue H), and climatology double-shift bug (Issue D); established automated recomputation audit in scripts/audit_metrics.py.
- **2026-11-03 22:36:00 IST**: Phase 3 Execution: user approved Options H1 and C1 and strategic pivot; regenerated all 6 model results CSVs, recomputed 10 Plotly interactive HTML widgets with audited metrics, updated benchmark-data.js and index.html on gh-pages, and aligned all technical documentation.
- **2026-11-03 22:59:06 IST**: Completed metric recomputation, frontend rewrite, and documentation alignment
- **2026-11-04 15:38:35 IST**: Audited zero-shot HydroLSTM units and paired drought FAR metrics
- **2026-11-05 00:38:36 IST**: Add 2011-2020 CAMELS-CWC overlap lineage audit note
- **2026-11-05 01:14:38 IST**: Add drought severity sensitivity audit (Q80, Q90, Q95) and update documentation
- **2026-11-05 02:15:00 IST**: Finalize merged handoff: pipeline hardening (dynamic metrics lookups, all-lead climatology/persistence medians), report documentation corrections (lineage audit citation, 10 river systems, test-period outlier clarification, 2016-2020 CAMELS forcings sub-period analysis), and frontend deployment sync.
- **2026-11-05 16:06:51 IST**: Rebuild 2021-2025 catchment-mean rainfall, mask validation zeros, add log-flow models and CAMELS-era robustness
- **2026-11-05 16:52:00 IST**: Fix website text inconsistencies, update chart notes, and add 4-week persistence drought comparison
- **2026-11-06 00:12:29 IST**: Polish website text consistency, escape HTML entities, and document log-target and CAMELS-era scripts
- **2026-11-10**: Corrected retraining, Log-NSE and aridity documentation; fixed audited heatmap deltas and clipped scores; added linked station/lead controls, shareable selections, and five-model flow/drought comparisons with fixed severity levels. Playwright Chromium desktop/mobile checks and all 240 flow-model / 720 drought-row export checks pass. Historical replay expansion (Priority 3) remains pending.
