# Shared rules for ML course project agents

These rules apply to every agent in `.agents/agents/` and to any global copy.
Read the repository root `AGENTS.md` first. Treat files in `sources/` as
read-only reference material.

## Evidence before claims

Do not present a README assertion as evidence. Verify actual source metadata and
data before claiming a station identity, record continuity, data coverage,
weather-variable availability, catchment mapping, or comparability between CWC,
IMD, and CAMELS-IND sources. The current materials do not establish all of
these. State unknowns, failed checks, and scope limits plainly.

## Approval gates

Stop and request explicit human approval before you implement or recommend as
settled any of the following: a research-question change; raw-data deletion,
move, or replacement; a split, forecast-issuance, drought-threshold, or
primary-metric change; or a major model-architecture change. Record the
approval in the project decision log when one exists. No draft, filename, or
prior score counts as approval.

## Evaluation integrity

Use only information available at forecast issue time. Fit imputation, scaling,
feature selection, transformations, tuning, and drought thresholds on training
data only. Keep the final test block out of model selection. Report failed and
negative results; do not delete them.

## ES 335 website constraint

The final deliverable is a static GitHub Pages site. Interactions must expose
fixed, precomputed, traceable results and cannot imply live training or a server
backend. Every displayed conclusion needs an identifiable experiment artifact.

## Response contract

State: scope, evidence inspected, findings, limitations, recommended next step,
and whether an approval gate was reached. Make the smallest defensible claim.
