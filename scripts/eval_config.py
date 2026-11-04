"""Shared evaluation settings, imported by every script that scores or reports results."""

# Stations whose gauge records exactly 0.0 m³/s for weeks that were not measured.
# CWC ran these as monsoon-only stations after ~2010 (see docs/AUDIT_LOG.md, Issue C):
# test-period zero share is 62–70% at Karad / Warunji / Yadgir and 14% at Huvinhedigi
# (versus 0–20% in training), and ~20% of those zeros fall in Jun–Oct.
# At these stations a recorded zero is treated as unobserved: any week whose issue-time
# flow or target flow is a recorded zero is excluded from scoring for every model.
# Haralahalli's zeros are dry-season only and present in training, so they are kept.
ZERO_AS_MISSING_STATIONS = ('Karad', 'Warunji', 'Yadgir', 'Huvinhedigi')
