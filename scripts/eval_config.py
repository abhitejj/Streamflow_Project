"""Shared evaluation settings, imported by every script that scores or reports results."""

# These four stations use the approved recorded-zero-as-missing scoring policy.
# Their post-2010 increase in recorded zeros suggests changed monitoring coverage;
# monsoon-only operation is a possible explanation, not independently verified CWC
# metadata. Overlap agreement with CAMELS-IND is evidence about matching source
# records, not proof of the institutional cause or absence of every merge error.
# Any week whose issue-time or target flow is zero is excluded for every model.
# Haralahalli's dry-season zeros also occur in training and remain eligible.
ZERO_AS_MISSING_STATIONS = ('Karad', 'Warunji', 'Yadgir', 'Huvinhedigi')
