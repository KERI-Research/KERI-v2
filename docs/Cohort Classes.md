# Cohort Classes

The legacy simulation cohort classes are strictly separated:

- `ordinary_incidence`: native Synthea incidence for realism-oriented pipeline checks.
- `enriched_pancreatic`: an explicit sampling stratum intended to provide development coverage for pancreas events. It is not population-calibrated.
- `enriched_multicancer`: explicit strata for pancreas, colorectal, breast, prostate, and lung. Each site remains a separate endpoint and reporting stratum.

Step 7 also defines `enriched_incidence` as a separate production stress-test domain. It is not pooled with `ordinary_incidence` or with the legacy endpoint-specific enriched classes.

An operation that receives more than one cohort class raises `CohortClassMismatchError`. Ordinary and enriched classes cannot be pooled for fitting, splitting, metrics, calibration, or reports. Enrichment provenance, stratum, and weighting policy belong in the manifest and capability reports; enriched counts must not be read as population prevalence.

Synthea and these augmentation modules validate software plumbing, reproducibility, data contracts, and study protocol. They do not establish clinical performance or support use outside simulation-only research. Step 4 creates splits; Step 3 always emits `split_status: not_created`.
