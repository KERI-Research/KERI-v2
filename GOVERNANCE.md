# MetaboGuard v2 Governance

## Intended use

MetaboGuard v2 is a research system for synthetic data. It may produce clinician-review signals for research analysis only. It does not diagnose, triage, recommend treatment, estimate prognosis, or replace clinical judgment.

Real patient inference is disabled by default and must return HTTP 409 until an approved real cohort is linked under the applicable governance process.

## Allowlist

Only these inputs may be used for development prediction and typing experiments:

- Pre-index dated metabolic measurements and their missingness indicators
- Patient profile variables declared in the canonical schema
- Incident outcomes dated after the index date, as labels only
- Synthetic augmentation outputs, explicitly marked as generated

## Denylist

The following fields must never be used as predictive features:

- Tumour site, stage, grade, histology, post-diagnosis pathology, or treatment
- Survival or any post-outcome information
- Diabetes diagnosis age or insulin use when predicting diabetes development
- Any event occurring after the index date
- Patient identifiers, direct identifiers, or source-system artifacts that can leak labels

Loading a denied column is a validation error, not a warning. The denylist will be enforced in executable code before feature generation.

## Release and result requirements

Every run must emit a manifest containing seeds, source artifact hashes, dataset SHA-256, split fingerprints, configuration hash, and Git SHA. Artifacts are invalid when the source data changes.

No horizon may be selected without at least 50 events and 50 eligible non-events. Test splits with fewer than 30 events are marked underpowered and must not emit point estimates without confidence intervals. Selected models must meet the configured usability floor and sequence models must pass the time-reversal control.
