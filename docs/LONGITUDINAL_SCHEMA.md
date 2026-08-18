# Longitudinal Schema

MetaboGuard v2 uses the Pydantic v2 models in `src/metaboguard/data/schema.py` as the single source of truth for longitudinal data shape. The exported schema is versioned as `2.0.0` and written to `artifacts/schema/longitudinal_schema.json` with a SHA-256 sidecar.

## Canonical records

- `Patient` contains stable Synthea identity, birth date, sex, ethnicity, and optional death date.
- `ClinicalEvent` contains one dated measurement, explicit missingness, canonical units, provenance, and any required augmentation module.
- `ConditionRecord` contains dated conditions and requires diabetes type or cancer site when the category requires it.
- `OutcomeRecord` contains dated outcome labels linked to their source condition code.

Models run in strict mode with extra fields forbidden. Dates and numeric values are not implicitly coerced. CA 19-9, C-peptide, and insulin are always generated/augmented and cannot be represented as native observations.

The schema does not authorize post-index features. Temporal eligibility and feature-window rules are enforced by downstream cohort and feature modules in later build steps.
