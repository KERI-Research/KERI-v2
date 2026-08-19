# Data Validation

`metaboguard.data.validation.validate` is the fail-closed validation boundary for a `CanonicalDataset`. Error checks include an offending row count plus up to ten example IDs and use `status: "failed"` when they fail. Warning checks are non-blocking: they keep `passed: true`, use `status: "warning"` when findings exist, and report the count in `warning_count`.

## Error checks

The validator checks referential integrity, chronology, death boundaries, derived age, plausible feature ranges, explicit missingness, canonical units, the feature denylist, augmentation module names, diabetes onset, cancer sites, and unique patient IDs.

The feature registry in `src/metaboguard/features/dictionary.py` is the only source of canonical units, plausible ranges, LOINC mappings, and allow/deny metadata. A denied feature raises `DeniedFeatureError` when loaded.

## Conversion and provenance

`to_canonical` reads Synthea-layout CSV exports, maps known LOINC codes, records every dropped code and count, performs only explicit unit conversions, deduplicates `(patient_id, event_date, feature_name)`, and writes sorted Parquet tables under the input directory's `canonical/` folder. Generated biomarkers retain `provenance="augmented"` and their module name.

The validator writes `artifacts/validation/dataset_validation_report.json`, including the dataset SHA-256, schema version, row counts, check results, and dropped-code table. This is the canonical validation-report filename. Parquet output is sorted and hashed across all canonical tables for reproducibility.

The deterministic fixture intentionally contains one missing glucose measurement. Its `missingness_rate` check therefore has `status: "warning"`, `passed: true`, and `warning_count: 1`; this is an expected non-blocking quality signal, not a failed validation.

Feature-registry plausible ranges are synthetic-export acceptance bounds for pipeline validation. They are not clinical reference intervals or evidence about human physiology. The live Synthea adapter records any source-specific date normalization separately; it does not relax the chronology validator for arbitrary pre-birth records.
