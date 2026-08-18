# Step 5 Artifact Inspection

Inspection date: 2026-08-16

Source Synthea run:

`data/synthetic_longitudinal/smoke/ordinary_incidence/ordinary_incidence-20260815T153605Z-20260815-4b7a75b7`

Inspected feature artifact set:

`features/type2_diabetes_inspection/`

The inspection used ten real eligible patient-index records from the accepted Synthea 3.3.0 ordinary-incidence run. It consumed the real canonical Parquet, real cohort indexes, and real split assignments. Feature extraction received no horizon labels, outcome tables, censoring states, or competing-death labels.

## Results

- Feature matrix rows: 10
- Feature matrix columns: 2,256 total, including identifier/provenance columns
- Forbidden label/outcome-derived columns: none detected
- Lineage rows: 22,440
- Lineage rows with `latest_source_date > index_date`: 0
- Feature validation: `passed: true`
- Feature warning records: non-blocking, `status: "warning"`, `passed: true`
- Feature manifest hashes: present for matrix, lineage, registry, validation, and quality reports
- `simulation_only`: `true`
- Parent `split_status`: `created`
- Parent `feature_status`: `created`
- Parent `model_status`: `not_created`

The artifact writer updated the actual parent cohort manifest only after feature validation passed. No model-related artifact or status was created. This inspection does not authorize model fitting by itself; it records that the real canonical/cohort/split path respected the Step 5 feature contracts for the inspected sample.
