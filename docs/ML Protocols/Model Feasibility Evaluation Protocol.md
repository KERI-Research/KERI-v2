# Model Feasibility Evaluation Protocol (Step 8)

This protocol defines how endpoint and horizon feasibility evaluation is performed for Step 8 synthetic experiments.

## Frozen endpoint and horizon protocol

- Evaluate one endpoint and one horizon per experiment.
- Do not pool endpoints, horizons, cohort classes, or partitions.
- Use endpoint-specific frozen feature and label artifacts from one immutable source run.

## Split and partition rules

- Split identity is validated against the split manifest hash chain.
- Patient isolation is required: a patient cannot appear in more than one partition.
- `temporal_holdout` is evaluation-only and must never influence fitting or selection.

## Label-state inclusion and exclusion

Supervised fitting/evaluation includes only:

- `positive`
- `eligible_negative`

The following are excluded from supervised fitting and metric computation:

- `censored`
- `competing_death`
- `excluded`

Excluded label states are still counted in reports.

## Train-only preprocessing and fitting

For both models:

- Imputation, encoding, and scaling are fit on training rows only.
- All-null and zero-variance columns are removed based on training data only and recorded.
- Hyperparameter search uses validation only.
- Test and temporal holdout are never used for preprocessing fit or selection.

## Metrics and not-evaluable behavior

Partition reports include:

- row and label-state counts
- AUROC (when both classes exist)
- average precision / AUPRC (when both classes exist)
- Brier score (descriptive synthetic metric)
- prediction distribution checks
- missing/finite prediction checks

Every metric name includes synthetic technical feasibility only language.

A partition is `not_evaluable` when:

- supervised rows are absent after label filtering, or
- both classes are not present, or
- configured minimum event count is not met.

## Leakage and shortcut checks

Step 8 runs deterministic audits for:

- feature policy and denylist violations
- patient isolation violations
- post-index / future-data lineage violations
- label-permutation sanity behavior
- compact metabolic/laboratory feature ablation
- train-to-temporal-holdout score-distribution drift (warning-only)

## Reporting language restrictions

- Do not label metrics as clinical performance.
- Do not present outputs as risk scores for care.
- Do not make diagnostic, screening, treatment, or patient-level claims.
- Treat all outputs as synthetic technical feasibility evidence only.
