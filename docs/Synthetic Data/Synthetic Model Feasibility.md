# Synthetic Model Feasibility (Step 8)

Step 8 adds a bounded, synthetic-only model feasibility workflow under explicit professor approval while real longitudinal data is unavailable.

## Scope and approvals

- Scope is one immutable production run, one endpoint, one horizon per experiment.
- Primary cohort for this step is `ordinary_incidence`.
- Primary endpoint for demonstration is `pancreatic_cancer`.
- Supported horizons are `1`, `3`, and `5` years when matching frozen labels exist.
- Execution requires both:
  - `--synthetic-feasibility`
  - `--approval-reference <reference>`

No run may proceed without completed synthetic provenance evidence, validated feature artifacts, cohort and split manifests, and source hash-chain checks.

## CLI usage

Preview only (no writes):

```bash
uv run metaboguard-model-feasibility \
  data/synthetic_longitudinal/production/ordinary_incidence/<run_id> \
  pancreatic_cancer \
  --horizon 3 \
  --synthetic-feasibility \
  --approval-reference professor-approval-YYYY-MM-DD
```

Execute and write artifacts:

```bash
uv run metaboguard-model-feasibility \
  data/synthetic_longitudinal/production/ordinary_incidence/<run_id> \
  pancreatic_cancer \
  --horizon 3 \
  --synthetic-feasibility \
  --approval-reference professor-approval-YYYY-MM-DD \
  --seed 1729 \
  --execute
```

## Artifact layout

Artifacts are written only under:

```text
artifacts/model_feasibility/
  <source_run_id>/
    <endpoint_id>/
      <horizon>y/
        <experiment_id>/
```

This path is separated from production generation and serving artifacts.

## What this demonstrates

- Train-only preprocessing and model fitting.
- Reproducible supervised feasibility baselines on frozen synthetic labels.
- Representation-head fitting with a train-fitted synthetic encoder.
- Held-out scoring for validation, test, and temporal holdout.
- Automated leakage and shortcut stress checks.
- Experiment manifests and mandatory model-card restrictions.

## What this cannot demonstrate

- Clinical validity or clinical utility.
- Diagnostic ability, screening ability, treatment guidance, or patient care suitability.
- Prevalence estimation, calibrated risk prediction validity, or generalization to real patients.

## Interpretation rules

Every output is tagged with:

- `simulation_only: true`
- `pipeline_rehearsal_only: true`
- `clinical_use_prohibited: true`
- `research_feasibility_only: true`

All metrics are reported as synthetic technical feasibility only. They are workflow evidence, not clinical performance evidence.
