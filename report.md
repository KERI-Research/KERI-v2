# Step 8 Synthetic Feasibility Viewer Report

> Synthetic technical-feasibility artifacts only. Nothing rendered here is a clinical performance, calibration, or utility claim, and none of it may be used for diagnosis, screening, patient-level risk, or care decisions.

Experiments rendered: 2

## type2_diabetes / 3y / enriched_incidence

- Experiment ID: `enriched_incidence-5000-20260901-bf2f7ca20c96-enriched_incidence-type2_diabetes-3y-1729-2a1f8e43eb5f2306`
- Source run: `enriched_incidence-5000-20260901-bf2f7ca20c96`
- Status: completed
- Provenance discrepancy: False

| Model | Partition | Rows | AUROC | AUPRC | Brier |
| --- | --- | --- | --- | --- | --- |
| baseline | validation | 1286 | 0.533 | 0.245 | 0.243 |
| baseline | test | 1460 | 0.481 | 0.163 | 0.251 |
| baseline | temporal_holdout | 44818 | 0.610 | 0.018 | 0.229 |
| representation_head | validation | 1286 | 0.522 | 0.224 | 0.270 |
| representation_head | test | 1460 | 0.472 | 0.155 | 0.270 |
| representation_head | temporal_holdout | 44818 | 0.561 | 0.015 | 0.269 |

## pancreatic_cancer / 3y / ordinary_incidence

- Experiment ID: `run-001-ordinary_incidence-pancreatic_cancer-3y-1729-02b947d8173edae2`
- Source run: `run-001`
- Status: completed
- Provenance discrepancy: False

| Model | Partition | Rows | AUROC | AUPRC | Brier |
| --- | --- | --- | --- | --- | --- |
| baseline | validation | 2 | not_evaluable | not_evaluable | not_evaluable |
| baseline | test | 2 | not_evaluable | not_evaluable | not_evaluable |
| baseline | temporal_holdout | 2 | not_evaluable | not_evaluable | not_evaluable |
| representation_head | validation | 2 | not_evaluable | not_evaluable | not_evaluable |
| representation_head | test | 2 | not_evaluable | not_evaluable | not_evaluable |
| representation_head | temporal_holdout | 2 | not_evaluable | not_evaluable | not_evaluable |
