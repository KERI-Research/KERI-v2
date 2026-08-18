# Feature Engineering

Step 5 creates deterministic, endpoint-agnostic feature rows from canonical observations and immutable Step 4 patient indexes/split assignments. It does not load horizon labels, outcomes, censoring states, competing-death states, or model artifacts.

Finite windows use `(index_date - window_days, index_date]`; lifetime uses `[birth_date, index_date]`. Latest values, summaries, baseline changes, descriptive ordinary-least-squares trajectories, measurement density, and binary missingness are computed without imputation. All source events are filtered before calculation and every feature group emits lineage.

Feature artifacts are simulation-only. They retain source hashes, registry/window hashes, split assignment, cohort and endpoint identifiers, and `model_status: not_created`. Feature engineering does not establish clinical validity or produce patient-facing output.
