# Capability Gating

Step 6 applies ordered mechanical gates: required artifacts, feature completeness, integrity, leakage/split integrity, event counts, feature quality, and the synthetic-data restriction. Decisions are endpoint- and horizon-specific.

The real smoke feature build is `complete`: 1,019 feature rows cover 1,019 eligible indexes. Its capability decision is `not_eligible` because the source is synthetic and the one-year event count is below threshold. It remains `pipeline_rehearsal_only`; neither state implies a model result.

Capability decisions are endpoint- and horizon-specific and apply in order: required artifacts, complete feature build, artifact integrity, leakage/split integrity, event and eligible-negative thresholds, feature quality, then the simulation-only restriction.

`blocked` means a required artifact, completeness, integrity, leakage, or split prerequisite is not satisfied. `not_eligible` means the mechanics are inspectable but event counts, feature quality, or simulation-only policy prevent later research-model work. `eligible_for_future_model_research` is reserved for a complete, non-synthetic artifact set that passes all configured mechanical gates.

Synthetic data is always marked `simulation_only` and `pipeline_rehearsal_only` under the current configuration. Censored and competing-death records are never counted as eligible negatives. Warnings use `status: "warning"` and `passed: true`; they remain visible without invalidating the readiness bundle.
