# Capability Gating

Step 6 applies ordered mechanical gates: required artifacts, feature completeness, integrity, leakage/split integrity, event counts, feature quality, and the synthetic-data restriction. Decisions are endpoint- and horizon-specific.

The real smoke feature build is `complete`: 1,019 feature rows cover 1,019 eligible indexes. Its capability decision is `not_eligible` because the source is synthetic and the one-year event count is below threshold. It remains `pipeline_rehearsal_only`; neither state implies a model result.

Capability decisions are endpoint- and horizon-specific and apply in order: required artifacts, complete feature build, artifact integrity, leakage/split integrity, event and eligible-negative thresholds, and feature quality. Synthetic status changes the successful outcome to `prototype_ready`; it does not replace the mechanical gates.

`blocked` means a required artifact, completeness, integrity, leakage, or split prerequisite is not satisfied. `not_eligible` means the mechanics are inspectable but event counts or feature quality prevent the requested work. `prototype_ready` means a synthetic artifact set passes the mechanical gates and may be used for approved prototype modeling, while clinical model research remains unauthorized. `eligible_for_future_model_research` is reserved for a complete, non-synthetic artifact set that passes all configured mechanical gates.

Synthetic data is always marked `simulation_only` and `pipeline_rehearsal_only` under the current configuration. A passing synthetic run is marked `prototype_ready`, with `prototype_modeling_authorized: true` and `clinical_model_research_authorized: false`. Censored and competing-death records are never counted as eligible negatives. Warnings use `status: "warning"` and `passed: true`; they remain visible without invalidating the readiness bundle.
