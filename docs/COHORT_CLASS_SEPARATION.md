# Cohort Class Separation

Step 7 maintains separate synthetic data domains:

- `ordinary_incidence`: native Synthea incidence for baseline pipeline rehearsal;
- `enriched_incidence`: generation-controlled stress-test coverage for endpoint feasibility.

They are never concatenated, pooled, balanced, jointly split, jointly summarized, or jointly assessed. Every run path, manifest, canonical table, cohort artifact, feature artifact, readiness report, and feasibility row carries its cohort class. Mixed classes fail closed through the shared cohort-class guard and production table validation.

Enriched incidence is not representative prevalence, real-world performance, or clinical evidence. Any augmented record retains `provenance: augmented`, its augmentation module, assumptions, and code/module hashes. Enrichment must not inject labels, treatment, pathology, survival, censoring state, or post-index outcome information into features.

Permitted comparisons are separate per-class operational summaries, such as generated counts, artifact completeness, runtime behavior, and endpoint feasibility. Prohibited comparisons include pooled event rates, pooled feature statistics, pooled split counts, pooled readiness decisions, and any cross-class model preparation.

All Step 7 outputs remain `simulation_only` and `pipeline_rehearsal_only`, with `model_status: not_created`.
