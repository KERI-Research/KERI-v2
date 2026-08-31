# Feature Build Status

Step 6 reads feature completeness from actual matrix/index counts rather than trusting a parent feature-status claim. `not_started`, `partial`, `complete`, and `unknown` are distinct states.

Feature build status is one of `not_started`, `partial`, `complete`, or `unknown`. It is derived from actual feature-row count versus eligible-index count, not from a parent manifest claim alone.

The accepted smoke run now has a complete 1,019-row Type 2 diabetes feature matrix for its 1,019 eligible indexes, with 2,286,636 lineage rows and validation passing. Its readiness status is `complete`, derived from those artifact counts. The rebuild uses bounded extraction and explicit Parquet schemas to avoid the earlier Windows native-runtime failure.

The completed Step 7 production run (`ordinary_incidence-500-202608303-cb4d24b6e243`, 570 generated patients) also has complete feature builds for both configured endpoints: 6,236 pancreatic-cancer feature rows and 5,396 Type 2 diabetes feature rows, both validated. Both endpoints remain `not_eligible` at every horizon because the source is synthetic and event counts stay below the configured 50-event threshold; the production top-level manifest is finalized as `completed_not_ready`.

The complete synthetic feature set is suitable for pipeline rehearsal, reproducibility, and leakage inspection. It remains ineligible for clinical model research because the source is synthetic.
