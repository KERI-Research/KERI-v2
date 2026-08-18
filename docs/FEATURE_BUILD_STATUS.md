# Feature Build Status

Step 6 reads feature completeness from actual matrix/index counts rather than trusting a parent feature-status claim. `not_started`, `partial`, `complete`, and `unknown` are distinct states.

Feature build status is one of `not_started`, `partial`, `complete`, or `unknown`. It is derived from actual feature-row count versus eligible-index count, not from a parent manifest claim alone.

The accepted smoke run now has a complete 1,019-row Type 2 diabetes feature matrix for its 1,019 eligible indexes, with 2,286,636 lineage rows and validation passing. Its readiness status is `complete`, derived from those artifact counts. The rebuild uses bounded extraction and explicit Parquet schemas to avoid the earlier Windows native-runtime failure.

The complete synthetic feature set is suitable for pipeline rehearsal, reproducibility, and leakage inspection. It remains ineligible for clinical model research because the source is synthetic.
