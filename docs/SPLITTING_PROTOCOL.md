# Splitting Protocol

Splits are created only after endpoint labels and cohort validation are frozen. Patient identity is the grouping unit: all rolling indexes for one patient receive one assignment, and no patient fingerprint appears in more than one split.

The development allocation is configured as train 65%, validation 15%, and test 20%. A separate temporal holdout is based on index-date quantiles and capped at 25% of eligible patients. A configured fixed date that would exceed the cap falls back to the quantile cutoff and records that fallback in the split manifest. Calendar years are not used as the only temporal rule.

Human-readable split reports contain SHA-256 patient fingerprints rather than raw identifiers. Split manifests include the root seed, endpoint and cohort hashes, cutoff/fallback rule, state counts, and `simulation_only: true`.

No metrics or model fitting are performed in Step 4. This split artifact is a frozen input for later feature and evaluation work.
