# Feature Leakage Policy

Feature extraction accepts canonical data, patient indexes, and split assignments only. Outcome labels and future records are not function arguments. Source selection is bounded by `event_date <= index_date`, and lineage records the bounds and `contains_post_index_record` flag.

The existing Step 2 dictionary remains the source of truth for allowed features and denylist enforcement. Cancer/treatment/pathology/survival fields, diabetes diagnosis-age and insulin-use fields, endpoint labels, enrichment stratum, split names, and manifest metadata cannot become numerical features.

Warnings are non-blocking and serialized with `status: warning` and `passed: true`. Any future-dated lineage, unregistered feature, outcome-derived column, or mixed cohort class fails validation.
