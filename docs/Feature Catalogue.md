# Feature Catalogue

The registry is generated from the Step 2 feature dictionary and configured windows. Each numeric source receives latest-value, summary, baseline/change, trajectory, and missingness families. Finite windows are recent 30 days, short 90 days, medium 365 days, and long 1095 days; lifetime starts at birth.

Latest and summary values preserve canonical units and return null when unavailable. Changes require two dated measurements; trajectories require three distinct dates spanning at least 30 days. Missingness is binary data availability and is never imputed. Measurement density describes observation intensity and is not a biological variable.

All definitions are versioned, simulation-only, pre-index-only, and non-diagnostic.
