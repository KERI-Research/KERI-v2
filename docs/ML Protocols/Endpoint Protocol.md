# Endpoint Protocol

The endpoint registry in `configs/default.yaml` is the single protocol source for endpoint family, site/type, horizons, minimum history, rolling interval, prevalent exclusion, washout, competing-death handling, and enabled status.

Each endpoint uses the first matching canonical condition after the index date as its incident event. A condition on or before the index is prevalent and excludes that patient-index for that endpoint. Cancer site and diabetes type come from canonical condition metadata, not text recomputation in the labeling layer.

Labels are independently assigned at 1-, 3-, and 5-year horizons:

- `positive`: the first endpoint occurs on or before the horizon.
- `eligible_negative`: no endpoint occurs during the horizon and follow-up reaches the horizon without an earlier competing death.
- `censored`: follow-up ends before the horizon with no endpoint observed.
- `competing_death`: death occurs before the endpoint or horizon.
- `excluded`: reserved for explicit protocol exclusions in persisted exclusion records.

Cancer conditions are outcomes only. They are never included as pre-index feature inputs. Pancreatic washout is a reverse-causation sensitivity control and is recorded with an explicit reason. Type 1 development-risk labels are disabled in this step because the required governance and longitudinal biomarker protocol is not present.
