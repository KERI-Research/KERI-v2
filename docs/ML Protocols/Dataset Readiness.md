# Dataset Readiness

Step 6 is a model-free gate over one immutable synthetic run and endpoint at a time. It inventories canonical, cohort, split, and feature artifacts; counts frozen labels; summarizes feature availability and lineage; and emits endpoint/horizon capability decisions.

The current smoke run has 1,019 eligible Type 2 diabetes indexes and a complete 1,019-row feature matrix. Readiness classifies this as `feature_build_status: complete`; its one-year decision remains `not_eligible` because the event count is below threshold. Synthetic runs that pass the mechanical gates are `prototype_ready`, while clinical model research remains unauthorized.

Readiness is not predictive performance, clinical validity, or permission for patient-level risk inference. Synthetic output is pipeline-rehearsal-only under the current configuration.

Step 6 inspects one frozen synthetic run at a time. It inventories canonical, cohort, split, and feature artifacts; counts labels independently by horizon; summarizes feature availability and lineage; and produces model-free capability decisions.

The current ordinary-incidence smoke run is complete: 1,019 eligible indexes are represented by 1,019 feature rows. The readiness result is therefore mechanically inspectable but not eligible for future clinical research-model work because the run is synthetic and the one-year event count is under threshold. It remains pipeline-rehearsal-only.

Readiness is not predictive performance, clinical validity, or permission for patient-level inference. No model, score, calibration, or predictive metric is produced.
