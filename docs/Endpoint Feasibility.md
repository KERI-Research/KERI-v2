# Endpoint Feasibility

Step 7 reports feasibility independently for each cohort class, run, endpoint, and horizon. It reads frozen label tables, split summaries, feature inventory, leakage audit, and readiness decisions. It does not fit a model or calculate a predictive metric.

`positive` records count observed endpoint events by horizon. `eligible_negative` records count only indexes with complete follow-up and no endpoint or earlier competing death. `censored` and `competing_death` records remain separate and are never counted as eligible negatives.

The configured gates are independent:

- minimum events per horizon;
- minimum eligible non-events per horizon;
- minimum events in evaluation partitions;
- complete feature rows relative to eligible indexes;
- passed leakage audit;
- patient-isolated split integrity.

A synthetic run can therefore have sufficient mechanical pipeline capacity and be `prototype_ready` without being clinically authorized. Every feasibility report states `simulation_only: true`, `pipeline_rehearsal_only: true`, `prototype_modeling_authorized: true`, `clinical_model_research_authorized: false`, and `model_status: not_created`.

Feasibility reports do not contain AUROC, AUPRC, Brier score, calibration, accuracy, precision, recall, F1, hazard ratios, risk scores, rankings, or patient-level predictions. They do not measure clinical prevalence, predictive performance, clinical utility, or clinical validity.
