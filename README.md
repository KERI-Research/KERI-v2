# MetaboGuard v2

[![wakatime](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/bac39155-6d8b-4e9d-ad6d-48260cdd076c.svg)](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/bac39155-6d8b-4e9d-ad6d-48260cdd076c) [![CI](https://github.com/JayNightmare/KERI-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/JayNightmare/KERI-v2/actions/workflows/ci.yml)

---

MetaboGuard v2 is a self-supervised metabolic early-warning research system built exclusively for synthetic longitudinal patient records and an approved fixed validation suite.

The system produces clinician-review signals. It is **non-diagnostic**, does not provide clinical advice, and must not be used for patient care. Real patient inference is disabled until an approved real cohort is linked.

## Status

Steps 1 through 8 are complete. Steps 1–6 established the canonical schema, deterministic conversion/validation, endpoint/cohort construction, patient-isolated splitting, label-blind feature engineering, and model-free readiness/capability gating. Step 7 hardened production orchestration with an idempotent manifest-reconciliation routine (`metaboguard-production --reconcile <run>`), so a completed synthetic run always carries a truthful top-level status instead of a stale pre-run manifest.

A 500-patient `ordinary_incidence` production run is complete end to end (canonical, cohort, split, feature, readiness, and feasibility artifacts for both `type2_diabetes` and `pancreatic_cancer`). Its top-level status is `completed_not_ready`: the pipeline is mechanically sound, but both endpoints remain `not_eligible` because the source is synthetic and neither endpoint reaches the configured 50-event threshold. No production-scale (5,000/25,000-patient) run has been executed yet, and no clinical model, score, evaluation result, API, or patient-level predictive output is authorized.

Step 8 adds a bounded, professor-approved, synthetic-only research track — see [docs/Synthetic Model Feasibility.md](docs/Synthetic%20Model%20Feasibility.md):

- `metaboguard-prototype` fits a label-free robust-PCA representation encoder on one approved synthetic run and writes an isolated artifact under `artifacts/model_prototypes/`, always with a model card disclaiming clinical use.
- `metaboguard-model-feasibility` runs endpoint- and horizon-specific supervised feasibility experiments (a logistic-regression baseline and a representation-head model) under an explicit `SyntheticFeasibilityAuthorization`, requiring both `--synthetic-feasibility` and a non-empty `--approval-reference`. Outputs are written only to `artifacts/model_feasibility/`, are permanently tagged `simulation_only`, `pipeline_rehearsal_only`, and `clinical_use_prohibited`, and ship with a model card and evaluation report.

Neither command enables real-patient inference, produces a clinical risk score, or constitutes a claim of clinical performance, calibration, prevalence, or generalisability. Real patient inference remains disabled until an approved real cohort is linked.

The next operational work is Step 9: large-scale synthetic rehearsal at 5,000 and 25,000 patients, run separately per cohort class. See [docs/Next Steps.md](docs/Next%20Steps.md) for the full remaining plan.

## Development

This project uses `uv` for reproducible Python environments.

```powershell
uv sync --dev
uv run ruff check .
uv run mypy src
uv run pytest
```

The test command enforces the configured coverage gate (currently 70%; the full suite passes with coverage well above that floor). No data, model binaries, generated biomarkers, or credentials belong in Git.

## Research constraints

- All source records are synthetic unless an explicitly approved cohort is configured.
- Augmented biomarkers are generated data and must never be represented as observed measurements.
- Feature construction, splitting, fitting, calibration, and evaluation will be governed by executable tests.
- Any emitted result must ship with a model card documenting synthetic data, non-diagnostic use, no clinical use, and the need for external validation on a real cohort.
