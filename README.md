# MetaboGuard v2

[![wakatime](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/bac39155-6d8b-4e9d-ad6d-48260cdd076c.svg)](https://wakatime.com/badge/user/2d4d1d3d-9942-415a-87fc-0530a909486d/project/bac39155-6d8b-4e9d-ad6d-48260cdd076c) [![CI](https://github.com/JayNightmare/KERI-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/JayNightmare/KERI-v2/actions/workflows/ci.yml)

---

MetaboGuard v2 is a self-supervised metabolic early-warning research system built exclusively for synthetic longitudinal patient records and an approved fixed validation suite.

The system produces clinician-review signals. It is **non-diagnostic**, does not provide clinical advice, and must not be used for patient care. Real patient inference is disabled until an approved real cohort is linked.

## Status

Steps 1 through 6 are complete, and the Step 7 production-generation and endpoint-feasibility workflow is implemented. The current synthetic ordinary-incidence smoke run has a complete feature build and passes leakage validation, but its readiness decision is `not_eligible` because synthetic data is restricted to pipeline rehearsal and the one-year endpoint has fewer than the configured 50 events. No production-scale run has been executed automatically, and no clinical model, score, evaluation result, API, or patient-level predictive output is authorized.

The next operational work is executing the configured production-scale synthetic cohort plans and reviewing endpoint-specific feasibility reports. This work must preserve ordinary and enriched cohort separation and must not begin modelling, scoring, or prediction outside the single explicitly approved exception described below.

That exception is Step 8: bounded, professor-approved, synthetic-only model-feasibility experiments run through `metaboguard-model-feasibility`, requiring both `--synthetic-feasibility` and a non-empty `--approval-reference`. Outputs are written only to `artifacts/model_feasibility/`, are permanently tagged simulation-only, and carry a non-removable restriction against clinical use, patient screening, risk assessment, or care decisions. This exception does not enable real-patient inference, does not produce a clinical risk score, and does not constitute a claim of clinical performance, calibration, prevalence, or generalisability; real patient inference remains disabled until an approved real cohort is linked.

## Development

This project uses `uv` for reproducible Python environments.

```powershell
uv sync --dev
uv run ruff check .
uv run mypy src
uv run pytest
```

The test command enforces the configured coverage gate. No data, model binaries, generated biomarkers, or credentials belong in Git.

## Research constraints

- All source records are synthetic unless an explicitly approved cohort is configured.
- Augmented biomarkers are generated data and must never be represented as observed measurements.
- Feature construction, splitting, fitting, calibration, and evaluation will be governed by executable tests.
- Any emitted result must ship with a model card documenting synthetic data, non-diagnostic use, no clinical use, and the need for external validation on a real cohort.
