# MetaboGuard v2 - Remaining Project Plan

**Status:** Steps 1–6 complete  
**Current stage:** Step 7 - production orchestration hardening  
**Scope:** Research infrastructure for clinician-reviewed prevention research. This project is non-diagnostic and does not produce patient-level clinical risk claims.

## Completed Foundation

Steps 1–6 established the canonical longitudinal schema, deterministic conversion and validation, endpoint/cohort construction, patient-isolated splitting, label-blind feature engineering, and model-free readiness/capability gating.

The 500-patient ordinary-incidence rehearsal completed generation and downstream artifact production successfully. It produced canonical data, endpoint cohorts, patient-isolated splits, feature artifacts, validation outputs, readiness outputs, and feasibility outputs. The run remains synthetic, simulation-only, and pipeline-rehearsal-only.

No predictive model, calibration metric, discrimination metric, clinical utility claim, or patient-level risk output has been created.

## Step 7 - Production Orchestration Hardening

### Objective

Make production synthetic runs auditable, resumable, and truthful at the top-level run-manifest boundary.

### Current issue

The completed 500-patient ordinary-incidence rehearsal contains completed generation, canonical, cohort, split, feature, and readiness artifacts. However, its top-level `manifest.json` remains in the initial pre-run state, reporting `status: running`, `completed_at: null`, empty batch records, zero generated patients, and `not_created` downstream statuses.

This stale manifest must not be treated as authoritative. The underlying `generation_manifest.json`, batch manifests, canonical artifacts, feature artifacts, and readiness artifacts establish that the pipeline work completed.

### Deliverables

- Add an idempotent production-manifest reconciliation/finalization routine.
- Rebuild the top-level `manifest.json` from completed underlying artifacts:
  - `generation_manifest.json`
  - all batch manifests
  - canonical validation reports
  - endpoint cohort manifests
  - split manifests
  - feature manifests
  - readiness reports
  - feasibility reports
- Record accurate:
  - run status
  - completion timestamp
  - requested and actual generated patient counts
  - batch records and return codes
  - canonical, cohort, split, feature, readiness, and feasibility stage statuses
  - source artifact hashes
  - bounded failure diagnostics where applicable
- Use atomic file writing for the top-level production manifest.
- Ensure the production CLI finalizes or records failure through a common exit path, including resumed or interrupted runs.
- Preserve `simulation_only: true`, `pipeline_rehearsal_only: true`, and `model_status: not_created` for synthetic runs.

### Acceptance criteria

- A completed 500-patient run receives a truthful top-level status such as `completed` or `completed_not_ready`.
- A partially completed run cannot be marked `completed`.
- A failed run records `failed`, the failing stage, and bounded diagnostics.
- Re-running finalization on an unchanged completed run yields byte-stable or semantically identical output.
- Automated tests cover:
  - successful completed-run reconciliation
  - incomplete/missing required artifacts
  - failed batch handling
  - repeated idempotent reconciliation
  - resumed-run finalization
- Full quality gates pass: pytest, coverage, Ruff, mypy, and lockfile verification.

### Exit condition

The repaired 500-patient ordinary-incidence rehearsal has a finalized, internally consistent top-level production manifest.

---

## Step 8 - Large-Scale Synthetic Rehearsal

### Objective

Demonstrate stable, reproducible throughput and artifact integrity at larger synthetic cohort sizes without making clinical or predictive claims.

### Sequence

Run cohort classes separately and never pool them:

1. `ordinary_incidence` at 5,000 patients.
2. Inspect all manifests, canonical validation, endpoint cohorts, splits, feature artifacts, readiness, and feasibility outputs.
3. If stable, run `ordinary_incidence` at 25,000 patients.
4. Only after ordinary-incidence review, run `enriched_incidence` separately as an explicit stress-test stratum.

### Required controls

- Keep run identity immutable: cohort class, population target, seed, batch size, and configuration digest.
- Preserve configured Java executable, pinned Synthea JAR, JVM options, and batch configuration in manifests.
- Use the tested Windows JVM configuration:
  - `-Xmx4g`
  - `-XX:+UseSerialGC`
  - `-XX:ActiveProcessorCount=4`
- Resume only batches whose canonical artifacts match their recorded hashes.
- Never silently change seeds, population targets, or cohort classes.
- Never silently reduce a requested production target.
- Never run ordinary-incidence and enriched-incidence together.
- Never pool ordinary-incidence and enriched-incidence data, labels, metrics, or feasibility conclusions.

### Required outputs per run

- Production `manifest.json`
- Step 3 `generation_manifest.json`
- Per-batch manifests and logs
- Frozen canonical Parquet data
- Dataset validation report
- Endpoint-specific cohort artifacts
- Patient-isolated split artifacts
- Label-blind pre-index feature artifacts and lineage
- Readiness report
- Endpoint and horizon feasibility report
- Cross-run summary report

### Interpretation limits

Synthetic runs can test determinism, stability, throughput, artifact contracts, leakage controls, and endpoint feasibility. They cannot establish disease prevalence, predictive performance, calibration, discrimination, clinical utility, generalizability, or patient-level risk.

### Exit condition

At least one large ordinary-incidence run completes with truthful top-level manifest finalization and all required downstream artifacts. Enriched-incidence, if run, is explicitly documented and retained as a separate stress-test domain.

---

## Step 9 - Real Longitudinal Data Onboarding

### Objective

Map the first approved real longitudinal source into the already-frozen schema, cohort, split, feature, and readiness contracts.

### Required work

- Select one governed, approved longitudinal data source.
- Define data-access, privacy, governance, and provenance requirements before ingestion.
- Build a source-specific converter into the canonical schema.
- Version and document source mappings, unit conversions, code mappings, missingness handling, and dropped-code reporting.
- Run canonical validation with fail-closed rules.
- Construct endpoint-specific cohorts using frozen endpoint protocols.
- Generate patient-isolated development, validation, test, and temporal-holdout splits.
- Build label-blind pre-index features and feature lineage.
- Run readiness and capability gating endpoint by endpoint and horizon by horizon.

### Non-negotiable controls

- No post-index records in features.
- No outcome-derived variables in features.
- No patient overlap across splits.
- No silent mapping or unit-conversion failures.
- No claim of model readiness when any required artifact, integrity, leakage, split, event-count, or feature-quality gate fails.

### Exit condition

A complete non-synthetic artifact set exists for at least one endpoint and is evaluated by the existing readiness/capability process.

---

## Step 10 - Gated Research-Model Preparation

### Objective

Begin research-model preparation only for non-synthetic endpoint-and-horizon datasets that pass all mechanical gates.

### Entry criteria

The relevant non-synthetic artifact set must pass:

- Required-artifact existence and integrity checks
- Complete feature-build checks
- Feature lineage checks
- Leakage checks
- Patient-isolated split checks
- Temporal-holdout checks
- Event and eligible-negative thresholds
- Feature missingness and quality thresholds
- Dataset governance and provenance checks

The artifact set must be classified as `eligible_for_future_model_research`. Synthetic data is never sufficient for this classification.

### Allowed work

- Pre-register endpoint-specific modelling questions.
- Specify baselines, candidate models, evaluation metrics, and calibration procedures.
- Define development, validation, test, and temporal-holdout use rules before fitting.
- Specify subgroup, missingness, robustness, and sensitivity analyses.
- Define model cards and reporting templates.
- Implement reproducible research-only training and evaluation infrastructure.

### Prohibited work

- Clinical deployment
- Patient-specific medical advice
- Diagnostic claims
- Claims of clinical utility without appropriate external validation
- Pooling enriched synthetic data with ordinary-incidence or real-world cohorts
- Using synthetic data as clinical evidence

### Exit condition

A research-model protocol and reproducible evaluation plan are ready, with all model work still explicitly classified as research-only and subject to clinical review.

---

## Continuous Requirements

Across Steps 7–10:

- Keep all run artifacts versioned, immutable, and hash-linked.
- Preserve reproducibility through pinned dependencies, fixed seeds, and configuration digests.
- Fail closed when artifacts are missing, inconsistent, stale, or unverifiable.
- Keep simulation-only and real-data outputs visibly distinct.
- Keep enriched-incidence cohorts separate from ordinary-incidence cohorts.
- Maintain the project’s non-diagnostic, clinician-reviewed prevention-research framing.
- Run pytest, coverage, Ruff, mypy, and lockfile verification before closing each implementation step.
