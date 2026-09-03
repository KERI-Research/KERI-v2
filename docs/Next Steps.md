# MetaboGuard v2 - Remaining Project Plan

**Status:** Steps 1–8 complete  
**Current stage:** Step 9 - large-scale synthetic rehearsal  
**Scope:** Research infrastructure for clinician-reviewed prevention research. This project is non-diagnostic and does not produce patient-level clinical risk claims.

## Completed Foundation

Steps 1–6 established the canonical longitudinal schema, deterministic conversion and validation, endpoint/cohort construction, patient-isolated splitting, label-blind feature engineering, and model-free readiness/capability gating.

The 500-patient ordinary-incidence rehearsal completed generation and downstream artifact production successfully. It produced canonical data, endpoint cohorts, patient-isolated splits, feature artifacts, validation outputs, readiness outputs, and feasibility outputs. The run remains synthetic, simulation-only, and pipeline-rehearsal-only. Its top-level manifest is now finalized by Step 7 reconciliation as `completed_not_ready` (both endpoints remain `not_eligible` on event-count grounds).

Step 8 added a bounded, professor-approved, synthetic-only model-feasibility track, exercised end to end against this same 500-patient run. No predictive model, calibration metric, discrimination metric, clinical utility claim, or patient-level risk output has been produced for real patients, and none is authorized until Step 9's real-data onboarding.

## Step 7 - Production Orchestration Hardening (Complete)

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

### Exit condition (met)

The repaired 500-patient ordinary-incidence rehearsal has a finalized, internally consistent top-level production manifest. `reconcile_production_manifest` is idempotent, atomic, fails closed on missing/incomplete artifacts, and is exposed through `metaboguard-production --reconcile <run_path>`.

---

## Step 8 - Synthetic Model Feasibility Study (Complete)

### Objective

Demonstrate, under explicit professor approval and while real longitudinal data remains unavailable, that the pipeline can fit and evaluate simple research-only models on frozen synthetic labels without making any clinical claim.

### What shipped

- `SyntheticFeasibilityAuthorization` requires both an explicit `--synthetic-feasibility` flag and a non-empty `--approval-reference`; the CLI fails closed without both.
- `metaboguard-model-feasibility` previews an immutable experiment plan by default and only writes artifacts with `--execute`, deriving a deterministic experiment ID from the source run, endpoint, horizon, model configuration, seed, and artifact hashes.
- Two research-only models: a regularized logistic-regression baseline on train-only-preprocessed engineered features, and a representation-head model built on a train-fitted robust-PCA/self-supervised encoder.
- Label-state handling restricted to `positive`/`eligible_negative` for fitting and metrics; `censored`, `competing_death`, and `excluded` rows are counted but never used as negatives.
- Deterministic leakage/shortcut audits: feature-policy and denylist checks, patient-isolation checks, post-index/lineage checks, label-permutation sanity checks, and train-to-temporal-holdout drift warnings.
- Every artifact (`experiment_manifest.json`, `synthetic_feasibility_report.json`, `MODEL_CARD.md`, per-partition evaluation reports) is tagged `simulation_only: true`, `pipeline_rehearsal_only: true`, `clinical_use_prohibited: true`, and `research_feasibility_only: true`.
- A separate `metaboguard-prototype` command fits the same label-free robust-PCA encoder as a standalone research artifact under professor approval, with its own non-diagnostic model card.
- `src/metaboguard/models/` also carries bounded, capability-gated model boundary modules for future research-model preparation, each with a mandatory non-diagnostic model-card restriction and no clinical authorization today:
  - `risk_heads.py` (`LogisticRiskHead`): a train-split-only, denylist-audited logistic risk head, fittable either under real `capability_report` authorization or explicit synthetic-prototype approval.
  - `calibration.py` (`RiskCalibrator`): a validation-split-only calibration mapping (isotonic when ≥50 rows and ≥10 events, otherwise Platt/logistic) that never touches train, test, or temporal-holdout rows, per [docs/v1-docs/FUTURE_RISK_EVALUATION.md](v1-docs/FUTURE_RISK_EVALUATION.md).
  - `typing_heads.py` (`TypingHeadModel`): a hard-gated boundary that always fails closed for cancer-type and diabetes-type classification, since those values are matched from canonical condition records rather than predicted (see [docs/Cohort Management/Cohort Construction.md](Cohort%20Management/Cohort%20Construction.md)).

See [docs/Synthetic Model Feasibility.md](Synthetic%20Model%20Feasibility.md) and [docs/Model Feasibility Evaluation Protocol.md](Model%20Feasibility%20Evaluation%20Protocol.md) for the full contract.

### Interpretation limits

This step demonstrates train-only preprocessing, reproducible fitting, held-out scoring, and leakage stress-testing on synthetic data. It does not demonstrate clinical validity, diagnostic or screening ability, prevalence estimation, calibrated risk-prediction validity, or generalization to real patients.

### Exit condition (met)

At least one endpoint-and-horizon experiment (`ordinary_incidence`, `pancreatic_cancer`, 3-year horizon) completed for both models against the 500-patient run, with full artifact sets, mandatory disclaimers, and passing quality gates.

---

## Step 9 - Large-Scale Synthetic Rehearsal

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

## Step 10 - Real Longitudinal Data Onboarding

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

## Step 11 - Gated Research-Model Preparation

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

Across Steps 7–11:

- Keep all run artifacts versioned, immutable, and hash-linked.
- Preserve reproducibility through pinned dependencies, fixed seeds, and configuration digests.
- Fail closed when artifacts are missing, inconsistent, stale, or unverifiable.
- Keep simulation-only and real-data outputs visibly distinct.
- Keep enriched-incidence cohorts separate from ordinary-incidence cohorts.
- Maintain the project’s non-diagnostic, clinician-reviewed prevention-research framing.
- Run pytest, coverage, Ruff, mypy, and lockfile verification before closing each implementation step.
