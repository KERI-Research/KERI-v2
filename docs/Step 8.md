# Implement MetaboGuard v2 Step 8 only: Synthetic Model Feasibility Study

Do not modify, weaken, bypass, or reinterpret any real-data authorization, governance, readiness, clinical-use, or serving gates. This step is an explicitly bounded, professor-approved, synthetic-only research feasibility study while Biobank data remains unavailable.

## Purpose

Build a reproducible, endpoint-specific model-feasibility pipeline that demonstrates the technical possibility of:

- train-only preprocessing;
- fitting simple supervised research baselines on frozen synthetic labels;
- fitting an endpoint head on the existing synthetic self-supervised/robust-PCA representation;
- held-out scoring;
- reproducible, artifact-backed evaluation;
- leakage/shortcut stress testing;
- model cards and experiment manifests.

This step must not claim clinical utility, clinical validity, diagnostic ability, screening ability, risk-prediction validity, prevalence estimation, calibration validity, generalization, or patient-level medical applicability.

All outputs must be visibly tagged:

- `simulation_only: true`
- `pipeline_rehearsal_only: true`
- `clinical_use_prohibited: true`
- `research_feasibility_only: true`

## Scope

Use only one immutable completed Synthea-derived production run and one endpoint at a time.

Primary demonstration cohort:

- `ordinary_incidence` only

Primary endpoint:

- `pancreatic_cancer`

Supported horizons:

- `1`, `3`, and `5` years, only if the frozen labels and partition event counts permit execution.

Do not pool:

- ordinary-incidence with enriched-incidence data;
- outcomes across endpoints;
- different horizons;
- train, validation, test, or temporal-holdout partitions.

The enriched-incidence cohort may be supported later as a separate stress-test input, but it must never be mixed into primary training, threshold selection, calibration, or reported metrics for ordinary incidence.

## Pre-work

First inspect the current repository layout, model modules, existing `metaboguard-prototype` command, feature artifacts, split artifacts, readiness/capability logic, and tests.

Preserve the existing synthetic prototype command and its professor-approval model card path. Step 8 adds a separate supervised feasibility-evaluation path; it must not silently convert the existing unsupervised prototype into a clinical or general model path.

Run focused tests before implementation changes. Use the current project dependency and quality-gate conventions. Do not change repository layout unless an existing ownership boundary makes it necessary.

## Required safety gates

Create an explicit `SyntheticFeasibilityAuthorization` contract.

The command must require all of the following:

1. An explicit `--synthetic-feasibility` flag.
2. A non-empty `--approval-reference`.
3. A completed immutable source run.
4. Durable evidence that the run is Synthea-derived.
5. A feature manifest with successful feature validation.
6. A cohort manifest and split manifest.
7. A source-run configuration/hash chain.
8. An endpoint-specific feature matrix containing frozen split assignments.
9. An output directory separate from:
   - normal production model artifacts;
   - serving artifacts;
   - existing generic prototype artifacts.

The command must fail closed if any required artifact is missing, unreadable, mismatched, non-synthetic, has inconsistent cohort class/endpoint identity, or cannot establish the source hash chain.

Handle historical provenance inconsistencies conservatively:

- if Synthea provenance is durably established from generation evidence but one historical downstream manifest has a contradictory flag, permit execution only with explicit professor approval;
- persist `source_provenance_discrepancy: true`;
- enumerate the contradictory artifact paths and observed values;
- include the discrepancy in every experiment manifest, report, and model card;
- do not rewrite or conceal historical source manifests;
- reject any run where synthetic provenance cannot be established from durable generation evidence.

No CLI path may permit model feasibility evaluation without both flags:

- `--synthetic-feasibility`
- `--approval-reference <reference>`

## CLI

Add a dedicated command, for example:

`metaboguard-model-feasibility`

Example:

```bash
uv run metaboguard-model-feasibility \
  data/synthetic_longitudinal/production/ordinary_incidence/<run_id> \
  pancreatic_cancer \
  --horizon 3 \
  --synthetic-feasibility \
  --approval-reference professor-approval-YYYY-MM-DD \
  --seed 1729
```

The CLI must:

- preview a resolved immutable experiment plan unless `--execute` is supplied;
- require `--execute` to write artifacts;
- refuse an unsupported endpoint or missing horizon;
- default to no overwrite;
- derive a deterministic experiment ID from source run ID, cohort class, endpoint, horizon, model configuration, seed, feature/split hashes, and approval-reference digest;
- resume only if every existing artifact hash matches its manifest;
- never silently overwrite a completed experiment;
- emit concise, bounded failures without raw patient-level details.

## Inputs and label handling

Load the frozen endpoint-specific feature matrix, the corresponding horizon-label artifact, and the patient-isolated split assignments.

Use only labels with states:

- `positive`
- `eligible_negative`

Exclude from supervised fitting and metric calculation:

- `censored`
- `competing_death`
- `excluded`

Do not reinterpret censored or competing-death records as negatives.

Confirm before fitting that:

- all rows belong to exactly one requested endpoint;
- all rows belong to exactly one cohort class;
- feature matrix and label index keys match exactly;
- all training rows are assigned to `train`;
- all validation rows are assigned to `validation`;
- all final test rows are assigned to `test`;
- all temporal-holdout rows are assigned to `temporal_holdout`;
- no patient fingerprint or raw patient ID appears in multiple partitions;
- no post-index or label-derived fields are included as predictors.

Do not use `split`, patient ID, raw dates, endpoint labels, horizon labels, cohort class, enrichment status, manifest fields, source hashes, diagnosis/treatment/prognosis fields, or any denylisted feature as numerical model inputs.

Respect the existing feature dictionary and leakage policy as the source of truth. Any unregistered, future-dated, outcome-derived, post-index, or denylisted predictor must fail validation.

## Models

Implement exactly two initial research-only models.

### Model A: baseline

A regularized logistic regression trained on train-only preprocessed engineered features.

Requirements:

- use a deterministic random seed;
- imputation fit only on training data;
- categorical encoding fit only on training data;
- scaling fit only on training data;
- all-null training columns excluded and recorded;
- zero-variance training columns excluded and recorded;
- preserve an ordered retained-feature registry;
- use class weights or an explicitly documented imbalance strategy fit only from training labels;
- select a small pre-registered hyperparameter grid using validation data only;
- do not search hyperparameters on test or temporal holdout;
- persist chosen configuration and validation rationale.

### Model B: representation-head prototype

Use the existing synthetic robust-PCA/self-supervised encoder path.

Requirements:

- fit the representation encoder on training partition rows only;
- score validation, test, and temporal holdout using the persisted train-fitted encoder;
- fit a regularized logistic-regression endpoint head using train embeddings and train labels only;
- perform any head hyperparameter selection on validation only;
- do not use labels during encoder transformation/scoring;
- persist encoder configuration, retained source features, latent dimensions, and head configuration;
- use the same split restrictions and synthetic-only restrictions as Model A.

Do not add deep neural networks, survival modelling, calibration models, clinical thresholds, explainability claims, or serving endpoints in this step.

## Evaluation

Create a synthetic technical-feasibility report for each model and partition.

Compute only after models are fitted and predictions are frozen:

- row count;
- positive count;
- eligible-negative count;
- excluded/censored/competing-death counts;
- AUROC where both classes are present;
- average precision / AUPRC where both classes are present;
- Brier score as a descriptive synthetic metric where feasible;
- confusion matrices only at a clearly non-clinical, pre-specified demonstration threshold, if included;
- prediction distribution summaries;
- missing-score count;
- finite-score checks;
- deterministic reproducibility hashes.

All metric labels must contain the words:

`synthetic technical feasibility only`

Do not:

- call any metric “clinical performance”;
- claim a model is accurate or clinically predictive;
- report a “risk score”;
- create clinical thresholds;
- calibrate on test/holdout;
- pool partitions;
- pool horizons;
- pool endpoints;
- hide failed or underpowered partitions.

If an evaluation partition lacks both classes or does not meet configured minimum events, write an explicit `not_evaluable` status and reason instead of raising a silent error or manufacturing a metric.

The temporal holdout is evaluation-only. It must never influence preprocessing, feature selection, encoder fitting, baseline fitting, hyperparameter selection, threshold selection, or calibration.

## Leakage and shortcut checks

Implement deterministic, automated stress checks.

Required checks:

1. Feature policy audit:
   - confirm all predictors are registered and allowed;
   - confirm no denylisted/outcome-derived/post-index/split/metadata columns enter the model matrix.

2. Patient isolation audit:
   - confirm no patient appears in more than one partition.

3. Time reversal / future-data check:
   - use existing lineage and index boundaries;
   - fail if any selected feature has post-index evidence.

4. Label permutation sanity check:
   - train one small train-only baseline on deterministically permuted training labels;
   - score the same fixed test partition;
   - record the result as a sanity check only;
   - do not assert a particular AUROC because synthetic datasets may behave unusually;
   - flag, but do not conceal, suspiciously similar performance to the unpermuted model.

5. Feature ablation:
   - evaluate at least:
     - all allowed retained features;
     - a compact metabolic/laboratory subset derived strictly from the registry;
   - retain separate artifacts and reports;
   - do not cherry-pick the best result.

6. Train-to-holdout drift:
   - calculate descriptive feature-availability and score-distribution drift;
   - write warnings rather than clinical inference;
   - no medical subgroup conclusions.

Every audit must be written to the experiment output and linked from the main experiment manifest.

## Artifact layout

Write only under a separate path such as:

```text
artifacts/model_feasibility/
  <source_run_id>/
    <endpoint_id>/
      <horizon>y/
        <experiment_id>/
          experiment_manifest.json
          authorization.json
          input_artifact_hashes.json
          preprocessing_manifest.json
          feature_selection_manifest.json
          split_audit.json
          leakage_audit.json
          shortcut_checks.json
          baseline/
            model.pkl
            predictions_validation.parquet
            predictions_test.parquet
            predictions_temporal_holdout.parquet
            evaluation_report.json
          representation_head/
            encoder.pkl
            head.pkl
            predictions_validation.parquet
            predictions_test.parquet
            predictions_temporal_holdout.parquet
            evaluation_report.json
          MODEL_CARD.md
          SYNTHETIC_FEASIBILITY_REPORT.md
```

Use atomic writes: write to a temporary sibling path and replace only after each artifact has been fully written and validated.

The experiment manifest must include:

- experiment ID;
- created timestamp;
- source run ID;
- cohort class;
- endpoint and horizon;
- synthetic provenance evidence;
- provenance discrepancy state and details;
- approval reference or its digest;
- all input file paths and SHA-256 hashes;
- feature definition and endpoint-protocol versions/hashes;
- split manifest hash;
- random seed;
- package/version metadata and git SHA where available;
- train/validation/test/temporal row counts and label-state counts;
- model status;
- artifact paths/hashes;
- clinical-use restriction fields;
- explicit `not_for_clinical_use` text;
- final experiment terminal status.

No raw patient identifiers may appear in JSON, Markdown, logs, model cards, or reports.

Prediction parquet files may contain a privacy-minimised, non-reversible row key or existing approved fingerprint only, plus:

- partition;
- endpoint;
- horizon;
- model identifier;
- prediction/probability;
- experiment ID;
- required synthetic-only flags.

Do not include raw labels in user-facing prediction artifacts. If internal label linkage is needed for evaluation, store it in a restricted evaluation-only artifact with no raw patient identifiers and document the boundary.

## Mandatory model card

Every experiment must write `MODEL_CARD.md` containing this prominent restriction, with no ability to disable it:

> Synthetic technical-feasibility prototype only. This artifact was trained and evaluated exclusively on Synthea-derived synthetic data. It is not a diagnostic device and must not be used for clinical decision-making, patient screening, risk assessment, medical advice, treatment decisions, triage, or patient care. Reported results demonstrate software and research-workflow feasibility only; they do not establish predictive performance, calibration, clinical validity, clinical utility, prevalence, or generalisability. External validation on an approved real longitudinal cohort is required before any clinical interpretation.

Include:

- purpose and intended use;
- prohibited uses;
- data provenance;
- endpoint and horizon;
- cohort-class isolation statement;
- model descriptions;
- train/validation/test/temporal use rules;
- preprocessing and missing-data handling;
- evaluation limitations;
- source provenance discrepancy, if present;
- reproducibility references;
- external-validation requirement.

## Documentation

Add:

1. `docs/SYNTHETIC_MODEL_FEASIBILITY.md`
   - scope, approval requirements, CLI usage, artifact layout;
   - what the study demonstrates;
   - what it cannot demonstrate;
   - separation from real-data modelling and clinical use;
   - interpretation rules for all reported metrics.

2. `docs/MODEL_FEASIBILITY_EVALUATION.md`
   - endpoint/horizon evaluation protocol;
   - frozen split rules;
   - label-state inclusion/exclusion;
   - train-only preprocessing rules;
   - metric definitions and `not_evaluable` behavior;
   - shortcut/leakage stress checks;
   - reporting language restrictions.

Update the root README only with a concise status statement that synthetic feasibility experiments exist under explicit approval and remain non-diagnostic, simulation-only, and unsuitable for clinical use. Do not imply real-data approval or clinical readiness.

## Tests

Add focused tests covering at least:

- CLI refuses execution without `--synthetic-feasibility`;
- CLI refuses execution without an approval reference;
- only completed Synthea-evidenced runs are accepted;
- missing/mismatched feature, cohort, label, or split artifacts fail closed;
- cohort class, endpoint, and horizon mismatch fail closed;
- inconsistent historical flags are rejected unless durable Synthea evidence plus approval is present;
- provenance discrepancies appear in every required report;
- synthetic flags and clinical-use prohibitions cannot be disabled;
- only positive and eligible-negative labels are used for supervised fitting/evaluation;
- censored, competing-death, and excluded rows are excluded;
- preprocessing is fit using train data only;
- all-null and zero-variance training columns are excluded and recorded;
- `pd.NA` categorical values are safely normalized;
- test and temporal-holdout rows do not influence preprocessing, encoder, head, model selection, threshold selection, or calibration;
- patient isolation is enforced;
- feature leakage/denylist checks fail on injected prohibited columns;
- baseline and representation-head models write the required artifacts;
- prediction artifacts contain no raw patient IDs;
- empty/single-class evaluation partitions return `not_evaluable`;
- model card contains the exact mandatory restriction text;
- artifact writing is atomic and no partial artifact remains after forced failure;
- deterministic repeated run/resume behavior is tested;
- no model feasibility artifact is written beneath serving or normal production-model directories.

Use small deterministic fixtures so tests remain fast. Do not require a 5,000-patient run to execute the test suite.

## Quality gates

Before completion, run and report:

```bash
uv run pytest -q
uv run ruff format .
uv run ruff check .
uv run mypy src
uv lock --check
```

Do not begin real-data ingestion, clinical serving, UI work, deep learning, calibration research, deployment, or any step outside Step 8.
