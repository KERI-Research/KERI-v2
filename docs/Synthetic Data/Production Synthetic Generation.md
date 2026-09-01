# Production Synthetic Generation

Step 7 provides configuration-driven, simulation-only cohort generation for pipeline-capacity rehearsal. It runs ordinary-incidence and enriched-incidence plans as separate immutable domains. A run identity includes cohort class, population target, seed, and configuration digest; completed paths are never reused for a different plan.

Each production run contains a production `manifest.json`, the underlying Step 3 `generation_manifest.json`, canonical data, endpoint-specific cohorts, patient-isolated splits, label-blind pre-index features, readiness reports, and feasibility outputs. Batch failures are recorded with stage and bounded diagnostics. A failed or partial run cannot claim its requested population target completed.

The orchestration reuses the frozen canonical converter, validator, endpoint cohort constructor, split writer, feature extractor, and readiness bundle. Canonical validation precedes cohort construction. Feature and readiness stages remain separate for each endpoint. The production manifest always retains `model_status: not_created`.

Enriched-incidence data is an explicit stress-test stratum. Augmentation provenance remains record-level and auditable; enriched event counts are not population prevalence and must not be used as clinical evidence. Synthetic generation can test determinism, throughput, artifact contracts, and endpoint feasibility, but cannot establish prevalence, predictive performance, calibration, discrimination, clinical utility, or patient-level risk.

Production configuration lives under `production_generation` in `configs/default.yaml`. Production targets are not silently reduced, seeds are not silently changed, and ordinary and enriched runs are never pooled.

The production CLI stops after Step 1-3 dataset generation by default. Pass `--pipeline` to also run the frozen Steps 4-6 (cohort construction, splitting, feature extraction, and readiness) for every configured endpoint in one command. A run that stops without `--pipeline` can be extended later by rerunning the same command with `--pipeline`; only missing stages are built.

Every run, whether freshly generated, resumed, interrupted, or extended, ends by reconciling its top-level `manifest.json` from durable underlying artifacts (`generation_manifest.json`, batch manifests, canonical/cohort/split/feature/readiness/feasibility artifacts). This produces a truthful status such as `completed`, `completed_not_ready`, `completed_prototype_ready`, `partial`, or `failed`, and is idempotent: rerunning it against an unchanged completed run yields the same result. Reconcile an existing run directly without generating anything:

```bash
uv run metaboguard-production ordinary_incidence --reconcile data/synthetic_longitudinal/production/ordinary_incidence/<run_id>
```

## Production CLI

Use the `metaboguard-production` command for Step 7 runs. It uses the configured Java executable, JVM options, pinned Synthea JAR, output root, and cohort-specific production plan. It never runs both cohort classes from a single command.

The tested Windows/Temurin configuration uses a 4 GB heap, Serial GC, and four active processors. Serial GC is required because Temurin 17 G1 has produced native `jvm.dll` access violations on this host even when physical memory and pagefile capacity remain available. The `TotalPageFile` value in a JVM crash report is system capacity, not the amount requested by Java.

Preview a plan first. This prints the immutable plan and does not start Java or write artifacts:

```bash
uv run metaboguard-production ordinary_incidence
```

Start a run only with `--execute`:

```bash
uv run metaboguard-production ordinary_incidence --execute
```

`--patients` selects a requested Synthea population. The default is `500`, intended for an end-to-end operational rehearsal. `5000` and `25000` are larger explicit requests. `all` uses the configured production target for the selected class:

```bash
# Default 500-patient operational rehearsal.
uv run metaboguard-production ordinary_incidence --execute

# Request 5,000 patients using the configured seed and batch size.
uv run metaboguard-production ordinary_incidence --patients 5000 --execute

# Request 25,000 patients using the configured production target.
uv run metaboguard-production ordinary_incidence --patients all --execute
```

Use `--seed` to create a distinct deterministic run and `--batch-size` to control the requested Synthea population per batch. These values are included in run identity and batch manifests:

```bash
uv run metaboguard-production ordinary_incidence \
 --patients 5000 \
 --seed 20260819 \
 --batch-size 1000 \
 --execute
```

Monitor an active run by following its per-batch `synthea.log` files or a shell log captured with `tee`:

```bash
uv run metaboguard-production ordinary_incidence --patients 5000 --execute \
 2>&1 | tee data/synthetic_longitudinal/production/_run_logs/ordinary_incidence_5000.log
```

Production run IDs are deterministic from cohort class, requested population, seed, and configuration hash. Repeating the same command resumes only batches whose canonical artifacts match their recorded hashes; it does not rerun valid completed batches. Do not delete a run directory to recover from an interruption unless its manifest and retained batch artifacts have first been reviewed.

Synthea can generate more than the requested population because connected households or family members may be completed together. The manifest records both `population_target` and `generated_patient_count`; the latter is the actual generated count. A successful production run remains `simulation_only`, `pipeline_rehearsal_only`, and `model_status: not_created`.

Run `ordinary_incidence` first and review its canonical validation, cohort, feature, readiness, and feasibility artifacts before separately starting `enriched_incidence`:

```bash
uv run metaboguard-production enriched_incidence --patients 500 --execute
```

The completed 500-patient `ordinary_incidence` run (`ordinary_incidence-500-202608303-cb4d24b6e243`) has a finalized top-level status of `completed_not_ready`: canonical, cohort, split, feature, readiness, and feasibility artifacts all exist for both `type2_diabetes` and `pancreatic_cancer`, but both endpoints remain `not_eligible` because the source is synthetic and event counts fall below the configured threshold. Downstream, bounded, professor-approved research use of this run's artifacts is documented in [Synthetic Model Feasibility.md](Synthetic%20Model%20Feasibility.md).
