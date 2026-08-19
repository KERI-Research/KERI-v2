# Synthea Generation

MetaboGuard v2 Step 3 generates simulation-only longitudinal cohorts in three separate classes:

- `ordinary_incidence` uses native Synthea incidence for realism-oriented software checks.
- `enriched_pancreatic` uses an explicit sampling stratum for development data and is not population-calibrated.
- `enriched_multicancer` samples configured sites separately; sites are not pooled into one endpoint.

The runner requires a configured Synthea release, JAR SHA-256, Java runtime, geography, age range, root seed, and batch size. The live smoke audit command is:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv"
uv run python -m metaboguard.data.synthea_runner `
 --cohort-class ordinary_incidence `
 --batches 1 `
 --patients-per-batch 100 `
 --seed 20260815 `
 --output-root data/synthetic_longitudinal/smoke `
 --dry-run false
```

This command requires `data/tools/synthea-with-dependencies.jar` and an installed Java runtime. The configured JAR SHA-256 must be populated before running it. Call `generate_synthea_cohort(SyntheaGenerationConfig(...))` directly when a caller needs a programmatic configuration. Production targets are configuration values and are not run automatically in CI. A fake batch generator is supported for unit tests; tests never download a JAR or invoke Java.

The 2026-08-15 live audit installed Temurin 17 and verified the Synthea 3.3.0 JAR hash. The exact 100-patient ordinary-incidence run completed after the narrow adapter normalization: exactly seven allowlisted birth-context observations were normalized to `BIRTHDATE`, each was recorded in `date_normalisation_audit.json`, validation passed with no error-level failures, and the raw batch was deleted only afterward. Any other pre-birth record remains a chronology error.

Two further narrow, source- and version-gated adapter rules were added from a live 5,000-patient audit. Synthea 3.3.0 emits UCUM `[iU]/L` for a small number of `alt` and `alkaline_phosphatase` observations; these are accepted as an explicit 1:1 alias of the canonical `U/L`. Synthea 3.3.0 also emits a small number of serum creatinine values at micromole-per-litre magnitude under an `mg/dL` label. Those are converted at `1/88.4` only when the source is `synthea` version `3.3.0`, the code is `2160-0`, the value is impossible as `mg/dL`, and the converted value becomes plausible. Every converted record is written to `unit_normalisation_audit.json`, hashed into the canonical dataset digest, and recorded in the generation manifest. No plausibility range was widened, and any value that remains implausible after conversion still fails closed.

Java runtime options are configuration-controlled through `production_generation.execution.jvm_options`. Each batch manifest records the effective Java executable, JVM options, exporter base directory, target size, and return code, and each batch writes its own `synthea.log`.

Each batch is generated, checked for the six required CSV exports, converted through the Step 2 canonical converter, validated through the Step 2 validation interface, and only then has its raw directory deleted. Canonical batch Parquet files and batch manifests are retained for resumability. A completed batch is skipped only when its manifest and canonical Parquet hash agree. Missing or corrupted batches regenerate.

The output is simulation-only and includes a manifest, batch manifests, canonical Parquet, augmentation assumptions, generation and validation reports, capability counts, hashes, and `split_status: not_created`. A target shortfall is reported explicitly; the runner never silently changes enrichment or lowers a count gate. Disk usage is bounded by deleting raw exports after successful canonical validation, while retained canonical batches enable safe resume.
