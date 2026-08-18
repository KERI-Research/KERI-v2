from pathlib import Path

import pytest

from metaboguard.data.production_manifests import (
    ProductionBatchRecord,
    ProductionRunManifest,
    load_production_manifest,
    manifest_hash,
    manifest_payload,
    new_production_manifest,
    production_run_id,
)


def test_production_identity_and_round_trip(tmp_path: Path) -> None:
    first = production_run_id("ordinary_incidence", 100, 17, "a" * 64)
    second = production_run_id("ordinary_incidence", 100, 17, "a" * 64)
    enriched = production_run_id("enriched_incidence", 100, 17, "a" * 64)
    assert first == second
    assert first != enriched
    manifest = new_production_manifest(
        "ordinary_incidence", 100, 17, "baseline", "3.3.0", {"seed": 17}
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    assert load_production_manifest(path).model_status == "not_created"
    assert manifest_hash(path)
    assert manifest_payload(manifest)["model_status"] == "not_created"


def test_batch_record_and_strict_manifest() -> None:
    batch = ProductionBatchRecord(
        batch_index=0,
        seed=17,
        target_size=10,
        source_output_path="raw/batch_00000",
        status="failed",
        started_at="now",
        failure_stage="canonical_validation",
        diagnostic="bounded failure",
    )
    manifest = ProductionRunManifest(
        run_id="r",
        cohort_class="enriched_incidence",
        population_target=10,
        root_seed=17,
        augmentation_profile="endpoint_enriched",
        configuration_sha256="a" * 64,
        synthea_version="3.3.0",
        started_at="now",
        batch_records=[batch],
        status="partial",
    )
    assert manifest.batch_records[0].status == "failed"
    with pytest.raises(ValueError, match="extra"):
        ProductionRunManifest(
            run_id="r",
            cohort_class="ordinary_incidence",
            population_target=1,
            root_seed=1,
            augmentation_profile="baseline",
            configuration_sha256="a" * 64,
            synthea_version="3.3.0",
            started_at="now",
            unexpected=True,
        )
