from metaboguard.features.definitions import build_feature_registry, registry_hash


def test_registry_is_configured_and_hashed() -> None:
    registry = build_feature_registry()
    assert "glucose__latest_lifetime" in registry
    assert "glucose__mean_recent" in registry
    assert registry["glucose__mean_recent"].unit == "mg/dL"
    assert registry_hash(registry) == registry_hash(build_feature_registry())
