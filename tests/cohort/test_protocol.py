import pytest

from metaboguard.cohort.protocol import (
    EndpointProtocol,
    _int_value,
    add_years,
    load_endpoint_registry,
)


def test_protocol_registry_and_calendar_years(diabetes_endpoint: EndpointProtocol) -> None:
    registry = load_endpoint_registry()
    assert "pancreatic_cancer" in registry
    assert diabetes_endpoint.to_dict()["endpoint_id"] == "type2_diabetes"
    assert add_years(__import__("datetime").date(2020, 2, 29), 1).isoformat() == "2021-02-28"
    with pytest.raises(TypeError):
        _int_value(object())
