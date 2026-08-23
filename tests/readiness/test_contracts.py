import pytest

from metaboguard.readiness.contracts import ReadinessCheck


def test_warning_is_nonblocking() -> None:
    warning = ReadinessCheck(
        name="partial",
        level="warning",
        status="warning",
        passed=True,
        offending_count=1,
    )
    assert warning.passed is True
    assert warning.status == "warning"


def test_contract_forbids_extra_fields() -> None:
    with pytest.raises(ValueError, match="extra"):
        ReadinessCheck(
            name="x",
            level="error",
            status="passed",
            passed=True,
            offending_count=0,
            extra="bad",
        )
