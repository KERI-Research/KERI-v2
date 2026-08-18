"""Smoke tests for the step-one repository scaffold."""

from metaboguard import __version__


def test_package_version_is_declared() -> None:
    assert __version__ == "0.1.0"
