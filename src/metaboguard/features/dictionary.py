"""Canonical feature registry and executable governance denylist."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


class DeniedFeatureError(ValueError):
    """Raised when a denied feature is loaded into a canonical dataset."""


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """Shape and governance metadata for one canonical feature."""

    canonical_unit: str
    plausible_min: float
    plausible_max: float
    code: str
    category: str
    allowed: bool = True


FEATURE_DICTIONARY: Final[dict[str, FeatureDefinition]] = {
    "hba1c": FeatureDefinition("%", 2.0, 20.0, "4548-4", "glycaemic"),
    "glucose": FeatureDefinition("mg/dL", 20.0, 1000.0, "2339-0", "glycaemic"),
    "insulin": FeatureDefinition(
        "uIU/mL", 0.0, 1000.0, "augmented:insulin", "glycaemic"
    ),
    "c_peptide": FeatureDefinition(
        "ng/mL", 0.0, 50.0, "augmented:c_peptide", "glycaemic"
    ),
    "ca_19_9": FeatureDefinition("U/mL", 0.0, 10000.0, "augmented:ca19_9", "oncology"),
    "total_cholesterol": FeatureDefinition("mg/dL", 20.0, 1500.0, "2093-3", "lipid"),
    "hdl": FeatureDefinition("mg/dL", 1.0, 300.0, "2085-9", "lipid"),
    "ldl": FeatureDefinition("mg/dL", 1.0, 1000.0, "2089-1", "lipid"),
    "triglycerides": FeatureDefinition("mg/dL", 1.0, 3000.0, "2571-8", "lipid"),
    "haemoglobin": FeatureDefinition("g/dL", 2.0, 25.0, "718-7", "cbc"),
    "platelets": FeatureDefinition("10^9/L", 1.0, 2000.0, "777-3", "cbc"),
    "alt": FeatureDefinition("U/L", 0.0, 5000.0, "1742-6", "liver"),
    "creatinine": FeatureDefinition("mg/dL", 0.05, 30.0, "2160-0", "renal"),
    "alkaline_phosphatase": FeatureDefinition("U/L", 0.0, 5000.0, "6768-6", "liver"),
    "bmi": FeatureDefinition("kg/m^2", 5.0, 100.0, "39156-5", "anthropometry"),
    "weight": FeatureDefinition("kg", 1.0, 500.0, "29463-7", "anthropometry"),
    "height": FeatureDefinition("cm", 20.0, 250.0, "8302-2", "anthropometry"),
    "waist_circumference": FeatureDefinition(
        "cm", 20.0, 300.0, "8280-0", "anthropometry"
    ),
    "systolic_bp": FeatureDefinition("mmHg", 20.0, 300.0, "8480-6", "vitals"),
    "diastolic_bp": FeatureDefinition("mmHg", 20.0, 200.0, "8462-4", "vitals"),
    "smoking_status": FeatureDefinition("coded", 0.0, 1.0, "72166-2", "lifestyle"),
    "alcohol_status": FeatureDefinition("coded", 0.0, 1.0, "74013-4", "lifestyle"),
    "tumour_stage": FeatureDefinition(
        "coded", 0.0, 0.0, "denied:tumour_stage", "denied", False
    ),
    "tumour_grade": FeatureDefinition(
        "coded", 0.0, 0.0, "denied:tumour_grade", "denied", False
    ),
    "histology": FeatureDefinition(
        "coded", 0.0, 0.0, "denied:histology", "denied", False
    ),
    "tumour_status": FeatureDefinition(
        "coded", 0.0, 0.0, "denied:tumour_status", "denied", False
    ),
    "treatment": FeatureDefinition(
        "coded", 0.0, 0.0, "denied:treatment", "denied", False
    ),
    "survival_time": FeatureDefinition(
        "days", 0.0, 0.0, "denied:survival_time", "denied", False
    ),
    "progression_time": FeatureDefinition(
        "days", 0.0, 0.0, "denied:progression_time", "denied", False
    ),
    "post_diagnosis_pathology": FeatureDefinition(
        "coded", 0.0, 0.0, "denied:pathology", "denied", False
    ),
    "diabetes_diagnosis_age": FeatureDefinition(
        "years", 0.0, 150.0, "denied:diabetes_diagnosis_age", "denied", False
    ),
    "insulin_use": FeatureDefinition(
        "coded", 0.0, 1.0, "denied:insulin_use", "denied", False
    ),
}

LOINC_TO_FEATURE: Final[dict[str, str]] = {
    definition.code: feature_name
    for feature_name, definition in FEATURE_DICTIONARY.items()
    if not definition.code.startswith(("augmented:", "denied:"))
}

AUGMENTED_CODE_TO_FEATURE: Final[dict[str, str]] = {
    "aug-ca19-9": "ca_19_9",
    "aug-c-peptide": "c_peptide",
    "aug-insulin": "insulin",
}


def get_feature_definition(
    feature_name: str, *, target: str | None = None
) -> FeatureDefinition:
    """Return a feature definition or reject it as unknown/denied."""
    definition = FEATURE_DICTIONARY.get(feature_name)
    diabetes_only_denied = {"diabetes_diagnosis_age", "insulin_use"}
    if (
        definition is None
        or not definition.allowed
        or (target == "diabetes_development" and feature_name in diabetes_only_denied)
    ):
        raise DeniedFeatureError(f"Feature is denied or unknown: {feature_name}")
    return definition


def validate_feature_name(feature_name: str) -> None:
    """Reject unknown or denied feature names at model/load time."""
    get_feature_definition(feature_name)


def feature_names() -> tuple[str, ...]:
    """Return the allowed canonical feature names in stable order."""
    return tuple(
        name for name, definition in FEATURE_DICTIONARY.items() if definition.allowed
    )
