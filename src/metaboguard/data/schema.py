"""Strict Pydantic v2 canonical models and schema export."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from metaboguard.config import load_config
from metaboguard.features.dictionary import get_feature_definition, validate_feature_name

SCHEMA_VERSION = "2.0.0"
EncounterType = Literal["wellness", "ambulatory", "emergency", "inpatient", "other"]
Provenance = Literal["synthea_native", "augmented"]
ConditionCategory = Literal["diabetes", "cancer", "other"]
DiabetesType = Literal["type1", "type2", "gestational"]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Patient(StrictModel):
    patient_id: str = Field(min_length=1)
    birth_date: date
    sex: str | Literal["male", "female", "other"] | None
    ethnicity: str
    death_date: date | None = None


class ClinicalEvent(StrictModel):
    patient_id: str = Field(min_length=1)
    event_date: date
    age_at_event: float
    encounter_type: EncounterType
    feature_name: str
    value: float | None
    unit: str
    is_missing: bool
    provenance: Provenance
    augmentation_module: str | None = None

    @model_validator(mode="after")
    def validate_feature_and_missingness(self) -> ClinicalEvent:
        validate_feature_name(self.feature_name)
        definition = get_feature_definition(self.feature_name)
        if self.is_missing != (self.value is None):
            raise ValueError("is_missing must be true exactly when value is None")
        if self.unit != definition.canonical_unit:
            raise ValueError(
                f"Unit mismatch for {self.feature_name}: expected {definition.canonical_unit}"
            )
        if self.provenance == "augmented" and not self.augmentation_module:
            raise ValueError("Augmented events must name an augmentation module")
        if self.provenance == "synthea_native" and self.augmentation_module is not None:
            raise ValueError("Native events cannot name an augmentation module")
        augmented_features = {"ca_19_9", "c_peptide", "insulin"}
        if self.feature_name in augmented_features and self.provenance != "augmented":
            raise ValueError(f"{self.feature_name} must be marked augmented")
        return self


class ConditionRecord(StrictModel):
    patient_id: str = Field(min_length=1)
    condition_code: str
    condition_system: str
    condition_display: str
    onset_date: date
    resolved_date: date | None = None
    category: Literal["diabetes", "cancer", "other"]
    diabetes_type: Literal["type1", "type2", "gestational"] | None = None
    cancer_site: str | None = None

    @model_validator(mode="after")
    def validate_condition_requirements(self) -> ConditionRecord:
        if self.category == "cancer" and self.cancer_site is None:
            raise ValueError("Cancer conditions require cancer_site")
        if self.category == "diabetes" and self.diabetes_type is None:
            raise ValueError("Diabetes conditions require diabetes_type")
        return self


class OutcomeRecord(StrictModel):
    patient_id: str = Field(min_length=1)
    outcome_type: str
    outcome_date: date
    source_condition_code: str


def export_json_schema(
    output_path: Path | None = None,
) -> tuple[str, str]:
    """Write the versioned JSON schema and return its path and SHA-256."""
    if output_path is None:
        output_path = Path(load_config()["paths"]["schema_artifact"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "metaboguard://schema/longitudinal/2.0.0",
        "x-schema-version": SCHEMA_VERSION,
        "title": "MetaboGuard v2 longitudinal canonical schema",
        "definitions": {
            "Patient": Patient.model_json_schema(),
            "ClinicalEvent": ClinicalEvent.model_json_schema(),
            "ConditionRecord": ConditionRecord.model_json_schema(),
            "OutcomeRecord": OutcomeRecord.model_json_schema(),
        },
    }
    payload = json.dumps(schema, indent=2, sort_keys=True).encode("utf-8")
    output_path.write_bytes(payload + b"\n")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8"
    )
    return str(output_path), digest
