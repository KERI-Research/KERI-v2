"""Convert raw Synthea CSV exports into deterministic canonical tables."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

from metaboguard.data.schema import (
    ClinicalEvent,
    ConditionCategory,
    ConditionRecord,
    DiabetesType,
    EncounterType,
    OutcomeRecord,
    Patient,
    Provenance,
)
from metaboguard.features.dictionary import (
    AUGMENTED_CODE_TO_FEATURE,
    LOINC_TO_FEATURE,
    get_feature_definition,
)


@dataclass(slots=True)
class CanonicalDataset:
    """Typed in-memory representation of the four canonical tables."""

    patients: list[Patient]
    events: list[ClinicalEvent]
    conditions: list[ConditionRecord]
    outcomes: list[OutcomeRecord]
    dropped_loinc_codes: dict[str, int] = field(default_factory=dict)
    output_dir: Path | None = None
    dataset_sha256: str = ""
    cohort_class: str | None = None
    cohort_metadata: dict[str, object] = field(default_factory=dict)
    date_normalisation_audit: list[dict[str, object]] = field(default_factory=list)
    date_normalisation_audit_sha256: str = ""
    unit_normalisation_audit: list[dict[str, object]] = field(default_factory=list)
    unit_normalisation_audit_sha256: str = ""


SYNTHEA_DATE_NORMALISATION_VERSION = "1.0.0"
SYNTHEA_BIRTH_CONTEXT_CODES = frozenset(
    {"8302-2", "29463-7", "8462-4", "8480-6", "718-7", "777-3", "72166-2"}
)

SYNTHEA_UNIT_NORMALISATION_VERSION = "1.0.0"
SYNTHEA_MICROMOLAR_CREATININE_CODE = "2160-0"
# Synthea 3.3.0 emits some serum creatinine at micromole-per-litre magnitude under an mg/dL label.
MICROMOLES_PER_LITRE_TO_MILLIGRAMS_PER_DECILITRE = 1 / 88.4


UNIT_CONVERSIONS: dict[tuple[str, str, str], tuple[float, float]] = {
    ("glucose", "mmol/L", "mg/dL"): (18.0, 0.0),
    ("total_cholesterol", "mmol/L", "mg/dL"): (38.67, 0.0),
    ("hdl", "mmol/L", "mg/dL"): (38.67, 0.0),
    ("ldl", "mmol/L", "mg/dL"): (38.67, 0.0),
    ("triglycerides", "mmol/L", "mg/dL"): (88.57, 0.0),
    ("hba1c", "mmol/mol", "%"): (1 / 10.929, 2.15),
    ("weight", "lb", "kg"): (0.45359237, 0.0),
    ("height", "in", "cm"): (2.54, 0.0),
    ("waist_circumference", "in", "cm"): (2.54, 0.0),
    ("systolic_bp", "mm[Hg]", "mmHg"): (1.0, 0.0),
    ("diastolic_bp", "mm[Hg]", "mmHg"): (1.0, 0.0),
    ("bmi", "kg/m2", "kg/m^2"): (1.0, 0.0),
    ("platelets", "10*3/uL", "10^9/L"): (1.0, 0.0),
    ("smoking_status", "", "coded"): (1.0, 0.0),
    # UCUM international units per litre are numerically 1:1 with enzyme units per litre.
    ("alt", "[iU]/L", "U/L"): (1.0, 0.0),
    ("alkaline_phosphatase", "[iU]/L", "U/L"): (1.0, 0.0),
}


def _parse_date(value: object) -> date:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def convert_unit(feature_name: str, value: float, source_unit: str, canonical_unit: str) -> float:
    """Convert an explicitly supported unit or fail closed on a mismatch."""
    if source_unit == canonical_unit:
        return value
    conversion = UNIT_CONVERSIONS.get((feature_name, source_unit, canonical_unit))
    if conversion is None:
        raise ValueError(
            f"Unsupported unit mismatch for {feature_name}: {source_unit} -> {canonical_unit}"
        )
    multiplier, offset = conversion
    return value * multiplier + offset


def _observation_value(feature_name: str, raw_value: str, source_unit: str) -> float:
    """Normalize Synthea numeric and coded observation values explicitly."""
    if feature_name == "smoking_status":
        return 0.0 if "never" in raw_value.lower() or "no" in raw_value.lower() else 1.0
    return float(raw_value)


def _read_csv(raw_export_dir: Path, name: str) -> pd.DataFrame:
    path = raw_export_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _encounter_type(encounter_class: str) -> EncounterType:
    return cast(
        EncounterType,
        {
            "WELLNESS": "wellness",
            "AMBULATORY": "ambulatory",
            "EMERGENCY": "emergency",
            "INPATIENT": "inpatient",
        }.get(encounter_class.upper(), "other"),
    )


def _condition_category(display: str) -> ConditionCategory:
    lower = display.lower()
    if "prediabet" in lower:
        return "other"
    if "diabet" in lower and any(
        term in lower for term in ("type 1", "type1", "type 2", "type2", "gestational")
    ):
        return "diabetes"
    if any(term in lower for term in ("benign neoplasm", "uncertain behavior")):
        return "other"
    if any(term in lower for term in ("cancer", "malignant", "neoplasm")):
        return "cancer"
    return "other"


def _diabetes_type(display: str) -> DiabetesType | None:
    lower = display.lower()
    if "type 1" in lower or "type1" in lower:
        return "type1"
    if "gestational" in lower:
        return "gestational"
    if "type 2" in lower or "type2" in lower:
        return "type2"
    return None


def _cancer_site(display: str) -> str | None:
    lower = display.lower()
    for site in ("pancreatic", "pancreas", "breast", "lung", "colorectal", "prostate"):
        if site in lower:
            return site
    return (
        "unspecified"
        if any(term in lower for term in ("cancer", "malignant", "neoplasm"))
        else None
    )


def _stable_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(columns, kind="mergesort").reset_index(drop=True)


def _write_table(rows: list[Any], columns: list[str], output_dir: Path, name: str) -> None:
    data = [{column: getattr(row, column) for column in columns} for row in rows]
    frame = _stable_frame(data, columns)
    frame.to_parquet(
        output_dir / f"{name}.parquet", index=False, engine="pyarrow", compression="zstd"
    )


def to_canonical(
    raw_export_dir: str | Path,
    *,
    source: str = "unknown",
    source_version: str | None = None,
) -> CanonicalDataset:
    """Read Synthea CSVs, construct canonical models, and write deterministic Parquet."""
    raw_dir = Path(raw_export_dir)
    output_dir = raw_dir / "canonical"
    output_dir.mkdir(parents=True, exist_ok=True)
    patients_csv = _read_csv(raw_dir, "patients")
    encounters_csv = _read_csv(raw_dir, "encounters")
    observations_csv = _read_csv(raw_dir, "observations")
    conditions_csv = _read_csv(raw_dir, "conditions")

    patients = [
        Patient(
            patient_id=row["Id"],
            birth_date=_parse_date(row["BIRTHDATE"]),
            sex={"m": "male", "f": "female"}.get(row["GENDER"].lower(), row["GENDER"].lower()),
            ethnicity=row.get("RACE", row.get("ETHNICITY", "unknown")),
            death_date=_parse_date(row["DEATHDATE"]) if row.get("DEATHDATE") else None,
        )
        for _, row in patients_csv.iterrows()
    ]
    patient_by_id = {patient.patient_id: patient for patient in patients}
    encounters: dict[str, tuple[str, date]] = {
        row["Id"]: (_encounter_type(row.get("ENCOUNTERCLASS", "other")), _parse_date(row["START"]))
        for _, row in encounters_csv.iterrows()
    }
    encounter_counts = encounters_csv.groupby("PATIENT").size().to_dict()

    dropped: dict[str, int] = {}
    date_normalisation_audit: list[dict[str, object]] = []
    unit_normalisation_audit: list[dict[str, object]] = []
    event_rows: dict[tuple[str, date, str], ClinicalEvent] = {}
    for _, row in observations_csv.iterrows():
        code = row["CODE"]
        feature_name = LOINC_TO_FEATURE.get(code, AUGMENTED_CODE_TO_FEATURE.get(code))
        if feature_name is None:
            dropped[code] = dropped.get(code, 0) + 1
            continue
        definition = get_feature_definition(feature_name)
        patient = patient_by_id[row["PATIENT"]]
        original_event_date = _parse_date(row["DATE"])
        event_date = original_event_date
        if (
            source == "synthea"
            and source_version == "3.3.0"
            and row["CODE"] in SYNTHEA_BIRTH_CONTEXT_CODES
            and original_event_date == patient.birth_date - timedelta(days=1)
        ):
            event_date = patient.birth_date
            date_normalisation_audit.append(
                {
                    "patient_id": row["PATIENT"],
                    "event_identifier": ":".join(
                        [row.get("ENCOUNTER", ""), row["CODE"], row["DATE"]]
                    ),
                    "feature_name": feature_name,
                    "source_code": row["CODE"],
                    "original_event_date": original_event_date.isoformat(),
                    "normalised_event_date": event_date.isoformat(),
                    "reason": "synthea_3_3_0_birth_context_date_only_boundary",
                    "source": source,
                    "source_version": source_version,
                    "normalisation_version": SYNTHEA_DATE_NORMALISATION_VERSION,
                }
            )
        encounter_type, _ = encounters.get(row.get("ENCOUNTER", ""), ("other", event_date))
        age = event_date.toordinal() - patient.birth_date.toordinal()
        age_years = age / 365.2425
        missing = not row.get("VALUE", "")
        value = (
            None
            if missing
            else convert_unit(
                feature_name,
                _observation_value(feature_name, row["VALUE"], row.get("UNITS", "")),
                row.get("UNITS", ""),
                definition.canonical_unit,
            )
        )
        provenance = "augmented" if definition.code.startswith("augmented:") else "synthea_native"
        module = definition.code.removeprefix("augmented:") if provenance == "augmented" else None
        if (
            value is not None
            and source == "synthea"
            and source_version == "3.3.0"
            and row["CODE"] == SYNTHEA_MICROMOLAR_CREATININE_CODE
            and value > definition.plausible_max
        ):
            converted = value * MICROMOLES_PER_LITRE_TO_MILLIGRAMS_PER_DECILITRE
            if definition.plausible_min <= converted <= definition.plausible_max:
                unit_normalisation_audit.append(
                    {
                        "patient_id": row["PATIENT"],
                        "event_identifier": ":".join(
                            [row.get("ENCOUNTER", ""), row["CODE"], row["DATE"]]
                        ),
                        "feature_name": feature_name,
                        "source_code": row["CODE"],
                        "original_value": value,
                        "original_unit": row.get("UNITS", ""),
                        "normalised_value": converted,
                        "normalised_unit": definition.canonical_unit,
                        "reason": "synthea_3_3_0_creatinine_micromolar_magnitude_under_mg_dl_label",
                        "source": source,
                        "source_version": source_version,
                        "normalisation_version": SYNTHEA_UNIT_NORMALISATION_VERSION,
                    }
                )
                value = converted
        event = ClinicalEvent(
            patient_id=row["PATIENT"],
            event_date=event_date,
            age_at_event=age_years,
            encounter_type=cast(EncounterType, encounter_type),
            feature_name=feature_name,
            value=value,
            unit=definition.canonical_unit,
            is_missing=missing,
            provenance=cast(Provenance, provenance),
            augmentation_module=module,
        )
        event_rows.setdefault((event.patient_id, event.event_date, event.feature_name), event)

    conditions: list[ConditionRecord] = []
    for _, row in conditions_csv.iterrows():
        display = row.get("DESCRIPTION", row.get("CODE", ""))
        category = _condition_category(display)
        conditions.append(
            ConditionRecord(
                patient_id=row["PATIENT"],
                condition_code=row["CODE"],
                condition_system="SNOMED-CT",
                condition_display=display,
                onset_date=_parse_date(row["START"]),
                resolved_date=_parse_date(row["STOP"]) if row.get("STOP") else None,
                category=category,
                diabetes_type=_diabetes_type(display),
                cancer_site=_cancer_site(display),
            )
        )
    outcomes = [
        OutcomeRecord(
            patient_id=condition.patient_id,
            outcome_type=condition.category,
            outcome_date=condition.onset_date,
            source_condition_code=condition.condition_code,
        )
        for condition in conditions
        if condition.category in {"diabetes", "cancer"}
    ]
    events = sorted(
        event_rows.values(),
        key=lambda event: (event.patient_id, event.event_date, event.feature_name),
    )
    conditions.sort(
        key=lambda condition: (condition.patient_id, condition.onset_date, condition.condition_code)
    )
    outcomes.sort(
        key=lambda outcome: (outcome.patient_id, outcome.outcome_date, outcome.outcome_type)
    )
    _write_table(
        patients,
        ["patient_id", "birth_date", "sex", "ethnicity", "death_date"],
        output_dir,
        "patients",
    )
    _write_table(
        events,
        [
            "patient_id",
            "event_date",
            "age_at_event",
            "encounter_type",
            "feature_name",
            "value",
            "unit",
            "is_missing",
            "provenance",
            "augmentation_module",
        ],
        output_dir,
        "events",
    )
    _write_table(
        conditions,
        [
            "patient_id",
            "condition_code",
            "condition_system",
            "condition_display",
            "onset_date",
            "resolved_date",
            "category",
            "diabetes_type",
            "cancer_site",
        ],
        output_dir,
        "conditions",
    )
    _write_table(
        outcomes,
        ["patient_id", "outcome_type", "outcome_date", "source_condition_code"],
        output_dir,
        "outcomes",
    )
    digest = hashlib.sha256()
    for path in sorted(output_dir.glob("*.parquet")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    audit_payload = json.dumps(date_normalisation_audit, indent=2, sort_keys=True).encode("utf-8")
    audit_path = output_dir / "date_normalisation_audit.json"
    audit_path.write_bytes(audit_payload + b"\n")
    audit_sha256 = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    digest.update(audit_path.name.encode("utf-8"))
    digest.update(audit_path.read_bytes())
    unit_audit_payload = json.dumps(unit_normalisation_audit, indent=2, sort_keys=True).encode(
        "utf-8"
    )
    unit_audit_path = output_dir / "unit_normalisation_audit.json"
    unit_audit_path.write_bytes(unit_audit_payload + b"\n")
    unit_audit_sha256 = hashlib.sha256(unit_audit_path.read_bytes()).hexdigest()
    digest.update(unit_audit_path.name.encode("utf-8"))
    digest.update(unit_audit_path.read_bytes())
    report = {
        "dropped_loinc_codes": dict(sorted(dropped.items())),
        "encounter_counts": encounter_counts,
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return CanonicalDataset(
        patients,
        events,
        conditions,
        outcomes,
        dropped,
        output_dir,
        digest.hexdigest(),
        date_normalisation_audit=date_normalisation_audit,
        date_normalisation_audit_sha256=audit_sha256,
        unit_normalisation_audit=unit_normalisation_audit,
        unit_normalisation_audit_sha256=unit_audit_sha256,
    )
