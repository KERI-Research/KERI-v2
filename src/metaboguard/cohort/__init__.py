"""Leakage-safe endpoint-specific cohort construction boundaries."""

from metaboguard.cohort.eligibility import build_eligible_patient_indexes
from metaboguard.cohort.endpoints import assign_outcomes
from metaboguard.cohort.index_dates import generate_rolling_index_dates
from metaboguard.cohort.protocol import EndpointProtocol, HorizonLabel, PatientIndex

__all__ = [
    "EndpointProtocol",
    "HorizonLabel",
    "PatientIndex",
    "assign_outcomes",
    "build_eligible_patient_indexes",
    "generate_rolling_index_dates",
]
