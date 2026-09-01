# Cohort Construction

Step 4 turns a validated canonical longitudinal dataset into immutable patient-index records and endpoint-specific horizon labels. It does not create feature tables, fit models, or generate metrics.

An index date is a patient-specific prediction cutoff. Only canonical events with `event_date <= index_date` are retained in the index history summary. Rolling candidates begin after the configured history requirement, advance by the configured interval, stop at the final valid observed date, and never continue after death or endpoint onset.

Eligibility is endpoint-specific. Patients below the minimum age, without enough history, encounters, or measurement dates, or with prevalent endpoint disease are excluded with reason codes. Cancer site and diabetes type are matched from canonical condition records. Type 1 and gestational diabetes remain disabled for future-development endpoints.

Ordinary-incidence and enriched cohort classes are never pooled. Every constructed cohort carries one class and is marked `simulation_only: true`. No model-ready feature table is written in this step.

Pancreatic cancer uses a configured 365-day washout sensitivity protocol. Washout exclusions are explicit and recorded in `washout_impact.json`; they are not hidden deletions.
