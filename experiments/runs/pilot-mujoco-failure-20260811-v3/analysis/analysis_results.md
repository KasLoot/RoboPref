# Campaign analysis

- Campaign: `pilot-mujoco-failure-20260811-v3`
- Protocol: `124e57a852f062388306fe254b6957cd121aa9d1fbe4018657f9a628fc2cb927`
- Tier: `pilot_nonconfirmatory`
- Experimental unit: episode
- Outcome: `contract_success`

## ITT summaries

- `T5`: 0.0000; assigned=1, missing=0

## Machine-readable report

```json
{
  "analysis_gate": {
    "execution_authorized": true,
    "live_source_checked": true,
    "registry_complete": true,
    "request_matches_frozen_analysis_spec": true,
    "resolved_attempts_without_passing_audit": 0,
    "unresolved_outcomes": 0
  },
  "analysis_spec": {
    "confirmatory_outcomes": [
      "contract_success"
    ],
    "experimental_unit": "episode",
    "families": [
      {
        "comparison_profiles": [],
        "matrix_id": "instrumentation_mujoco",
        "reference_profile": "T5"
      }
    ],
    "multiplicity_method": "Holm within matrix",
    "pairing_key": "crn_key",
    "reference_profile": "T5",
    "schema_version": 1
  },
  "analysis_tier": "pilot_nonconfirmatory",
  "attempt_history": {
    "artifact_audit_failures": 0,
    "preserved_attempts": 1,
    "retry_attempts": 0,
    "scheduled_episodes": 1,
    "status_counts": {
      "ABORTED_SAFETY": 0,
      "INFRA_INTERRUPTED": 0,
      "INVALID_HARNESS": 0,
      "NOT_RUN": 0,
      "VALID_PASS": 0,
      "VALID_SYSTEM_FAILURE": 1
    }
  },
  "by_profile": [
    {
      "assigned": 1,
      "confidence": 0.95,
      "estimable": true,
      "estimate": 0.0,
      "experimental_unit": "episode",
      "failures": 1,
      "matrix_id": "instrumentation_mujoco",
      "missing": 0,
      "missing_by_status": {},
      "observed_rate": 0.0,
      "observed_wilson_interval": [
        0.0,
        0.7934506856227626
      ],
      "outcome": "contract_success",
      "profile_id": "T5",
      "resolved": 1,
      "success_rate_bounds": [
        0.0,
        0.0
      ],
      "successes": 0,
      "wilson_interval": [
        0.0,
        0.7934506856227626
      ]
    }
  ],
  "campaign_id": "pilot-mujoco-failure-20260811-v3",
  "episode_count": 1,
  "failure_taxonomy": {
    "episodes": 1,
    "experimental_unit": "episode",
    "failure_layer_counts": {
      "execution": 1
    },
    "note": "oracle outcomes and post-run diagnoses are distinct fields",
    "primary_diagnosis_counts": {
      "Controlled MuJoCo calibration left the target predicate unmet.": 1
    },
    "status_counts": {
      "ABORTED_SAFETY": 0,
      "INFRA_INTERRUPTED": 0,
      "INVALID_HARNESS": 0,
      "NOT_RUN": 0,
      "VALID_PASS": 0,
      "VALID_SYSTEM_FAILURE": 1
    },
    "unresolved_or_invalid": 0
  },
  "live_attempt_audits": [
    {
      "attempt_number": 1,
      "audit_report_sha256": "261020a245750de0afa5712dd0fba9d60f49fd50415ff2b15d7e6de71796174d",
      "errors": [],
      "passed": true,
      "registry_audit_passed": true,
      "schedule_id": "S000001"
    }
  ],
  "mode": "pilot",
  "multiplicity_family": "reference-profile pairwise comparisons within matrix",
  "multiplicity_method": "Holm",
  "outcome": "contract_success",
  "paired_comparisons": {},
  "protocol_sha256": "124e57a852f062388306fe254b6957cd121aa9d1fbe4018657f9a628fc2cb927",
  "reference_profile": "T5",
  "schema_version": 1
}
```
