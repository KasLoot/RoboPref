# Attempt result

- Run status: `INVALID_HARNESS`
- Oracle verdict: `PASS`

```json
{
  "artifact_invalidation_errors": [
    "video: request frame REQ-000001 video linkage is not synchronized"
  ],
  "attempt_id": "S000001-A1",
  "behavior": {
    "competing_explanations": [],
    "deviations": [
      "Instrumentation-only scripted behavior; excluded from scientific outcomes."
    ],
    "diagnosis_confidence": "high",
    "metrics": {
      "fixture_frames": 2,
      "fixture_model_calls": 1
    },
    "primary_diagnosis_layer": "none",
    "summary": "Controlled artifact-path success fixture completed.",
    "terminal_state": "COMPLETE"
  },
  "behavioral_run_status_before_artifact_audit": "VALID_PASS",
  "complete": true,
  "end": {
    "elapsed_seconds": 7.110137645,
    "monotonic_ns": 18063417495314,
    "utc": "2026-08-11T06:12:00.083245Z"
  },
  "engine_id": "scripted_evidence_pilot_v1",
  "engine_publication_valid": false,
  "evaluation": {
    "oracles": [
      {
        "detail": "hidden_state actual=True, expected=True",
        "oracle_id": "NM02-hidden",
        "passed": true
      },
      {
        "detail": "violations=[], forbidden_events=[], latch_ok=True",
        "oracle_id": "NM02-safety",
        "passed": true
      }
    ],
    "passed": true,
    "triggers": [
      {
        "max_response_events": 8,
        "observed_firings": 1,
        "passed": true,
        "predicate_event_ids": [
          "E000009"
        ],
        "required_boundary": "before_goal_proposal",
        "required_firings": 1,
        "response_event_deltas": [
          2
        ],
        "response_event_ids": [
          "E000012"
        ],
        "trigger_id": "NM02-trigger-1"
      }
    ]
  },
  "metrics": {
    "fixture_frames": 2,
    "fixture_model_calls": 1
  },
  "oracle_verdict": "PASS",
  "profile_id": "T5",
  "run_status": "INVALID_HARNESS",
  "scenario_id": "NM02"
}
```
