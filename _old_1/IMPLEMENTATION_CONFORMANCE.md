# RoboPref Design–Implementation Conformance Audit

## Audit Scope

This audit compares the implementation with the normative requirements in `DESIGN_PARADIGM.md`. The result is **conformant within the recorded-episode prototype boundary**. Preference consent, deterministic dispatch gates, outcome classification, and recovery recommendations are implemented and tested. Physical VLA execution, actuator safety, automatic retry execution, persistent outcome logs, and learned capability estimates remain outside the current repository and therefore cannot be claimed as implemented.

Audit date: 18 July 2026.

## Requirement Traceability

| Requirement | Status | Implementation evidence | Verification evidence |
|---|---|---|---|
| R1: distinguish action approval from memory consent | Satisfied | The memory prompt assigns typed evidence origins; `memory/store.py` rejects `assistant_plan_confirmation` and bare affirmations; `agents/hri.py` applies durable consent only through `answer_memory_confirmation`. | `test_yes_to_action_proposal_is_not_preference_evidence`; `test_repeated_candidate_requires_dedicated_memory_confirmation`. |
| R2: do not persist a fully specified current command | Satisfied | `current_task` is a non-preference origin and fails closed in `PreferenceStore`. | `test_fully_specified_current_task_is_not_preference_evidence`. |
| R3: no silent promotion from repetition | Satisfied | Two independent observations return `CONFIRM_REQUIRED`; status remains `candidate`. Only explicit future language or `confirm_candidate` can create durable status. Missing evidence provenance fails closed. | `test_repeated_evidence_requests_consent_but_does_not_auto_promote`; `test_missing_evidence_origin_fails_closed`; `test_dedicated_memory_yes_promotes_candidate`. |
| R4: current instruction wins; durable replacement requires consent | Satisfied | Conflicting unmarked evidence creates a candidate and replacement question while preserving the durable record. Explicit future correction may replace it. | `test_present_task_correction_cannot_overwrite_durable_default`; `test_explicit_future_correction_replaces_durable_default`. |
| R5: check feasibility and final postconditions | Satisfied for recorded episodes | Planner schema exposes preconditions/status; `TaskAssurance.assess_plan` blocks low confidence, unmet preconditions, empty plans, malformed output, unsafe and unsupported requests. Validator schema evaluates goal conditions from the final frame; success requires adequate confidence and `task_complete=true`. | `test_blocked_plan_does_not_call_validator`; task-assurance tests for missing blocks, unmet preconditions, malformed/low-confidence output, partial state, and inconsistent success. |
| R6: proportional, bounded recovery | Policy satisfied; physical action pending | `TaskAssurance` deterministically assigns `REOBSERVE`, `AUTO_LOCAL`, `REPLAN`, `USER_ASSIST`, `ABORT_UNSUPPORTED`, or `ABORT_SAFETY`. `bounded_recovery` allows one re-observation, local retry, and replan before escalation; safety abort bypasses retries. The current runtime reports the decision but has no physical executive that can perform the action. | Tests cover partial completion, exhausted local retry/replan, exhausted re-observation, user assistance, unsupported skill, and safety abort. `SUCCESS_RECOVERED` requires a nonzero logged attempt argument. |
| R7: separate preference, outcome, and capability evidence | Satisfied at boundary; stores incomplete | Every failure record has `memory_effect: NONE`; the memory curator rejects scene, plan, execution, and validation evidence. A persistent task-outcome store and capability-learning store are intentionally not implemented in this prototype. | Scenario matrix asserts that every non-success failure has no memory effect; memory tests reject non-user evidence. |
| R8: traceable user-visible claims | Satisfied for current agents | HRI, planner, validator, and task assurance use structured JSON. Failures include stage, code, evidence, confidence, recovery, next action, and message. `last_task_result` exposes the final deterministic decision. | HRI integration and task-assurance schema tests; transcript tests. |

## Preference-Policy Audit

The following transition checks were performed:

| Evidence sequence | Required result | Observed implementation result |
|---|---|---|
| Assistant proposes action → user says “Yes” | Execute only; no preference | Rejected by evidence-origin and affirmation gates. |
| User fully specifies current RGB order | No preference | Rejected as `current_task`. |
| One answer to open order question | Candidate | Candidate with one independent evidence item. |
| Two independent matching answers | Ask to remember; still candidate | `CONFIRM_REQUIRED`, `confirmation_pending=true`, status candidate. |
| Dedicated memory question → “Yes” | Durable | `confirm_candidate` promotes and retracts competing active values. |
| Dedicated memory question → “No” | Remain candidate | Candidate retained; prompt closed until new independent evidence. |
| Durable RGB → present-task opposite correction | Execute override; retain RGB; ask before replacement | Alternate candidate with `conflicts_with_durable`; original remains durable until explicit answer. |
| Durable RGB → “opposite order in the future” | Durable BGR | Explicit future correction replaces the active value. |
| Legacy durable record created under old auto-promotion policy | Do not silently use as durable | Read-time compatibility gate exposes it as candidate unless direct durable evidence exists. The existing repository record is currently read as candidate without rewriting the user’s file. |

## Task-Assurance Audit

Ten reusable end-to-end policy cases are recorded in `tests/scenario_cases.json`, with exact scene setup, user query, planner output, validator output, expected outcome, and recovery action. They cover two verified successes, an already-satisfied task, no blocks, uncertain vision, partial completion, unreachable object, unsafe workspace, unsupported skill, and unverifiable final state.

The deterministic assurance layer guarantees:

1. unparseable planner output is never dispatched;
2. `READY` cannot bypass a declared unsatisfied precondition;
3. a low-confidence plan cannot be dispatched;
4. a blocked, unsupported, unsafe, or already-satisfied plan does not invoke validation as if execution occurred;
5. low-confidence validation becomes `UNKNOWN`, not success or failure;
6. a `SUCCESS` label without `task_complete=true` is rejected;
7. failure and partial outcomes carry a recovery recommendation and no memory effect.

## Verification Result

Command:

```bash
export PYTHONDONTWRITEBYTECODE=1
./.venv/bin/python -m unittest discover -s tests -v
```

Result: **60 tests passed**. This includes dataset loading, HRI/memory integration, consent-gated preference transitions, component-unavailability handling, user cancellation, bounded recovery escalation, the scenario matrix, task assurance, transcript behaviour, and the pre-existing visual-image preprocessing tests.

`python ./main.py --help` also completed successfully and exposed participant-specific dataset, memory-store, user-ID, and transcript options for data collection.

## Residual Risks and Required Next Integration

1. **No physical executor:** `Planner_Agent` returns a plan and the prototype evaluates a pre-recorded final frame; no VLA action API is called here. A physical implementation must place task assurance around the executor and log every attempt.
2. **Model-grounded preconditions:** planner preconditions are produced by a VLM. An independent object detector, safety monitor, or calibrated perception module should cross-check safety-critical preconditions.
3. **Single-view validation:** occlusion may prevent reliable outcome judgement. The system correctly supports `UNKNOWN`, but active viewpoint acquisition is not yet available.
4. **No actuator safety claim:** software prompts and classifications are not a certified emergency-stop or collision-avoidance system.
5. **No empirical threshold validation:** two independent observations are a prototype question trigger. The proposed data collection must evaluate question burden and false/missed preference rates before deployment.
6. **No persistent outcome/capability stores:** the design specifies separation; future work should add these stores without feeding their records into preference persistence.

Subject to these explicit boundaries, the implementation matches the design paradigm and does not overstate unimplemented physical recovery.
