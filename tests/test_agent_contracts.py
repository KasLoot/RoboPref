from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from prefmem.agents.contracts import (
    ConditionState,
    ExpectedOutcome,
    HRIDecision,
    MemoryContext,
    MonitorResult,
    PlanResult,
    PlanningRequest,
    TaskContract,
    ValidationRequest,
    ValidationResult,
    ValidationSpec,
)


def validation_spec() -> dict:
    return {
        "spec_id": "spec-1",
        "confirmed_intent": "Put the book in the left zone.",
        "goal_conditions": [
            {
                "goal_id": "goal-1",
                "description": "The book is fully in the left zone.",
                "predicate": "INSIDE_ZONE",
                "arguments": ["book", "zone:left"],
                "required": True,
                "observable": True,
                "evidence_modalities": ["terminal_observation"],
            }
        ],
    }


class HRIContractTests(unittest.TestCase):
    def test_retrieve_memory_is_internal_and_requires_request(self) -> None:
        decision = HRIDecision.model_validate(
            {
                "decision": "RETRIEVE_MEMORY",
                "reply_to_user": None,
                "memory_request": {
                    "memory_types": ["PERSISTENT_PREFERENCE_MEMORY"],
                    "search_text": "preferred table-tidying layout",
                    "task_hint": "tidy_table",
                    "scene_entities": ["book"],
                    "reason_code": "PREFERENCE_SENSITIVE_AMBIGUITY",
                },
                "interaction": None,
                "task_contract": None,
                "memory_action": {
                    "action": "NONE",
                    "statement": None,
                    "scope": {},
                    "structured_value": {},
                },
                "memory_refs_used": [],
                "reason_code": "MEMORY_NEEDED",
            }
        )

        self.assertIsNone(decision.user_text)
        with self.assertRaises(ValidationError):
            HRIDecision.model_validate(
                {
                    **decision.model_dump(mode="json"),
                    "reply_to_user": "I am searching memory.",
                }
            )

    def test_only_available_memory_context_may_contain_records(self) -> None:
        record_fields = {
            "relevant_preferences": [
                {"record_id": "pref-1", "statement": "Put books on the left."}
            ]
        }
        MemoryContext.model_validate(
            {"status": "AVAILABLE", **record_fields}
        )
        for status in ("EMPTY", "NOT_RETRIEVED", "UNAVAILABLE"):
            with self.subTest(status=status), self.assertRaisesRegex(
                ValidationError,
                "cannot contain records",
            ):
                MemoryContext.model_validate(
                    {"status": status, **record_fields}
                )


class PlannerContractTests(unittest.TestCase):
    def test_ready_plan_requires_monitorable_subtask_and_spec(self) -> None:
        plan = PlanResult.model_validate(
            {
                "planning_status": "READY",
                "plan_id": "plan-1",
                "plan_version": 1,
                "planning_mode": "ON_DEVIATION",
                "preconditions": [],
                "subtasks": [
                    {
                        "subtask_id": "subtask-1",
                        "task_instruction": "Place the book in the left zone.",
                        "expected_outcome": {
                            "conditions": [
                                {
                                    "condition_id": "condition-1",
                                    "description": "The book is in the left zone.",
                                    "predicate": "INSIDE_ZONE",
                                    "arguments": ["book", "zone:left"],
                                    "required": True,
                                    "observable": True,
                                }
                            ],
                            "failure_conditions": [],
                            "progress_cues": ["The book approaches the left zone."],
                        },
                        "timeout_policy": {
                            "timeout_seconds": 60,
                            "stall_seconds": 15,
                            "on_timeout": "REPLAN",
                            "on_stall": "REOBSERVE",
                        },
                    }
                ],
                "validation_spec": validation_spec(),
                "failure": None,
                "planner_confidence": 0.9,
            }
        )

        self.assertEqual(plan.subtasks[0].subtask_id, "subtask-1")
        self.assertEqual(
            plan.validation_spec.goal_conditions[0].goal_id,
            "goal-1",
        )
        plan.require_request(
            PlanningRequest(
                plan_id="plan-1",
                plan_version=1,
                planning_mode="ON_DEVIATION",
                validation_spec_id="spec-1",
            )
        )
        with self.assertRaisesRegex(ValueError, "validation spec ID"):
            plan.require_request(
                PlanningRequest(
                    plan_id="plan-1",
                    plan_version=1,
                    planning_mode="ON_DEVIATION",
                    validation_spec_id="spec-other",
                )
            )

    def test_recovery_spec_must_match_exactly(self) -> None:
        frozen = ValidationSpec.model_validate(validation_spec())
        changed_payload = validation_spec()
        changed_payload["goal_conditions"][0]["arguments"] = [
            "book",
            "zone:right",
        ]
        changed = ValidationSpec.model_validate(changed_payload)

        with self.assertRaisesRegex(ValueError, "frozen validation spec"):
            frozen.require_exact_match(changed)

    def test_ready_plan_rejects_unsatisfied_preconditions(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "unsatisfied preconditions",
        ):
            PlanResult.model_validate(
                {
                    "planning_status": "READY",
                    "plan_id": "plan-1",
                    "plan_version": 1,
                    "planning_mode": "ON_DEVIATION",
                    "preconditions": [
                        {
                            "condition_id": "clear-path",
                            "description": "The path to the left zone is clear.",
                            "satisfied": False,
                            "evidence": "The path is occluded.",
                        }
                    ],
                    "subtasks": [
                        {
                            "subtask_id": "subtask-1",
                            "task_instruction": "Place the book in the left zone.",
                            "expected_outcome": {
                                "conditions": [
                                    {
                                        "condition_id": "condition-1",
                                        "description": "The book is in the left zone.",
                                        "predicate": "INSIDE_ZONE",
                                        "arguments": ["book", "zone:left"],
                                    }
                                ],
                                "failure_conditions": [],
                                "progress_cues": [],
                            },
                            "timeout_policy": {
                                "timeout_seconds": 60,
                                "stall_seconds": 15,
                                "on_timeout": "REPLAN",
                                "on_stall": "REOBSERVE",
                            },
                        }
                    ],
                    "validation_spec": validation_spec(),
                    "failure": None,
                    "planner_confidence": 0.9,
                }
            )

    def test_validation_spec_covers_resolved_task_details(self) -> None:
        spec = ValidationSpec.model_validate(validation_spec())
        covered = TaskContract(
            task_type="placement",
            confirmed_intent="Put the book in the left zone.",
            objects=["books"],
            constraints=["Keep the book in the left zone."],
            parameters={"destination": "left"},
        )
        spec.require_task_coverage(covered)

        with self.assertRaisesRegex(ValueError, "do not cover"):
            spec.require_task_coverage(
                covered.model_copy(update={"objects": ["book", "laptop"]})
            )


class MonitorAndValidatorContractTests(unittest.TestCase):
    def test_unsafe_monitor_result_is_a_hard_stop(self) -> None:
        result = MonitorResult.model_validate(
            {
                "dispatch_id": "dispatch-1",
                "subtask_id": "subtask-1",
                "task_status": "FAILURE",
                "progress": "DEVIATED",
                "observation_quality": "ADEQUATE",
                "safety_status": "UNSAFE",
                "failure_kind": "SAFETY",
                "recommended_action": "ABORT_SAFETY",
                "condition_checks": [
                    {
                        "condition_id": "condition-1",
                        "state": "UNKNOWN",
                        "evidence": "Execution stopped after a visible hazard.",
                        "confidence": 0.95,
                    }
                ],
                "confidence": 0.95,
            }
        )

        result.require_current_dispatch(
            dispatch_id="dispatch-1",
            subtask_id="subtask-1",
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            result.require_current_dispatch(
                dispatch_id="dispatch-2",
                subtask_id="subtask-1",
            )

    def test_monitor_success_requires_quality_safety_and_required_goals(self) -> None:
        payload = {
            "dispatch_id": "dispatch-1",
            "subtask_id": "subtask-1",
            "task_status": "SUCCESS",
            "progress": "VERIFYING",
            "observation_quality": "ADEQUATE",
            "safety_status": "SAFE",
            "failure_kind": None,
            "recommended_action": "ADVANCE",
            "condition_checks": [
                {
                    "condition_id": "required-1",
                    "state": "UNKNOWN",
                    "evidence": "The target is partly occluded.",
                    "confidence": 0.4,
                },
                {
                    "condition_id": "optional-1",
                    "state": "UNKNOWN",
                    "evidence": "The optional alignment is not visible.",
                    "confidence": 0.3,
                },
            ],
            "confidence": 0.8,
        }
        for field, invalid_value in (
            ("observation_quality", "OCCLUDED"),
            ("safety_status", "UNKNOWN"),
        ):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                MonitorResult.model_validate(
                    {**payload, field: invalid_value}
                )

        result = MonitorResult.model_validate(payload)
        expected = ExpectedOutcome.model_validate(
            {
                "conditions": [
                    {
                        "condition_id": "required-1",
                        "description": "The book is in the left zone.",
                        "predicate": "INSIDE_ZONE",
                        "arguments": ["book", "zone:left"],
                        "required": True,
                    },
                    {
                        "condition_id": "optional-1",
                        "description": "The book is aligned with the table.",
                        "predicate": "ALIGNED",
                        "arguments": ["book", "table"],
                        "required": False,
                    },
                ]
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "every required condition SATISFIED",
        ):
            result.require_expected_outcome(expected)

        satisfied = result.model_copy(
            update={
                "condition_checks": [
                    result.condition_checks[0].model_copy(
                        update={
                            "state": ConditionState.SATISFIED,
                            "confidence": 0.9,
                        }
                    ),
                    result.condition_checks[1],
                ]
            }
        )
        satisfied.require_expected_outcome(expected)

    def test_unsafe_safety_status_requires_unsafe_abort(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "unsafe safety status",
        ):
            ValidationResult.model_validate(
                {
                    "spec_id": "spec-1",
                    "outcome": "FAILURE",
                    "task_complete": False,
                    "goal_checks": [
                        {
                            "goal_id": "goal-1",
                            "state": "VIOLATED",
                            "evidence": "A hazard is visible.",
                            "confidence": 0.9,
                        }
                    ],
                    "safety_status": "UNSAFE",
                    "recoverability": "REPLAN",
                    "user_message": "Unsafe.",
                    "validator_confidence": 0.9,
                }
            )

    def test_validator_outcome_is_derived_from_required_goal_states(self) -> None:
        spec = ValidationSpec.model_validate(validation_spec())
        cases = (
            ("VIOLATED", "UNKNOWN", "FAILURE"),
            ("UNKNOWN", "FAILURE", "UNKNOWN"),
        )
        for goal_state, reported_outcome, expected_outcome in cases:
            with self.subTest(goal_state=goal_state):
                inconsistent = ValidationResult.model_validate(
                    {
                        "spec_id": "spec-1",
                        "outcome": reported_outcome,
                        "task_complete": False,
                        "goal_checks": [
                            {
                                "goal_id": "goal-1",
                                "state": goal_state,
                                "evidence": "The required goal is not verified.",
                                "confidence": 0.5,
                            }
                        ],
                        "safety_status": "SAFE",
                        "recoverability": "REPLAN",
                        "user_message": "The goal is not met.",
                        "validator_confidence": 0.5,
                    }
                )
                with self.assertRaisesRegex(
                    ValueError,
                    f"expected {expected_outcome}",
                ):
                    inconsistent.require_exact_spec(spec)

    def test_validator_evidence_refs_must_be_host_supplied(self) -> None:
        spec = ValidationSpec.model_validate(validation_spec())
        request = ValidationRequest.model_validate(
            {
                "validation_spec": spec.model_dump(mode="json"),
                "terminal_observations": [
                    {
                        "observation_id": "observation-2",
                        "captured_at": datetime.now(timezone.utc),
                        "image_block": {"type": "image_url"},
                    }
                ],
                "execution_evidence": {
                    "dispatches": [
                        {"command": {"dispatch_id": "dispatch-1"}}
                    ]
                },
            }
        )
        result_payload = {
            "spec_id": "spec-1",
            "outcome": "SUCCESS",
            "task_complete": True,
            "goal_checks": [
                {
                    "goal_id": "goal-1",
                    "state": "SATISFIED",
                    "evidence": "The book is visibly inside the zone.",
                    "evidence_refs": ["observation-2"],
                    "confidence": 0.98,
                }
            ],
            "safety_status": "SAFE",
            "discrepancies": [],
            "recoverability": "NONE",
            "user_message": "Task complete.",
            "validator_confidence": 0.98,
        }
        ValidationResult.model_validate(result_payload).require_request(request)

        wrong_modality = {
            **result_payload,
            "goal_checks": [
                {
                    **result_payload["goal_checks"][0],
                    "evidence_refs": ["dispatch-1"],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "declared modalities"):
            ValidationResult.model_validate(wrong_modality).require_request(
                request
            )

        hallucinated = {
            **result_payload,
            "goal_checks": [
                {
                    **result_payload["goal_checks"][0],
                    "evidence_refs": ["observation-not-supplied"],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "not supplied by the host"):
            ValidationResult.model_validate(hallucinated).require_request(request)

    def test_validator_must_check_frozen_goals_exactly(self) -> None:
        spec = ValidationSpec.model_validate(validation_spec())
        result = ValidationResult.model_validate(
            {
                "spec_id": "spec-1",
                "outcome": "SUCCESS",
                "task_complete": True,
                "goal_checks": [
                    {
                        "goal_id": "goal-1",
                        "state": "SATISFIED",
                        "evidence": "The book is visibly inside the zone.",
                        "evidence_refs": ["observation-2"],
                        "confidence": 0.98,
                    }
                ],
                "safety_status": "NOT_EVALUATED",
                "discrepancies": [],
                "recoverability": "NONE",
                "user_message": "Task complete.",
                "validator_confidence": 0.98,
            }
        )

        result.require_exact_spec(spec)
        self.assertEqual(
            result.goal_checks[0].state,
            ConditionState.SATISFIED,
        )


if __name__ == "__main__":
    unittest.main()
