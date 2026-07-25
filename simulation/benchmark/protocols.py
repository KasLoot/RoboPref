from __future__ import annotations

import copy
from typing import Any, Iterable


def memory_protocol_fixtures() -> dict[str, dict[str, Any]]:
    """Return semantic fixture requests, not repository-internal JSON records."""

    preference_request = {
        "user_id": "participant-a",
        "requested_action": "UPSERT",
        "instruction": (
            "Use red, green, blue from bottom to top when stacking coloured "
            "blocks in future."
        ),
        "preference": {
            "statement": (
                "When stacking the red, green, and blue blocks, use red at the "
                "bottom, green in the middle, and blue at the top."
            ),
            "scope": "contextual",
            "applicability": {
                "task_family": "stack_blocks",
                "objects": ["red block", "green block", "blue block"],
            },
            "structured_value": {
                "order_bottom_to_top": ["red", "green", "blue"],
            },
        },
    }
    approved_rgb = {
        "fixture_kind": "approved_preference_request",
        "user_id": "participant-a",
        "preference_request": preference_request,
        "consent": {
            "kind": "explicit_future_language",
            "quote": "Remember that order from now on.",
            "turn_id": "fixture-approved-rgb-stack-preference",
            "authorized_action": "UPSERT",
            "proposal": copy.deepcopy(preference_request),
        },
    }
    return {
        "approved-rgb-stack-preference": copy.deepcopy(approved_rgb),
        "participant-a-approved-rgb-only": copy.deepcopy(approved_rgb),
    }


def build_memory_protocols(scenarios: Iterable[Any]) -> list[dict[str, Any]]:
    """Build semantic multi-conversation checks around one RGB stack endpoint.

    Protocols intentionally describe expected state changes rather than exact model
    wording.  A future experiment runner can bind ``scenario_selector`` to any
    matching opaque episode without exposing the oracle manifest to PrefMem.
    """

    scenario_list = list(scenarios)
    available = {
        (
            scenario.family,
            scenario.target_id,
            scenario.outcome,
        )
        for scenario in scenario_list
    }
    required = ("block_stack", "rgb_bottom_to_top", "success")
    if required not in available:
        return []

    rgb_selector = {
        "family": "block_stack",
        "target_id": "rgb_bottom_to_top",
        "outcome": "success",
    }
    bgr_selector = {
        "family": "block_stack",
        "target_id": "bgr_bottom_to_top",
        "outcome": "success",
    }
    protocols = [
        {
            "protocol_id": "memory-first-choice-history-only",
            "description": "A first one-off choice becomes history, never preference.",
            "user_id": "participant-a",
            "requires_fresh_memory": True,
            "steps": [
                {
                    "query": "Stack the blocks.",
                    "scenario_selector": rgb_selector,
                    "expected": {
                        "clarification_required": True,
                        "history_delta": 0,
                        "preference_delta": 0,
                    },
                },
                {
                    "reply": "Red, green, then blue from bottom to top.",
                    "expected": {
                        "min_dispatches": 1,
                        "validator_outcome": "SUCCESS",
                        "history_delta": 1,
                        "preference_delta": 0,
                        "task_semantic_score": True,
                    },
                }
            ],
        },
        {
            "protocol_id": "memory-repeat-defer-then-consent",
            "description": (
                "A repeated choice can trigger a post-task save question; defer is "
                "not consent, while a later explicit yes creates one compact record."
            ),
            "user_id": "participant-a",
            "requires_fresh_memory": True,
            "steps": [
                {
                    "query": "Stack the blocks.",
                    "scenario_selector": rgb_selector,
                    "expected": {
                        "clarification_required": True,
                        "history_delta": 0,
                        "preference_delta": 0,
                    },
                },
                {
                    "reply": "Red, green, then blue from bottom to top.",
                    "expected": {
                        "history_delta": 1,
                        "preference_delta": 0,
                        "validator_outcome": "SUCCESS",
                        "min_dispatches": 1,
                        "post_task_preference_question": False,
                        "task_semantic_score": True,
                    },
                },
                {
                    "query": "Stack the blocks.",
                    "scenario_selector": rgb_selector,
                    "expected": {
                        "clarification_required": True,
                        "history_delta": 0,
                        "preference_delta": 0,
                    },
                },
                {
                    "reply": "RGB again.",
                    "expected": {
                        "history_delta": 0,
                        "preference_delta": 0,
                        "validator_outcome": "SUCCESS",
                        "min_dispatches": 1,
                        "post_task_preference_question": True,
                        "task_semantic_score": True,
                    },
                },
                {
                    "reply": "I want to decide later.",
                    "expected": {
                        "history_delta": 1,
                        "preference_delta": 0,
                        "pending_question_cleared": True,
                    },
                },
                {
                    "query": "Please stack those cubes the same way as before.",
                    "scenario_selector": rgb_selector,
                    "expected": {
                        "history_retrieval_required": True,
                        "validator_outcome": "SUCCESS",
                        "min_dispatches": 1,
                        "history_delta": 0,
                        "preference_delta": 0,
                        "post_task_preference_question": True,
                        "task_semantic_score": True,
                    },
                },
                {
                    "reply": "Yes, remember that order from now on.",
                    "expected": {
                        "history_delta": 1,
                        "preference_delta": 1,
                        "active_equivalent_preferences": 1,
                    },
                },
            ],
        },
        {
            "protocol_id": "memory-retrieval-and-one-off-override",
            "description": (
                "A semantic paraphrase retrieves the approved RGB preference. A "
                "one-off BGR instruction overrides execution but does not replace it."
            ),
            "user_id": "participant-a",
            "requires_fresh_memory": True,
            "initial_memory_fixture": "approved-rgb-stack-preference",
            "steps": [
                {
                    "query": "Build a tower with the coloured cubes.",
                    "scenario_selector": rgb_selector,
                    "expected": {
                        "clarification_required": False,
                        "preference_retrieval_required": True,
                        "min_dispatches": 1,
                        "validator_outcome": "SUCCESS",
                        "history_delta": 1,
                        "preference_delta": 0,
                        "task_semantic_score": True,
                    },
                },
                {
                    "query": (
                        "This time only, stack blue, green, red from bottom to top."
                    ),
                    "scenario_selector": bgr_selector,
                    "expected": {
                        "uses_one_off_override": True,
                        "active_rgb_preference_preserved": True,
                        "min_dispatches": 1,
                        "validator_outcome": "SUCCESS",
                        "history_delta": 1,
                        "preference_delta": 0,
                        "post_task_preference_question": False,
                        "task_semantic_score": True,
                    },
                },
            ],
        },
        {
            "protocol_id": "memory-user-isolation",
            "description": "Preferences never cross participant boundaries.",
            "user_id": "participant-b",
            "requires_fresh_memory": True,
            "initial_memory_fixture": "participant-a-approved-rgb-only",
            "steps": [
                {
                    "query": "Stack the blocks.",
                    "scenario_selector": rgb_selector,
                    "expected": {
                        "clarification_required": True,
                        "foreign_preference_refs": 0,
                    },
                }
            ],
        },
    ]
    if any(
        getattr(scenario, "control_kind", None) == "already_satisfied"
        and scenario.family == "block_stack"
        and scenario.target_id == "rgb_bottom_to_top"
        for scenario in scenario_list
    ):
        protocols.append(
            {
                "protocol_id": "control-already-satisfied",
                "description": (
                    "The requested RGB stack already exists; validate it without "
                    "dispatching a VLA subtask."
                ),
                "user_id": "participant-a",
                "requires_fresh_memory": True,
                "steps": [
                    {
                        "query": "Stack the blocks in RGB order from bottom to top.",
                        "scenario_selector": {
                            "family": "block_stack",
                            "target_id": "rgb_bottom_to_top",
                            "outcome": "success",
                            "control_kind": "already_satisfied",
                        },
                        "expected": {
                            "planner_status": "ALREADY_SATISFIED",
                            "dispatches": 0,
                            "validator_outcome": "SUCCESS",
                            "history_delta": 1,
                            "preference_delta": 0,
                            "task_semantic_score": True,
                        },
                    }
                ],
            }
        )
    return protocols
