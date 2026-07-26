from __future__ import annotations

import unittest

from agents.hri import HRIContractError, HRIOrchestrator


def proposal() -> dict[str, object]:
    return {
        "confirmed_intent": "Stack red, green, blue from bottom to top.",
        "task_type": "stack_blocks",
        "objects": ["red block", "green block", "blue block"],
        "parameters": {
            "color_positions": {
                "bottom": "red",
                "middle": "green",
                "top": "blue",
            }
        },
    }


class HRIConfirmationContractTests(unittest.TestCase):
    def test_confirmation_requires_and_preserves_structured_proposal(self) -> None:
        response = {
            "mode": "CONFIRM",
            "user_message": "Should I use RGB order?",
            "task_contract": None,
            "pending_question": {
                "kind": "TASK_CONFIRMATION",
                "payload": {"proposed_task": proposal()},
            },
            "memory_action": {"action": "NONE"},
            "report": None,
        }

        normalized = HRIOrchestrator._normalize_hri_transition_output(
            response,
            pending_before=None,
            post_task_memory_reply=False,
        )

        self.assertEqual(
            normalized["pending_question"]["payload"]["proposed_task"],
            proposal(),
        )

    def test_confirmation_missing_proposal_fails_closed(self) -> None:
        response = {
            "mode": "CONFIRM",
            "user_message": "Is that okay?",
            "task_contract": None,
            "pending_question": {
                "kind": "TASK_CONFIRMATION",
                "payload": {},
            },
            "memory_action": {"action": "NONE"},
            "report": None,
        }
        with self.assertRaisesRegex(HRIContractError, "payload.proposed_task"):
            HRIOrchestrator._normalize_hri_transition_output(
                response,
                pending_before=None,
                post_task_memory_reply=False,
            )

    def test_nested_oracle_fields_are_rejected(self) -> None:
        for key in (
            "scenario_id",
            "target_id",
            "expected_outcome",
            "control_kind",
            "goal_predicates",
        ):
            with self.subTest(key=key):
                value = proposal()
                value["parameters"]["metadata"] = {key: "private"}  # type: ignore[index]
                with self.assertRaisesRegex(
                    HRIContractError, "oracle-only field"
                ):
                    HRIOrchestrator._normalize_task_proposal(value)

    def test_only_clarification_null_payload_is_losslessly_normalized(self) -> None:
        clarification = HRIOrchestrator._validate_hri_output(
            {
                "mode": "ASK",
                "user_message": "Which order?",
                "pending_question": {
                    "kind": "TASK_CLARIFICATION",
                    "payload": None,
                },
            }
        )
        self.assertEqual(clarification["pending_question"]["payload"], {})

        confirmation = HRIOrchestrator._validate_hri_output(
            {
                "mode": "CONFIRM",
                "user_message": "Use RGB?",
                "pending_question": {
                    "kind": "TASK_CONFIRMATION",
                    "payload": None,
                },
            }
        )
        with self.assertRaisesRegex(HRIContractError, "object payload"):
            HRIOrchestrator._normalize_hri_transition_output(
                confirmation,
                pending_before=None,
                post_task_memory_reply=False,
            )


if __name__ == "__main__":
    unittest.main()
