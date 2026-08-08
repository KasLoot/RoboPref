from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from prefmem.agents.planner import (
    PlannerOutputError,
    Planner_Agent,
    parse_json_object,
)
from prefmem.contracts import (
    ExecutionOutcome,
    ExecutionRecord,
    GoalContract,
    PlanStatus,
    PlannerCycleRequest,
    PlannerDecisionType,
    PlannerTrigger,
)


FRAME = {
    "type": "image_url",
    "image_url": {"url": "data:image/jpeg;base64,AA=="},
}


class QueueModel:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list, dict]] = []

    def invoke(self, messages: list, **kwargs) -> AIMessage:
        self.calls.append((messages, kwargs))
        if not self.responses:
            raise AssertionError("model received more calls than expected")
        return AIMessage(content=self.responses.pop(0))


def planner_with_model(model: QueueModel) -> Planner_Agent:
    planner = object.__new__(Planner_Agent)
    planner.llm = model
    planner.system_prompt = "planner system"
    planner.thinking_enabled = False
    planner.metrics = None
    planner.print_raw = False
    return planner


def preview_json(**updates) -> str:
    payload = {
        "status": "READY",
        "goal": "Build a stable tower from all three blocks.",
        "final_expected_observation": [
            "All three blocks form one stable upright tower."
        ],
        "constraints": [],
        "nominal_tasks": [
            "Place the red block flat as the base.",
            "Stack the green and blue blocks centrally on the base.",
        ],
        "reason": None,
    }
    payload.update(updates)
    return json.dumps(payload)


def cycle_request() -> PlannerCycleRequest:
    return PlannerCycleRequest(
        goal_contract=GoalContract(
            goal_id="tower-1",
            revision=1,
            goal="Build a stable tower from all three blocks.",
            final_expected_observation=(
                "All three blocks form one stable upright tower.",
            ),
            constraints=("Keep every block on the table.",),
            nominal_tasks=("Build a flat base, then stack centrally.",),
        ),
        cycle_id=2,
        trigger=PlannerTrigger.TASK_FAIL,
        execution_history=(
            ExecutionRecord(
                cycle_id=1,
                publication_id="tower-1:c1:a1",
                instruction="Place the green block on the red block.",
                expected_observation=(
                    "The green block rests centrally on the red block.",
                ),
                outcome=ExecutionOutcome.FAIL,
                observation="The two-block stack collapsed.",
                failure_reason="The green block slid off after release.",
                evidence_frame_sequence=17,
            ),
        ),
        frame_sequence=18,
    )


def act_json(**updates) -> str:
    payload = {
        "decision": "ACT",
        "candidate_tasks": [
            {
                "instruction": "Place the red block flat in the tower area.",
                "expected_observation": [
                    "The red block lies flat in the tower area.",
                    "The red block is stable and every block remains on the table.",
                ],
                "known_failure_conditions": [],
            }
        ],
        "reason": "The collapsed base must be rebuilt.",
        "blocked_reason": None,
        "user_question": None,
    }
    payload.update(updates)
    return json.dumps(payload)


class PlannerJsonTests(unittest.TestCase):
    def test_parser_tolerates_fence_and_ghost_thinking(self) -> None:
        response = AIMessage(
            content=(
                "<think>This block is transport-only reasoning.</think>\n"
                f"```json\n{preview_json()}\n```"
            )
        )

        payload = parse_json_object(response)

        self.assertEqual(payload["status"], "READY")
        self.assertEqual(len(payload["nominal_tasks"]), 2)

    def test_parser_rejects_multiple_or_outside_json(self) -> None:
        with self.assertRaisesRegex(PlannerOutputError, "one JSON object"):
            parse_json_object('{"first": 1}\n{"second": 2}')
        with self.assertRaisesRegex(PlannerOutputError, "outside"):
            parse_json_object('```json\n{"first": 1}\n```\n{"second": 2}')


class PlannerBoundaryTests(unittest.TestCase):
    def test_preview_is_typed_and_orders_system_image_text(self) -> None:
        model = QueueModel(preview_json())
        planner = planner_with_model(model)

        proposal = planner.preview(
            "Build a stable tower from all three blocks.",
            FRAME,
            constraints=("Keep every block on the table.",),
        )

        self.assertIs(proposal.status, PlanStatus.READY)
        self.assertEqual(len(model.calls), 1)
        messages, options = model.calls[0]
        self.assertEqual(options, {})
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        self.assertEqual(messages[1].content[0], FRAME)
        request = json.loads(messages[1].content[1]["text"])
        self.assertEqual(request["request_kind"], "PREVIEW")
        self.assertEqual(
            request["clarified_goal"],
            "Build a stable tower from all three blocks.",
        )

    def test_cycle_contains_frozen_goal_trigger_and_terminal_history(self) -> None:
        model = QueueModel(act_json())
        planner = planner_with_model(model)

        decision = planner.plan_cycle(cycle_request(), FRAME)

        self.assertIs(decision.decision, PlannerDecisionType.ACT)
        self.assertEqual(len(decision.candidate_tasks), 1)
        messages, _ = model.calls[0]
        request = json.loads(messages[1].content[1]["text"])
        self.assertEqual(request["request_kind"], "PLAN_CYCLE")
        self.assertEqual(request["trigger"], "TASK_FAIL")
        self.assertEqual(request["goal_contract"]["goal_id"], "tower-1")
        self.assertEqual(
            request["execution_history"][-1]["failure_reason"],
            "The green block slid off after release.",
        )
        self.assertNotIn("messages", request)

    def test_schema_failure_gets_targeted_correction(self) -> None:
        model = QueueModel(
            preview_json(unexpected="not allowed"),
            preview_json(),
        )
        planner = planner_with_model(model)

        proposal = planner.preview("Build the tower.", FRAME)

        self.assertIs(proposal.status, PlanStatus.READY)
        self.assertEqual(len(model.calls), 2)
        correction = json.loads(model.calls[1][0][1].content[1]["text"])
        self.assertIn("schema_correction", correction)
        self.assertIn("unknown fields", correction["schema_correction"]["error"])
        self.assertEqual(
            correction["schema_correction"]["contract"],
            "GoalProposal",
        )

    def test_third_invalid_response_is_not_retried_again(self) -> None:
        model = QueueModel("not json", "still not json", "also not json")
        planner = planner_with_model(model)

        with self.assertRaisesRegex(
            PlannerOutputError,
            "after 2 schema corrections",
        ):
            planner.preview("Build the tower.", FRAME)

        self.assertEqual(len(model.calls), 3)

    def test_oversized_task_checklist_gets_second_targeted_correction(self) -> None:
        oversized_payload = json.loads(act_json())
        oversized_payload["candidate_tasks"][0]["expected_observation"] = [
            "The red block is on the platform.",
            "The green block remains visible.",
            "The blue block remains visible.",
            "The full tower is stable.",
        ]
        oversized = json.dumps(oversized_payload)
        model = QueueModel(oversized, oversized, act_json())
        planner = planner_with_model(model)

        decision = planner.plan_cycle(cycle_request(), FRAME)

        self.assertIs(decision.decision, PlannerDecisionType.ACT)
        self.assertEqual(len(model.calls), 3)
        for call_index in (1, 2):
            correction = json.loads(
                model.calls[call_index][0][1].content[1]["text"]
            )["schema_correction"]
            self.assertEqual(correction["contract"], "PlannerDecision")
            self.assertEqual(correction["attempt"], call_index)
            self.assertTrue(
                any(
                    "expected_observation" in rule and "1 to 3" in rule
                    for rule in correction["contract_rules"]
                )
            )

    def test_passed_frame_never_triggers_hidden_live_fetch(self) -> None:
        planner = planner_with_model(QueueModel(preview_json()))
        with patch(
            "prefmem.agents.planner.get_live_frame",
            side_effect=AssertionError("unexpected live fetch"),
        ):
            proposal = planner.preview("Build the tower.", FRAME)

        self.assertIs(proposal.status, PlanStatus.READY)

    def test_legacy_run_uses_explicit_frame_without_refetching(self) -> None:
        planner = object.__new__(Planner_Agent)
        received: list[object] = []
        planner.invoke_agent = lambda messages, current_frame: received.append(
            current_frame
        ) or {"messages": []}
        with patch(
            "prefmem.agents.planner.get_live_frame",
            side_effect=AssertionError("unexpected live fetch"),
        ):
            planner.run([HumanMessage(content="request")], current_frame=FRAME)

        self.assertEqual(received, [FRAME])


class PlannerPromptTests(unittest.TestCase):
    def test_prompt_defines_receding_horizon_and_full_post_state(self) -> None:
        prompt = Path(
            "src/prefmem/agents/prompt/planner/planner-prompt-v2.md"
        ).read_text(encoding="utf-8")

        self.assertIn("publish only `candidate_tasks[0]`", prompt)
        self.assertIn("discard the remaining prediction tail", prompt)
        self.assertIn("complete relevant post-state", prompt)
        self.assertIn("Never copy the full goal-wide final", prompt)
        self.assertIn("prior stack", prompt)
        self.assertIn("TASK_FAIL", prompt)
        self.assertIn("NEEDS_USER_INPUT", prompt)


if __name__ == "__main__":
    unittest.main()
