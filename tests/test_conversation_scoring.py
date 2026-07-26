from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from simulation.benchmark.conversation_cases import load_episode_catalog
from simulation.benchmark.conversation_models import CommandSpec, EpisodeDescriptor
from simulation.benchmark.conversation_scoring import score_conversation_run


ROOT = Path(__file__).resolve().parents[1] / "dataset" / "sim_datasets"


def validation_spec(descriptor: EpisodeDescriptor) -> dict[str, Any]:
    conditions = []
    for index, value in enumerate(descriptor.manifest["target"]["goal_predicates"]):
        predicate, remainder = value.split("(", 1)
        arguments_text = remainder.rsplit(")", 1)[0]
        conditions.append(
            {
                "id": f"goal-{index + 1}",
                "description": value,
                "predicate": predicate,
                "arguments": [
                    item.strip()
                    for item in arguments_text.split(",")
                    if item.strip()
                ],
                "required": True,
                "observable": True,
            }
        )
    return {
        "spec_id": "spec-evaluation",
        "confirmed_intent": descriptor.instruction,
        "goal_conditions": conditions,
    }


def run_record(
    descriptor: EpisodeDescriptor,
    *,
    validator_outcome: str | None = None,
    task_complete: bool | None = None,
    terminal_outcome: str | None = None,
) -> dict[str, Any]:
    expectations = descriptor.manifest["benchmark_expectations"]
    expected_validator = expectations["validator"]
    validator_outcome = (
        expected_validator["outcome"]
        if validator_outcome is None
        else validator_outcome
    )
    task_complete = (
        expected_validator["task_complete"]
        if task_complete is None
        else task_complete
    )
    terminal_outcome = (
        expectations["hri"]["terminal_outcome"]
        if terminal_outcome is None
        else terminal_outcome
    )
    command = CommandSpec(
        command_id="explicit-endpoint",
        episode_id=descriptor.scenario_id,
        query=descriptor.instruction,
        target_id=descriptor.target_id,
        query_kind="explicit",
        expected_initial_modes=("EXECUTE",),
    )
    validation = {
        "outcome": validator_outcome,
        "task_complete": task_complete,
    }
    attempt = {
        "attempt": 1,
        "plan": {
            "planning_status": expectations["planner"]["expected_status"],
            "validation_spec": validation_spec(descriptor),
            "subtasks": [{"task_instruction": descriptor.instruction}],
        },
        "execution": {
            "status": expectations["execution"]["expected_status"],
            "subtask_results": [{"status": "RECORDED_EXTERNAL_ATTEMPT"}],
        },
        "initial_validation": validation,
        "initial_validation_assurance": {
            "next_action": expectations["recovery"]
        },
        "validation": validation,
        "validation_assurance": {"next_action": expectations["recovery"]},
    }
    task = {"outcome": terminal_outcome, "attempts": [attempt]}
    return {
        "schema_version": "robopref.conversation-run.v1",
        "case": {
            "case_id": "endpoint-case",
            "suite": "endpoint",
            "profile": "explicit",
            "cluster_id": descriptor.initial_frame_sha256,
            "metadata": descriptor.public_metadata(),
            "commands": [command.to_dict()],
        },
        "commands": [
            {
                "command": command.to_dict(),
                "episode": descriptor.public_metadata(),
                "turns": [
                    {
                        "decision_mode": "EXECUTE",
                        "result": {
                            "hri_decision": {
                                "mode": "EXECUTE",
                                "trace": {
                                    "history_refs": [],
                                    "memory_refs": [],
                                },
                            },
                            "hri": {
                                "mode": "REPORT",
                                "report": {"outcome": terminal_outcome},
                            },
                            "task": task,
                        },
                        "memory_delta": {
                            "history_delta": 1,
                            "preference_delta": 0,
                            "history_state_changed": True,
                            "preference_state_changed": False,
                        },
                        "delivered_memory": {
                            "owned_history_ids": [],
                            "owned_preference_ids": [],
                            "foreign_history_ids": [],
                            "foreign_preference_ids": [],
                            "history_content_mismatch_ids": [],
                            "preference_content_mismatch_ids": [],
                            "malformed_history_records": 0,
                            "malformed_preference_records": 0,
                            "history_ownership_verified": True,
                            "preference_ownership_verified": True,
                        },
                    }
                ],
                "script_events": [],
                "task": task,
                "resolved_task": {
                    "confirmed_intent": descriptor.instruction,
                    "parameters": {},
                    "objects": [],
                    "preference_refs": [],
                },
                "memory_delta": {
                    "history_delta": 1,
                    "preference_delta": 0,
                    "history_state_changed": True,
                    "preference_state_changed": False,
                },
                "command_complete": True,
                "turn_limit_reached": False,
            }
        ],
        "artifact_status": {"complete": True},
    }


class ConversationScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog = load_episode_catalog(ROOT)
        cls.by_id = {item.scenario_id: item for item in catalog}
        cls.success = next(
            item
            for item in catalog
            if item.outcome == "success" and item.control_kind is None
        )
        cls.wrong = next(item for item in catalog if item.outcome == "wrong_complete")
        cls.control = next(
            item
            for item in catalog
            if item.control_kind == "already_satisfied"
        )

    def score(self, descriptor: EpisodeDescriptor, **changes: Any) -> dict[str, Any]:
        return score_conversation_run(
            run_record(descriptor, **changes),
            self.by_id,
            near_miss_policy="strict",
        )

    def test_correct_success_chain_passes(self) -> None:
        result = self.score(self.success)
        self.assertEqual(result["benchmark_result"], "PASS", result["checks"])

    def test_exact_predicates_cannot_hide_unrelated_vla_actions(self) -> None:
        raw = run_record(self.success)
        plan = raw["commands"][0]["task"]["attempts"][0]["plan"]
        plan["subtasks"] = [
            {"task_instruction": "Do nothing unrelated to the requested task."}
        ]
        result = score_conversation_run(
            raw,
            self.by_id,
            near_miss_policy="strict",
        )
        check = next(
            item
            for item in result["checks"]
            if item["name"] == "planner_vla_subtask_target"
        )
        self.assertEqual(check["status"], "FAIL")
        self.assertEqual(result["benchmark_result"], "FAIL")

    def test_already_satisfied_plan_requires_explicit_empty_subtasks(self) -> None:
        raw = run_record(self.control)
        plan = raw["commands"][0]["task"]["attempts"][0]["plan"]
        plan["subtasks"] = []
        empty = score_conversation_run(
            raw,
            self.by_id,
            near_miss_policy="strict",
        )
        empty_check = next(
            item
            for item in empty["checks"]
            if item["name"] == "planner_subtasks_empty"
        )
        self.assertEqual(empty_check["status"], "PASS")

        plan["subtasks"] = [
            {"task_instruction": "Move a block despite the completed scene."}
        ]
        nonempty = score_conversation_run(
            raw,
            self.by_id,
            near_miss_policy="strict",
        )
        nonempty_check = next(
            item
            for item in nonempty["checks"]
            if item["name"] == "planner_subtasks_empty"
        )
        self.assertEqual(nonempty_check["status"], "FAIL")

    def test_correctly_detected_wrong_completion_is_a_benchmark_pass(self) -> None:
        result = self.score(self.wrong)
        self.assertEqual(result["benchmark_result"], "PASS", result["checks"])
        check = next(
            item for item in result["checks"] if item["name"] == "no_false_completion"
        )
        self.assertEqual(check["status"], "PASS")

    def test_successful_recovery_is_not_scored_as_false_completion(self) -> None:
        raw = run_record(self.wrong)
        raw["case"]["suite"] = "recovery"
        task = raw["commands"][0]["task"]
        task["attempts"].append(
            {
                "attempt": 2,
                "plan": task["attempts"][0]["plan"],
                "execution": {
                    "status": "OBSERVED_RECORDED_ATTEMPT",
                    "final_observation": "recovery-final.png",
                    "subtask_results": [
                        {"status": "RECORDED_EXTERNAL_ATTEMPT"}
                    ],
                },
                "validation": {
                    "outcome": "SUCCESS",
                    "task_complete": True,
                },
                "validation_assurance": {"next_action": "NONE"},
            }
        )
        task["outcome"] = "SUCCESS_RECOVERED"
        result = score_conversation_run(
            raw,
            self.by_id,
            near_miss_policy="strict",
        )
        self.assertEqual(result["benchmark_result"], "PASS", result["checks"])
        statuses = {
            item["name"]: item["status"]
            for item in result["checks"]
        }
        self.assertEqual(statuses["terminal_outcome"], "NOT_APPLICABLE")
        self.assertEqual(statuses["no_false_completion"], "PASS")

    def test_missing_planned_commands_cannot_vacuously_pass(self) -> None:
        raw = run_record(self.success)
        raw["commands"] = []
        result = score_conversation_run(
            raw,
            self.by_id,
            near_miss_policy="strict",
        )
        self.assertEqual(result["benchmark_result"], "FAIL")
        coverage = next(
            item
            for item in result["checks"]
            if item["name"] == "command_sequence_complete"
        )
        self.assertEqual(coverage["status"], "FAIL")

    def test_false_success_on_negative_packet_is_explicitly_failed(self) -> None:
        result = self.score(
            self.wrong,
            validator_outcome="SUCCESS",
            task_complete=True,
            terminal_outcome="SUCCESS",
        )
        self.assertEqual(result["benchmark_result"], "FAIL")
        check = next(
            item for item in result["checks"] if item["name"] == "no_false_completion"
        )
        self.assertEqual(check["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
