from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from typing import Any

from agents.configs import PrefMemConfig
from agents.contracts import ExecutionResult, GoalCondition, PlanResult, ValidationResult, ValidationSpec
from agents.hri import HRIContractError, HRIOrchestrator
from agents.memory import MemoryAgent
from memory.models import ConsentEvidence, MemoryContext
from memory.repositories import (
    HistoryOutboxRepository,
    HistoryRepository,
    MemoryRepositoryError,
    PreferenceRepository,
)
from tests.fakes import FixedPlanner, FixedValidator, RecordingExecutor, RecordingMemoryAgent, ScriptedJsonModel


ROOT = Path(__file__).resolve().parents[1]


def make_config(directory: str) -> PrefMemConfig:
    return PrefMemConfig(
        workspace_root=str(ROOT),
        dataset_path=str(ROOT / "dataset" / "v3"),
        history_store_path=str(Path(directory) / "history.json"),
        preference_store_path=str(Path(directory) / "preferences.json"),
        user_id="participant-a",
        max_replans=0,
        max_reobservations=0,
        compact_preferences_after_write=False,
    )


def ready_components() -> tuple[FixedPlanner, FixedValidator, RecordingExecutor]:
    intent = "Stack red, green, and blue from bottom to top."
    spec = ValidationSpec(
        spec_id="spec-rgb",
        confirmed_intent=intent,
        goal_conditions=(GoalCondition("goal-rgb", "Red is below green and green is below blue."),),
    )
    plan = PlanResult(
        status="READY",
        subtasks=[{"task_instruction": "Place green on red, then blue on green."}],
        validation_spec=spec,
        confidence=0.99,
    )
    validation = ValidationResult(
        outcome="SUCCESS",
        task_complete=True,
        goal_checks=[{"goal_id": "goal-rgb", "satisfied": True}],
        discrepancies=[],
        confidence=0.99,
        recoverability="NONE",
        user_message="Task Complete.",
    )
    return (
        FixedPlanner(plan),
        FixedValidator(validation),
        RecordingExecutor(str(ROOT / "dataset" / "v3" / "12.png")),
    )


class TransactionRevisionTests(unittest.TestCase):
    def test_corrupt_primary_recovers_from_last_valid_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.json"
            repository = PreferenceRepository(path)
            repository.apply_transaction(
                user_id="participant-a",
                transaction_id="txn-backup",
                operations=[
                    {
                        "operation_id": "op-backup",
                        "action": "ADD",
                        "preference": {"statement": "Use RGB for block stacks."},
                    }
                ],
                authorization=ConsentEvidence(
                    kind="explicit_future_language",
                    quote="Remember RGB.",
                    turn_id="turn-backup",
                    authorized_action="UPSERT",
                    proposal={"instruction": "Remember RGB."},
                ),
            )
            path.write_text("{not valid json", encoding="utf-8")
            recovered = PreferenceRepository(path).list_preferences("participant-a")
            self.assertEqual(len(recovered), 1)
            self.assertTrue(list(Path(directory).glob("preferences.json.corrupt-*")))

    def test_history_outbox_is_idempotent_and_removable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outbox = HistoryOutboxRepository(Path(directory) / "history_outbox.json")
            source = {
                "episode_id": "episode-retry",
                "user_id": "participant-a",
                "user_request": "Stack the blocks.",
            }
            outbox.enqueue(source, "temporary model failure")
            outbox.enqueue(source, "temporary model failure")
            pending = outbox.list_pending("participant-a")
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["attempts"], 2)
            outbox.remove("episode-retry")
            self.assertEqual(outbox.list_pending("participant-a"), [])

    def test_stale_expected_revision_rejects_without_lost_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = PreferenceRepository(Path(directory) / "preferences.json")
            consent = ConsentEvidence(
                kind="explicit_future_language",
                quote="Remember RGB from now on.",
                turn_id="turn-1",
                authorized_action="UPSERT",
                proposal={"statement": "RGB"},
            )
            operation = {
                "operation_id": "op-1",
                "action": "ADD",
                "preference": {"statement": "Use RGB for block stacks."},
            }
            repository.apply_transaction(
                user_id="participant-a",
                transaction_id="txn-1",
                operations=[operation],
                authorization=consent,
                requested_action="UPSERT",
                expected_version=0,
                allowed_preference_ids=set(),
            )
            with self.assertRaisesRegex(MemoryRepositoryError, "Stale preference revision"):
                repository.apply_transaction(
                    user_id="participant-a",
                    transaction_id="txn-2",
                    operations=[{**operation, "operation_id": "op-2"}],
                    authorization=ConsentEvidence(
                        kind="explicit_future_language",
                        quote="Remember it.",
                        turn_id="turn-2",
                        authorized_action="UPSERT",
                        proposal={"statement": "RGB"},
                    ),
                    requested_action="UPSERT",
                    expected_version=0,
                    allowed_preference_ids=set(),
                )
            self.assertEqual(repository.version, 1)
            self.assertEqual(len(repository.list_preferences("participant-a")), 1)
            audit_events = [
                json.loads(line)
                for line in repository.audit_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [event["status"] for event in audit_events], ["ACCEPTED", "REJECTED"]
            )


class PostTaskPreferenceProposalTests(unittest.TestCase):
    def test_second_matching_task_prompts_at_boundary_and_defer_writes_only_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(directory)
            history = HistoryRepository(config.history_store_path)
            preferences = PreferenceRepository(config.preference_store_path)
            history.append(
                {
                    "episode_id": "episode-prior",
                    "user_id": "participant-a",
                    "summary": "The participant chose RGB bottom-to-top for a block stack.",
                    "user_request": "Stack the cubes.",
                    "user_choices": {"order": "RGB bottom-to-top"},
                    "terminal_outcome": "SUCCESS",
                }
            )

            def retrieve_history(payload: dict[str, Any]) -> dict[str, Any]:
                return {
                    "matches": [
                        {
                            "episode_id": payload["episodes"][0]["episode_id"],
                            "confidence": 0.98,
                            "reason": "Semantically the same stacking task.",
                        }
                    ]
                }

            def propose(payload: dict[str, Any]) -> dict[str, Any]:
                current_id = payload["current_episode_id"]
                return {
                    "should_ask": True,
                    "proposal": {
                        "question": "Would you like me to remember RGB as your future block-stacking default?",
                        "preference_request": {
                            "instruction": "Remember RGB bottom-to-top for future block stacks.",
                            "preference": {
                                "statement": "Use RGB bottom-to-top for block stacks.",
                                "scope": "contextual",
                                "applicability": {"task": "block stacking"},
                                "structured_value": {"bottom": "red", "middle": "green", "top": "blue"},
                            },
                        },
                        "source_episode_ids": ["episode-prior", current_id],
                        "confidence": 0.97,
                        "reason": "Two independent equivalent choices.",
                    },
                }

            def summarize(payload: dict[str, Any]) -> dict[str, Any]:
                conversation = payload["conversation"]
                return {
                    "episode": {
                        key: conversation[key]
                        for key in (
                            "user_request",
                            "resolved_task",
                            "actions",
                            "execution",
                            "validation",
                            "user_choices",
                            "user_visible_result",
                            "terminal_outcome",
                            "assurance",
                            "memory_events",
                        )
                    }
                }

            memory_model = ScriptedJsonModel(
                {
                    "retrieve_history": retrieve_history,
                    "propose_preference_question": propose,
                    "summarize_history_episode": summarize,
                    "compact_history": lambda payload: {
                        "summary": "Two RGB stacking episodes are recorded as history.",
                        "source_episode_ids": [item["episode_id"] for item in payload["episodes"]],
                    },
                }
            )
            memory = MemoryAgent(
                config.memory,
                history,
                preferences,
                model=memory_model,
                compact_after_write=False,
            )
            hri_model = ScriptedJsonModel(
                {
                    "resolve_hri_turn": [
                        {
                            "mode": "EXECUTE",
                            "user_message": "I will use RGB from bottom to top.",
                            "task_contract": {
                                "confirmed_intent": "Stack red, green, and blue from bottom to top.",
                                "parameters": {"order": "RGB"},
                            },
                            "memory_action": {"action": "NONE"},
                        },
                        {
                            "mode": "REPORT",
                            "user_message": "No problem; I will leave it unsaved.",
                            "report": {"outcome": "DEFERRED", "next_action": "NONE"},
                            "memory_action": {"action": "DEFER"},
                        },
                    ]
                }
            )
            planner, validator, executor = ready_components()
            hri = HRIOrchestrator(
                config,
                hri_model=hri_model,
                memory_agent=memory,
                planner_agent=planner,
                validator_agent=validator,
                executor=executor,
            )

            task_result = hri.handle_user_message("Stack the blocks RGB bottom-to-top")
            self.assertTrue(task_result["awaiting_user"])
            self.assertEqual(task_result["hri"]["mode"], "MEMORY_CONFIRM")
            self.assertIn("Task Complete", task_result["hri"]["user_message"])
            self.assertEqual(len(history.list_episodes("participant-a")), 1)
            self.assertEqual(
                len(hri.history_outbox.list_pending("participant-a")), 1
            )

            defer_result = hri.handle_user_message("I want to decide later")
            self.assertFalse(defer_result["awaiting_user"])
            self.assertEqual(preferences.list_preferences("participant-a"), [])
            episodes = history.list_episodes("participant-a")
            self.assertEqual(len(episodes), 2)
            self.assertEqual(hri.history_outbox.list_pending("participant-a"), [])
            current = next(item for item in episodes if item["episode_id"] != "episode-prior")
            self.assertEqual(current["memory_events"][0]["action"], "DEFER")


class ExecutionAndConsentGateTests(unittest.TestCase):
    def test_explicit_future_language_is_verified_then_can_save_and_execute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = RecordingMemoryAgent([MemoryContext()])
            planner, validator, executor = ready_components()
            model = ScriptedJsonModel(
                {
                    "resolve_hri_turn": {
                        "mode": "EXECUTE",
                        "user_message": "I will remember and use RGB.",
                        "task_contract": {
                            "confirmed_intent": "Stack red, green, and blue from bottom to top."
                        },
                        "memory_action": {
                            "action": "COMMIT",
                            "explicit_consent": True,
                            "request": {
                                "instruction": "Remember RGB for future block stacks."
                            },
                        },
                    },
                    "verify_direct_memory_consent": {
                        "entailed": True,
                        "requested_action": "UPSERT",
                        "confidence": 0.99,
                        "reason": "The user explicitly said remember for future tasks.",
                    },
                }
            )
            hri = HRIOrchestrator(
                make_config(directory),
                hri_model=model,
                memory_agent=memory,
                planner_agent=planner,
                validator_agent=validator,
                executor=executor,
            )
            result = hri.handle_user_message(
                "Remember RGB for future block stacks, and stack these blocks now."
            )
            self.assertEqual(result["task"]["outcome"], "SUCCESS")
            self.assertTrue(result["memory"]["committed"])
            self.assertEqual(memory.preference_updates[0]["consent"].authorized_action, "UPSERT")

    def test_unsafe_execution_never_reaches_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(directory)
            planner, validator, _ = ready_components()

            class UnsafeExecutor:
                def execute(self, plan: PlanResult, *, attempt: int = 1) -> ExecutionResult:
                    del plan
                    return ExecutionResult(
                        status="UNSAFE",
                        final_observation=str(ROOT / "dataset" / "v3" / "12.png"),
                        evidence={"attempt": attempt},
                        error="Safety stop.",
                    )

            hri = HRIOrchestrator(
                config,
                hri_model=ScriptedJsonModel(
                    {
                        "resolve_hri_turn": {
                            "mode": "EXECUTE",
                            "user_message": "I will perform the task.",
                            "task_contract": {
                                "confirmed_intent": "Stack red, green, and blue from bottom to top."
                            },
                            "memory_action": {"action": "NONE"},
                        }
                    }
                ),
                memory_agent=RecordingMemoryAgent([MemoryContext()]),
                planner_agent=planner,
                validator_agent=validator,
                executor=UnsafeExecutor(),
            )
            result = hri.handle_user_message("Stack the blocks")
            self.assertEqual(result["task"]["outcome"], "ABORTED_SAFETY")
            self.assertEqual(validator.calls, [])
            blocked_retry = hri.handle_user_message("Retry the task")
            self.assertTrue(blocked_retry["awaiting_user"])
            self.assertEqual(
                blocked_retry["hri"]["report"]["next_action"], "SAFETY_CLEARANCE"
            )

    def test_model_flag_alone_cannot_authorize_first_query_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            memory = RecordingMemoryAgent([MemoryContext()])
            model = ScriptedJsonModel(
                {
                    "resolve_hri_turn": {
                        "mode": "REPORT",
                        "user_message": "I saved RGB.",
                        "report": {"outcome": "MEMORY_UPDATED", "next_action": "NONE"},
                        "memory_action": {
                            "action": "COMMIT",
                            "explicit_consent": True,
                            "request": {"instruction": "Save RGB."},
                        },
                    },
                    "verify_direct_memory_consent": {
                        "entailed": False,
                        "requested_action": "UPSERT",
                        "reason": "The utterance describes only the current task.",
                    },
                }
            )
            hri = HRIOrchestrator(
                make_config(directory),
                hri_model=model,
                memory_agent=memory,
                planner_agent=ready_components()[0],
                validator_agent=ready_components()[1],
                executor=ready_components()[2],
            )
            with self.assertRaisesRegex(HRIContractError, "does not clearly authorize"):
                hri.handle_user_message("Stack the blocks")
            self.assertEqual(memory.preference_updates, [])
            self.assertFalse(hri.command_active)


if __name__ == "__main__":
    unittest.main()
