from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from agents.configs import PrefMemConfig
from agents.contracts import GoalCondition, PlanResult, ValidationResult, ValidationSpec
from agents.hri import HRIContractError, HRIOrchestrator
from agents.memory import MemoryAgent
from agents.planner import PlannerAgent, PlannerAgentError
from agents.validator import ValidatorAgent
from memory.models import MemoryContext
from memory.repositories import HistoryRepository, PreferenceRepository
from tests.fakes import (
    FixedPlanner,
    FixedValidator,
    RecordingExecutor,
    RecordingMemoryAgent,
    ScriptedJsonModel,
)


ROOT = Path(__file__).resolve().parents[1]


def config_for(directory: str, *, max_replans: int = 0) -> PrefMemConfig:
    config = PrefMemConfig(
        workspace_root=str(ROOT),
        dataset_path=str(ROOT / "dataset" / "v3"),
        history_store_path=str(Path(directory) / "history.json"),
        preference_store_path=str(Path(directory) / "preferences.json"),
        user_id="participant-a",
        max_replans=max_replans,
        compact_preferences_after_write=False,
    )
    config.hri.model = "offline-only"
    config.memory.model = "offline-only"
    config.planner.model = "offline-only"
    config.validator.model = "offline-only"
    return config


def successful_task_components(final_frame: str) -> tuple[FixedPlanner, FixedValidator, RecordingExecutor]:
    spec = ValidationSpec(
        spec_id="spec-rgb",
        confirmed_intent="Stack red, green, and blue from bottom to top.",
        goal_conditions=(
            GoalCondition("goal-order", "Red is below green and green is below blue."),
            GoalCondition("goal-stack", "The three blocks form one stable stack."),
        ),
    )
    plan = PlanResult(
        status="READY",
        subtasks=[{"task_instruction": "Place green on red, then blue on green."}],
        validation_spec=spec,
        preconditions=[],
        confidence=0.99,
    )
    validation = ValidationResult(
        outcome="SUCCESS",
        task_complete=True,
        goal_checks=[
            {"goal_id": "goal-order", "satisfied": True},
            {"goal_id": "goal-stack", "satisfied": True},
        ],
        discrepancies=[],
        confidence=0.99,
        recoverability="NONE",
        user_message="Task Complete.",
    )
    return FixedPlanner(plan), FixedValidator(validation), RecordingExecutor(final_frame)


class PlannerStructuredGoalContractTests(unittest.TestCase):
    @staticmethod
    def _planner(
        directory: str,
        response: dict[str, Any],
    ) -> PlannerAgent:
        config = config_for(directory)
        return PlannerAgent(
            config.planner,
            vision=config.vision,
            model=ScriptedJsonModel({"plan_task": response}),
        )

    def test_ready_plan_rejects_description_only_goal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            planner = self._planner(
                directory,
                {
                    "planning_status": "READY",
                    "planner_confidence": 0.9,
                    "preconditions": [],
                    "subtasks": [{"task_instruction": "Stack the blocks."}],
                    "validation_spec": {
                        "confirmed_intent": "Stack RGB bottom-to-top.",
                        "goal_conditions": [
                            {
                                "id": "goal-prose-only",
                                "description": "The RGB stack is complete.",
                            }
                        ],
                    },
                },
            )

            with self.assertRaisesRegex(
                PlannerAgentError,
                "structured predicate",
            ):
                planner.plan(
                    "Stack RGB bottom-to-top.",
                    str(ROOT / "dataset" / "v3" / "1.png"),
                )

    def test_ready_plan_rejects_predicate_without_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            planner = self._planner(
                directory,
                {
                    "planning_status": "READY",
                    "planner_confidence": 0.9,
                    "preconditions": [],
                    "subtasks": [{"task_instruction": "Stack the blocks."}],
                    "validation_spec": {
                        "confirmed_intent": "Stack RGB bottom-to-top.",
                        "goal_conditions": [
                            {
                                "id": "goal-empty-arguments",
                                "description": "The stack is stable.",
                                "predicate": "STABLE_STACK",
                                "arguments": [],
                            }
                        ],
                    },
                },
            )

            with self.assertRaisesRegex(
                PlannerAgentError,
                "at least one semantic predicate argument",
            ):
                planner.plan(
                    "Stack RGB bottom-to-top.",
                    str(ROOT / "dataset" / "v3" / "1.png"),
                )

    def test_ready_plan_accepts_semantic_goals_and_safety_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            planner = self._planner(
                directory,
                {
                    "planning_status": "READY",
                    "planner_confidence": 0.9,
                    "preconditions": [],
                    "subtasks": [{"task_instruction": "Stack the blocks."}],
                    "validation_spec": {
                        "spec_id": "spec-structured",
                        "confirmed_intent": "Stack RGB bottom-to-top.",
                        "goal_conditions": [
                            {
                                "id": "goal-stack",
                                "description": "The blocks form one RGB stack.",
                                "predicate": "STABLE_STACK",
                                "arguments": [
                                    "red block",
                                    "green block",
                                    "blue block",
                                ],
                            },
                            {
                                "id": "goal-safety",
                                "description": "Execution remained safe.",
                                "observable": False,
                                "required": False,
                                "predicate": "SAFE_EXECUTION",
                                "arguments": ["robot"],
                                "evidence_modalities": [
                                    "execution_evidence"
                                ],
                            },
                        ],
                    },
                },
            )

            plan = planner.plan(
                "Stack RGB bottom-to-top.",
                str(ROOT / "dataset" / "v3" / "1.png"),
            )

            self.assertEqual(plan.status, "READY")
            self.assertIsNotNone(plan.validation_spec)
            assert plan.validation_spec is not None
            safety = plan.validation_spec.goal_conditions[1]
            self.assertEqual(safety.predicate, "SAFE_EXECUTION")
            self.assertEqual(safety.arguments, ("robot",))
            self.assertFalse(safety.required)
            self.assertFalse(safety.observable)
            self.assertEqual(
                safety.evidence_modalities,
                ("execution_evidence",),
            )
            self.assertIn("SUPPORTED_BY", planner.system_prompt)
            self.assertIn("INSIDE_SORT_ZONE", planner.system_prompt)
            self.assertIn(
                "ALL_PLACE_SETTING_ITEMS_PLACED",
                planner.system_prompt,
            )
            self.assertIn(
                "hidden simulator object IDs",
                planner.system_prompt,
            )

    def test_non_ready_plan_does_not_require_validation_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            planner = self._planner(
                directory,
                {
                    "planning_status": "BLOCKED",
                    "planner_confidence": 0.9,
                    "preconditions": [],
                    "subtasks": [],
                    "failure": {
                        "stage": "PLANNING",
                        "code": "OBJECT_MISSING",
                        "recoverability": "USER_ASSIST",
                        "user_message": "A required object is missing.",
                    },
                },
            )

            plan = planner.plan(
                "Stack RGB bottom-to-top.",
                str(ROOT / "dataset" / "v3" / "1.png"),
            )

            self.assertEqual(plan.status, "BLOCKED")
            self.assertIsNone(plan.validation_spec)


class HRIHistoryIntegrationTests(unittest.TestCase):
    def test_first_task_is_history_only_and_next_task_receives_semantic_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = config_for(directory)
            history = HistoryRepository(config.history_store_path)
            preferences = PreferenceRepository(config.preference_store_path)

            def summarize(payload: dict[str, Any]) -> dict[str, Any]:
                conversation = payload["conversation"]
                return {
                    "episode": {
                        "summary": conversation["user_visible_result"],
                        "user_request": conversation["user_request"],
                        "resolved_task": conversation["resolved_task"],
                        "actions": conversation["actions"],
                        "execution": conversation["execution"],
                        "validation": conversation["validation"],
                        "user_choices": conversation["user_choices"],
                        "user_visible_result": conversation["user_visible_result"],
                        "memory_events": conversation["memory_events"],
                    }
                }

            def compact_history(payload: dict[str, Any]) -> dict[str, Any]:
                return {
                    "summary": "The participant previously requested an RGB block stack.",
                    "source_episode_ids": [item["episode_id"] for item in payload["episodes"]],
                }

            def retrieve_history(payload: dict[str, Any]) -> dict[str, Any]:
                self.assertEqual(payload["query"]["request"], "Stack the blocks again")
                return {
                    "matches": [
                        {
                            "episode_id": payload["episodes"][0]["episode_id"],
                            "confidence": 0.97,
                            "reason": "This repeats the prior block-stacking task.",
                        }
                    ]
                }

            memory_model = ScriptedJsonModel(
                {
                    "summarize_history_episode": summarize,
                    "compact_history": compact_history,
                    "retrieve_history": retrieve_history,
                }
            )
            memory_agent = MemoryAgent(
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
                            "mode": "ASK",
                            "user_message": "In what order should I stack them?",
                            "pending_question": {
                                "kind": "TASK_CLARIFICATION",
                                "payload": {"field": "stack_order"},
                            },
                            "memory_action": {"action": "NONE"},
                        },
                        {
                            "mode": "EXECUTE",
                            "user_message": "I will use red, green, then blue from bottom to top.",
                            "task_contract": {
                                "confirmed_intent": (
                                    "Stack red, green, and blue from bottom to top."
                                ),
                                "parameters": {
                                    "bottom": "red",
                                    "middle": "green",
                                    "top": "blue",
                                },
                                "preference_refs": [],
                            },
                            "memory_action": {"action": "NONE"},
                        },
                        {
                            "mode": "REPORT",
                            "user_message": "I found the related previous stacking episode.",
                            "report": {"outcome": "BLOCKED", "next_action": "NONE"},
                            "memory_action": {"action": "NONE"},
                        },
                    ]
                }
            )
            planner, validator, executor = successful_task_components(
                str(ROOT / "dataset" / "v3" / "12.png")
            )
            hri = HRIOrchestrator(
                config,
                hri_model=hri_model,
                memory_agent=memory_agent,
                planner_agent=planner,
                validator_agent=validator,
                executor=executor,
            )

            first_turn = hri.handle_user_message("Stack the blocks")
            self.assertTrue(first_turn["awaiting_user"])
            self.assertEqual(preferences.list_preferences("participant-a"), [])
            self.assertEqual(history.list_episodes("participant-a"), [])

            first_result = hri.handle_user_message("RGB from bottom to top")
            self.assertEqual(first_result["task"]["outcome"], "SUCCESS")
            self.assertEqual(len(history.list_episodes("participant-a")), 1)
            self.assertEqual(preferences.list_preferences("participant-a"), [])

            hri.handle_user_message("Stack the blocks again")
            second_command_payload = hri_model.calls_for("resolve_hri_turn")[2]["payload"]
            context = second_command_payload["memory_context"]
            self.assertIn("previously requested", context["history_summary"])
            self.assertEqual(len(context["relevant_history"]), 1)
            self.assertEqual(
                context["relevant_history"][0]["user_choices"],
                {"bottom": "red", "middle": "green", "top": "blue"},
            )


class HRIQuestionTypingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.config = config_for(self.temporary_directory.name)
        final_frame = str(ROOT / "dataset" / "v3" / "12.png")
        self.planner, self.validator, self.executor = successful_task_components(final_frame)

    def _orchestrator(
        self, responses: list[dict[str, Any]], memory: RecordingMemoryAgent
    ) -> HRIOrchestrator:
        return HRIOrchestrator(
            self.config,
            hri_model=ScriptedJsonModel({"resolve_hri_turn": responses}),
            memory_agent=memory,
            planner_agent=self.planner,
            validator_agent=self.validator,
            executor=self.executor,
        )

    def test_null_task_question_payload_is_normalized_to_empty_object(self) -> None:
        memory = RecordingMemoryAgent()
        hri = self._orchestrator(
            [
                {
                    "mode": "ASK",
                    "user_message": "In what order should I stack the blocks?",
                    "pending_question": {
                        "kind": "TASK_CLARIFICATION",
                        "payload": None,
                    },
                    "memory_action": {"action": "NONE"},
                }
            ],
            memory,
        )

        result = hri.handle_user_message("Stack the blocks")

        self.assertTrue(result["awaiting_user"])
        self.assertEqual(result["hri"]["pending_question"]["payload"], {})
        self.assertIsNotNone(hri.pending_question)
        self.assertEqual(hri.pending_question.payload, {})

    def test_ask_with_task_confirmation_kind_is_normalized_without_retry(self) -> None:
        """Regression for the display-all transcript's repeated-task question."""
        memory = RecordingMemoryAgent()
        hri_model = ScriptedJsonModel(
            {
                "resolve_hri_turn": [
                    {
                        "mode": "ASK",
                        "user_message": (
                            "Should I stack them in the same RGB order as last time?"
                        ),
                        "task_contract": None,
                        "pending_question": {
                            "kind": "TASK_CONFIRMATION",
                            "prompt_id": "confirm_stacking_order",
                            "payload": {},
                        },
                        "memory_action": {
                            "action": "NONE",
                            "explicit_consent": False,
                            "request": None,
                        },
                        "report": None,
                    }
                ]
            }
        )
        hri = HRIOrchestrator(
            self.config,
            hri_model=hri_model,
            memory_agent=memory,
            planner_agent=self.planner,
            validator_agent=self.validator,
            executor=self.executor,
        )

        result = hri.handle_user_message("Stack the blocks again")

        self.assertEqual(result["hri"]["mode"], "CONFIRM")
        self.assertTrue(result["awaiting_user"])
        self.assertIsNotNone(hri.pending_question)
        self.assertEqual(hri.pending_question.kind, "TASK_CONFIRMATION")
        self.assertEqual(hri.pending_question.prompt_id, "confirm_stacking_order")
        self.assertEqual(len(hri_model.calls_for("resolve_hri_turn")), 1)
        self.assertEqual(memory.preference_updates, [])
        self.assertEqual(self.executor.calls, [])

    def test_confirm_with_task_clarification_kind_normalizes_to_ask(self) -> None:
        memory = RecordingMemoryAgent()
        hri = self._orchestrator(
            [
                {
                    "mode": "CONFIRM",
                    "user_message": "In what order should I stack the blocks?",
                    "pending_question": {
                        "kind": "TASK_CLARIFICATION",
                        "prompt_id": "clarify_stacking_order",
                        "payload": {},
                    },
                    "memory_action": {"action": "NONE"},
                }
            ],
            memory,
        )

        result = hri.handle_user_message("Stack the blocks")

        self.assertEqual(result["hri"]["mode"], "ASK")
        self.assertTrue(result["awaiting_user"])
        self.assertIsNotNone(hri.pending_question)
        self.assertEqual(hri.pending_question.kind, "TASK_CLARIFICATION")
        self.assertEqual(memory.preference_updates, [])

    def test_task_mode_cannot_normalize_or_synthesize_memory_consent(self) -> None:
        invalid = {
            "mode": "ASK",
            "user_message": "Should I remember RGB for future stacking tasks?",
            "pending_question": {
                "kind": "MEMORY_CONSENT",
                "prompt_id": "save-rgb",
                "payload": {
                    "preference_request": {
                        "instruction": "Save RGB as the future stacking default."
                    }
                },
            },
            "memory_action": {"action": "NONE"},
        }
        memory = RecordingMemoryAgent()
        hri = self._orchestrator([invalid, invalid], memory)

        with self.assertRaisesRegex(
            HRIContractError, "failed its output contract twice"
        ):
            hri.handle_user_message("Stack the blocks")

        self.assertFalse(hri.command_active)
        self.assertIsNone(hri.pending_question)
        self.assertEqual(memory.preference_updates, [])
        self.assertEqual(self.executor.calls, [])

    def test_post_task_memory_commit_with_null_report_commits_once_and_finishes(self) -> None:
        memory = RecordingMemoryAgent()
        hri_model = ScriptedJsonModel(
            {
                "resolve_hri_turn": [
                    {
                        "mode": "MEMORY_CONFIRM",
                        "user_message": "Should I remember RGB for future stacking tasks?",
                        "task_contract": None,
                        "pending_question": {
                            "kind": "MEMORY_CONSENT",
                            "prompt_id": "remember-rgb-after-task",
                            "payload": {
                                "preference_request": {
                                    "instruction": "Save RGB as the future stacking default.",
                                    "preference": {
                                        "statement": "Use RGB bottom-to-top for block stacks.",
                                        "scope": "block stacking",
                                    },
                                },
                                "requested_action": "UPSERT",
                            },
                        },
                        "memory_action": {"action": "NONE"},
                        "report": None,
                    },
                    {
                        "mode": "REPORT",
                        "user_message": "I've saved that stacking preference.",
                        "task_contract": None,
                        "pending_question": None,
                        "memory_action": {
                            "action": "COMMIT",
                            "explicit_consent": False,
                            "request": {
                                "instruction": "Save this exact future preference and scope."
                            },
                        },
                        "report": None,
                    },
                ]
            }
        )
        hri = HRIOrchestrator(
            self.config,
            hri_model=hri_model,
            memory_agent=memory,
            planner_agent=self.planner,
            validator_agent=self.validator,
            executor=self.executor,
        )

        proposal = hri.handle_user_message("Consider remembering RGB after this task")
        self.assertTrue(proposal["awaiting_user"])
        self.assertEqual(hri.pending_question.kind, "MEMORY_CONSENT")

        # This state is normally installed by the immediately preceding successful
        # task before HRI emits the MEMORY_CONFIRM boundary response.
        hri._terminal_task_contract = {
            "confirmed_intent": "Stack RGB bottom-to-top.",
            "parameters": {"order": "RGB"},
        }
        hri._terminal_task_result = {
            "phase": "VALIDATION",
            "outcome": "SUCCESS",
            "proceed": False,
            "next_action": "NONE",
            "message": "Task Complete.",
            "failure": None,
            "attempts": [],
        }
        hri._terminal_hri_response = {
            "mode": "REPORT",
            "user_message": "Task Complete.",
            "pending_question": None,
            "memory_action": {"action": "NONE"},
            "report": {"outcome": "SUCCESS", "next_action": "NONE"},
        }

        result = hri.handle_user_message("Yeah sure")

        self.assertFalse(result["awaiting_user"])
        self.assertIsInstance(result["hri"]["report"], dict)
        self.assertEqual(len(memory.preference_updates), 1)
        self.assertTrue(result["memory"]["committed"])
        self.assertEqual(memory.preference_updates[0]["consent"].kind, "memory_confirmation")
        self.assertEqual(
            memory.preference_updates[0]["consent"].prompt_id,
            "remember-rgb-after-task",
        )
        self.assertEqual(len(memory.history_updates), 1)
        self.assertFalse(hri.command_active)
        self.assertIsNone(hri.pending_question)
        self.assertEqual(len(hri_model.calls_for("resolve_hri_turn")), 2)

    def test_ordinary_report_with_null_report_remains_rejected(self) -> None:
        invalid = {
            "mode": "REPORT",
            "user_message": "The request cannot continue.",
            "task_contract": None,
            "pending_question": None,
            "memory_action": {"action": "NONE"},
            "report": None,
        }
        memory = RecordingMemoryAgent()
        hri_model = ScriptedJsonModel(
            {"resolve_hri_turn": [invalid, invalid]}
        )
        hri = HRIOrchestrator(
            self.config,
            hri_model=hri_model,
            memory_agent=memory,
            planner_agent=self.planner,
            validator_agent=self.validator,
            executor=self.executor,
        )

        with self.assertRaisesRegex(
            HRIContractError, "REPORT requires a structured report"
        ):
            hri.handle_user_message("Stack the blocks")

        self.assertEqual(len(hri_model.calls_for("resolve_hri_turn")), 2)
        self.assertEqual(memory.preference_updates, [])
        self.assertEqual(memory.history_updates, [])
        self.assertFalse(hri.command_active)

    def test_task_confirmation_yes_cannot_authorize_memory_commit(self) -> None:
        memory = RecordingMemoryAgent()
        hri = self._orchestrator(
            [
                {
                    "mode": "CONFIRM",
                    "user_message": "Use RGB for this task?",
                    "pending_question": {
                        "kind": "TASK_CONFIRMATION",
                        "payload": {"proposed_order": "RGB"},
                    },
                    "memory_action": {"action": "NONE"},
                },
                {
                    "mode": "EXECUTE",
                    "user_message": "I will remember and use RGB.",
                    "task_contract": {
                        "confirmed_intent": "Stack RGB bottom-to-top.",
                        "parameters": {"order": "RGB"},
                    },
                    "memory_action": {
                        "action": "COMMIT",
                        "request": {"instruction": "Remember RGB."},
                    },
                },
            ],
            memory,
        )

        hri.handle_user_message("Stack the blocks")
        with self.assertRaisesRegex(
            HRIContractError, "task-confirmation reply cannot authorize"
        ):
            hri.handle_user_message("Yes")
        self.assertEqual(memory.preference_updates, [])
        self.assertEqual(self.executor.calls, [])

    def test_dedicated_memory_confirmation_commits_with_prompt_provenance(self) -> None:
        memory = RecordingMemoryAgent()
        hri = self._orchestrator(
            [
                {
                    "mode": "MEMORY_CONFIRM",
                    "user_message": "Should I remember RGB as your default stacking order?",
                    "pending_question": {
                        "kind": "MEMORY_CONSENT",
                        "prompt_id": "prompt-save-rgb",
                        "payload": {
                            "preference_request": {
                                "instruction": "Save RGB as the default block-stacking order."
                            }
                        },
                    },
                    "memory_action": {"action": "NONE"},
                },
                {
                    "mode": "REPORT",
                    "user_message": "I saved that preference.",
                    "report": {"outcome": "MEMORY_UPDATED", "next_action": "NONE"},
                    "memory_action": {"action": "COMMIT"},
                },
            ],
            memory,
        )

        hri.handle_user_message("Please consider remembering the order")
        result = hri.handle_user_message("Yes")

        self.assertTrue(result["memory"]["committed"])
        self.assertEqual(len(memory.preference_updates), 1)
        update = memory.preference_updates[0]
        self.assertEqual(update["consent"].kind, "memory_confirmation")
        self.assertEqual(update["consent"].quote, "Yes")
        self.assertEqual(update["consent"].prompt_id, "prompt-save-rgb")
        self.assertEqual(update["request"]["requested_action"], "UPSERT")

    def test_defer_closes_memory_question_without_mutation_or_loop(self) -> None:
        memory = RecordingMemoryAgent()
        hri = self._orchestrator(
            [
                {
                    "mode": "MEMORY_CONFIRM",
                    "user_message": "Should I remember RGB for future stacks?",
                    "pending_question": {
                        "kind": "MEMORY_CONSENT",
                        "prompt_id": "prompt-defer-rgb",
                        "payload": {
                            "preference_request": {
                                "instruction": "Save RGB for future block stacks."
                            }
                        },
                    },
                    "memory_action": {"action": "NONE"},
                },
                {
                    "mode": "REPORT",
                    "user_message": "No problem; I will not save it now.",
                    "report": {"outcome": "DEFERRED", "next_action": "NONE"},
                    "memory_action": {"action": "DEFER"},
                },
            ],
            memory,
        )

        hri.handle_user_message("Maybe save my stacking order")
        result = hri.handle_user_message("I want to decide later")

        self.assertFalse(result["awaiting_user"])
        self.assertEqual(result["memory"]["action"], "DEFER")
        self.assertEqual(result["memory"]["prompt_id"], "prompt-defer-rgb")
        self.assertEqual(memory.preference_updates, [])
        self.assertIsNone(hri.pending_question)
        self.assertEqual(memory.history_updates[0]["memory_events"][0]["action"], "DEFER")


class FrozenValidationFlowTests(unittest.TestCase):
    def test_repeated_planner_failures_have_distinct_event_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = config_for(directory, max_replans=1)
            hri_model = ScriptedJsonModel(
                {
                    "resolve_hri_turn": {
                        "mode": "EXECUTE",
                        "user_message": "I will stack the blocks.",
                        "task_contract": {
                            "confirmed_intent": "Stack RGB bottom-to-top.",
                            "parameters": {"order": "RGB"},
                        },
                        "memory_action": {"action": "NONE"},
                    }
                }
            )

            class RepeatedlyFailingPlanner:
                def __init__(self) -> None:
                    self.calls = 0

                def plan(self, *args: Any, **kwargs: Any) -> PlanResult:
                    del args, kwargs
                    self.calls += 1
                    raise PlannerAgentError("same invalid planner response")

            planner = RepeatedlyFailingPlanner()
            _, validator, executor = successful_task_components(
                str(ROOT / "dataset" / "v3" / "12.png")
            )
            hri = HRIOrchestrator(
                config,
                hri_model=hri_model,
                memory_agent=RecordingMemoryAgent(),
                planner_agent=planner,
                validator_agent=validator,
                executor=executor,
            )

            result = hri.handle_user_message("Stack the blocks")

            failures = [
                attempt["failure"] for attempt in result["task"]["attempts"]
            ]
            self.assertEqual(planner.calls, 2)
            self.assertEqual([item["attempt"] for item in failures], [1, 2])
            self.assertEqual(
                {item["event"] for item in failures},
                {"planner-call"},
            )
            self.assertEqual(len({item["event_id"] for item in failures}), 2)
            self.assertEqual(
                {item["message"] for item in failures},
                {"same invalid planner response"},
            )

    def test_planner_validation_spec_reaches_executor_and_validator_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = config_for(directory)
            memory = RecordingMemoryAgent([MemoryContext()])
            hri_model = ScriptedJsonModel(
                {
                    "resolve_hri_turn": {
                        "mode": "EXECUTE",
                        "user_message": "I will execute the confirmed RGB stack.",
                        "task_contract": {
                            "confirmed_intent": (
                                "Put red at the bottom, green in the middle, and blue on top."
                            ),
                            "parameters": {
                                "bottom": "red",
                                "middle": "green",
                                "top": "blue",
                            },
                        },
                        "memory_action": {"action": "NONE"},
                    }
                }
            )
            planner_model = ScriptedJsonModel(
                {
                    "plan_task": {
                        "planning_status": "READY",
                        "planner_confidence": 0.98,
                        "preconditions": [],
                        "subtasks": [
                            {"task_instruction": "Place green on red."},
                            {"task_instruction": "Place blue on green."},
                        ],
                        "validation_spec": {
                            "spec_id": "frozen-spec-42",
                            "confirmed_intent": (
                                "Put red at the bottom, green in the middle, and blue on top."
                            ),
                            "goal_conditions": [
                                {
                                    "id": "frozen-order",
                                    "description": "Red is below green and green is below blue.",
                                    "predicate": "VERTICALLY_ALIGNED",
                                    "arguments": [
                                        "red block",
                                        "green block",
                                        "blue block",
                                    ],
                                },
                                {
                                    "id": "frozen-single-stack",
                                    "description": "All three blocks form one stack.",
                                    "predicate": "STABLE_STACK",
                                    "arguments": [
                                        "red block",
                                        "green block",
                                        "blue block",
                                    ],
                                },
                            ],
                        },
                    }
                }
            )

            def validation_response(payload: dict[str, Any]) -> dict[str, Any]:
                spec = payload["validation_spec"]
                return {
                    "spec_id": spec["spec_id"],
                    "outcome": "SUCCESS",
                    "task_complete": True,
                    "goal_checks": [
                        {"goal_id": item["id"], "satisfied": True}
                        for item in spec["goal_conditions"]
                    ],
                    "discrepancies": [],
                    "validator_confidence": 0.99,
                    "recoverability": "NONE",
                    "user_message": "Task Complete.",
                }

            validator_model = ScriptedJsonModel({"validate_task": validation_response})
            planner = PlannerAgent(config.planner, vision=config.vision, model=planner_model)
            validator = ValidatorAgent(
                config.validator, vision=config.vision, model=validator_model
            )
            executor = RecordingExecutor(str(ROOT / "dataset" / "v3" / "12.png"))
            hri = HRIOrchestrator(
                config,
                hri_model=hri_model,
                memory_agent=memory,
                planner_agent=planner,
                validator_agent=validator,
                executor=executor,
            )

            result = hri.handle_user_message("Stack the blocks RGB from bottom to top")

            self.assertEqual(result["task"]["outcome"], "SUCCESS")
            self.assertEqual(result["task"]["message"], "Task Complete.")
            executed_spec = executor.calls[0]["plan"].validation_spec
            validator_spec = validator_model.calls_for("validate_task")[0]["payload"][
                "validation_spec"
            ]
            self.assertEqual(executed_spec.spec_id, "frozen-spec-42")
            self.assertEqual(validator_spec["spec_id"], "frozen-spec-42")
            self.assertEqual(
                [goal["id"] for goal in validator_spec["goal_conditions"]],
                ["frozen-order", "frozen-single-stack"],
            )
            attempt = result["task"]["attempts"][0]
            self.assertEqual(
                attempt["plan"]["validation_spec"], validator_spec
            )

    def test_corrective_replan_cannot_replace_the_frozen_goal_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = config_for(directory, max_replans=1)
            memory = RecordingMemoryAgent([MemoryContext()])
            hri_model = ScriptedJsonModel(
                {
                    "resolve_hri_turn": {
                        "mode": "EXECUTE",
                        "user_message": "I will execute the confirmed stack.",
                        "task_contract": {
                            "confirmed_intent": "Stack RGB bottom-to-top.",
                            "parameters": {"order": "RGB"},
                        },
                        "memory_action": {"action": "NONE"},
                    }
                }
            )
            frozen_spec = ValidationSpec(
                spec_id="spec-frozen-original",
                confirmed_intent="Stack RGB bottom-to-top.",
                goal_conditions=(
                    GoalCondition("goal-original-order", "The stack is RGB bottom-to-top."),
                ),
            )
            substituted_spec = ValidationSpec(
                spec_id="spec-illegal-replacement",
                confirmed_intent="Put only two blocks together.",
                goal_conditions=(
                    GoalCondition("goal-easier", "Any two blocks touch."),
                ),
            )

            class ReplanningPlanner:
                def __init__(self) -> None:
                    self.calls = 0

                def plan(
                    self,
                    task_contract: dict[str, Any],
                    image_path: str,
                    *,
                    recovery_context: dict[str, Any] | None = None,
                ) -> PlanResult:
                    del task_contract, image_path, recovery_context
                    self.calls += 1
                    spec = frozen_spec if self.calls == 1 else substituted_spec
                    return PlanResult(
                        status="READY",
                        subtasks=[{"task_instruction": "Perform the next corrective move."}],
                        validation_spec=spec,
                        preconditions=[],
                        confidence=0.99,
                    )

            class ReplanningValidator:
                def __init__(self) -> None:
                    self.spec_ids: list[str] = []

                def validate(self, spec: ValidationSpec, execution: Any) -> ValidationResult:
                    del execution
                    self.spec_ids.append(spec.spec_id)
                    checks = [
                        {"goal_id": goal.id, "satisfied": len(self.spec_ids) > 1}
                        for goal in spec.goal_conditions
                    ]
                    return ValidationResult(
                        outcome="SUCCESS" if len(self.spec_ids) > 1 else "FAILURE",
                        task_complete=len(self.spec_ids) > 1,
                        goal_checks=checks,
                        discrepancies=[] if len(self.spec_ids) > 1 else ["Goal remains unmet."],
                        confidence=0.99,
                        recoverability="NONE" if len(self.spec_ids) > 1 else "REPLAN",
                        user_message=(
                            "Task Complete."
                            if len(self.spec_ids) > 1
                            else "A corrective move is required."
                        ),
                    )

            planner = ReplanningPlanner()
            validator = ReplanningValidator()
            executor = RecordingExecutor(str(ROOT / "dataset" / "v3" / "12.png"))
            hri = HRIOrchestrator(
                config,
                hri_model=hri_model,
                memory_agent=memory,
                planner_agent=planner,
                validator_agent=validator,
                executor=executor,
            )

            result = hri.handle_user_message("Stack the blocks")

            self.assertEqual(planner.calls, 2)
            self.assertEqual(
                validator.spec_ids,
                ["spec-frozen-original"],
                "A changed corrective schema must never reach validation.",
            )
            self.assertEqual(len(executor.calls), 1)
            self.assertEqual(result["task"]["outcome"], "UNKNOWN")
            self.assertEqual(
                result["task"]["failure"]["code"], "VALIDATION_SPEC_DRIFT"
            )


if __name__ == "__main__":
    unittest.main()
