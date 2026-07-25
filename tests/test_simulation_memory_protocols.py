from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from PIL import Image

from agents.contracts import ExecutionResult
from memory.models import MemoryContext, MemoryQuery
from simulation.benchmark.catalog import build_catalog, build_control_catalog
from simulation.benchmark.generator import _scenario_manifest, generate_benchmark
from simulation.benchmark.mujoco_render import _build_model, _render
from simulation.benchmark.protocol_runner import (
    SUPPORTED_EXPECTATIONS,
    run_memory_protocol,
)
from simulation.benchmark.protocols import (
    build_memory_protocols,
    memory_protocol_fixtures,
)
from simulation.benchmark.scorer import score_agent_result
from simulation.benchmark.validator import validate_benchmark


MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None


class _HistoryRepository:
    def __init__(
        self,
        episodes: list[dict[str, Any]] | None = None,
        *,
        summary: str = "",
    ) -> None:
        self.episodes = copy.deepcopy(episodes or [])
        self.summary = summary

    def list_episodes(
        self, user_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        episodes = [
            copy.deepcopy(item)
            for item in self.episodes
            if item.get("user_id") == user_id
        ]
        return episodes if limit is None else episodes[:limit]

    def get_summary(self, user_id: str) -> dict[str, Any]:
        owned_ids = [
            str(item["episode_id"])
            for item in self.episodes
            if item.get("user_id") == user_id and item.get("episode_id")
        ]
        return {
            "text": self.summary,
            "source_episode_ids": owned_ids,
            "updated_at": None,
        }


class _PreferenceRepository:
    def __init__(self, preferences: list[dict[str, Any]] | None = None) -> None:
        self.preferences = copy.deepcopy(preferences or [])

    def list_preferences(
        self,
        user_id: str,
        *,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        del include_inactive
        return [
            copy.deepcopy(item)
            for item in self.preferences
            if item.get("user_id") == user_id
        ]


class _OutboxRepository:
    def __init__(self, pending: list[dict[str, Any]] | None = None) -> None:
        self.pending = copy.deepcopy(pending or [])

    def list_pending(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(item)
            for item in self.pending
            if user_id is None
            or item.get("user_id") == user_id
            or item.get("source", {}).get("user_id") == user_id
        ]


class _FakeMemoryAgent:
    def __init__(
        self,
        history: _HistoryRepository,
        preferences: _PreferenceRepository,
        contexts: list[MemoryContext] | None = None,
    ) -> None:
        self.history_repository = history
        self.preference_repository = preferences
        self._contexts = [copy.deepcopy(item) for item in contexts or []]

    def get_memory_context(self, query: MemoryQuery) -> MemoryContext:
        if self._contexts:
            return copy.deepcopy(self._contexts.pop(0))
        return MemoryContext(
            relevant_history=self.history_repository.list_episodes(
                query.user_id
            ),
            relevant_preferences=self.preference_repository.list_preferences(
                query.user_id
            ),
        )


TurnHandler = Callable[["_FakeOrchestrator"], dict[str, Any]]


class _FakeOrchestrator:
    def __init__(
        self,
        handlers: list[TurnHandler] | None = None,
        *,
        user_id: str = "participant-a",
        histories: list[dict[str, Any]] | None = None,
        preferences: list[dict[str, Any]] | None = None,
        pending_outbox: list[dict[str, Any]] | None = None,
        contexts: list[MemoryContext] | None = None,
    ) -> None:
        self.config = SimpleNamespace(user_id=user_id)
        self.history = _HistoryRepository(histories)
        self.preferences = _PreferenceRepository(preferences)
        self.memory_agent = _FakeMemoryAgent(
            self.history,
            self.preferences,
            contexts,
        )
        self.history_outbox = _OutboxRepository(pending_outbox)
        self.pending_question: dict[str, Any] | None = None
        self._handlers = list(handlers or [])
        self.messages: list[str] = []
        self.dataset_switches: list[str] = []

    def switch_dataset(self, path: str) -> None:
        self.dataset_switches.append(path)

    def handle_user_message(self, message: str) -> dict[str, Any]:
        self.messages.append(message)
        if not self._handlers:
            raise AssertionError("No fake HRI turn remains.")
        self.memory_agent.get_memory_context(
            MemoryQuery(
                user_id=self.config.user_id,
                request=message,
                scene={},
            )
        )
        return copy.deepcopy(self._handlers.pop(0)(self))


def _attempt_result(
    *,
    mode: str,
    memory_refs: list[str],
    history_refs: list[str],
) -> dict[str, Any]:
    return {
        "hri": {
            "mode": mode,
            "trace": {
                "memory_refs": memory_refs,
                "history_refs": history_refs,
            },
        },
        "task": {
            "outcome": "SUCCESS",
            "attempts": [
                {
                    "plan": {"planning_status": "READY"},
                    "execution": {
                        "status": "OBSERVED_RECORDED_ATTEMPT",
                        "subtask_results": [{"status": "COMPLETED"}],
                    },
                    "validation": {
                        "outcome": "SUCCESS",
                        "task_complete": True,
                    },
                }
            ],
        },
    }


def _scorable_success_result(
    manifest: Mapping[str, Any],
    *,
    mode: str,
    memory_refs: list[str],
    history_refs: list[str],
) -> dict[str, Any]:
    target_id = str(manifest["target"]["target_id"])
    intent = str(manifest["target"]["instruction"])
    order = (
        ["red", "green", "blue"]
        if target_id == "rgb_bottom_to_top"
        else ["blue", "green", "red"]
    )
    validation_spec = {
        "spec_id": f"spec-{target_id}",
        "confirmed_intent": intent,
        "goal_conditions": [
            {
                "id": f"goal-{index}",
                "description": "Structured protocol task goal.",
                "predicate": predicate,
                "arguments": [],
            }
            for index, predicate in enumerate(
                manifest["target"]["goal_predicates"],
                start=1,
            )
        ],
    }
    return {
        "hri": {
            "mode": mode,
            "report": (
                {"outcome": "SUCCESS"} if mode == "REPORT" else None
            ),
            "trace": {
                "memory_refs": memory_refs,
                "history_refs": history_refs,
            },
        },
        "resolved_task": {
            "confirmed_intent": intent,
            "parameters": {
                "target_id": target_id,
                "order_bottom_to_top": order,
            },
            "preference_refs": memory_refs,
        },
        "task": {
            "outcome": "SUCCESS",
            "next_action": "NONE",
            "attempts": [
                {
                    "plan": {
                        "planning_status": "READY",
                        "validation_spec": validation_spec,
                    },
                    "execution": {
                        "status": "OBSERVED_RECORDED_ATTEMPT",
                        "subtask_results": [{"status": "COMPLETED"}],
                    },
                    "validation": {
                        "outcome": "SUCCESS",
                        "task_complete": True,
                    },
                    "validation_assurance": {"next_action": "NONE"},
                }
            ],
        },
    }


class MemoryProtocolDefinitionTests(unittest.TestCase):
    def _protocols(self) -> list[dict[str, Any]]:
        scenarios = [
            *build_catalog(families=["block_stack"], seeds=[5]),
            *build_control_catalog(families=["block_stack"], seeds=[5]),
        ]
        return build_memory_protocols(scenarios)

    def test_all_emitted_expectations_are_supported_and_steps_are_one_turn(
        self,
    ) -> None:
        protocols = self._protocols()
        self.assertTrue(protocols)
        emitted_expectations: set[str] = set()

        for protocol in protocols:
            for index, step in enumerate(protocol["steps"], start=1):
                with self.subTest(
                    protocol=protocol["protocol_id"],
                    step=index,
                ):
                    query = str(step.get("query", "")).strip()
                    reply = str(step.get("reply", "")).strip()
                    self.assertEqual(
                        int(bool(query)) + int(bool(reply)),
                        1,
                        "each protocol step must represent exactly one user turn",
                    )
                    emitted_expectations.update(step["expected"])

        self.assertTrue(emitted_expectations)
        self.assertLessEqual(emitted_expectations, SUPPORTED_EXPECTATIONS)

    def test_fixtures_are_canonical_independent_and_consent_adaptable(
        self,
    ) -> None:
        fixtures = memory_protocol_fixtures()
        approved = fixtures["approved-rgb-stack-preference"]
        isolated = fixtures["participant-a-approved-rgb-only"]

        self.assertEqual(approved, isolated)
        self.assertIsNot(approved, isolated)
        self.assertIsNot(
            approved["preference_request"],
            isolated["preference_request"],
        )
        self.assertEqual(
            approved["consent"]["authorized_action"],
            approved["preference_request"]["requested_action"],
        )
        self.assertEqual(
            approved["consent"]["proposal"],
            approved["preference_request"],
        )
        self.assertIsNot(
            approved["consent"]["proposal"],
            approved["preference_request"],
        )

        canonical = copy.deepcopy(fixtures)
        received: list[dict[str, Any]] = []

        def adapt_consent(
            fixture_id: str,
            fixture: Mapping[str, Any],
            orchestrator: Any,
        ) -> None:
            del fixture_id, orchestrator
            adapted = dict(fixture)
            adapted["consent"] = copy.deepcopy(adapted["consent"])
            adapted["consent"]["turn_id"] = "experiment-specific-turn"
            received.append(adapted)

        orchestrator = _FakeOrchestrator(
            [
                lambda _orchestrator: {
                    "hri": {"mode": "ASK", "trace": {}},
                    "task": {"attempts": []},
                }
            ]
        )
        report = run_memory_protocol(
            {
                "protocol_id": "fixture-copy-boundary",
                "user_id": "participant-a",
                "requires_fresh_memory": True,
                "initial_memory_fixture": "approved-rgb-stack-preference",
                "steps": [
                    {
                        "query": "Stack the blocks.",
                        "expected": {"clarification_required": True},
                    }
                ],
            },
            orchestrator,
            resolve_scenario=lambda _selector: Path("unused"),
            apply_fixture=adapt_consent,
            fixtures=fixtures,
        )

        self.assertTrue(report.passed, report.steps[0].checks)
        self.assertEqual(
            received[0]["consent"]["turn_id"],
            "experiment-specific-turn",
        )
        self.assertEqual(
            fixtures,
            canonical,
            "consent adaptation must not mutate the canonical fixture bundle",
        )


class MemoryProtocolRunnerTests(unittest.TestCase):
    @staticmethod
    def _protocol(step: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocol_id": "runner-guard",
            "user_id": "participant-a",
            "requires_fresh_memory": True,
            "steps": [step],
        }

    def test_runner_rejects_query_plus_reply_and_unknown_expectations(
        self,
    ) -> None:
        cases = {
            "query-and-reply": {
                "step": {
                    "query": "Stack the blocks.",
                    "reply": "RGB.",
                    "expected": {"clarification_required": True},
                },
                "message": "exactly one query or reply",
            },
            "unknown-expectation": {
                "step": {
                    "query": "Stack the blocks.",
                    "expected": {"unobservable_future_metric": True},
                },
                "message": "unsupported expectations",
            },
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                orchestrator = _FakeOrchestrator()
                with self.assertRaisesRegex(ValueError, case["message"]):
                    run_memory_protocol(
                        self._protocol(case["step"]),
                        orchestrator,
                        resolve_scenario=lambda _selector: Path("unused"),
                    )
                self.assertEqual(orchestrator.messages, [])

    def test_runner_rejects_each_non_fresh_memory_store(self) -> None:
        occupied_stores = {
            "history": {
                "histories": [
                    {
                        "episode_id": "episode-existing",
                        "user_id": "participant-a",
                    }
                ]
            },
            "preference": {
                "preferences": [
                    {
                        "id": "preference-existing",
                        "user_id": "participant-a",
                    }
                ]
            },
            "outbox": {
                "pending_outbox": [
                    {
                        "user_id": "participant-a",
                        "source": {
                            "episode_id": "episode-pending",
                            "user_id": "participant-a",
                        },
                    }
                ]
            },
        }
        protocol = self._protocol(
            {
                "query": "Stack the blocks.",
                "expected": {"clarification_required": True},
            }
        )

        for store, arguments in occupied_stores.items():
            with self.subTest(store=store):
                orchestrator = _FakeOrchestrator(**arguments)
                with self.assertRaisesRegex(
                    ValueError,
                    "requires a fresh per-user history, preference, and outbox",
                ):
                    run_memory_protocol(
                        protocol,
                        orchestrator,
                        resolve_scenario=lambda _selector: Path("unused"),
                    )
                self.assertEqual(orchestrator.messages, [])

    def test_runner_requires_foreign_fixture_owner_namespace_to_be_fresh(
        self,
    ) -> None:
        orchestrator = _FakeOrchestrator(
            user_id="participant-b",
            preferences=[
                {
                    "id": "participant-a-existing",
                    "user_id": "participant-a",
                }
            ],
        )
        protocol = {
            "protocol_id": "foreign-fixture-freshness",
            "user_id": "participant-b",
            "requires_fresh_memory": True,
            "initial_memory_fixture": "participant-a-approved-rgb-only",
            "steps": [
                {
                    "query": "Stack the blocks.",
                    "expected": {"clarification_required": True},
                }
            ],
        }
        fixture_called = False

        def apply_fixture(*_args: Any) -> None:
            nonlocal fixture_called
            fixture_called = True

        with self.assertRaisesRegex(
            ValueError,
            "namespace for user 'participant-a'",
        ):
            run_memory_protocol(
                protocol,
                orchestrator,
                resolve_scenario=lambda _selector: Path("unused"),
                apply_fixture=apply_fixture,
                fixtures=memory_protocol_fixtures(),
            )

        self.assertFalse(fixture_called)
        self.assertEqual(orchestrator.messages, [])

    def test_hallucinated_hri_trace_refs_do_not_count_as_retrieval(self) -> None:
        orchestrator = _FakeOrchestrator(
            [
                lambda _orchestrator: {
                    "hri": {
                        "mode": "ASK",
                        "trace": {
                            "history_refs": ["hallucinated-history"],
                            "memory_refs": ["hallucinated-preference"],
                        },
                    },
                    "task": {"attempts": []},
                }
            ]
        )
        protocol = self._protocol(
            {
                "query": "Stack the blocks as before.",
                "expected": {
                    "history_retrieval_required": True,
                    "preference_retrieval_required": True,
                },
            }
        )

        report = run_memory_protocol(
            protocol,
            orchestrator,
            resolve_scenario=lambda _selector: Path("unused"),
        )

        self.assertFalse(report.passed)
        checks = {
            check["name"]: check for check in report.steps[0].checks
        }
        self.assertEqual(
            checks["history_retrieval_required"]["actual"],
            False,
        )
        self.assertFalse(
            checks["history_retrieval_required"]["passed"]
        )
        self.assertEqual(
            checks["preference_retrieval_required"]["actual"],
            False,
        )
        self.assertFalse(
            checks["preference_retrieval_required"]["passed"]
        )
        self.assertEqual(
            report.steps[0].delivered_memory_context["history_ids"],
            [],
        )
        self.assertEqual(
            report.steps[0].delivered_memory_context["preference_ids"],
            [],
        )

    def test_delivered_foreign_preference_is_detected_without_hri_ref(
        self,
    ) -> None:
        foreign = {
            "id": "foreign-preference",
            "user_id": "participant-b",
            "structured_value": {
                "order_bottom_to_top": ["red", "green", "blue"]
            },
        }
        orchestrator = _FakeOrchestrator(
            [
                lambda _orchestrator: {
                    "hri": {
                        "mode": "ASK",
                        "trace": {
                            "history_refs": [],
                            "memory_refs": [],
                        },
                    },
                    "task": {"attempts": []},
                }
            ],
            preferences=[foreign],
            contexts=[
                MemoryContext(
                    relevant_preferences=[copy.deepcopy(foreign)]
                )
            ],
        )
        protocol = self._protocol(
            {
                "query": "Stack the blocks.",
                "expected": {
                    "clarification_required": True,
                    "history_retrieval_required": False,
                    "preference_retrieval_required": True,
                },
            }
        )

        report = run_memory_protocol(
            protocol,
            orchestrator,
            resolve_scenario=lambda _selector: Path("unused"),
        )

        self.assertFalse(report.passed)
        checks = {
            check["name"]: check for check in report.steps[0].checks
        }
        self.assertEqual(checks["foreign_preference_refs"]["expected"], 0)
        self.assertEqual(checks["foreign_preference_refs"]["actual"], 1)
        self.assertFalse(checks["foreign_preference_refs"]["passed"])
        self.assertEqual(
            checks["preference_retrieval_required"]["actual"],
            False,
        )
        self.assertFalse(
            checks["preference_retrieval_required"]["passed"]
        )
        delivered = report.steps[0].delivered_memory_context
        self.assertEqual(
            delivered["preference_ids"],
            ["foreign-preference"],
        )
        self.assertEqual(
            delivered["foreign_preference_ids"],
            ["foreign-preference"],
        )
        self.assertFalse(delivered["preference_ownership_verified"])
        self.assertEqual(
            delivered["preference_content_mismatch_ids"],
            ["foreign-preference"],
        )

    def test_idless_delivered_record_makes_ownership_unverifiable(
        self,
    ) -> None:
        orchestrator = _FakeOrchestrator(
            [
                lambda _orchestrator: {
                    "hri": {"mode": "ASK", "trace": {}},
                    "task": {"attempts": []},
                }
            ],
            contexts=[
                MemoryContext(
                    relevant_preferences=[
                        {
                            "user_id": "participant-b",
                            "statement": "Foreign content without an ID.",
                        }
                    ]
                )
            ],
        )

        report = run_memory_protocol(
            self._protocol(
                {
                    "query": "Stack the blocks.",
                    "expected": {"clarification_required": True},
                }
            ),
            orchestrator,
            resolve_scenario=lambda _selector: Path("unused"),
        )

        self.assertFalse(report.passed)
        delivered = report.steps[0].delivered_memory_context
        self.assertEqual(delivered["malformed_preference_records"], 1)
        self.assertFalse(delivered["preference_ownership_verified"])
        ownership = next(
            check
            for check in report.steps[0].checks
            if check["name"] == "memory_context_ownership_verified"
        )
        self.assertFalse(ownership["passed"])

    def test_delivered_history_summary_must_match_user_repository(
        self,
    ) -> None:
        orchestrator = _FakeOrchestrator(
            [
                lambda _orchestrator: {
                    "hri": {"mode": "ASK", "trace": {}},
                    "task": {"attempts": []},
                }
            ],
            contexts=[MemoryContext(history_summary="Another user's summary.")],
        )

        report = run_memory_protocol(
            self._protocol(
                {
                    "query": "Stack the blocks.",
                    "expected": {"clarification_required": True},
                }
            ),
            orchestrator,
            resolve_scenario=lambda _selector: Path("unused"),
        )

        self.assertFalse(report.passed)
        delivered = report.steps[0].delivered_memory_context
        self.assertFalse(delivered["history_summary_matches_repository"])
        self.assertFalse(delivered["history_ownership_verified"])

    def test_runner_executes_distinct_turns_and_passes_semantic_checks(
        self,
    ) -> None:
        scenarios = {
            target_id: next(
                scenario
                for scenario in build_catalog(
                    families=["block_stack"],
                    seeds=[31],
                )
                if scenario.target_id == target_id
                and scenario.outcome == "success"
            )
            for target_id in (
                "rgb_bottom_to_top",
                "bgr_bottom_to_top",
            )
        }
        manifests = {
            target_id: _scenario_manifest(
                scenario,
                backend="synthetic",
            )
            for target_id, scenario in scenarios.items()
        }

        def semantic_result(
            *,
            target_id: str,
            mode: str,
            memory_refs: list[str],
            history_refs: list[str],
        ) -> dict[str, Any]:
            result = _attempt_result(
                mode=mode,
                memory_refs=memory_refs,
                history_refs=history_refs,
            )
            manifest = manifests[target_id]
            confirmed_intent = str(manifest["target"]["instruction"])
            order = (
                ["red", "green", "blue"]
                if target_id == "rgb_bottom_to_top"
                else ["blue", "green", "red"]
            )
            result["resolved_task"] = {
                "confirmed_intent": confirmed_intent,
                "parameters": {
                    "target_id": target_id,
                    "order_bottom_to_top": order,
                },
            }
            result["task"]["next_action"] = "NONE"
            attempt = result["task"]["attempts"][0]
            attempt["plan"]["validation_spec"] = {
                "confirmed_intent": confirmed_intent,
                "goal_conditions": [
                    {
                        "id": f"goal-{index}",
                        "predicate": predicate,
                    }
                    for index, predicate in enumerate(
                        manifest["target"]["goal_predicates"],
                        start=1,
                    )
                ],
            }
            attempt["validation_assurance"] = {"next_action": "NONE"}
            return result

        def first_turn(orchestrator: _FakeOrchestrator) -> dict[str, Any]:
            orchestrator.preferences.preferences.append(
                {
                    "id": "pref-rgb",
                    "user_id": "participant-a",
                    "structured_value": {
                        "order_bottom_to_top": ["red", "green", "blue"]
                    },
                }
            )
            orchestrator.history.episodes.append(
                {
                    "episode_id": "episode-rgb",
                    "user_id": "participant-a",
                    "resolved_task": {
                        "parameters": {
                            "order_bottom_to_top": [
                                "red",
                                "green",
                                "blue",
                            ]
                        },
                        "preference_refs": ["pref-rgb"],
                    },
                }
            )
            orchestrator.pending_question = {"kind": "MEMORY_CONSENT"}
            return semantic_result(
                target_id="rgb_bottom_to_top",
                mode="MEMORY_CONFIRM",
                memory_refs=["pref-rgb"],
                history_refs=["episode-prior"],
            )

        def second_turn(orchestrator: _FakeOrchestrator) -> dict[str, Any]:
            orchestrator.history.episodes.append(
                {
                    "episode_id": "episode-bgr",
                    "user_id": "participant-a",
                    "resolved_task": {
                        "parameters": {
                            "order_bottom_to_top": [
                                "blue",
                                "green",
                                "red",
                            ]
                        },
                        "preference_refs": ["pref-rgb"],
                    },
                }
            )
            orchestrator.pending_question = None
            return semantic_result(
                target_id="bgr_bottom_to_top",
                mode="REPORT",
                memory_refs=["pref-rgb"],
                history_refs=["episode-rgb"],
            )

        orchestrator = _FakeOrchestrator([first_turn, second_turn])
        protocol = {
            "protocol_id": "observable-semantics",
            "user_id": "participant-a",
            "requires_fresh_memory": True,
            "steps": [
                {
                    "query": "Stack the blocks as before.",
                    "scenario_selector": {
                        "family": "block_stack",
                        "target_id": "rgb_bottom_to_top",
                        "outcome": "success",
                    },
                    "expected": {
                        "active_equivalent_preferences": 1,
                        "active_rgb_preference_preserved": False,
                        "clarification_required": False,
                        "dispatches": 1,
                        "foreign_preference_refs": 0,
                        "history_delta": 1,
                        "history_retrieval_required": False,
                        "min_dispatches": 1,
                        "pending_question_cleared": False,
                        "planner_status": "READY",
                        "post_task_preference_question": True,
                        "preference_delta": 1,
                        "preference_retrieval_required": False,
                        "task_semantic_score": True,
                        "uses_one_off_override": False,
                        "validator_outcome": "SUCCESS",
                    },
                },
                {
                    "reply": "This time only, use BGR.",
                    "scenario_selector": {
                        "family": "block_stack",
                        "target_id": "bgr_bottom_to_top",
                        "outcome": "success",
                    },
                    "expected": {
                        "active_equivalent_preferences": 1,
                        "active_rgb_preference_preserved": True,
                        "clarification_required": False,
                        "dispatches": 1,
                        "foreign_preference_refs": 0,
                        "history_delta": 1,
                        "history_retrieval_required": True,
                        "min_dispatches": 1,
                        "pending_question_cleared": True,
                        "planner_status": "READY",
                        "post_task_preference_question": False,
                        "preference_delta": 0,
                        "preference_retrieval_required": True,
                        "task_semantic_score": True,
                        "uses_one_off_override": True,
                        "validator_outcome": "SUCCESS",
                    },
                },
            ],
        }

        report = run_memory_protocol(
            protocol,
            orchestrator,
            resolve_scenario=lambda selector: Path(
                "episodes"
            )
            / str(selector["target_id"]),
            load_manifest=lambda path: manifests[Path(path).name],
        )

        self.assertTrue(report.passed, report.steps)
        self.assertEqual(
            orchestrator.messages,
            ["Stack the blocks as before.", "This time only, use BGR."],
        )
        self.assertEqual(
            orchestrator.dataset_switches,
            [
                str(Path("episodes") / "rgb_bottom_to_top"),
                str(Path("episodes") / "bgr_bottom_to_top"),
            ],
        )
        self.assertTrue(all(step.passed for step in report.steps))
        self.assertTrue(all(step.task_score_required for step in report.steps))
        self.assertTrue(
            all(
                step.task_score is not None
                and step.task_score["semantic_passed"] is True
                for step in report.steps
            )
        )
        first_task_score = report.steps[0].task_score
        assert first_task_score is not None
        self.assertFalse(first_task_score["full_passed"])
        preference_delta_check = next(
            check
            for check in first_task_score["checks"]
            if check["name"] == "preference_delta_without_consent"
        )
        self.assertEqual(preference_delta_check["actual"], 1)
        self.assertFalse(preference_delta_check["passed"])
        self.assertNotIn(
            "preference_delta_without_consent",
            {
                check["name"]
                for check in first_task_score["semantic_checks"]
            },
        )
        self.assertTrue(
            all(
                check["evaluated"] and check["passed"]
                for step in report.steps
                for check in step.checks
            )
        )

    def test_selected_rgb_manifest_rejects_native_bgr_task_result(self) -> None:
        scenarios = build_catalog(families=["block_stack"], seeds=[47])
        rgb = next(
            scenario
            for scenario in scenarios
            if scenario.target_id == "rgb_bottom_to_top"
            and scenario.outcome == "success"
        )
        bgr = next(
            scenario
            for scenario in scenarios
            if scenario.target_id == "bgr_bottom_to_top"
            and scenario.outcome == "success"
        )
        rgb_manifest = _scenario_manifest(rgb, backend="synthetic")
        bgr_manifest = _scenario_manifest(bgr, backend="synthetic")
        events: list[str] = []

        def wrong_turn(orchestrator: _FakeOrchestrator) -> dict[str, Any]:
            events.append("hri")
            orchestrator.history.episodes.append(
                {
                    "episode_id": "episode-wrong-bgr",
                    "user_id": "participant-a",
                }
            )
            return _scorable_success_result(
                bgr_manifest,
                mode="REPORT",
                memory_refs=[],
                history_refs=[],
            )

        def load_manifest(_path: str | Path) -> Mapping[str, Any]:
            events.append("oracle")
            return rgb_manifest

        orchestrator = _FakeOrchestrator([wrong_turn])
        report = run_memory_protocol(
            self._protocol(
                {
                    "query": "Stack the blocks RGB from bottom to top.",
                    "scenario_selector": {
                        "family": "block_stack",
                        "target_id": "rgb_bottom_to_top",
                        "outcome": "success",
                    },
                    "expected": {
                        "clarification_required": False,
                        "dispatches": 1,
                        "history_delta": 1,
                        "min_dispatches": 1,
                        "planner_status": "READY",
                        "preference_delta": 0,
                        "task_semantic_score": True,
                        "validator_outcome": "SUCCESS",
                    },
                }
            ),
            orchestrator,
            resolve_scenario=lambda _selector: Path("opaque-rgb-episode"),
            load_manifest=load_manifest,
        )

        self.assertEqual(events, ["hri", "oracle"])
        self.assertFalse(report.passed)
        step = report.steps[0]
        self.assertTrue(step.task_score_required)
        self.assertIsNotNone(step.task_score)
        assert step.task_score is not None
        self.assertFalse(step.task_score["semantic_passed"])
        self.assertFalse(step.task_score["full_passed"])
        hri_target = next(
            check
            for check in step.task_score["checks"]
            if check["name"] == "hri_resolved_target"
        )
        self.assertEqual(hri_target["actual"], "bgr_bottom_to_top")
        self.assertFalse(hri_target["passed"])
        ordinary_checks = [
            check
            for check in step.checks
            if check["name"] != "task_semantic_score"
        ]
        self.assertTrue(
            all(
                check["evaluated"] and check["passed"]
                for check in ordinary_checks
            ),
            ordinary_checks,
        )
        semantic_gate = next(
            check
            for check in step.checks
            if check["name"] == "task_semantic_score"
        )
        self.assertFalse(semantic_gate["actual"])
        self.assertFalse(semantic_gate["passed"])

    def test_clarification_only_step_does_not_require_task_score(self) -> None:
        orchestrator = _FakeOrchestrator(
            [
                lambda _orchestrator: {
                    "hri": {
                        "mode": "ASK",
                        "trace": {"history_refs": [], "memory_refs": []},
                    },
                    "task": {"attempts": []},
                }
            ]
        )
        report = run_memory_protocol(
            self._protocol(
                {
                    "query": "Stack the blocks.",
                    "scenario_selector": {
                        "family": "block_stack",
                        "target_id": "rgb_bottom_to_top",
                        "outcome": "success",
                    },
                    "expected": {"clarification_required": True},
                }
            ),
            orchestrator,
            resolve_scenario=lambda _selector: Path("opaque-episode"),
        )

        self.assertTrue(report.passed, report.steps)
        self.assertFalse(report.steps[0].task_score_required)
        self.assertIsNone(report.steps[0].task_score)


class MemoryProtocolScorerTests(unittest.TestCase):
    def test_success_manifest_accepts_memory_confirm_checkpoint_boundary(
        self,
    ) -> None:
        success = next(
            scenario
            for scenario in build_catalog(
                families=["block_stack"],
                seeds=[11],
            )
            if scenario.target_id == "rgb_bottom_to_top"
            and scenario.outcome == "success"
        )
        manifest = _scenario_manifest(success, backend="synthetic")
        confirmed_intent = str(manifest["target"]["instruction"])
        validation_spec = {
            "spec_id": "spec-memory-boundary",
            "confirmed_intent": confirmed_intent,
            "goal_conditions": [
                {
                    "id": f"goal-{index}",
                    "description": "Structured benchmark goal.",
                    "predicate": predicate,
                    "arguments": [],
                }
                for index, predicate in enumerate(
                    manifest["target"]["goal_predicates"],
                    start=1,
                )
            ],
        }
        result = {
            "hri": {
                "mode": "MEMORY_CONFIRM",
                "report": {"outcome": "SUCCESS"},
            },
            "resolved_task": {
                "confirmed_intent": confirmed_intent,
                "parameters": {
                    "target_id": str(manifest["target"]["target_id"])
                },
            },
            "history_checkpointed": True,
            "memory": {"preference_delta": 0},
            "task": {
                "outcome": "SUCCESS",
                "next_action": "NONE",
                "attempts": [
                    {
                        "plan": {
                            "planning_status": "READY",
                            "validation_spec": validation_spec,
                        },
                        "execution": {
                            "status": "OBSERVED_RECORDED_ATTEMPT",
                            "subtask_results": [{"status": "COMPLETED"}],
                        },
                        "validation": {
                            "outcome": "SUCCESS",
                            "task_complete": True,
                        },
                        "validation_assurance": {"next_action": "NONE"},
                    }
                ],
            },
        }

        report = score_agent_result(manifest, result)

        self.assertTrue(report.passed, report.checks)
        self.assertEqual(report.skipped, 0)
        history_check = next(
            check for check in report.checks if check["name"] == "history_delta"
        )
        self.assertEqual(history_check["actual"], 1)
        self.assertTrue(history_check["passed"])


class BenchmarkVisualOracleRegressionTests(unittest.TestCase):
    def test_execution_evidence_id_is_pixel_derived_not_path_derived(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "success-private-label.png"
            second = root / "failure-private-label.png"
            first.write_bytes(b"same-observation-bytes")
            second.write_bytes(first.read_bytes())

            first_model = ExecutionResult(
                status="OBSERVED_RECORDED_ATTEMPT",
                final_observation=str(first),
            ).to_model_dict()
            second_model = ExecutionResult(
                status="OBSERVED_RECORDED_ATTEMPT",
                final_observation=str(second),
            ).to_model_dict()

            self.assertEqual(
                first_model["final_observation_id"],
                second_model["final_observation_id"],
            )
            serialized = json.dumps([first_model, second_model])
            self.assertNotIn("success-private-label", serialized)
            self.assertNotIn("failure-private-label", serialized)

    def test_validator_binds_optional_protocol_bundle_to_declared_matrix(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_benchmark(
                root,
                families=["block_stack"],
                seeds=[23],
                backend="synthetic",
                include_controls=True,
            )
            scenarios = [
                *build_catalog(families=["block_stack"], seeds=[23]),
                *build_control_catalog(
                    families=["block_stack"],
                    seeds=[23],
                ),
            ]
            bundle = {
                "schema_version": "robopref.memory-protocols.v1",
                "fixtures": memory_protocol_fixtures(),
                "protocols": build_memory_protocols(scenarios),
            }
            protocols_path = root / "protocols.json"
            protocols_path.write_text(
                json.dumps(bundle, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.assertTrue(validate_benchmark(root).valid)

            bundle["protocols"][0]["description"] = "tampered"
            protocols_path.write_text(
                json.dumps(bundle, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report = validate_benchmark(root)

            self.assertFalse(report.valid)
            self.assertTrue(
                any("protocols.json differs" in error for error in report.errors),
                report.errors,
            )

    def test_validator_rejects_render_noise_as_visual_distinction_with_new_hash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generate_benchmark(
                root,
                families=["block_stack"],
                seeds=[17],
                backend="synthetic",
            )
            packets: dict[
                tuple[str, str, int, str],
                dict[str, Path],
            ] = {}
            for manifest_path in root.rglob("manifest.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                key = (
                    str(manifest["scene"]["family"]),
                    str(manifest["scene"]["variant"]),
                    int(manifest["scene"]["seed"]),
                    str(manifest["target"]["target_id"]),
                )
                packets.setdefault(key, {})[
                    str(manifest["expected_outcome"])
                ] = manifest_path

            siblings = next(
                outcomes
                for outcomes in packets.values()
                if {"success", "near_miss"} <= outcomes.keys()
            )
            success_final = siblings["success"].parent / "2.png"
            near_miss_manifest_path = siblings["near_miss"]
            near_miss_final = near_miss_manifest_path.parent / "2.png"
            near_miss_final.write_bytes(success_final.read_bytes())
            with Image.open(near_miss_final) as source:
                noisy = source.convert("RGB")
            pixels = noisy.load()
            for index in range(200):
                x = index * 37 % noisy.width
                y = index * 53 % noisy.height
                red, green, blue = pixels[x, y]
                pixels[x, y] = (
                    red + 1 if red < 255 else red - 1,
                    green,
                    blue,
                )
            noisy.save(near_miss_final, format="PNG", optimize=False)
            self.assertNotEqual(
                success_final.read_bytes(),
                near_miss_final.read_bytes(),
                "the regression must use different encoded frames",
            )
            near_miss_manifest = json.loads(
                near_miss_manifest_path.read_text(encoding="utf-8")
            )
            near_miss_manifest["frame_sha256"]["2.png"] = hashlib.sha256(
                near_miss_final.read_bytes()
            ).hexdigest()
            near_miss_manifest_path.write_text(
                json.dumps(near_miss_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            report = validate_benchmark(root)

            self.assertFalse(report.valid)
            self.assertTrue(
                any("pixel-identical" in error for error in report.errors),
                report.errors,
            )
            self.assertFalse(
                any("does not match its recorded digest" in error for error in report.errors),
                "the test must exercise semantic pixel validation, not stale hashing",
            )

    @unittest.skipUnless(MUJOCO_AVAILABLE, "optional mujoco dependency is absent")
    def test_mujoco_place_endpoints_and_category_semantic_geoms_differ(
        self,
    ) -> None:
        place_scenarios = build_catalog(
            families=["place_setting"],
            seeds=[19],
        )
        success = next(
            scenario
            for scenario in place_scenarios
            if scenario.scene_variant == "front_scatter"
            and scenario.target_id == "right_handed"
            and scenario.outcome == "success"
        )
        near_miss = next(
            scenario
            for scenario in place_scenarios
            if scenario.scene_variant == success.scene_variant
            and scenario.target_id == success.target_id
            and scenario.outcome == "near_miss"
        )

        success_image = _render(
            success,
            success.final_objects,
            occluded=False,
        )
        near_miss_image = _render(
            near_miss,
            near_miss.final_objects,
            occluded=False,
        )
        self.assertNotEqual(
            success_image.tobytes(),
            near_miss_image.tobytes(),
            "MuJoCo must make the inward-blade near miss visually observable",
        )

        category = next(
            scenario
            for scenario in build_catalog(
                families=["category_sort"],
                seeds=[19],
            )
            if scenario.outcome == "success"
        )
        model = _build_model(category, category.final_objects)
        import mujoco

        geom_names = {
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            for geom_id in range(model.ngeom)
        }
        objects_by_type = {
            item.object_type: item for item in category.final_objects
        }
        self.assertIn(
            f"{objects_by_type['laptop'].object_id}_screen",
            geom_names,
            "the laptop_screen visual primitive is missing",
        )
        self.assertIn(
            f"{objects_by_type['magazine'].object_id}_pages",
            geom_names,
            "the magazine_pages visual primitive is missing",
        )


if __name__ == "__main__":
    unittest.main()
