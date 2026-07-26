from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from simulation.benchmark.conversation_cases import (
    build_conversation_cases,
    load_episode_catalog,
)
from simulation.benchmark.conversation_evaluation import (
    ConversationEvaluationError,
    _classify_exception,
    evaluate_conversations,
)
from simulation.benchmark.conversation_models import ConversationEvaluationConfig


ROOT = Path(__file__).resolve().parents[1] / "dataset" / "sim_datasets"


def spec(descriptor: Any) -> dict[str, Any]:
    conditions = []
    for index, value in enumerate(descriptor.manifest["target"]["goal_predicates"]):
        predicate, remainder = value.split("(", 1)
        arguments = remainder.rsplit(")", 1)[0]
        conditions.append(
            {
                "id": f"goal-{index + 1}",
                "description": value,
                "predicate": predicate,
                "arguments": [item.strip() for item in arguments.split(",")],
            }
        )
    return {
        "spec_id": "spec-fake",
        "confirmed_intent": descriptor.instruction,
        "goal_conditions": conditions,
    }


class Repository:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.version = 0

    def list_episodes(self, user_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        values = [item for item in self.items if item["user_id"] == user_id]
        return values[:limit] if limit is not None else values
    def get_summary(self, user_id: str) -> dict[str, Any]:
        return {"text": "", "source_episode_ids": []}



    def list_preferences(
        self, user_id: str, *, include_inactive: bool = False
    ) -> list[dict[str, Any]]:
        return []


class FakeMemory:
    def __init__(self, history: Repository, preferences: Repository) -> None:
        self.history_repository = history
        self.preference_repository = preferences

    def begin_evaluation_turn(self) -> None:
        return None

    def delivery_for_current_turn(self, **_: Any) -> dict[str, Any]:
        return {
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
        }


class FakeOrchestrator:
    def __init__(self, context: Any) -> None:
        self.context = context
        self.descriptor = context.episodes[context.case.commands[0].episode_id]
        self.history = Repository()
        self.preferences = Repository()
        self.memory_agent = FakeMemory(self.history, self.preferences)
        self.history_outbox = SimpleNamespace(list_pending=lambda _user: [])
        self.history_context = None
        self.command_active = False
        self.evaluation_artifact_status = lambda: {
            "complete": True,
            "reason": "fake orchestrator",
        }

    def switch_dataset(self, selected_path: str) -> bool:
        return Path(selected_path).resolve() == self.descriptor.path

    def handle_user_message(self, message: str) -> dict[str, Any]:
        expectations = self.descriptor.manifest["benchmark_expectations"]
        validation = {"outcome": "SUCCESS", "task_complete": True}
        attempt = {
            "attempt": 1,
            "plan": {
                "planning_status": "READY",
                "validation_spec": spec(self.descriptor),
                "subtasks": [{"task_instruction": message}],
            },
            "execution": {
                "status": "OBSERVED_RECORDED_ATTEMPT",
                "subtask_results": [{"status": "RECORDED_EXTERNAL_ATTEMPT"}],
            },
            "initial_validation": validation,
            "initial_validation_assurance": {"next_action": "NONE"},
            "validation": validation,
            "validation_assurance": {"next_action": "NONE"},
        }
        task = {
            "outcome": expectations["hri"]["terminal_outcome"],
            "attempts": [attempt],
        }
        self.history.items.append(
            {
                "episode_id": "episode-fake",
                "user_id": self.context.case.user_id,
            }
        )
        self.history.version += 1
        return {
            "hri_decision": {
                "mode": "EXECUTE",
                "trace": {"history_refs": [], "memory_refs": []},
            },
            "hri": {
                "mode": "REPORT",
                "report": {"outcome": task["outcome"]},
            },
            "memory": None,
            "task": task,
            "resolved_task": {
                "confirmed_intent": self.descriptor.instruction,
                "parameters": {},
                "objects": [],
                "preference_refs": [],
            },
            "awaiting_user": False,
        }


def fake_factory(context: Any) -> FakeOrchestrator:
    return FakeOrchestrator(context)


class ConversationEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scenario = next(
            item
            for item in load_episode_catalog(ROOT)
            if item.outcome == "success" and item.control_kind is None
        )

    def test_batch_persists_and_resumes_one_isolated_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = ConversationEvaluationConfig(
                benchmark_root=ROOT,
                output_dir=directory,
                suites=("endpoint",),
                scenario_ids=(self.scenario.scenario_id,),
                bootstrap_replicates=10,
            )
            first = evaluate_conversations(
                config,
                orchestrator_factory=fake_factory,
            )
            self.assertEqual(first.executed_runs, 1)
            self.assertEqual(first.passed_runs, 1)
            self.assertTrue(first.results_path.is_file())
            self.assertTrue(first.summary_path.is_file())

            second = evaluate_conversations(
                config,
                orchestrator_factory=fake_factory,
            )
            self.assertEqual(second.executed_runs, 0)
            self.assertEqual(second.resumed_runs, 1)
            self.assertEqual(second.passed_runs, 1)

    def test_resume_rejects_a_tampered_command_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = ConversationEvaluationConfig(
                benchmark_root=ROOT,
                output_dir=directory,
                suites=("endpoint",),
                scenario_ids=(self.scenario.scenario_id,),
                bootstrap_replicates=10,
            )
            report = evaluate_conversations(
                config,
                orchestrator_factory=fake_factory,
            )
            record = json.loads(report.results_path.read_text())
            record["commands"] = []
            report.results_path.write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConversationEvaluationError):
                evaluate_conversations(
                    config,
                    orchestrator_factory=fake_factory,
                )

    def test_interrupted_directory_is_preserved_and_retry_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = ConversationEvaluationConfig(
                benchmark_root=ROOT,
                output_dir=directory,
                suites=("endpoint",),
                scenario_ids=(self.scenario.scenario_id,),
                bootstrap_replicates=10,
            )
            case = build_conversation_cases(
                load_episode_catalog(ROOT),
                config,
            )[0]
            interrupted = (
                Path(directory)
                / "runs"
                / case.case_id
                / "rep-001"
            )
            interrupted.mkdir(parents=True)
            stale = interrupted / "interrupted-artifact.jsonl"
            stale.write_text("stale\n", encoding="utf-8")

            report = evaluate_conversations(
                config,
                orchestrator_factory=fake_factory,
            )
            record = json.loads(report.results_path.read_text())
            self.assertEqual(
                Path(record["run_directory"]).name,
                "rep-001-retry-001",
            )
            self.assertEqual(stale.read_text(encoding="utf-8"), "stale\n")

    def test_operational_controls_do_not_change_semantic_digest_input(self) -> None:
        first = ConversationEvaluationConfig(
            benchmark_root=ROOT,
            output_dir="/tmp/prefmem-first",
            bootstrap_replicates=10,
        )
        second = ConversationEvaluationConfig(
            benchmark_root=ROOT,
            output_dir="/tmp/prefmem-second",
            resume=False,
            fail_fast=True,
            display_all=True,
            show_progress=False,
            bootstrap_replicates=20,
            bootstrap_seed=99,
        )
        self.assertEqual(first.semantic_dict(), second.semantic_dict())

    def test_progress_tracks_fresh_and_resumed_runs(self) -> None:
        bars: list[Any] = []

        class Progress:
            def __init__(self, **options: Any) -> None:
                self.options = options
                self.updates: list[int] = []
                self.closed = False
                bars.append(self)

            def update(self, amount: int) -> None:
                self.updates.append(amount)

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            config = ConversationEvaluationConfig(
                benchmark_root=ROOT,
                output_dir=directory,
                suites=("endpoint",),
                scenario_ids=(self.scenario.scenario_id,),
                bootstrap_replicates=10,
            )
            with patch(
                "simulation.benchmark.conversation_evaluation.tqdm",
                Progress,
            ):
                evaluate_conversations(
                    config,
                    orchestrator_factory=fake_factory,
                )
                evaluate_conversations(
                    config,
                    orchestrator_factory=fake_factory,
                )

        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].options["total"], 1)
        self.assertEqual(bars[0].options["initial"], 0)
        self.assertEqual(bars[0].updates, [1])
        self.assertTrue(bars[0].closed)
        self.assertEqual(bars[1].options["total"], 1)
        self.assertEqual(bars[1].options["initial"], 1)
        self.assertEqual(bars[1].updates, [])
        self.assertTrue(bars[1].closed)

    def test_wrapped_model_timeout_is_classified_as_timeout(self) -> None:
        try:
            try:
                raise RuntimeError("vLLM request timed out (APITimeoutError).")
            except RuntimeError as inner:
                raise ValueError("HRI contract failed twice") from inner
        except ValueError as error:
            self.assertEqual(_classify_exception(error), "TIMEOUT")

    def test_wrapped_connection_error_is_infrastructure(self) -> None:
        try:
            raise ValueError("vLLM request failed (ConnectError).")
        except ValueError as error:
            self.assertEqual(
                _classify_exception(error),
                "INFRASTRUCTURE_ERROR",
            )


if __name__ == "__main__":
    unittest.main()
