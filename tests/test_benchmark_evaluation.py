from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from simulation.benchmark.evaluation import (
    EvaluationConfig,
    EvaluationDatasetError,
    EvaluationError,
    EvaluationResumeError,
    _EpisodeMetadata,
    _SelectedEpisode,
    _scene_cluster_bootstrap_ci95,
    _stratified_prefix,
    _summarize,
    run_cold_memory_evaluation,
)
from simulation.benchmark.generator import generate_benchmark
from simulation.benchmark.provenance import benchmark_content_sha256


class _ListRepository:
    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self.items = list(items or [])

    def list_episodes(self, user_id: str) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in self.items
            if item.get("user_id") == user_id
        ]

    def list_preferences(self, user_id: str) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in self.items
            if item.get("user_id") == user_id
        ]


class _Outbox:
    def list_pending(self, user_id: str) -> list[dict[str, Any]]:
        del user_id
        return []


def _successful_result(episode: Any) -> dict[str, Any]:
    target = episode.manifest["target"]
    confirmed_intent = str(target["instruction"])
    validation_spec = {
        "confirmed_intent": confirmed_intent,
        "goal_conditions": [
            {"id": f"goal-{index}", "predicate": predicate}
            for index, predicate in enumerate(
                target["goal_predicates"],
                start=1,
            )
        ],
    }
    return {
        "hri": {"mode": "REPORT", "report": {"outcome": "SUCCESS"}},
        "memory": None,
        "resolved_task": {
            "confirmed_intent": confirmed_intent,
            "parameters": {"target_id": target["target_id"]},
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
                        "subtask_results": [{"status": "SUCCESS"}],
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


class _SuccessfulOrchestrator:
    def __init__(
        self,
        episode: Any,
        user_id: str,
        *,
        prepopulate: bool = False,
    ) -> None:
        self.episode = episode
        self.user_id = user_id
        histories = (
            [{"episode_id": "old", "user_id": user_id}]
            if prepopulate
            else []
        )
        self.history = _ListRepository(histories)
        self.preferences = _ListRepository()
        self.memory_agent = SimpleNamespace(
            history_repository=self.history,
            preference_repository=self.preferences,
        )
        self.history_outbox = _Outbox()
        self.messages: list[str] = []

    def handle_user_message(self, message: str) -> dict[str, Any]:
        self.messages.append(message)
        if getattr(self.memory_agent, "history_enabled", True):
            self.history.items.append(
                {
                    "episode_id": f"history-{len(self.history.items) + 1}",
                    "user_id": self.user_id,
                }
            )
        return _successful_result(self.episode)


class BenchmarkEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_directory = tempfile.TemporaryDirectory()
        cls.dataset_root = Path(cls.dataset_directory.name)
        generate_benchmark(
            cls.dataset_root,
            families=["block_stack"],
            seeds=[101],
            backend="synthetic",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dataset_directory.cleanup()

    def _assert_resume_rejects_mutation(
        self,
        mutator: Any,
        *,
        memory_mode: str = "full",
        track_artifacts: bool = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            def factory(config, episode, executor, trial_dir, user_id):
                del config, executor, trial_dir
                orchestrator = _SuccessfulOrchestrator(episode, user_id)
                if track_artifacts:
                    orchestrator.evaluation_artifact_status = lambda: {
                        "complete": True,
                    }
                return orchestrator

            config = EvaluationConfig(
                benchmark_root=self.dataset_root,
                output_dir=output,
                outcomes=("success",),
                max_scenarios=1,
                memory_mode=memory_mode,
            )
            run_cold_memory_evaluation(
                config,
                orchestrator_factory=factory,
            )
            path = output / "episode_results.jsonl"
            record = json.loads(path.read_text(encoding="utf-8"))
            mutator(record)
            path.write_text(
                json.dumps(record, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(EvaluationResumeError):
                run_cold_memory_evaluation(
                    config,
                    orchestrator_factory=factory,
                )

    def test_cold_repetitions_persist_and_resume_without_rerunning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            calls: list[dict[str, Any]] = []

            def factory(config, episode, executor, trial_dir, user_id):
                del executor
                calls.append(
                    {
                        "history_path": config.history_store_path,
                        "user_id": user_id,
                        "scenario_id": episode.scenario_id,
                        "trial_dir": str(trial_dir),
                        "seed": config.hri.seed,
                        "temperature": config.hri.temperature,
                    }
                )
                return _SuccessfulOrchestrator(episode, user_id)

            config = EvaluationConfig(
                benchmark_root=self.dataset_root,
                output_dir=output,
                repetitions=2,
                outcomes=("success",),
                max_scenarios=1,
                shuffle_seed=73,
                model_seed=400,
                temperature=0.2,
            )
            report = run_cold_memory_evaluation(
                config,
                orchestrator_factory=factory,
            )

            self.assertEqual(report.planned_trials, 2)
            self.assertEqual(report.executed_trials, 2)
            self.assertEqual(report.passed_trials, 2)
            self.assertEqual(report.error_trials, 0)
            self.assertEqual({call["seed"] for call in calls}, {400, 401})
            self.assertEqual(
                {call["temperature"] for call in calls},
                {0.2},
            )
            self.assertEqual(len({call["history_path"] for call in calls}), 2)
            self.assertEqual(
                {call["user_id"] for call in calls},
                {"evaluation-participant"},
            )
            for call in calls:
                self.assertNotIn(call["scenario_id"], call["user_id"])

            jsonl_path = output / "episode_results.jsonl"
            records = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["memory"]["history_delta"] == 1 for record in records))
            self.assertTrue(all(record["memory"]["preference_delta"] == 0 for record in records))
            self.assertTrue((output / "run_config.json").is_file())
            self.assertTrue((output / "checks.csv").is_file())
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue(
                all(Path(record["trial_artifact_dir"], "result.json").is_file() for record in records)
            )
            run_config = json.loads(
                (output / "run_config.json").read_text(encoding="utf-8")
            )
            runtime = run_config["evaluation"]["runtime"]
            self.assertTrue(runtime["python_version"])
            self.assertTrue(runtime["platform"])
            self.assertIn("git_commit", runtime)
            self.assertEqual(len(runtime["source_tree_sha256"]), 64)
            self.assertIn("source_sha256", run_config["evaluation"]["orchestrator_factory"])
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertIsNotNone(
                summary["overall"]["strict_pass_rate_wilson_ci95"]
            )
            self.assertIsNotNone(
                summary["overall"]["duration_seconds"]["p50"]
            )
            self.assertIn("by_scene_variant", summary)
            self.assertIn("by_target", summary)
            self.assertIn("by_seed", summary)
            self.assertIn("regular", summary["by_control_kind"])
            self.assertGreater(summary["check_totals"]["evaluated"], 0)
            self.assertIn("validator", summary["headline"])
            # Legacy results without runtime provenance must never be
            # retroactively attributed to the current implementation.
            run_config["evaluation"].pop("runtime")
            (output / "run_config.json").write_text(
                json.dumps(run_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                EvaluationResumeError,
                "lacks complete provenance",
            ):
                run_cold_memory_evaluation(
                    config,
                    orchestrator_factory=factory,
                )

    def test_pilot_prefix_balances_families_before_secondary_strata(
        self,
    ) -> None:
        selected: list[_SelectedEpisode] = []
        for family in ("block_stack", "category_sort", "place_setting"):
            for target_index, target in enumerate(("target-a", "target-b")):
                for outcome_index, outcome in enumerate(
                    ("near_miss", "success", "unsafe")
                ):
                    for seed in (1, 2, 3):
                        scenario_id = (
                            f"{family}-{target_index}-{outcome_index}-{seed}"
                        )
                        selected.append(
                            _SelectedEpisode(
                                episode=object(),  # type: ignore[arg-type]
                                metadata=_EpisodeMetadata(
                                    scenario_id=scenario_id,
                                    family=family,
                                    scene_variant=f"variant-{seed}",
                                    target_id=target,
                                    outcome=outcome,
                                    seed=seed,
                                    control_kind=None,
                                ),
                            )
                        )

        sample = _stratified_prefix(selected, 24)
        repeated = _stratified_prefix(list(reversed(selected)), 24)

        self.assertEqual(
            [item.metadata.scenario_id for item in sample],
            [item.metadata.scenario_id for item in repeated],
        )
        family_counts = Counter(item.metadata.family for item in sample)
        self.assertEqual(set(family_counts.values()), {8})
        for family in family_counts:
            targets = Counter(
                item.metadata.target_id
                for item in sample
                if item.metadata.family == family
            )
            seeds = Counter(
                item.metadata.seed
                for item in sample
                if item.metadata.family == family
            )
            self.assertLessEqual(max(targets.values()) - min(targets.values()), 1)
            self.assertLessEqual(max(seeds.values()) - min(seeds.values()), 1)

    def test_trial_exception_is_a_durable_failed_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            class FailingOrchestrator(_SuccessfulOrchestrator):
                def handle_user_message(self, message: str) -> dict[str, Any]:
                    del message
                    raise TimeoutError("offline model timed out")

            def factory(config, episode, executor, trial_dir, user_id):
                del config, executor, trial_dir
                return FailingOrchestrator(episode, user_id)

            config = EvaluationConfig(
                benchmark_root=self.dataset_root,
                output_dir=output,
                outcomes=("success",),
                max_scenarios=1,
            )
            report = run_cold_memory_evaluation(
                config,
                orchestrator_factory=factory,
            )

            self.assertEqual(report.executed_trials, 1)
            self.assertEqual(report.error_trials, 1)
            record = json.loads(
                (output / "episode_results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(record["status"], "ERROR")
            self.assertEqual(record["error"]["type"], "TimeoutError")
            self.assertEqual(record["failure_category"], "TIMEOUT")
            self.assertEqual(record["error"]["phase"], "agent_runtime")
            self.assertFalse(record["passed"])
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "checks.csv").is_file())
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["check_totals"]["expected_observations"],
                14,
            )
            self.assertEqual(summary["check_totals"]["missing_evidence"], 14)
            self.assertEqual(
                summary["check_totals"]["incomplete_evidence_trials"],
                1,
            )
            hri_target = summary["checks"]["hri_resolved_target"]
            self.assertEqual(hri_target["expected_observations"], 1)
            self.assertEqual(hri_target["missing_evidence"], 1)
            self.assertEqual(hri_target["coverage_rate"], 0.0)
            self.assertEqual(hri_target["unconditional_accuracy"], 0.0)

    def test_structured_subagent_failures_are_persisted_and_aggregated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            first = {
                "event_id": "planning-attempt-1-planner-call",
                "attempt": 1,
                "event": "planner-call",
                "stage": "PLANNING",
                "error_type": "PlannerAgentError",
                "classification": "OUTPUT_CONTRACT",
                "message": "READY plan omitted structured predicates",
            }
            second = {
                **first,
                "event_id": "planning-attempt-2-planner-call",
                "attempt": 2,
            }

            class EmbeddedFailureOrchestrator(_SuccessfulOrchestrator):
                def handle_user_message(self, message: str) -> dict[str, Any]:
                    result = super().handle_user_message(message)
                    # The first event is embedded twice by HRI/assurance
                    # telemetry; the second is an identical retry failure.
                    result["diagnostic_failures"] = [
                        dict(first),
                        dict(first),
                        dict(second),
                    ]
                    return result

            def factory(config, episode, executor, trial_dir, user_id):
                del config, executor, trial_dir
                return EmbeddedFailureOrchestrator(episode, user_id)

            run_cold_memory_evaluation(
                EvaluationConfig(
                    benchmark_root=self.dataset_root,
                    output_dir=output,
                    outcomes=("success",),
                    max_scenarios=1,
                ),
                orchestrator_factory=factory,
            )

            record = json.loads(
                (output / "episode_results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(record["agent_failures"], [first, second])
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["subagent_failures"]["by_classification"],
                {"OUTPUT_CONTRACT": 2},
            )
            self.assertEqual(
                summary["subagent_failures"]["by_stage"],
                {"PLANNING": 2},
            )
            self.assertEqual(
                summary["subagent_failures"]["failure_events"],
                2,
            )
            self.assertEqual(
                summary["subagent_failures"]["affected_trials"],
                1,
            )
            self.assertEqual(
                summary["subagent_failures"]["affected_trial_rate"],
                1.0,
            )
            self.assertEqual(
                summary["subagent_failures"][
                    "affected_trials_by_classification"
                ],
                {"OUTPUT_CONTRACT": 1},
            )
            self.assertEqual(
                summary["subagent_failures"]["affected_trials_by_stage"],
                {"PLANNING": 1},
            )

    def test_selected_packet_digest_changes_when_numbered_frame_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copied_dataset = Path(directory) / "benchmark"
            shutil.copytree(self.dataset_root, copied_dataset)
            output = Path(directory) / "run"

            def factory(config, episode, executor, trial_dir, user_id):
                del config, executor, trial_dir
                return _SuccessfulOrchestrator(episode, user_id)

            run_cold_memory_evaluation(
                EvaluationConfig(
                    benchmark_root=copied_dataset,
                    output_dir=output,
                    outcomes=("success",),
                    max_scenarios=1,
                ),
                orchestrator_factory=factory,
            )
            record = json.loads(
                (output / "episode_results.jsonl").read_text(encoding="utf-8")
            )
            configured = json.loads(
                (output / "run_config.json").read_text(encoding="utf-8")
            )["evaluation"]["selected_packets_sha256"]
            selected_ids = [record["scenario_id"]]
            self.assertEqual(
                configured,
                benchmark_content_sha256(copied_dataset, selected_ids),
            )

            frame = (
                copied_dataset
                / "episodes"
                / record["scenario_id"]
                / "1.png"
            )
            frame.write_bytes(frame.read_bytes() + b"frame-mutation")
            self.assertNotEqual(
                configured,
                benchmark_content_sha256(copied_dataset, selected_ids),
            )

    def test_nonempty_factory_memory_is_rejected_before_the_user_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            orchestrators: list[_SuccessfulOrchestrator] = []

            def factory(config, episode, executor, trial_dir, user_id):
                del config, executor, trial_dir
                orchestrator = _SuccessfulOrchestrator(
                    episode,
                    user_id,
                    prepopulate=True,
                )
                orchestrators.append(orchestrator)
                return orchestrator

            report = run_cold_memory_evaluation(
                EvaluationConfig(
                    benchmark_root=self.dataset_root,
                    output_dir=output,
                    outcomes=("success",),
                    max_scenarios=1,
                ),
                orchestrator_factory=factory,
            )

            self.assertEqual(report.error_trials, 1)
            self.assertEqual(orchestrators[0].messages, [])
            record = json.loads(
                (output / "episode_results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(
                record["error"]["type"],
                "EvaluationIsolationError",
            )
            self.assertEqual(
                record["failure_category"],
                "MEMORY_ISOLATION",
            )

    def test_uninspectable_memory_is_rejected_before_the_user_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            class UninspectableOrchestrator:
                def __init__(self) -> None:
                    self.messages: list[str] = []

                def handle_user_message(self, message: str) -> dict[str, Any]:
                    self.messages.append(message)
                    raise AssertionError("the user turn must not run")

            orchestrator = UninspectableOrchestrator()

            def factory(config, episode, executor, trial_dir, user_id):
                del config, episode, executor, trial_dir, user_id
                return orchestrator

            report = run_cold_memory_evaluation(
                EvaluationConfig(
                    benchmark_root=self.dataset_root,
                    output_dir=output,
                    outcomes=("success",),
                    max_scenarios=1,
                ),
                orchestrator_factory=factory,
            )

            self.assertEqual(report.error_trials, 1)
            self.assertEqual(orchestrator.messages, [])
            record = json.loads(
                (output / "episode_results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(
                record["failure_category"],
                "MEMORY_ISOLATION",
            )
            self.assertIn(
                "could not be verified",
                record["error"]["message"],
            )

    def test_resume_rejects_changed_semantic_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            def factory(config, episode, executor, trial_dir, user_id):
                del config, executor, trial_dir
                return _SuccessfulOrchestrator(episode, user_id)

            base = EvaluationConfig(
                benchmark_root=self.dataset_root,
                output_dir=output,
                outcomes=("success",),
                max_scenarios=1,
            )
            run_cold_memory_evaluation(base, orchestrator_factory=factory)

            changed = EvaluationConfig(
                benchmark_root=self.dataset_root,
                output_dir=output,
                outcomes=("success",),
                max_scenarios=1,
                max_replans=7,
            )
            with self.assertRaises(EvaluationResumeError):
                run_cold_memory_evaluation(
                    changed,
                    orchestrator_factory=factory,
                )

    def test_resume_rejects_corrupt_durable_trial_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            def factory(config, episode, executor, trial_dir, user_id):
                del config, executor, trial_dir
                return _SuccessfulOrchestrator(episode, user_id)

            config = EvaluationConfig(
                benchmark_root=self.dataset_root,
                output_dir=output,
                outcomes=("success",),
                max_scenarios=1,
            )
            run_cold_memory_evaluation(
                config,
                orchestrator_factory=factory,
            )
            result_path = output / "episode_results.jsonl"
            record = json.loads(result_path.read_text(encoding="utf-8"))
            record["family"] = "corrupt-family"
            result_path.write_text(
                json.dumps(record, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                EvaluationResumeError,
                "inconsistent 'family'",
            ):
                run_cold_memory_evaluation(
                    config,
                    orchestrator_factory=factory,
                )

    def test_resume_rejects_boolean_integer_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            def factory(config, episode, executor, trial_dir, user_id):
                del config, executor, trial_dir
                return _SuccessfulOrchestrator(episode, user_id)

            config = EvaluationConfig(
                benchmark_root=self.dataset_root,
                output_dir=output,
                outcomes=("success",),
                max_scenarios=1,
                model_seed=1,
            )
            run_cold_memory_evaluation(
                config,
                orchestrator_factory=factory,
            )
            result_path = output / "episode_results.jsonl"
            record = json.loads(result_path.read_text(encoding="utf-8"))
            record["repetition"] = True
            record["model_seed"] = True
            result_path.write_text(
                json.dumps(record, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                EvaluationResumeError,
                "condition, repetition, and scenario_id",
            ):
                run_cold_memory_evaluation(
                    config,
                    orchestrator_factory=factory,
                )

    def test_resume_rejects_malformed_or_incomplete_strict_evidence(
        self,
    ) -> None:
        def empty_checks(record: dict[str, Any]) -> None:
            record["score"]["checks"] = []

        def incomplete_coverage(record: dict[str, Any]) -> None:
            record["raw_score"]["checks"].pop()

        def duplicate_check(record: dict[str, Any]) -> None:
            record["raw_score"]["checks"].append(
                dict(record["raw_score"]["checks"][0])
            )

        def invalid_check_shape(record: dict[str, Any]) -> None:
            record["score"]["checks"][0].pop("actual")

        def inconsistent_terminal_pass(record: dict[str, Any]) -> None:
            record["passed"] = not record["passed"]

        def self_consistent_but_oracle_false(record: dict[str, Any]) -> None:
            for score_name in ("raw_score", "score"):
                score = record[score_name]
                score["checks"][0]["passed"] = False
                score["correct"] -= 1
                score["passed"] = False
            record["passed"] = False

        for name, mutation in (
            ("empty", empty_checks),
            ("incomplete", incomplete_coverage),
            ("duplicate", duplicate_check),
            ("invalid-shape", invalid_check_shape),
            ("terminal-pass", inconsistent_terminal_pass),
            ("oracle-recompute", self_consistent_but_oracle_false),
        ):
            with self.subTest(name=name):
                self._assert_resume_rejects_mutation(mutation)

    def test_resume_enforces_ablation_and_tracked_artifact_evidence(
        self,
    ) -> None:
        def remove_exclusions(record: dict[str, Any]) -> None:
            record["score_exclusions"] = []

        self._assert_resume_rejects_mutation(
            remove_exclusions,
            memory_mode="no-memory",
        )

        def write_disabled_history(record: dict[str, Any]) -> None:
            memory = record["memory"]
            memory["after"]["history_ids"] = ["forged-history"]
            memory["after"]["history_count"] = 1
            memory["history_delta"] = 1

        self._assert_resume_rejects_mutation(
            write_disabled_history,
            memory_mode="preference-only",
        )

        def remove_artifact_gate(record: dict[str, Any]) -> None:
            record["score"]["checks"] = [
                check
                for check in record["score"]["checks"]
                if check["name"] != "evaluation_artifacts_complete"
            ]

        self._assert_resume_rejects_mutation(
            remove_artifact_gate,
            track_artifacts=True,
        )

    def test_zero_selected_packets_reports_filters_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = EvaluationConfig(
                benchmark_root=self.dataset_root,
                output_dir=Path(directory) / "run",
                outcomes=("not-a-real-outcome",),
            )
            with self.assertRaises(EvaluationDatasetError) as caught:
                run_cold_memory_evaluation(config)

            message = str(caught.exception)
            self.assertIn("Zero benchmark packets were selected", message)
            self.assertIn("not-a-real-outcome", message)
            self.assertFalse(Path(config.output_dir).exists())

    def test_evaluation_and_dataset_roots_must_be_disjoint(self) -> None:
        nested_output = self.dataset_root / "evaluation-output"
        with self.assertRaises(EvaluationError):
            run_cold_memory_evaluation(
                EvaluationConfig(
                    benchmark_root=self.dataset_root,
                    output_dir=nested_output,
                    outcomes=("success",),
                    max_scenarios=1,
                )
            )
        self.assertFalse(nested_output.exists())

        with self.assertRaises(EvaluationError):
            run_cold_memory_evaluation(
                EvaluationConfig(
                    benchmark_root=self.dataset_root,
                    output_dir=self.dataset_root.parent,
                    outcomes=("success",),
                    max_scenarios=1,
                )
            )

    def test_failed_checks_receive_component_taxonomy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            class WrongValidatorOrchestrator(_SuccessfulOrchestrator):
                def handle_user_message(self, message: str) -> dict[str, Any]:
                    result = super().handle_user_message(message)
                    result["task"]["attempts"][0]["validation"][
                        "outcome"
                    ] = "FAILURE"
                    return result

            def factory(config, episode, executor, trial_dir, user_id):
                del config, executor, trial_dir
                return WrongValidatorOrchestrator(episode, user_id)

            report = run_cold_memory_evaluation(
                EvaluationConfig(
                    benchmark_root=self.dataset_root,
                    output_dir=output,
                    outcomes=("success",),
                    max_scenarios=1,
                ),
                orchestrator_factory=factory,
            )

            self.assertEqual(report.failed_trials, 1)
            self.assertEqual(report.error_trials, 0)
            record = json.loads(
                (output / "episode_results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(
                record["failure_category"],
                "STRICT_CHECK_FAILURE",
            )
            self.assertIn("VALIDATOR", record["failure_categories"])
            self.assertIn(
                "validator_outcome",
                record["failed_checks_by_category"]["VALIDATOR"],
            )

    def test_ablation_primary_score_excludes_only_disabled_memory_checks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            def factory(config, episode, executor, trial_dir, user_id):
                del config, executor, trial_dir
                return _SuccessfulOrchestrator(episode, user_id)

            report = run_cold_memory_evaluation(
                EvaluationConfig(
                    benchmark_root=self.dataset_root,
                    output_dir=output,
                    outcomes=("success",),
                    max_scenarios=1,
                    memory_mode="no-memory",
                    condition="no-memory",
                ),
                orchestrator_factory=factory,
            )

            self.assertEqual(report.passed_trials, 1)
            record = json.loads(
                (output / "episode_results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertTrue(record["score"]["passed"])
            self.assertFalse(record["raw_score"]["passed"])
            excluded = {
                item["check_name"] for item in record["score_exclusions"]
            }
            self.assertEqual(
                excluded,
                {
                    "history_delta",
                    "preference_delta_without_consent",
                },
            )
            retained_names = {
                check["name"] for check in record["score"]["checks"]
            }
            self.assertNotIn("history_delta", retained_names)
            self.assertNotIn(
                "preference_delta_without_consent",
                retained_names,
            )
            self.assertIsNone(record["failure_category"])
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["check_totals"]["ablation_exclusions"],
                2,
            )

    def test_incomplete_production_artifacts_fail_the_evaluation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            def factory(config, episode, executor, trial_dir, user_id):
                del config, executor, trial_dir
                orchestrator = _SuccessfulOrchestrator(episode, user_id)
                orchestrator.evaluation_artifact_status = lambda: {
                    "complete": False,
                    "streams": {
                        "model_calls": {
                            "complete": False,
                            "events_written": 0,
                            "error_count": 1,
                        },
                        "agent_events": {
                            "complete": True,
                            "events_written": 1,
                            "error_count": 0,
                        },
                    },
                }
                return orchestrator

            report = run_cold_memory_evaluation(
                EvaluationConfig(
                    benchmark_root=self.dataset_root,
                    output_dir=output,
                    outcomes=("success",),
                    max_scenarios=1,
                ),
                orchestrator_factory=factory,
            )

            self.assertEqual(report.failed_trials, 1)
            record = json.loads(
                (output / "episode_results.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertTrue(record["raw_score"]["passed"])
            self.assertFalse(record["score"]["passed"])
            self.assertFalse(record["passed"])
            self.assertEqual(
                record["failure_category"],
                "ARTIFACT_INCOMPLETE",
            )
            artifact_check = next(
                check
                for check in record["score"]["checks"]
                if check["name"] == "evaluation_artifacts_complete"
            )
            self.assertFalse(artifact_check["passed"])
            self.assertTrue(record["artifact_status"]["tracked"])

    def test_scene_cluster_bootstrap_is_deterministic_and_nests_repetitions(
        self,
    ) -> None:
        records: list[dict[str, Any]] = []
        for seed, passed in ((1, True), (2, False)):
            for repetition in (1, 2, 3):
                records.append(
                    {
                        "condition": "full",
                        "repetition": repetition,
                        "scenario_id": f"scenario-{seed}-{repetition}",
                        "family": "block_stack",
                        "scene_variant": "scattered",
                        "target_id": "rgb_bottom_to_top",
                        "outcome": "success",
                        "seed": seed,
                        "control_kind": None,
                        "status": "COMPLETED",
                        "passed": passed,
                        "duration_seconds": 1.0,
                        "score": {"checks": []},
                    }
                )

        first = _summarize(records)["overall"][
            "strict_pass_rate_scene_cluster_bootstrap_ci95"
        ]
        second = _summarize(list(reversed(records)))["overall"][
            "strict_pass_rate_scene_cluster_bootstrap_ci95"
        ]

        self.assertEqual(first, second)
        self.assertEqual(
            first["method"],
            "percentile_scene_cluster_bootstrap",
        )
        self.assertEqual(
            first["cluster_unit"],
            [
                "packet_kind",
                "family",
                "scene_variant",
                "seed",
                "target_id_if_control",
            ],
        )
        self.assertEqual(first["cluster_count"], 2)
        self.assertEqual(first["replicate_count"], 2_000)
        self.assertEqual(first["lower"], 0.0)
        self.assertEqual(first["upper"], 1.0)

    def test_scene_bootstrap_separates_target_specific_control_scenes(
        self,
    ) -> None:
        def record(target_id: str, *, control: bool) -> dict[str, Any]:
            return {
                "family": "block_stack",
                "scene_variant": (
                    "control_already_satisfied"
                    if control
                    else "wide_scatter"
                ),
                "seed": 1,
                "target_id": target_id,
                "control_kind": (
                    "already_satisfied" if control else None
                ),
                "passed": True,
            }

        core = _scene_cluster_bootstrap_ci95(
            [
                record("rgb_bottom_to_top", control=False),
                record("bgr_bottom_to_top", control=False),
            ],
            replicate_count=10,
        )
        controls = _scene_cluster_bootstrap_ci95(
            [
                record("rgb_bottom_to_top", control=True),
                record("bgr_bottom_to_top", control=True),
            ],
            replicate_count=10,
        )

        self.assertEqual(core["cluster_count"], 1)
        self.assertEqual(controls["cluster_count"], 2)

    def test_validator_outcome_headline_excludes_expected_no_call_cases(
        self,
    ) -> None:
        def record(
            scenario_id: str,
            outcome: str,
            checks: list[dict[str, Any]],
        ) -> dict[str, Any]:
            return {
                "condition": "full",
                "repetition": 1,
                "scenario_id": scenario_id,
                "family": "block_stack",
                "scene_variant": "scattered",
                "target_id": "rgb_bottom_to_top",
                "outcome": outcome,
                "seed": 101,
                "control_kind": None,
                "status": "COMPLETED",
                "passed": True,
                "duration_seconds": 1.0,
                "score": {"checks": checks},
            }

        summary = _summarize(
            [
                record(
                    "success",
                    "success",
                    [
                        {
                            "name": "validator_called",
                            "expected": True,
                            "actual": True,
                            "evaluated": True,
                            "passed": True,
                        },
                        {
                            "name": "validator_outcome",
                            "expected": "SUCCESS",
                            "actual": "SUCCESS",
                            "evaluated": True,
                            "passed": True,
                        },
                    ],
                ),
                record(
                    "failure",
                    "failure",
                    [
                        {
                            "name": "validator_called",
                            "expected": True,
                            "actual": True,
                            "evaluated": True,
                            "passed": True,
                        },
                        {
                            "name": "validator_outcome",
                            "expected": "FAILURE",
                            "actual": "SUCCESS",
                            "evaluated": True,
                            "passed": False,
                        },
                    ],
                ),
                record(
                    "unsafe",
                    "unsafe",
                    [
                        {
                            "name": "validator_called",
                            "expected": False,
                            "actual": False,
                            "evaluated": True,
                            "passed": True,
                        },
                        {
                            "name": "validator_outcome",
                            "expected": None,
                            "actual": None,
                            "evaluated": True,
                            "passed": True,
                        },
                    ],
                ),
            ]
        )

        validator = summary["headline"]["validator"]
        self.assertEqual(validator["outcome_accuracy"]["successes"], 1)
        self.assertEqual(validator["outcome_accuracy"]["trials"], 2)
        self.assertEqual(validator["outcome_accuracy"]["rate"], 0.5)
        self.assertIsNotNone(
            validator["outcome_accuracy"]["wilson_ci95"]
        )
        self.assertEqual(
            validator["confusion_matrix"],
            {
                "FAILURE": {"SUCCESS": 1},
                "SUCCESS": {"SUCCESS": 1},
            },
        )
        # The raw scorer/check summary remains unchanged for auditability.
        self.assertEqual(summary["checks"]["validator_outcome"]["observations"], 3)
        self.assertEqual(summary["checks"]["validator_outcome"]["evaluated"], 3)
        self.assertEqual(summary["checks"]["validator_outcome"]["passed"], 2)

    def test_false_completion_headline_covers_all_non_success_cases(
        self,
    ) -> None:
        def task_complete_check(
            expected: bool | None,
            actual: Any,
            *,
            evaluated: bool = True,
        ) -> dict[str, Any]:
            return {
                "name": "task_complete",
                "expected": expected,
                "actual": actual,
                "evaluated": evaluated,
                "passed": (
                    actual == expected if evaluated else None
                ),
            }

        records: list[dict[str, Any]] = []
        cases = [
            ("failure", task_complete_check(False, False)),
            ("partial", task_complete_check(False, True)),
            ("near_miss", task_complete_check(False, None)),
            (
                "unknown",
                task_complete_check(False, None, evaluated=False),
            ),
            ("unsafe", task_complete_check(None, True)),
            ("unsafe", task_complete_check(None, None)),
            ("success", task_complete_check(True, True)),
        ]
        for index, (outcome, check) in enumerate(cases, start=1):
            records.append(
                {
                    "condition": "full",
                    "repetition": 1,
                    "scenario_id": f"scenario-{index}",
                    "family": "block_stack",
                    "scene_variant": "scattered",
                    "target_id": "rgb_bottom_to_top",
                    "outcome": outcome,
                    "seed": index,
                    "control_kind": None,
                    "status": "COMPLETED",
                    "passed": True,
                    "duration_seconds": 1.0,
                    "score": {"checks": [check]},
                }
            )

        validator = _summarize(records)["headline"]["validator"]

        self.assertEqual(validator["false_task_completions"], 2)
        self.assertEqual(validator["non_success_terminal_cases"], 6)
        self.assertEqual(validator["non_success_task_checks"], 6)
        self.assertEqual(
            validator["false_task_completion_rate"]["successes"],
            2,
        )
        self.assertEqual(
            validator["false_task_completion_rate"]["trials"],
            6,
        )
        self.assertAlmostEqual(
            validator["false_task_completion_rate"]["rate"],
            1 / 3,
        )
        self.assertEqual(
            validator["task_completion_states"],
            {
                "reported_true": 2,
                "reported_false": 1,
                "reported_none": 2,
                "missing_or_unevaluated": 1,
                "invalid": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
