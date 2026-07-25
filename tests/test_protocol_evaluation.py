from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agents.configs import PrefMemConfig
from memory.models import ConsentEvidence, MemoryContext, MemoryQuery
from simulation.benchmark.ablations import MemoryAblationAgent
from simulation.benchmark.protocol_evaluation import (
    _build_summary,
    _protocol_cluster_bootstrap_ci95,
    ProtocolEvaluationError,
    apply_consent_fixture,
    evaluate_memory_protocols,
)
from simulation.benchmark.protocols import memory_protocol_fixtures
from simulation.benchmark.protocol_runner import run_memory_protocol


class _HistoryRepository:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.version = 0

    def list_episodes(
        self, user_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        selected = [
            copy.deepcopy(item)
            for item in self.items
            if item.get("user_id") == user_id
        ]
        return selected[-limit:] if limit is not None else selected

    def get_summary(self, user_id: str) -> dict[str, Any]:
        return {
            "text": "",
            "source_episode_ids": [
                str(item["episode_id"])
                for item in self.items
                if item.get("user_id") == user_id and item.get("episode_id")
            ],
            "updated_at": None,
        }


class _PreferenceRepository:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.version = 0

    def list_preferences(
        self, user_id: str, include_inactive: bool = False
    ) -> list[dict[str, Any]]:
        selected = [
            item
            for item in self.items
            if item.get("user_id") == user_id
            and (include_inactive or item.get("status") == "active")
        ]
        return copy.deepcopy(selected)


class _MemoryAgent:
    def __init__(self) -> None:
        self.history_repository = _HistoryRepository()
        self.preference_repository = _PreferenceRepository()
        self.fixture_calls: list[dict[str, Any]] = []

    def get_memory_context(self, query: MemoryQuery) -> MemoryContext:
        return MemoryContext(
            relevant_history=self.history_repository.list_episodes(
                query.user_id
            ),
            relevant_preferences=self.preference_repository.list_preferences(
                query.user_id
            ),
        )

    def update_preference_memory(
        self,
        request: dict[str, Any],
        *,
        consent: ConsentEvidence,
        transaction_id: str,
    ) -> dict[str, Any]:
        consent.validate()
        self.fixture_calls.append(
            {
                "request": copy.deepcopy(request),
                "consent": consent,
                "transaction_id": transaction_id,
            }
        )
        record = {
            "id": f"pref-{len(self.preference_repository.items) + 1}",
            "user_id": request["user_id"],
            "status": "active",
            **copy.deepcopy(request["preference"]),
            "consent": consent.to_dict(),
        }
        self.preference_repository.items.append(record)
        self.preference_repository.version += 1
        return {
            "transaction": {
                "transaction_id": transaction_id,
                "results": [{"action": "ADD", "preference_id": record["id"]}],
            },
            "compaction": None,
        }


class _Outbox:
    def list_pending(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return []


class _Orchestrator:
    def __init__(self, config: PrefMemConfig, *, fail: bool = False) -> None:
        self.config = config
        self.memory_agent = _MemoryAgent()
        self.history_outbox = _Outbox()
        self.pending_question: dict[str, Any] | None = None
        self.fail = fail
        self.switches: list[str] = []

    def switch_dataset(self, selected_path: str) -> bool:
        self.switches.append(selected_path)
        return True

    def handle_user_message(self, message: str) -> dict[str, Any]:
        if self.fail:
            raise TimeoutError("scripted model timeout")
        self.memory_agent.get_memory_context(
            MemoryQuery(
                user_id=self.config.user_id,
                request=message,
                scene={},
            )
        )
        self.pending_question = {
            "kind": "TASK_CLARIFICATION",
            "payload": {},
        }
        return {
            "hri": {
                "mode": "ASK",
                "user_message": "Which order?",
                "trace": {
                    "history_refs": [],
                    "memory_refs": [],
                },
            },
            "task": {"attempts": []},
        }


def _scenario_id(index: int) -> str:
    return f"ep-{index:024x}"


def _write_benchmark(root: Path) -> None:
    scenario_ids: list[str] = []
    index = 0
    for variant in ("compact_scatter", "wide_scatter"):
        for seed in (1, 2):
            index += 1
            scenario_id = _scenario_id(index)
            scenario_ids.append(scenario_id)
            episode = root / "episodes" / scenario_id
            episode.mkdir(parents=True)
            manifest = {
                "scenario_id": scenario_id,
                "scene": {
                    "family": "block_stack",
                    "variant": variant,
                    "seed": seed,
                    "counterfactual_group_id": f"grp-{index:024x}",
                    "control_kind": None,
                },
                "target": {"target_id": "rgb_bottom_to_top"},
                "expected_outcome": "success",
            }
            (episode / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
    (root / "index.json").write_text(
        json.dumps(
            {
                "scenario_ids": scenario_ids,
                "catalog_digest": "catalog-for-test",
            }
        ),
        encoding="utf-8",
    )


def _protocol(
    protocol_id: str,
    *,
    fixture_id: str | None = None,
    message: str = "Stack the blocks.",
) -> dict[str, Any]:
    result = {
        "protocol_id": protocol_id,
        "user_id": "participant-a",
        "requires_fresh_memory": True,
        "steps": [
            {
                "query": message,
                "scenario_selector": {
                    "family": "block_stack",
                    "target_id": "rgb_bottom_to_top",
                    "outcome": "success",
                },
                "expected": {
                    "clarification_required": True,
                    "history_delta": 0,
                    "preference_delta": 0,
                },
            }
        ],
    }
    if fixture_id is not None:
        result["initial_memory_fixture"] = fixture_id
    return result


def _write_protocols(
    root: Path,
    protocols: list[dict[str, Any]],
    *,
    fixtures: dict[str, dict[str, Any]] | None = None,
) -> None:
    (root / "protocols.json").write_text(
        json.dumps(
            {
                "schema_version": "robopref.memory-protocols.v1",
                "fixtures": fixtures or {},
                "protocols": protocols,
            }
        ),
        encoding="utf-8",
    )


class ProtocolEvaluationTests(unittest.TestCase):
    @staticmethod
    def _evaluate(*args, **kwargs):
        validation = SimpleNamespace(valid=True, errors=())
        with patch(
            "simulation.benchmark.protocol_evaluation.validate_benchmark",
            return_value=validation,
        ):
            return evaluate_memory_protocols(*args, **kwargs)

    def test_rotates_compatible_scenarios_and_isolates_stores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "results"
            benchmark.mkdir()
            _write_benchmark(benchmark)
            _write_protocols(benchmark, [_protocol("rotation")])

            normal_memory = root / "normal-memory"
            normal_memory.mkdir()
            normal_history = normal_memory / "history.json"
            normal_history.write_text('{"sentinel": true}', encoding="utf-8")
            base = PrefMemConfig(
                workspace_root=str(root),
                history_store_path=str(normal_history),
                history_outbox_path=str(normal_memory / "outbox.json"),
                preference_store_path=str(normal_memory / "preferences.json"),
            )
            contexts = []

            def factory(context):
                contexts.append(context)
                return _Orchestrator(context.config)

            result = self._evaluate(
                benchmark,
                output,
                repetitions=4,
                base_config=base,
                orchestrator_factory=factory,
                memory_mode="history-only",
                model_seed=50,
            )

            self.assertEqual(result.summary["run_count"], 4)
            self.assertEqual(result.summary["passed"], 4)
            self.assertEqual(result.summary["memory_mode"], "history-only")
            self.assertEqual(
                result.summary["model_seeds_by_repetition"],
                [50, 51, 52, 53],
            )
            overall_ci = result.summary["pass_rate_wilson_ci95"]
            self.assertEqual(overall_ci["confidence"], 0.95)
            self.assertLess(overall_ci["lower"], 1.0)
            self.assertEqual(overall_ci["upper"], 1.0)
            self.assertEqual(
                result.summary["duration_seconds"]["observations"],
                4,
            )
            self.assertIsNotNone(
                result.summary["duration_seconds"]["p50"]
            )
            self.assertGreaterEqual(
                result.summary["duration_seconds"]["p95"],
                result.summary["duration_seconds"]["p50"],
            )
            protocol_summary = result.summary["protocols"]["rotation"]
            self.assertEqual(protocol_summary["pass_rate"], 1.0)
            self.assertIsNotNone(
                protocol_summary["pass_rate_wilson_ci95"]
            )
            self.assertEqual(
                protocol_summary["duration_seconds"]["observations"],
                4,
            )
            self.assertEqual(result.summary["error_categories"], {})
            run_config = json.loads(
                (output / "run_config.json").read_text(encoding="utf-8")
            )["evaluation"]
            self.assertEqual(len(run_config["benchmark_content_sha256"]), 64)
            self.assertEqual(
                len(run_config["runtime"]["source_tree_sha256"]),
                64,
            )
            self.assertEqual(
                run_config["selector_policy"],
                "balanced-variant-seed-physical-scene-v3",
            )
            self.assertRegex(
                run_config["selection_plan_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertIn(
                "host",
                run_config["config"]["agents"]["hri"],
            )
            self.assertIn("image_width", run_config["config"]["vision"])
            lines = [
                json.loads(line)
                for line in result.results_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [line["selections"][0]["candidate_index"] for line in lines],
                [0, 1, 2, 3],
            )
            self.assertEqual(
                [
                    (
                        line["selections"][0]["scene_variant"],
                        line["selections"][0]["seed"],
                    )
                    for line in lines
                ],
                [
                    ("compact_scatter", 1),
                    ("wide_scatter", 2),
                    ("compact_scatter", 2),
                    ("wide_scatter", 1),
                ],
            )
            self.assertEqual(
                [line["model_seed"] for line in lines],
                [50, 51, 52, 53],
            )
            self.assertTrue(
                all(
                    line["report"]["steps"][0][
                        "delivered_memory_context"
                    ]["preference_ownership_verified"]
                    for line in lines
                )
            )
            self.assertEqual(
                [context.memory_mode for context in contexts],
                ["history-only"] * 4,
            )
            for context in contexts:
                self.assertTrue(
                    Path(context.config.history_store_path).is_relative_to(
                        context.run_directory
                    )
                )
                self.assertEqual(
                    {
                        context.config.hri.seed,
                        context.config.memory.seed,
                        context.config.planner.seed,
                        context.config.validator.seed,
                    },
                    {context.model_seed},
                )
            self.assertEqual(
                normal_history.read_text(encoding="utf-8"),
                '{"sentinel": true}',
            )

    def test_catches_one_protocol_failure_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "results"
            benchmark.mkdir()
            _write_benchmark(benchmark)
            _write_protocols(
                benchmark,
                [
                    _protocol("works"),
                    _protocol("times-out", message="boom"),
                ],
            )

            def factory(context):
                return _Orchestrator(
                    context.config,
                    fail=context.protocol["protocol_id"] == "times-out",
                )

            result = self._evaluate(
                benchmark,
                output,
                orchestrator_factory=factory,
                base_config=PrefMemConfig(workspace_root=str(root)),
            )

            self.assertEqual(result.summary["run_count"], 2)
            self.assertEqual(result.summary["passed"], 1)
            self.assertEqual(result.summary["errors"], 1)
            self.assertEqual(
                result.summary["error_categories"],
                {"INFRASTRUCTURE_TIMEOUT": 1},
            )
            self.assertEqual(
                result.summary["protocols"]["times-out"]["error_categories"],
                {"INFRASTRUCTURE_TIMEOUT": 1},
            )
            self.assertIsNotNone(
                result.summary["protocols"]["works"][
                    "pass_rate_wilson_ci95"
                ]
            )
            self.assertEqual(
                result.summary["duration_seconds"]["observations"],
                2,
            )
            records = [
                json.loads(line)
                for line in result.results_path.read_text(encoding="utf-8").splitlines()
            ]
            failure = next(
                record for record in records if record["protocol_id"] == "times-out"
            )
            self.assertEqual(failure["status"], "error")
            self.assertEqual(failure["error"]["type"], "TimeoutError")
            self.assertEqual(
                failure["error"]["category"],
                "INFRASTRUCTURE_TIMEOUT",
            )
            self.assertIn("TimeoutError", failure["error"]["traceback"])
            self.assertTrue(result.summary_path.is_file())

    def test_incomplete_production_artifacts_fail_protocol_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "results"
            benchmark.mkdir()
            _write_benchmark(benchmark)
            _write_protocols(benchmark, [_protocol("artifact-gate")])

            def factory(context):
                orchestrator = _Orchestrator(context.config)
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

            result = self._evaluate(
                benchmark,
                output,
                orchestrator_factory=factory,
                base_config=PrefMemConfig(workspace_root=str(root)),
            )

            self.assertEqual(result.summary["passed"], 0)
            self.assertEqual(
                result.summary["failure_categories"],
                {"ARTIFACT_INCOMPLETE": 1},
            )
            self.assertEqual(
                result.summary["evaluation_artifacts"],
                {
                    "tracked_runs": 1,
                    "complete_runs": 0,
                    "incomplete_runs": 1,
                    "complete_rate": 0.0,
                },
            )
            record = json.loads(
                result.results_path.read_text(encoding="utf-8").strip()
            )
            self.assertEqual(record["status"], "failed")
            self.assertFalse(record["passed"])
            self.assertTrue(record["report"]["passed"])
            self.assertEqual(
                record["failure_category"],
                "ARTIFACT_INCOMPLETE",
            )
            self.assertTrue(record["artifact_status"]["tracked"])
            self.assertFalse(record["artifact_status"]["complete"])

    def test_fixture_is_applied_through_consent_aware_agent(self) -> None:
        fixture = memory_protocol_fixtures()["approved-rgb-stack-preference"]
        memory_agent = _MemoryAgent()
        orchestrator = SimpleNamespace(
            memory_agent=MemoryAblationAgent(memory_agent, "history-only")
        )

        apply_consent_fixture(
            "approved-rgb-stack-preference",
            fixture,
            orchestrator,
        )

        self.assertEqual(len(memory_agent.fixture_calls), 1)
        call = memory_agent.fixture_calls[0]
        self.assertIsInstance(call["consent"], ConsentEvidence)
        self.assertEqual(
            call["consent"].proposal,
            fixture["preference_request"],
        )
        self.assertEqual(
            memory_agent.preference_repository.list_preferences("participant-a")[0][
                "statement"
            ],
            fixture["preference_request"]["preference"]["statement"],
        )

    def test_protocol_fixture_seeds_identical_state_before_recording_ablation(
        self,
    ) -> None:
        fixture_id = "approved-rgb-stack-preference"
        fixtures = memory_protocol_fixtures()
        stored_states: list[list[dict[str, Any]]] = []

        for mode in ("full", "history-only"):
            config = PrefMemConfig(user_id="participant-a")
            orchestrator = _Orchestrator(config)
            underlying = orchestrator.memory_agent
            if mode != "full":
                orchestrator.memory_agent = MemoryAblationAgent(
                    underlying,
                    mode,
                )
            report = run_memory_protocol(
                _protocol(
                    f"fixture-{mode}",
                    fixture_id=fixture_id,
                ),
                orchestrator,
                resolve_scenario=lambda _selector: Path("unused"),
                apply_fixture=apply_consent_fixture,
                fixtures=fixtures,
            )

            self.assertTrue(report.passed, report.steps[0].checks)
            self.assertEqual(len(underlying.fixture_calls), 1)
            stored_states.append(
                underlying.preference_repository.list_preferences(
                    "participant-a",
                    include_inactive=True,
                )
            )

        self.assertEqual(stored_states[0], stored_states[1])
        self.assertEqual(len(stored_states[0]), 1)

    def test_fixture_rejects_consent_proposal_drift(self) -> None:
        fixture = memory_protocol_fixtures()["approved-rgb-stack-preference"]
        fixture["consent"]["proposal"]["instruction"] = "A different request."
        memory_agent = _MemoryAgent()
        orchestrator = SimpleNamespace(memory_agent=memory_agent)

        with self.assertRaisesRegex(
            ProtocolEvaluationError,
            "not bound to the exact request",
        ):
            apply_consent_fixture("drifted", fixture, orchestrator)

        self.assertEqual(memory_agent.fixture_calls, [])

    def test_resumes_without_rerunning_completed_protocols(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "results"
            benchmark.mkdir()
            _write_benchmark(benchmark)
            _write_protocols(benchmark, [_protocol("once")])
            factory = lambda context: _Orchestrator(context.config)

            first = self._evaluate(
                benchmark,
                output,
                orchestrator_factory=factory,
                base_config=PrefMemConfig(workspace_root=str(root)),
            )
            resumed = self._evaluate(
                benchmark,
                output,
                orchestrator_factory=factory,
                base_config=PrefMemConfig(workspace_root=str(root)),
            )

            self.assertEqual(first.executed_runs, 1)
            self.assertEqual(resumed.executed_runs, 0)
            self.assertEqual(resumed.skipped_runs, 1)
            self.assertEqual(
                len(
                    resumed.results_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ),
                1,
            )
            with self.assertRaisesRegex(
                ProtocolEvaluationError,
                "resume is disabled",
            ):
                self._evaluate(
                    benchmark,
                    output,
                    orchestrator_factory=factory,
                    base_config=PrefMemConfig(workspace_root=str(root)),
                    resume=False,
                )

    def test_resume_rejects_corrupt_protocol_selection_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "results"
            benchmark.mkdir()
            _write_benchmark(benchmark)
            _write_protocols(benchmark, [_protocol("corrupt-resume")])
            factory = lambda context: _Orchestrator(context.config)

            self._evaluate(
                benchmark,
                output,
                orchestrator_factory=factory,
                base_config=PrefMemConfig(workspace_root=str(root)),
            )
            result_path = output / "results.jsonl"
            record = json.loads(result_path.read_text(encoding="utf-8"))
            record["selections"][0]["seed"] = 999
            result_path.write_text(
                json.dumps(record, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ProtocolEvaluationError,
                "inconsistent selections",
            ):
                self._evaluate(
                    benchmark,
                    output,
                    orchestrator_factory=factory,
                    base_config=PrefMemConfig(workspace_root=str(root)),
                )

    def test_error_resume_accepts_a_valid_selection_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "results"
            benchmark.mkdir()
            _write_benchmark(benchmark)
            protocol = _protocol("prefix-error")
            protocol["steps"].append(
                {
                    "query": "Stack them again.",
                    "scenario_selector": {
                        "family": "block_stack",
                        "target_id": "rgb_bottom_to_top",
                        "outcome": "success",
                        "variant": "compact_scatter",
                    },
                    "expected": {
                        "clarification_required": True,
                        "history_delta": 0,
                        "preference_delta": 0,
                    },
                }
            )
            _write_protocols(benchmark, [protocol])
            calls = 0

            def factory(context):
                nonlocal calls
                calls += 1
                return _Orchestrator(context.config, fail=True)

            first = self._evaluate(
                benchmark,
                output,
                orchestrator_factory=factory,
                base_config=PrefMemConfig(workspace_root=str(root)),
            )
            resumed = self._evaluate(
                benchmark,
                output,
                orchestrator_factory=factory,
                base_config=PrefMemConfig(workspace_root=str(root)),
            )

            self.assertEqual(first.executed_runs, 1)
            self.assertEqual(resumed.executed_runs, 0)
            self.assertEqual(resumed.skipped_runs, 1)
            self.assertEqual(calls, 1)
            record = json.loads(
                first.results_path.read_text(encoding="utf-8")
            )
            self.assertEqual(record["status"], "error")
            self.assertEqual(len(record["selections"]), 1)

    def test_invalid_benchmark_stops_before_factory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "results"
            benchmark.mkdir()
            factory_calls: list[Any] = []

            with patch(
                "simulation.benchmark.protocol_evaluation.validate_benchmark",
                return_value=SimpleNamespace(
                    valid=False,
                    errors=("manifest digest mismatch",),
                ),
            ):
                with self.assertRaisesRegex(
                    ProtocolEvaluationError,
                    "manifest digest mismatch",
                ):
                    evaluate_memory_protocols(
                        benchmark,
                        output,
                        orchestrator_factory=lambda context: factory_calls.append(
                            context
                        ),
                    )

            self.assertEqual(factory_calls, [])
            self.assertFalse(output.exists())

    def test_malformed_protocol_stops_before_factory_or_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "results"
            benchmark.mkdir()
            _write_benchmark(benchmark)
            malformed = _protocol("malformed")
            malformed["steps"][0]["expected"] = {
                "unsupported_expectation": True
            }
            _write_protocols(benchmark, [malformed])
            factory_calls: list[Any] = []
            fixture_calls: list[Any] = []

            with self.assertRaisesRegex(
                ProtocolEvaluationError,
                "unsupported expectations",
            ):
                self._evaluate(
                    benchmark,
                    output,
                    orchestrator_factory=lambda context: factory_calls.append(
                        context
                    ),
                    fixture_applier=lambda *args: fixture_calls.append(args),
                    base_config=PrefMemConfig(workspace_root=str(root)),
                )

            self.assertEqual(factory_calls, [])
            self.assertEqual(fixture_calls, [])
            self.assertFalse(output.exists())

    def test_preflight_rejects_semantic_score_before_selector_and_negative_dispatches(
        self,
    ) -> None:
        cases = {
            "semantic-before-selector": (
                "before any scenario selector",
                {
                    "query": "Do the task.",
                    "expected": {"task_semantic_score": True},
                },
            ),
            "negative-min-dispatches": (
                "must be non-negative",
                {
                    "query": "Stack the blocks.",
                    "scenario_selector": {
                        "family": "block_stack",
                        "target_id": "rgb_bottom_to_top",
                        "outcome": "success",
                    },
                    "expected": {"min_dispatches": -1},
                },
            ),
        }
        for name, (message, invalid_step) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                benchmark = root / "benchmark"
                output = root / "results"
                benchmark.mkdir()
                _write_benchmark(benchmark)
                protocol = _protocol(f"preflight-{name}")
                if name == "semantic-before-selector":
                    protocol["steps"].insert(0, invalid_step)
                else:
                    protocol["steps"][0] = invalid_step
                _write_protocols(benchmark, [protocol])
                factory_calls: list[Any] = []
                fixture_calls: list[Any] = []

                with self.assertRaisesRegex(ProtocolEvaluationError, message):
                    self._evaluate(
                        benchmark,
                        output,
                        orchestrator_factory=lambda context: factory_calls.append(
                            context
                        ),
                        fixture_applier=lambda *args: fixture_calls.append(args),
                        base_config=PrefMemConfig(workspace_root=str(root)),
                    )

                self.assertEqual(factory_calls, [])
                self.assertEqual(fixture_calls, [])
                self.assertFalse(output.exists())

    def test_resume_rejects_incomplete_or_inconsistent_report_evidence(
        self,
    ) -> None:
        mutations = {
            "empty-steps": (
                "incomplete report.steps",
                lambda record: record["report"].update({"steps": []}),
            ),
            "missing-check": (
                "incomplete.*report step",
                lambda record: record["report"]["steps"][0].update(
                    {
                        "checks": [
                            check
                            for check in record["report"]["steps"][0]["checks"]
                            if check["name"] != "clarification_required"
                        ]
                    }
                ),
            ),
            "record-pass-drift": (
                "inconsistent passed status",
                lambda record: record.update(
                    {"status": "failed", "passed": False}
                ),
            ),
        }
        for name, (message, mutate) in mutations.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                benchmark = root / "benchmark"
                output = root / "results"
                benchmark.mkdir()
                _write_benchmark(benchmark)
                _write_protocols(
                    benchmark,
                    [_protocol(f"resume-{name}")],
                )
                factory = lambda context: _Orchestrator(context.config)
                first = self._evaluate(
                    benchmark,
                    output,
                    orchestrator_factory=factory,
                    base_config=PrefMemConfig(workspace_root=str(root)),
                )
                record = json.loads(
                    first.results_path.read_text(encoding="utf-8")
                )
                mutate(record)
                first.results_path.write_text(
                    json.dumps(record, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                with self.assertRaisesRegex(ProtocolEvaluationError, message):
                    self._evaluate(
                        benchmark,
                        output,
                        orchestrator_factory=factory,
                        base_config=PrefMemConfig(workspace_root=str(root)),
                    )

    def test_preflight_rejects_cross_selector_physical_scene_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "results"
            benchmark.mkdir()
            _write_benchmark(benchmark)
            protocol = _protocol("scene-drift")
            protocol["steps"][0]["scenario_selector"]["variant"] = (
                "compact_scatter"
            )
            protocol["steps"].append(
                {
                    "query": "Stack the blocks again.",
                    "scenario_selector": {
                        "family": "block_stack",
                        "target_id": "rgb_bottom_to_top",
                        "outcome": "success",
                        "variant": "wide_scatter",
                    },
                    "expected": {"clarification_required": True},
                }
            )
            _write_protocols(benchmark, [protocol])
            factory_calls: list[Any] = []
            fixture_calls: list[Any] = []

            with self.assertRaisesRegex(
                ProtocolEvaluationError,
                "drift across physical scenes",
            ):
                self._evaluate(
                    benchmark,
                    output,
                    repetitions=2,
                    orchestrator_factory=lambda context: factory_calls.append(
                        context
                    ),
                    fixture_applier=lambda *args: fixture_calls.append(args),
                    base_config=PrefMemConfig(workspace_root=str(root)),
                )

            self.assertEqual(factory_calls, [])
            self.assertEqual(fixture_calls, [])
            self.assertFalse(output.exists())

    def test_scene_cluster_bootstrap_groups_cross_target_same_scene(self) -> None:
        records = [
            {
                "protocol_id": "physical-scene",
                "repetition": 1,
                "passed": True,
                "selections": [
                    {
                        "scenario_id": "ep-rgb-a",
                        "family": "block_stack",
                        "scene_variant": "wide_scatter",
                        "seed": 7,
                        "counterfactual_group_id": "scene-a",
                        "target_id": "rgb_bottom_to_top",
                    }
                ],
            },
            {
                "protocol_id": "physical-scene",
                "repetition": 2,
                "passed": False,
                "selections": [
                    {
                        "scenario_id": "ep-bgr-a",
                        "family": "block_stack",
                        "scene_variant": "wide_scatter",
                        "seed": 7,
                        "counterfactual_group_id": "scene-a",
                        "target_id": "bgr_bottom_to_top",
                    }
                ],
            },
            {
                "protocol_id": "physical-scene",
                "repetition": 3,
                "passed": True,
                "selections": [
                    {
                        "scenario_id": "ep-rgb-b",
                        "family": "block_stack",
                        "scene_variant": "wide_scatter",
                        "seed": 8,
                        "counterfactual_group_id": "",
                        "target_id": "rgb_bottom_to_top",
                    }
                ],
            },
            {
                "protocol_id": "physical-scene",
                "repetition": 4,
                "passed": True,
                "selections": [
                    {
                        "scenario_id": "ep-bgr-b",
                        "family": "block_stack",
                        "scene_variant": "wide_scatter",
                        "seed": 8,
                        "target_id": "bgr_bottom_to_top",
                    }
                ],
            },
        ]

        interval = _protocol_cluster_bootstrap_ci95(records)

        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertEqual(interval["clusters"], 2)
        self.assertEqual(
            interval["cluster_unit"],
            (
                "core:family_scene_variant_seed;"
                "control:family_scene_variant_seed_target;"
                "counterfactual_group_fallback"
            ),
        )

        control_interval = _protocol_cluster_bootstrap_ci95(
            [
                {
                    "protocol_id": "controls",
                    "repetition": 1,
                    "passed": True,
                    "selections": [
                        {
                            "scenario_id": "control-rgb",
                            "family": "block_stack",
                            "scene_variant": "control_already_satisfied",
                            "seed": 7,
                            "target_id": "rgb_bottom_to_top",
                            "control_kind": "already_satisfied",
                        }
                    ],
                },
                {
                    "protocol_id": "controls",
                    "repetition": 2,
                    "passed": True,
                    "selections": [
                        {
                            "scenario_id": "control-bgr",
                            "family": "block_stack",
                            "scene_variant": "control_already_satisfied",
                            "seed": 7,
                            "target_id": "bgr_bottom_to_top",
                            "control_kind": "already_satisfied",
                        }
                    ],
                },
            ]
        )
        self.assertIsNotNone(control_interval)
        assert control_interval is not None
        self.assertEqual(control_interval["cluster_count"], 2)

    def test_unexpected_auto_semantic_checks_do_not_inflate_accuracy(
        self,
    ) -> None:
        protocol = {
            "protocol_id": "aligned-checks",
            "user_id": "participant-a",
            "steps": [
                {
                    "query": "First.",
                    "scenario_selector": {"scenario_id": "scene-a"},
                    "expected": {"clarification_required": False},
                },
                {
                    "query": "Second.",
                    "scenario_selector": {"scenario_id": "scene-a"},
                    "expected": {"task_semantic_score": True},
                },
            ],
        }

        def check(name: str) -> dict[str, Any]:
            return {
                "name": name,
                "expected": True,
                "actual": True,
                "evaluated": True,
                "passed": True,
            }

        report_steps = []
        for index in (1, 2):
            report_steps.append(
                {
                    "index": index,
                    "checks": [check("task_semantic_score")],
                    "task_score": {
                        "checks": [check("hri_resolved_target")]
                    },
                }
            )
        record = {
            "protocol_id": "aligned-checks",
            "repetition": 1,
            "status": "passed",
            "passed": True,
            "failure_category": None,
            "artifact_status": {"tracked": False, "complete": None},
            "agent_failures": [],
            "report": {"steps": report_steps},
            "selections": [],
        }
        summary = _build_summary(
            records=[record],
            benchmark_root=Path("benchmark"),
            protocol_path=Path("protocols.json"),
            protocol_digest="digest",
            index={},
            repetitions=1,
            config=PrefMemConfig(),
            results_path=Path("results.jsonl"),
            memory_mode="full",
            model_seed=None,
            planned_runs=1,
            executed_runs=1,
            skipped_runs=0,
            protocols=[protocol],
        )

        for name in (
            "task_semantic_score",
            "task_score.hri_resolved_target",
        ):
            counts = summary["checks"][name]
            self.assertEqual(counts["expected"], 1)
            self.assertEqual(counts["total"], 2)
            self.assertEqual(counts["expected_evidence"], 1)
            self.assertEqual(counts["unexpected_evidence"], 1)
            self.assertEqual(counts["excess_evidence"], 1)
            self.assertEqual(counts["coverage_rate"], 1.0)
            self.assertEqual(counts["unconditional_accuracy"], 1.0)
            self.assertLessEqual(counts["coverage_rate"], 1.0)
            self.assertLessEqual(counts["unconditional_accuracy"], 1.0)

    def test_subagent_failure_summary_separates_events_from_trials(self) -> None:
        failures = [
            {
                "stage": "planner",
                "error_type": "TimeoutError",
                "classification": "INFRASTRUCTURE_TIMEOUT",
                "message": "attempt one",
            },
            {
                "stage": "planner",
                "error_type": "TimeoutError",
                "classification": "INFRASTRUCTURE_TIMEOUT",
                "message": "attempt two",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "results"
            benchmark.mkdir()
            _write_benchmark(benchmark)
            _write_protocols(benchmark, [_protocol("retry-events")])

            with patch(
                "simulation.benchmark.protocol_evaluation._agent_failure_evidence",
                return_value=copy.deepcopy(failures),
            ):
                result = self._evaluate(
                    benchmark,
                    output,
                    orchestrator_factory=lambda context: _Orchestrator(
                        context.config
                    ),
                    base_config=PrefMemConfig(workspace_root=str(root)),
                )

        overall = result.summary["subagent_failures"]
        self.assertEqual(overall["event_count"], 2)
        self.assertEqual(overall["affected_trial_count"], 1)
        self.assertEqual(overall["events"]["total"], 2)
        self.assertEqual(overall["affected_trials"]["total"], 1)
        self.assertEqual(
            overall["events_by_classification"],
            {"INFRASTRUCTURE_TIMEOUT": 2},
        )
        self.assertEqual(
            overall["affected_trials_by_classification"],
            {"INFRASTRUCTURE_TIMEOUT": 1},
        )
        protocol = result.summary["protocols"]["retry-events"][
            "subagent_failures"
        ]
        self.assertEqual(protocol["event_count"], 2)
        self.assertEqual(protocol["affected_trial_count"], 1)
        self.assertEqual(protocol["events"]["total"], 2)
        self.assertEqual(protocol["affected_trials"]["total"], 1)
        self.assertEqual(protocol["affected_trial_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
