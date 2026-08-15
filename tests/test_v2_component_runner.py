from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from experiments_suite_v2.io import iter_jsonl, load_json, sha256_file
from experiments_suite_v2.runners.component import (
    COMPONENT_CASES_PATH,
    CapabilityResponseError,
    CallableFixtureAdapter,
    ComponentRunnerError,
    InterfacePipelineAdapter,
    LiveHRIAdapter,
    LiveMemoryAdapter,
    LiveMonitorAdapter,
    LivePlannerAdapter,
    LiveValidatorAdapter,
    expand_component_trials,
    load_component_registry,
    registered_component_aim_order,
    run_component_trial,
    validate_component_registry,
    write_component_traces,
)
from experiments_suite_v2.schemas import (
    AnalyticalClassification,
    ArtifactProfile,
    FormalVerdict,
    canonical_sha256,
)


class ComponentRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_component_registry()
        cls.trials = expand_component_trials(
            cls.registry,
            software_config_commit="component-test-commit",
        )
        cls.by_aim = defaultdict(list)
        for trial in cls.trials:
            cls.by_aim[trial.case_definition.aim_id].append(trial)

    def test_registry_is_exactly_40_aims_in_authoritative_order(self) -> None:
        expected = registered_component_aim_order()
        self.assertEqual(len(expected), 40)
        self.assertEqual(
            tuple(item["aim_id"] for item in self.registry["aims"]),
            expected,
        )
        self.assertEqual(expected[0], "C-HRI-ID")
        self.assertEqual(expected[-1], "I-HMP-PREF")

    def test_candidate_registry_is_hash_pinned_by_protocol_manifest(self) -> None:
        manifest = load_json(COMPONENT_CASES_PATH.parent / "protocol_manifest.json")
        record = next(
            item
            for item in manifest["candidate_freeze_inputs"]
            if item["path"] == "protocol/cases_component.json"
        )
        self.assertEqual(record["status"], "FROZEN_CANDIDATE")
        self.assertEqual(record["sha256"], sha256_file(COMPONENT_CASES_PATH))

    def test_expansion_is_8_contexts_by_5_variants_per_aim(self) -> None:
        self.assertEqual(len(self.trials), 1_600)
        self.assertEqual(len({trial.trial_id for trial in self.trials}), 1_600)
        self.assertEqual(
            [trial.order_index for trial in self.trials],
            list(range(1_600)),
        )
        for aim_id in registered_component_aim_order():
            rows = self.by_aim[aim_id]
            self.assertEqual(len(rows), 40)
            self.assertEqual(
                [(row.context_id, row.variant_id) for row in rows],
                [
                    (f"{context:02d}", f"{variant:02d}")
                    for context in range(1, 9)
                    for variant in range(1, 6)
                ],
            )
            self.assertTrue(
                all(row.case_definition is rows[0].case_definition for row in rows)
            )

    def test_exact_utterances_fixtures_scorers_and_seeds_are_frozen(self) -> None:
        variant_seeds = {
            item["variant_id"]: item["seed"]
            for item in self.registry["variants"]
        }
        registry_aims = {item["aim_id"]: item for item in self.registry["aims"]}
        for aim_id, rows in self.by_aim.items():
            aim = registry_aims[aim_id]
            self.assertEqual(aim["primary_oracle"]["aim_id"], aim_id)
            self.assertEqual(aim["primary_oracle"]["aggregation"], "ALL_MUST_PASS")
            contexts = {item["context_id"]: item for item in aim["contexts"]}
            for row in rows:
                context = contexts[row.context_id]
                self.assertEqual(
                    row.trial_tuple.exact_utterance,
                    context["exact_utterances"][row.variant_id],
                )
                self.assertEqual(
                    row.trial_tuple.random_seed,
                    variant_seeds[row.variant_id],
                )
                self.assertNotIn("$variant_id", json.dumps(dict(row.fixtures)))
                self.assertNotIn("$seed", json.dumps(dict(row.fixtures)))
                if row.fixtures["frame"] != "NOT_APPLICABLE":
                    self.assertEqual(
                        row.fixtures["frame"]["camera_role"],
                        "prefmem_third_person_camera",
                    )
                self.assertEqual(row.oracle.aim_id, aim_id)
                self.assertEqual(
                    tuple(item.name for item in row.oracle.predicates),
                    row.case_definition.must_pass,
                )
            for context_id in contexts:
                hashes = {
                    canonical_sha256(dict(row.fixtures))
                    for row in rows
                    if row.context_id == context_id
                }
                self.assertEqual(len(hashes), 5)

    def test_registry_contains_no_observed_or_invented_results(self) -> None:
        forbidden = {
            "actual_result",
            "analytical_classification",
            "formal_verdict",
            "observed_result",
        }

        def visit(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(self.registry)
        self.assertFalse(self.registry["observed_results_included"])

    def test_profiles_and_required_traces_follow_live_boundaries(self) -> None:
        for aim_id, rows in self.by_aim.items():
            case = rows[0].case_definition
            self.assertIn(
                case.artifact_profile,
                {ArtifactProfile.MODEL_COMPONENT, ArtifactProfile.VISUAL_REPLAY},
            )
            self.assertIn("component_trace.jsonl", case.required_artifacts)
            self.assertIn("model_calls.jsonl", case.required_artifacts)
            if "Memory" in case.live_components:
                self.assertIn("memory_trace.jsonl", case.required_artifacts)
            if case.artifact_profile is ArtifactProfile.VISUAL_REPLAY:
                self.assertIn("event_trace.jsonl", case.required_artifacts)

    def test_registry_rejects_a_cross_aim_primary_oracle(self) -> None:
        # Reload from disk so the shared class fixture remains immutable.
        registry = json.loads(COMPONENT_CASES_PATH.read_text(encoding="utf-8"))
        registry["aims"][0]["primary_oracle"]["aim_id"] = "C-HRI-VIS"
        with self.assertRaisesRegex(ComponentRunnerError, "explicitly scoped"):
            validate_component_registry(registry)


class ComponentVerdictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trials = expand_component_trials(
            software_config_commit="component-verdict-test",
        )

    def setUp(self) -> None:
        self.trial = self.trials[0]
        self.expected = self.trial.oracle.expected_values["oracle_label"]

    def test_deterministic_pass_and_capability_fail_are_single_attempts(self) -> None:
        passing = CallableFixtureAdapter(
            lambda _trial: {"oracle_label": self.expected}
        )
        passed = run_component_trial(self.trial, passing)
        self.assertEqual(passed.result.formal_verdict, FormalVerdict.PASS)
        self.assertEqual(
            passed.result.analytical_classification,
            AnalyticalClassification.PASS,
        )
        self.assertEqual(passing.calls, [self.trial.trial_id])
        self.assertEqual(
            [event["event_type"] for event in passed.trace],
            ["COMPONENT_CALL_STARTED", "COMPONENT_CALL_COMPLETED"],
        )
        self.assertEqual(
            passed.trace[0]["payload"]["invocation"],
            dict(self.trial.invocation),
        )
        self.assertEqual(
            passed.trace[1]["payload"]["result"],
            {"oracle_label": self.expected},
        )
        self.assertEqual(passed.trace[0]["call_id"], passed.trace[1]["call_id"])

        failing = CallableFixtureAdapter(
            lambda _trial: {"oracle_label": "wrong-boundary-result"}
        )
        failed = run_component_trial(self.trial, failing)
        self.assertEqual(failed.result.formal_verdict, FormalVerdict.FAIL)
        self.assertEqual(
            failed.result.analytical_classification,
            AnalyticalClassification.CAPABILITY_FAIL,
        )
        self.assertEqual(failing.calls, [self.trial.trial_id])
        self.assertFalse(failed.result.metrics["automatic_retry_performed"])
        self.assertIsNone(failed.result.invalidity)

    def test_schema_capability_error_is_not_retried_or_invalidated(self) -> None:
        def malformed(_trial):
            raise CapabilityResponseError("healthy endpoint returned malformed JSON")

        adapter = CallableFixtureAdapter(malformed)
        evaluation = run_component_trial(self.trial, adapter)
        self.assertEqual(adapter.calls, [self.trial.trial_id])
        self.assertEqual(
            evaluation.result.analytical_classification,
            AnalyticalClassification.CAPABILITY_FAIL,
        )
        self.assertFalse(evaluation.result.metrics["automatic_retry_performed"])
        self.assertIsNone(evaluation.result.invalidity)

    def test_unplanned_live_timeout_is_invalid_run_with_exact_failure_trace(self) -> None:
        class OfflineMonitor:
            def assess(self, *, task, frame):
                del task, frame
                raise TimeoutError("forwarded endpoint timed out")

        trial = replace(
            self.trial,
            invocation={
                "operation": "assess",
                "arguments": {"task": {"publication_id": "p1"}, "frame": {}},
            },
        )
        adapter = LiveMonitorAdapter(
            OfflineMonitor(),
            service_id="gemma-hri-planner-monitor-validator-v1",
        )
        evaluation = run_component_trial(trial, adapter)
        self.assertEqual(adapter.calls, [trial.trial_id])
        self.assertEqual(
            evaluation.result.analytical_classification,
            AnalyticalClassification.INVALID_RUN,
        )
        self.assertEqual(evaluation.result.formal_verdict, FormalVerdict.FAIL)
        self.assertEqual(
            evaluation.result.invalidity.reason_code,
            "MODEL_ENDPOINT_TIMEOUT",
        )
        self.assertEqual(
            [item["event_type"] for item in evaluation.trace],
            ["COMPONENT_CALL_STARTED", "COMPONENT_CALL_FAILED"],
        )
        self.assertEqual(evaluation.trace[0]["call_id"], evaluation.trace[1]["call_id"])
        self.assertTrue(all(not value for value in evaluation.result.must_pass.values()))
        self.assertFalse(evaluation.result.metrics["automatic_retry_performed"])

    def test_raw_third_party_connection_error_is_also_invalid_run(self) -> None:
        class ThirdPartyAdapter:
            component_name = "THIRD_PARTY"
            service_id = "fixture-service"

            def __init__(self):
                self.calls = 0

            def invoke(self, trial):
                del trial
                self.calls += 1
                raise ConnectionError("SSH forwarding disconnected")

        adapter = ThirdPartyAdapter()
        evaluation = run_component_trial(self.trial, adapter)
        self.assertEqual(adapter.calls, 1)
        self.assertEqual(
            evaluation.result.analytical_classification,
            AnalyticalClassification.INVALID_RUN,
        )
        self.assertEqual(
            evaluation.result.invalidity.reason_code,
            "MODEL_ENDPOINT_CONNECTION_LOST",
        )

    def test_oracle_cannot_be_reused_across_aims(self) -> None:
        other = next(
            trial
            for trial in self.trials
            if trial.case_definition.aim_id == "C-HRI-VIS"
        )
        with self.assertRaisesRegex(ComponentRunnerError, "oracle.*aim IDs differ"):
            replace(self.trial, oracle=other.oracle)


class InterfaceAndLiveSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        trials = expand_component_trials(
            software_config_commit="component-interface-test",
        )
        cls.direct = trials[0]
        cls.interface = next(
            trial
            for trial in trials
            if trial.case_definition.aim_id == "I-HP-CONTRACT"
        )

    def test_interface_pipeline_resolves_typed_step_references_and_traces_each_call(self) -> None:
        expected = self.interface.oracle.expected_values["oracle_label"]
        observed = []

        def first(trial):
            observed.append(("HRI", dict(trial.invocation)))
            return {"goal_contract": {"goal_id": "goal-1"}}

        def second(trial):
            observed.append(("Planner", dict(trial.invocation)))
            self.assertEqual(
                trial.invocation["arguments"]["goal_contract"],
                {"goal_id": "goal-1"},
            )
            return {"oracle_label": expected}

        trial = replace(
            self.interface,
            invocation={
                "steps": [
                    {
                        "step_id": "hri_contract",
                        "component": "HRI",
                        "invocation": {
                            "operation": "respond",
                            "arguments": {"utterance": "frozen utterance"},
                        },
                    },
                    {
                        "step_id": "planner_consume",
                        "component": "Planner",
                        "invocation": {
                            "operation": "preview",
                            "arguments": {
                                "goal_contract": "$steps.hri_contract.goal_contract"
                            },
                        },
                    },
                ]
            },
        )
        adapter = InterfacePipelineAdapter(
            {
                "HRI": CallableFixtureAdapter(first, component_name="HRI"),
                "Planner": CallableFixtureAdapter(second, component_name="Planner"),
            }
        )
        evaluation = run_component_trial(trial, adapter)
        self.assertEqual([name for name, _call in observed], ["HRI", "Planner"])
        self.assertEqual(
            evaluation.output["final"],
            {"oracle_label": expected},
        )
        self.assertEqual(
            evaluation.result.analytical_classification,
            AnalyticalClassification.PASS,
        )
        self.assertEqual(
            [event["event_type"] for event in evaluation.trace],
            [
                "COMPONENT_CALL_STARTED",
                "COMPONENT_CALL_COMPLETED",
                "COMPONENT_CALL_STARTED",
                "COMPONENT_CALL_COMPLETED",
            ],
        )

    def test_live_hri_planner_memory_monitor_validator_invocation_seams(self) -> None:
        class Metrics:
            def __init__(self):
                self.trace_sink = None

            def set_trace_sink(self, sink):
                self.trace_sink = sink

            def emit(self, component):
                if self.trace_sink is not None:
                    self.trace_sink({"component": component, "exact": True})

        class HRI:
            def __init__(self):
                self.metrics = Metrics()
                self.calls = []

            def invoke_agent(self, messages, current_frame):
                self.calls.append((messages[0].content, current_frame))
                self.metrics.emit("HRI")
                return {"messages": [{"content": "hri-result"}], "llm_calls": 1}

        class Planner:
            def __init__(self):
                self.calls = []

            def preview(self, clarified_goal, current_frame, **kwargs):
                self.calls.append((clarified_goal, current_frame, kwargs))
                return {"component": "Planner", "accepted": True}

        class Memory:
            def __init__(self):
                self.calls = []
                self.agent_calls = []
                self._trace_sink = None

            def set_trace_sink(self, sink):
                self._trace_sink = sink

            def retrieve(self, *, query):
                self.calls.append(query)
                if self._trace_sink is not None:
                    self._trace_sink({"stage": "final", "ids": ["M1"]})
                return [{"id": "M1", "text": "memory"}]

            def invoke_agent(self, messages, current_frame):
                self.agent_calls.append((messages[0].content, current_frame))
                return {"messages": [{"content": "memory-agent-result"}]}

        class Monitor:
            def __init__(self):
                self.calls = []

            def assess(self, *, task, frame):
                self.calls.append((task, frame))
                return {"component": "Monitor", "task_status": "ONGOING"}

        class Validator:
            def __init__(self):
                self.calls = []

            def assess_contract(self, contract, frame, **kwargs):
                self.calls.append((contract, frame, kwargs))
                return {"component": "Validator", "outcome": "INCOMPLETE"}

        hri = HRI()
        hri_trial = replace(
            self.direct,
            invocation={
                "operation": "agent_turn",
                "arguments": {
                    "utterance": "exact hri utterance",
                    "frame": {"type": "image_url", "image_url": {"url": "fixture://frame"}},
                },
            },
        )
        hri_capture = LiveHRIAdapter(hri).invoke(hri_trial)
        self.assertEqual(hri.calls[0][0], "exact hri utterance")
        self.assertEqual(
            hri.calls[0][1],
            {"type": "image_url", "image_url": {"url": "fixture://frame"}},
        )
        self.assertEqual(hri_capture.output["content"], "hri-result")
        self.assertEqual(len(hri_capture.model_calls), 1)

        planner = Planner()
        planner_trial = replace(
            self.direct,
            invocation={
                "operation": "preview",
                "arguments": {
                    "utterance": "place red on white",
                    "frame": {"fixture_id": "F1"},
                    "constraints": ["preserve blue"],
                },
            },
        )
        planner_capture = LivePlannerAdapter(planner).invoke(planner_trial)
        self.assertEqual(
            planner.calls,
            [("place red on white", {"fixture_id": "F1"}, {"constraints": ["preserve blue"]})],
        )
        self.assertTrue(planner_capture.output["accepted"])

        memory = Memory()
        memory_trial = replace(
            self.direct,
            invocation={"operation": "retrieve", "arguments": {"query": ["q1", "q2"]}},
        )
        memory_capture = LiveMemoryAdapter(memory).invoke(memory_trial)
        self.assertEqual(memory.calls, [["q1", "q2"]])
        self.assertEqual(memory_capture.output["items"][0]["id"], "M1")
        self.assertEqual(len(memory_capture.memory_traces), 1)

        memory_agent_trial = replace(
            self.direct,
            invocation={
                "operation": "agent_turn",
                "arguments": {
                    "utterance": "retrieve my board preference",
                    "frame": "NOT_APPLICABLE",
                },
            },
        )
        memory_agent_capture = LiveMemoryAdapter(memory).invoke(memory_agent_trial)
        self.assertEqual(
            memory.agent_calls[0][0],
            "retrieve my board preference",
        )
        self.assertEqual(
            memory_agent_capture.output["content"],
            "memory-agent-result",
        )

        monitor = Monitor()
        monitor_trial = replace(
            self.direct,
            invocation={
                "operation": "assess",
                "arguments": {"task": {"publication_id": "P1"}, "frame": {"id": "F2"}},
            },
        )
        monitor_capture = LiveMonitorAdapter(monitor).invoke(monitor_trial)
        self.assertEqual(monitor.calls, [({"publication_id": "P1"}, {"id": "F2"})])
        self.assertEqual(monitor_capture.output["task_status"], "ONGOING")

        validator = Validator()
        validator_trial = replace(
            self.direct,
            invocation={
                "operation": "assess_contract",
                "arguments": {
                    "validation_contract": {"validation_id": "V1"},
                    "frame": {"id": "F3"},
                    "publication_id": "P2",
                },
            },
        )
        validator_capture = LiveValidatorAdapter(validator).invoke(validator_trial)
        self.assertEqual(
            validator.calls,
            [({"validation_id": "V1"}, {"id": "F3"}, {"publication_id": "P2"})],
        )
        self.assertEqual(validator_capture.output["outcome"], "INCOMPLETE")

        for capture in (
            hri_capture,
            planner_capture,
            memory_capture,
            memory_agent_capture,
            monitor_capture,
            validator_capture,
        ):
            self.assertEqual(
                [event["event_type"] for event in capture.trace],
                ["COMPONENT_CALL_STARTED", "COMPONENT_CALL_COMPLETED"],
            )
            self.assertEqual(capture.trace[0]["call_id"], capture.trace[1]["call_id"])

    def test_trace_writer_preserves_exact_jsonl_records(self) -> None:
        expected = self.direct.oracle.expected_values["oracle_label"]
        evaluation = run_component_trial(
            self.direct,
            CallableFixtureAdapter(lambda _trial: {"oracle_label": expected}),
        )
        with tempfile.TemporaryDirectory() as directory:
            attempt_dir = Path(directory) / "attempt"
            write_component_traces(attempt_dir, evaluation)
            written = list(iter_jsonl(attempt_dir / "component_trace.jsonl"))
        self.assertEqual(written, [dict(event) for event in evaluation.trace])


if __name__ == "__main__":
    unittest.main()
