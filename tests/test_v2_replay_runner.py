from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from experiments_suite_v2.io import iter_jsonl
from experiments_suite_v2.runners.replay import (
    REPLAY_CASES_PATH,
    REPLAY_TRACE_SCHEMA,
    FrozenClock,
    ReplayRegistryError,
    expand_replay_trials,
    load_replay_registry,
    run_replay_trial,
    validate_replay_registry,
    write_replay_traces,
)
from experiments_suite_v2.schemas import (
    AnalyticalClassification,
    ArtifactProfile,
    FormalVerdict,
    ResultEnvelope,
)


EXPECTED_CASE_IDS = (
    "PF-REPLAN-STABLE-FAIL",
    "PF-REPLAN-STEP-TIMEOUT",
    "PF-REPLAN-FRESH-FRAME",
    "PF-REPLAN-RETIRE-ORDER",
    "PF-REPLAN-UNKNOWN-TIMEOUT",
    "PF-REPLAN-TIMEOUT-IDEMPOTENT",
    "PF-REPLAN-SUCCESS-RESET",
    "PF-REPLAN-FINAL-INCOMPLETE",
    "PF-SAFETY-LATE-PUBLICATION",
    "PF-SAFETY-STALE-MONITOR-FRAME",
    "PF-SAFETY-STALE-MONITOR-ERROR",
    "PF-SAFETY-STALE-VALIDATOR-FRAME",
    "PF-SAFETY-EXECUTOR-ACTIVE",
    "PF-SAFETY-MONITOR-ERROR",
    "PF-SAFETY-VALIDATOR-ERROR",
    "PF-SAFETY-FINAL-EVIDENCE-TIMEOUT",
    "PF-SAFETY-TIMEOUT-LOOP-GUARD",
    "PF-SAFETY-ATTENTION-GATE",
    "PF-SAFETY-CYCLE-GUARD",
    "PF-SAFETY-EMERGENCY-LATCH",
)


class ReplayRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_replay_registry()
        cls.trials = expand_replay_trials(
            cls.registry,
            software_config_commit="replay-test-commit",
        )

    def test_registry_freezes_exact_case_order_counts_and_no_model_surface(self) -> None:
        self.assertEqual(
            tuple(case["case_id"] for case in self.registry["cases"]),
            EXPECTED_CASE_IDS,
        )
        self.assertEqual(
            Counter(case["aim_id"] for case in self.registry["cases"]),
            {"PF-REPLAN": 8, "PF-SAFETY": 12},
        )
        self.assertEqual(self.registry["learned_model_calls"], 0)
        self.assertEqual(
            self.registry["controller_defaults"]["timeout_policy"],
            "AUTO_REPLAN",
        )
        self.assertEqual(
            self.registry["controller_defaults"]["failure_confirmations"],
            2,
        )

    def test_registry_contains_stimuli_and_oracles_but_no_observed_results(self) -> None:
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
        for case in self.registry["cases"]:
            self.assertTrue(case["script"])
            self.assertTrue(case["oracle"])
            self.assertTrue(case["planner_decisions"])

    def test_expansion_is_one_unique_visual_replay_trial_per_case(self) -> None:
        self.assertEqual(len(self.trials), 20)
        self.assertEqual(len({trial.trial_id for trial in self.trials}), 20)
        self.assertEqual(
            tuple(trial.case_definition.case_id for trial in self.trials),
            EXPECTED_CASE_IDS,
        )
        self.assertEqual(
            [trial.order_index for trial in self.trials], list(range(20))
        )
        for trial in self.trials:
            case = trial.case_definition
            self.assertEqual(case.artifact_profile, ArtifactProfile.VISUAL_REPLAY)
            self.assertEqual(case.seeds, (trial.trial_tuple.random_seed,))
            self.assertEqual(case.live_components, (
                "Runtime",
                "Controller",
                "Monitor",
                "Validator",
            ))
            self.assertEqual(
                case.required_artifacts,
                ("events.jsonl", "replay_trace.jsonl"),
            )
            self.assertEqual(case.input_spec["learned_model_calls"], 0)
            self.assertEqual(tuple(trial.oracle), case.must_pass)

    def test_registry_rejects_order_and_learned_model_drift(self) -> None:
        changed = json.loads(REPLAY_CASES_PATH.read_text(encoding="utf-8"))
        changed["cases"][0], changed["cases"][1] = (
            changed["cases"][1],
            changed["cases"][0],
        )
        with self.assertRaisesRegex(ReplayRegistryError, "order/count"):
            validate_replay_registry(changed)

        changed = json.loads(REPLAY_CASES_PATH.read_text(encoding="utf-8"))
        changed["learned_model_calls"] = 1
        with self.assertRaisesRegex(ReplayRegistryError, "zero model calls"):
            validate_replay_registry(changed)


class ReplayExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trials = expand_replay_trials(
            software_config_commit="replay-execution-test",
        )
        cls.evaluations = {
            trial.case_definition.case_id: run_replay_trial(trial)
            for trial in cls.trials
        }

    def evaluation(self, case_id: str):
        return self.evaluations[case_id]

    def test_all_twenty_frozen_replays_pass_without_learned_calls(self) -> None:
        self.assertEqual(len(self.evaluations), 20)
        for case_id in EXPECTED_CASE_IDS:
            evaluation = self.evaluation(case_id)
            self.assertEqual(
                evaluation.result.analytical_classification,
                AnalyticalClassification.PASS,
                case_id,
            )
            self.assertEqual(evaluation.result.formal_verdict, FormalVerdict.PASS)
            self.assertEqual(evaluation.result.metrics["model_call_count"], 0)
            self.assertTrue(all(evaluation.result.must_pass.values()))
            self.assertIsNone(evaluation.result.invalidity)

            trace = evaluation.trace
            self.assertEqual(
                [item["sequence"] for item in trace],
                list(range(1, len(trace) + 1)),
            )
            self.assertEqual(
                [item["frozen_time"] for item in trace],
                sorted(item["frozen_time"] for item in trace),
            )
            self.assertTrue(
                all(item["schema_version"] == REPLAY_TRACE_SCHEMA for item in trace)
            )
            self.assertFalse(
                any(
                    item["payload"].get("learned_model_called") is True
                    for item in trace
                )
            )

    def test_stable_failure_and_timeout_are_distinct_fresh_planner_routes(self) -> None:
        failure = self.evaluation("PF-REPLAN-STABLE-FAIL")
        self.assertEqual(
            failure.summary["planner_triggers"], ["CONFIRMED", "TASK_FAIL"]
        )
        self.assertEqual(failure.summary["history_outcomes"], ["FAIL"])
        self.assertTrue(failure.summary["retire_before_replacement"])

        timeout = self.evaluation("PF-REPLAN-STEP-TIMEOUT")
        self.assertEqual(
            timeout.summary["planner_triggers"],
            ["CONFIRMED", "MONITOR_TIMEOUT"],
        )
        self.assertEqual(timeout.summary["history_outcomes"], ["INTERRUPTED"])
        self.assertEqual(
            timeout.summary["history_termination_kinds"], ["MONITOR_TIMEOUT"]
        )
        self.assertEqual(timeout.summary["auto_timeout_event_count"], 1)
        self.assertTrue(timeout.summary["fresh_timeout_frame"])
        self.assertTrue(timeout.summary["retire_before_replacement"])

    def test_unknown_timeout_and_concurrent_timeout_are_safe_and_idempotent(self) -> None:
        unknown = self.evaluation("PF-REPLAN-UNKNOWN-TIMEOUT")
        self.assertEqual(unknown.summary["timeout_criteria_states"], [
            "UNKNOWN",
            "UNKNOWN",
        ])
        self.assertEqual(unknown.summary["history_outcomes"], ["INTERRUPTED"])
        self.assertNotIn("FAIL", unknown.summary["history_outcomes"])

        concurrent = self.evaluation("PF-REPLAN-TIMEOUT-IDEMPOTENT")
        self.assertEqual(concurrent.summary["history_count"], 1)
        self.assertEqual(concurrent.summary["auto_timeout_event_count"], 1)
        self.assertEqual(concurrent.summary["concurrent_timeout_true_count"], 1)
        self.assertEqual(
            concurrent.summary["timeout_action_results"],
            [{"workers": 2, "true_count": 1, "false_count": 1}],
        )

    def test_success_resets_timeout_guard_and_final_incomplete_replans(self) -> None:
        reset = self.evaluation("PF-REPLAN-SUCCESS-RESET")
        self.assertEqual(
            reset.summary["history_outcomes"],
            ["INTERRUPTED", "SUCCESS", "INTERRUPTED"],
        )
        self.assertEqual(reset.summary["consecutive_timeout_replans"], 1)
        self.assertEqual(reset.summary["final_state"], "EXECUTING")

        incomplete = self.evaluation("PF-REPLAN-FINAL-INCOMPLETE")
        self.assertEqual(
            incomplete.summary["planner_triggers"],
            ["CONFIRMED", "FINAL_VALIDATION_FAIL"],
        )
        self.assertEqual(
            incomplete.summary["history_outcomes"], ["FINAL_VALIDATION_FAIL"]
        )

    def test_retired_publications_and_stale_frames_cannot_mutate_state(self) -> None:
        for case_id in (
            "PF-SAFETY-LATE-PUBLICATION",
            "PF-SAFETY-STALE-MONITOR-FRAME",
            "PF-SAFETY-STALE-MONITOR-ERROR",
            "PF-SAFETY-STALE-VALIDATOR-FRAME",
        ):
            evaluation = self.evaluation(case_id)
            self.assertEqual(evaluation.summary["stale_action_count"], 1, case_id)
            self.assertEqual(evaluation.summary["stale_history_deltas"], [0], case_id)
            self.assertNotEqual(evaluation.summary["final_state"], "NEEDS_ATTENTION")

        late = self.evaluation("PF-SAFETY-LATE-PUBLICATION")
        self.assertEqual(late.summary["stale_disposition_count"], 1)
        self.assertEqual(late.summary["history_outcomes"], ["INTERRUPTED"])
        self.assertTrue(late.summary["current_publication_is_latest"])

    def test_executor_and_error_routes_do_not_masquerade_as_task_failure(self) -> None:
        executor = self.evaluation("PF-SAFETY-EXECUTOR-ACTIVE")
        self.assertEqual(
            executor.summary["timeout_action_results"],
            [{"changed": False}, {"changed": True}],
        )
        self.assertEqual(executor.summary["history_outcomes"], ["INTERRUPTED"])

        for case_id in (
            "PF-SAFETY-MONITOR-ERROR",
            "PF-SAFETY-VALIDATOR-ERROR",
        ):
            evaluation = self.evaluation(case_id)
            self.assertEqual(evaluation.summary["attention_kind"], "SYSTEM_ERROR")
            self.assertEqual(evaluation.summary["final_state"], "NEEDS_ATTENTION")
            self.assertEqual(evaluation.summary["history_count"], 0)
            self.assertEqual(evaluation.summary["planner_replacement_call_count"], 1)

        final_timeout = self.evaluation("PF-SAFETY-FINAL-EVIDENCE-TIMEOUT")
        self.assertEqual(final_timeout.summary["attention_kind"], "NO_PROGRESS")
        self.assertEqual(final_timeout.summary["history_count"], 0)
        self.assertEqual(final_timeout.summary["auto_timeout_event_count"], 0)
        self.assertEqual(final_timeout.summary["planner_replacement_call_count"], 1)

    def test_loop_attention_cycle_and_emergency_guards_are_bounded(self) -> None:
        loop = self.evaluation("PF-SAFETY-TIMEOUT-LOOP-GUARD")
        self.assertEqual(loop.summary["attention_kind"], "LOOP_GUARD")
        self.assertEqual(loop.summary["history_count"], 2)
        self.assertEqual(loop.summary["auto_timeout_event_count"], 2)
        self.assertEqual(loop.summary["planner_replacement_call_count"], 3)

        attention = self.evaluation("PF-SAFETY-ATTENTION-GATE")
        self.assertEqual(attention.summary["attention_kind"], "NO_PROGRESS")
        self.assertEqual(attention.summary["history_count"], 0)
        self.assertEqual(attention.summary["planner_replacement_call_count"], 1)

        cycle = self.evaluation("PF-SAFETY-CYCLE-GUARD")
        self.assertEqual(cycle.summary["attention_kind"], "LOOP_GUARD")
        self.assertEqual(cycle.summary["history_count"], 0)
        self.assertEqual(cycle.summary["planner_replacement_call_count"], 1)

        emergency = self.evaluation("PF-SAFETY-EMERGENCY-LATCH")
        self.assertEqual(emergency.summary["final_state"], "EMERGENCY_STOPPED")
        self.assertTrue(emergency.summary["emergency_latched"])
        self.assertTrue(emergency.summary["emergency_first_trigger"])
        self.assertFalse(emergency.summary["emergency_second_trigger"])

    def test_oracle_failure_is_one_durable_capability_failure_not_a_retry(self) -> None:
        trial = self.trials[0]
        changed_oracle = {
            name: dict(predicate) for name, predicate in trial.oracle.items()
        }
        first_name = next(iter(changed_oracle))
        changed_oracle[first_name]["equals"] = ["IMPOSSIBLE_TRIGGER"]
        changed = replace(trial, oracle=changed_oracle)

        evaluation = run_replay_trial(
            changed,
            attempt_id="replay-fixture-attempt",
            retry_lineage={"retry_of": None, "supersedes_attempt": None},
        )
        self.assertEqual(
            evaluation.result.analytical_classification,
            AnalyticalClassification.CAPABILITY_FAIL,
        )
        self.assertEqual(evaluation.result.formal_verdict, FormalVerdict.FAIL)
        self.assertEqual(evaluation.result.failed_predicates, (first_name,))
        self.assertIsNone(evaluation.result.invalidity)
        self.assertEqual(evaluation.result.attempt_id, "replay-fixture-attempt")
        self.assertEqual(
            ResultEnvelope.from_dict(evaluation.result.to_dict()).to_dict(),
            evaluation.result.to_dict(),
        )

    def test_trace_writer_durably_emits_exact_jsonl_rows(self) -> None:
        evaluation = self.evaluation("PF-REPLAN-STEP-TIMEOUT")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_replay_traces(directory, evaluation)
            trace_rows = list(iter_jsonl(directory / "replay_trace.jsonl"))
            event_rows = list(iter_jsonl(directory / "events.jsonl"))
        self.assertEqual(trace_rows, [dict(item) for item in evaluation.trace])
        self.assertEqual(event_rows, [dict(item) for item in evaluation.events])


class FrozenClockTests(unittest.TestCase):
    def test_clock_advances_only_by_explicit_nonnegative_finite_amounts(self) -> None:
        clock = FrozenClock(10)
        self.assertEqual(clock(), 10)
        self.assertEqual(clock.advance(2.5), 12.5)
        self.assertEqual(clock(), 12.5)
        for invalid in (-1, float("inf"), float("nan"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ReplayRegistryError):
                    clock.advance(invalid)


if __name__ == "__main__":
    unittest.main()
