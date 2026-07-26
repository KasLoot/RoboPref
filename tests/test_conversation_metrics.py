from __future__ import annotations

import unittest
from typing import Any

from simulation.benchmark.conversation_metrics import build_conversation_summary


def case_plan(
    case_id: str,
    *,
    outcome: str = "success",
    suite: str = "endpoint",
    profile: str = "explicit",
    cluster_id: str | None = None,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "family": "block_stack",
        "outcome": outcome,
    }
    if dependencies is not None:
        metadata["dependency_cluster_ids"] = dependencies
    return {
        "case_id": case_id,
        "suite": suite,
        "profile": profile,
        "cluster_id": cluster_id or f"pixels-{case_id}",
        "metadata": metadata,
        "commands": [
            {
                "command_id": f"command-{case_id}",
                "score_endpoint": True,
                "packet_outcome": outcome,
            }
        ],
    }


def check(
    name: str,
    status: str,
    *,
    command_id: str,
    expected: Any = None,
    actual: Any = None,
    stage: str = "validator",
    criticality: str = "HARD",
) -> dict[str, Any]:
    return {
        "name": name,
        "stage": stage,
        "criticality": criticality,
        "status": status,
        "command_id": command_id,
        "expected": expected,
        "actual": actual,
    }


def record(
    case: dict[str, Any],
    run_status: str,
    benchmark_result: str,
    *,
    repetition: int = 1,
    checks: list[dict[str, Any]] | None = None,
    artifact_complete: bool | None = None,
) -> dict[str, Any]:
    if artifact_complete is None:
        artifact_complete = run_status == "COMPLETED"
    return {
        "case": case,
        "repetition": repetition,
        "run_status": run_status,
        "benchmark_result": benchmark_result,
        "checks": checks or [],
        "artifact_status": {"complete": artifact_complete},
        "duration_seconds": 1.0,
    }


class ConversationMetricsTests(unittest.TestCase):
    def test_functional_and_audited_results_have_explicit_denominators(self) -> None:
        one = case_plan("one")
        two = case_plan("two")
        three = case_plan("three")
        summary = build_conversation_summary(
            [
                record(one, "COMPLETED", "PASS"),
                record(
                    two,
                    "ARTIFACT_ERROR",
                    "PASS",
                    artifact_complete=False,
                ),
                record(
                    three,
                    "AGENT_TERMINATED",
                    "NOT_SCORED",
                    artifact_complete=False,
                ),
            ],
            planned_runs=3,
            planned_cases=[one, two, three],
            bootstrap_replicates=20,
            bootstrap_seed=7,
        )
        self.assertTrue(summary["integrity"]["valid"])
        self.assertEqual(summary["results"]["functional"]["pass"], 2)
        self.assertEqual(summary["results"]["audited"]["pass"], 1)
        self.assertEqual(
            summary["results"]["audited"]
            ["planned_intention_to_treat_rate_naive"]["rate"],
            1 / 3,
        )
        self.assertEqual(summary["by_suite"]["endpoint"]["functional_passed"], 2)
        self.assertEqual(summary["by_suite"]["endpoint"]["audited_passed"], 1)

    def test_check_metrics_separate_missing_stage_and_criticality(self) -> None:
        one = case_plan("one")
        two = case_plan("two")
        command_one = "command-one"
        command_two = "command-two"
        summary = build_conversation_summary(
            [
                record(
                    one,
                    "COMPLETED",
                    "PASS",
                    checks=[
                        check(
                            "validator_outcome",
                            "PASS",
                            command_id=command_one,
                            expected="SUCCESS",
                            actual="SUCCESS",
                        )
                    ],
                ),
                record(
                    two,
                    "COMPLETED",
                    "FAIL",
                    checks=[
                        check(
                            "validator_outcome",
                            "MISSING",
                            command_id=command_two,
                            expected="SUCCESS",
                        )
                    ],
                ),
            ],
            planned_cases=[one, two],
            bootstrap_replicates=10,
        )
        metric = summary["checks"]["by_name"]["validator_outcome"]
        self.assertEqual(metric["eligible"], 2)
        self.assertEqual(metric["observed"], 1)
        self.assertEqual(metric["missing"], 1)
        self.assertEqual(metric["conditional_pass_rate"], 1.0)
        self.assertEqual(metric["unconditional_pass_rate"], 0.5)
        self.assertEqual(
            summary["checks"]["by_criticality"]["HARD"]["eligible"],
            2,
        )
        self.assertEqual(
            summary["checks"]["by_stage"]["validator"]["missing"],
            1,
        )

    def test_validator_counts_terminated_runs_and_zero_prediction_f1(self) -> None:
        wrong_one = case_plan("wrong-one", outcome="wrong_complete")
        wrong_two = case_plan("wrong-two", outcome="wrong_complete")
        unsafe = case_plan("unsafe", outcome="unsafe")
        summary = build_conversation_summary(
            [
                record(
                    wrong_one,
                    "COMPLETED",
                    "FAIL",
                    checks=[
                        check(
                            "validator_outcome",
                            "FAIL",
                            command_id="command-wrong-one",
                            expected="FAILURE",
                            actual="SUCCESS",
                        )
                    ],
                ),
                record(
                    wrong_two,
                    "AGENT_TERMINATED",
                    "NOT_SCORED",
                    artifact_complete=False,
                ),
                record(
                    unsafe,
                    "AGENT_TERMINATED",
                    "NOT_SCORED",
                    artifact_complete=False,
                ),
            ],
            planned_cases=[wrong_one, wrong_two, unsafe],
            bootstrap_replicates=10,
        )
        validator = summary["validator"]
        self.assertEqual(validator["planned_eligible"], 2)
        self.assertEqual(validator["observed"], 1)
        self.assertEqual(validator["missing"], 1)
        self.assertEqual(validator["per_class"]["FAILURE"]["f1"], 0.0)
        self.assertEqual(validator["macro_f1"], 0.0)

    def test_incomplete_batch_disables_bootstrap(self) -> None:
        one = case_plan("one")
        two = case_plan("two")
        summary = build_conversation_summary(
            [record(one, "COMPLETED", "PASS")],
            planned_runs=2,
            planned_cases=[one, two],
            bootstrap_replicates=10,
        )
        self.assertFalse(summary["integrity"]["valid"])
        self.assertEqual(summary["integrity"]["missing_run_count"], 1)
        self.assertEqual(summary["bootstrap"]["status"], "NOT_COMPUTED")

    def test_dependency_chain_forms_one_bootstrap_component(self) -> None:
        one = case_plan("one", dependencies=["a", "b"])
        two = case_plan("two", dependencies=["b", "c"])
        three = case_plan("three", dependencies=["c", "d"])
        summary = build_conversation_summary(
            [
                record(one, "COMPLETED", "PASS"),
                record(two, "COMPLETED", "PASS"),
                record(three, "COMPLETED", "FAIL"),
            ],
            planned_cases=[one, two, three],
            bootstrap_replicates=20,
            bootstrap_seed=3,
        )
        self.assertEqual(summary["bootstrap"]["status"], "COMPUTED")
        self.assertEqual(summary["bootstrap"]["cluster_count"], 1)
        self.assertEqual(
            summary["bootstrap"]["trial_micro"]["estimate"],
            2 / 3,
        )
        self.assertEqual(summary["bootstrap"]["seed"], 3)

    def test_headlines_filter_isolation_and_require_full_recovery_chain(self) -> None:
        ordinary = case_plan(
            "ordinary-memory",
            suite="memory",
            profile="learn-consent-reuse-twice",
        )
        isolation = case_plan(
            "isolation",
            suite="memory",
            profile="cross-user-isolation",
        )
        recovery = case_plan(
            "recovery",
            outcome="partial",
            suite="recovery",
            profile="partial-to-success",
        )
        summary = build_conversation_summary(
            [
                record(
                    ordinary,
                    "COMPLETED",
                    "PASS",
                    checks=[
                        check(
                            "memory_ownership",
                            "PASS",
                            command_id="command-ordinary-memory",
                            stage="memory",
                        )
                    ],
                ),
                record(
                    isolation,
                    "COMPLETED",
                    "FAIL",
                    checks=[
                        check(
                            "memory_ownership",
                            "FAIL",
                            command_id="command-isolation",
                            stage="memory",
                        )
                    ],
                ),
                record(
                    recovery,
                    "COMPLETED",
                    "FAIL",
                    checks=[
                        check(
                            "scripted_counterfactual_transition",
                            "FAIL",
                            command_id="command-recovery",
                            stage="recovery",
                        ),
                        check(
                            "recovery_terminal_outcome",
                            "PASS",
                            command_id="command-recovery",
                            stage="recovery",
                        ),
                        check(
                            "recovery_validator_outcome",
                            "PASS",
                            command_id="command-recovery",
                            stage="recovery",
                        ),
                        check(
                            "recovery_validator_task_complete",
                            "PASS",
                            command_id="command-recovery",
                            stage="recovery",
                        ),
                        check(
                            "recovery_observation_available",
                            "PASS",
                            command_id="command-recovery",
                            stage="recovery",
                        ),
                        check(
                            "recovery_planner_vla_subtask_target",
                            "PASS",
                            command_id="command-recovery",
                            stage="recovery",
                        ),
                    ],
                ),
            ],
            planned_cases=[ordinary, isolation, recovery],
            bootstrap_replicates=10,
        )
        ownership = summary["headlines"][
            "cross_user_memory_ownership_isolation"
        ]
        self.assertEqual(ownership["eligible_runs"], 1)
        self.assertEqual(ownership["failed_runs"], 1)
        recovered = summary["headlines"][
            "scripted_counterfactual_recovery_chain"
        ]
        self.assertEqual(recovered["eligible_runs"], 1)
        self.assertEqual(recovered["failed_runs"], 1)

    def test_repetition_and_status_integrity_are_not_tautological(self) -> None:
        repeated = case_plan("repeated")
        incomplete = build_conversation_summary(
            [record(repeated, "COMPLETED", "PASS", repetition=1)],
            planned_runs=2,
            planned_cases=[repeated],
            repetitions=2,
            bootstrap_replicates=10,
        )
        reliability = incomplete["repetition_reliability"]
        self.assertEqual(reliability["eligible_cases"], 1)
        self.assertEqual(reliability["complete_case_rate"], 0.0)
        self.assertEqual(reliability["all_audited_pass_rate"], 0.0)

        invalid = build_conversation_summary(
            [
                record(
                    repeated,
                    "ARTIFACT_ERROR",
                    "NOT_SCORED",
                    artifact_complete=True,
                )
            ],
            planned_cases=[repeated],
            bootstrap_replicates=10,
        )
        self.assertFalse(invalid["integrity"]["valid"])
        self.assertTrue(invalid["integrity"]["invalid_status_result_pairs"])
        self.assertTrue(invalid["integrity"]["artifact_status_conflicts"])


if __name__ == "__main__":
    unittest.main()
