from __future__ import annotations

from collections import Counter, defaultdict
import unittest

from experiments_suite_v2.cases.system_cases import (
    ambiguity_contexts,
    expected_trial_counts,
    load_system_case_registry,
    registered_states,
    system_case_definitions,
)
from experiments_suite_v2.runners.system import (
    CallableSystemAdapter,
    ExternalServiceInterruption,
    SystemBlock,
    count_trials_by_block,
    expand_system_trials,
    run_system_trial,
    summarize_execution_outcomes,
)
from experiments_suite_v2.schemas import AnalyticalClassification


class SystemCaseFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_system_case_registry()
        cls.trials = expand_system_trials(cls.registry)

    def test_exact_primary_and_supplementary_counts(self) -> None:
        counts = count_trials_by_block(self.trials)
        self.assertEqual(
            counts,
            {
                "E2E-N": 30,
                "E2E-A": 40,
                "E2E-R": 50,
                "AB-COMP": 400,
                "AB-RPL-F": 80,
                "AB-RPL-T": 80,
                "WORKFLOW-LADDER": 280,
            },
        )
        expected = expected_trial_counts()
        self.assertEqual(sum(counts.values()), expected["ALL_EXPANDED_TOTAL"])
        primary = sum(
            1 for item in self.trials if item.analysis_role == "PRIMARY"
        )
        self.assertEqual(primary, expected["PRIMARY_SYSTEM_TOTAL"])
        self.assertEqual(
            len(self.trials) - primary, expected["SUPPLEMENTARY_TOTAL"]
        )
        without_workflow = expand_system_trials(
            self.registry, include_workflow_ladder=False
        )
        self.assertEqual(len(without_workflow), 680)

    def test_registered_states_ambiguities_and_camera_routing_are_complete(self) -> None:
        states = registered_states(self.registry)
        self.assertEqual(tuple(states), tuple(f"V{i:02d}" for i in range(14)))
        self.assertTrue(
            all(len(state["cubes"]) == 6 and len(state["boards"]) == 2 for state in states.values())
        )
        ambiguity = ambiguity_contexts(self.registry)
        self.assertEqual(tuple(ambiguity), tuple(f"A{i:02d}" for i in range(1, 9)))
        self.assertTrue(
            all(tuple(item["exact_utterances"]) == ("01", "02", "03", "04", "05") for item in ambiguity.values())
        )
        routing = self.registry["camera_routing"]
        self.assertEqual(routing["sam_stream"]["camera_name"], "sam_camera")
        self.assertEqual(
            routing["prefmem_stream"]["camera_name"], "prefmem_camera"
        )
        self.assertEqual(
            routing["variant_policy"],
            "APPROVED_FIXED_POSE_NO_EXTRINSIC_INTRINSIC_OR_FOV_PERTURBATION",
        )
        self.assertTrue(
            all(
                trial.trial_tuple.extra_frozen_factors[
                    "approved_camera_variant"
                ]
                == "APPROVED_FIXED_POSE"
                for trial in self.trials
            )
        )

    def test_every_case_has_one_aggregate_primary_oracle(self) -> None:
        definitions = system_case_definitions(self.registry)
        self.assertEqual(len(definitions), 32)
        self.assertTrue(all(case.primary_oracle for case in definitions))
        self.assertTrue(all(case.must_pass for case in definitions))
        for trial in self.trials:
            self.assertEqual(
                trial.case_definition.primary_oracle, trial.oracle.oracle_id
            )
            self.assertEqual(
                trial.case_definition.must_pass,
                tuple(item.name for item in trial.oracle.predicates),
            )

    def test_paired_rows_share_identity_and_only_two_conditions(self) -> None:
        paired = [trial for trial in self.trials if trial.pair_id is not None]
        self.assertEqual(len(paired), 560)
        groups: dict[str, list] = defaultdict(list)
        for trial in paired:
            groups[str(trial.pair_id)].append(trial)
        self.assertEqual(len(groups), 280)
        for members in groups.values():
            self.assertEqual(len(members), 2)
            first, second = members
            self.assertNotEqual(first.condition_id, second.condition_id)
            self.assertEqual(first.context_id, second.context_id)
            self.assertEqual(first.variant_id, second.variant_id)
            self.assertEqual(first.exact_utterance, second.exact_utterance)
            self.assertEqual(first.initial_state, second.initial_state)
            self.assertEqual(first.memory_fixture, second.memory_fixture)
            self.assertEqual(first.controlled_event, second.controlled_event)
            self.assertEqual(
                first.trial_tuple.random_seed, second.trial_tuple.random_seed
            )
            self.assertEqual(
                first.trial_tuple.extra_frozen_factors["pair_common_sha256"],
                second.trial_tuple.extra_frozen_factors["pair_common_sha256"],
            )

    def test_recovery_trigger_semantics_are_frozen(self) -> None:
        recovery = {
            trial.case_definition.case_id: trial
            for trial in self.trials
            if trial.block_id is SystemBlock.E2E_R and trial.variant_id == "01"
        }
        timeout = recovery["E2E-R-02"].controlled_event
        self.assertEqual(
            timeout["expected_route"],
            [
                "AUTO_TIMEOUT_REPLAN",
                "INTERRUPTED",
                "MONITOR_TIMEOUT",
                "PLANNER_FRESH_FRAME",
                "RECOVERY_TASK",
            ],
        )
        self.assertTrue(
            {"executor_inactive", "executor_settled"}.issubset(
                timeout["preconditions"]
            )
        )
        stable_fail = recovery["E2E-R-01"].controlled_event
        self.assertEqual(
            stable_fail["expected_route"][:2],
            ["MONITOR_FAIL_STABLE", "TASK_FAIL"],
        )
        outage = recovery["E2E-R-10"].controlled_event
        self.assertTrue(outage["planned_external_interruption"])

        timeout_contexts = {
            trial.context_id: trial.controlled_event["kind"]
            for trial in self.trials
            if trial.block_id is SystemBlock.AB_RPL_T
            and trial.condition_id == "AUTO_REPLAN"
            and trial.variant_id == "01"
        }
        self.assertEqual(len(timeout_contexts), 8)
        self.assertIn("TWO_CONSECUTIVE_TIMEOUT_CYCLES", timeout_contexts.values())
        self.assertIn("LATE_RETIRED_PUBLICATION_RESULT", timeout_contexts.values())

    def test_workflow_is_supplementary_and_not_pooled(self) -> None:
        workflow = [
            trial
            for trial in self.trials
            if trial.block_id is SystemBlock.WORKFLOW_LADDER
        ]
        self.assertEqual(len(workflow), 280)
        self.assertTrue(
            all(
                item.analysis_role
                == "SUPPLEMENTARY_EXCLUDED_FROM_EXPLICIT_PRIMARY_MINIMUM"
                for item in workflow
            )
        )
        blocks = {
            item["applicable_task_block_id"]
            for item in self.registry["workflow_ladder"]["conditions"]
        }
        self.assertEqual(len(blocks), 7)
        self.assertEqual(
            self.registry["analysis_contract"][
                "workflow_pooling_across_task_blocks"
            ],
            "PROHIBITED",
        )


class SystemExecutionSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trial = expand_system_trials(include_workflow_ladder=False)[0]

    def _passing_output(self) -> dict:
        output: dict = {}
        for predicate in self.trial.oracle.predicates:
            self.assertNotIn(".", predicate.actual_path)
            output[predicate.actual_path] = predicate.expected
        return output

    def test_valid_pass_and_capability_fail_are_not_retried(self) -> None:
        passed = run_system_trial(
            self.trial, CallableSystemAdapter(lambda _: self._passing_output())
        )
        self.assertEqual(
            passed.result.analytical_classification,
            AnalyticalClassification.PASS,
        )
        self.assertFalse(passed.retry_required)

        failed_output = self._passing_output()
        failed_output[next(iter(failed_output))] = False
        failed = run_system_trial(
            self.trial, CallableSystemAdapter(lambda _: failed_output)
        )
        self.assertEqual(
            failed.result.analytical_classification,
            AnalyticalClassification.CAPABILITY_FAIL,
        )
        self.assertFalse(failed.retry_required)
        self.assertIsNone(failed.result.invalidity)

    def test_unplanned_service_interruption_is_invalid_run_with_exact_retry(self) -> None:
        def interrupted(_):
            raise ExternalServiceInterruption(
                "forward disconnected",
                service_id="gemma-hri-planner-monitor-validator-v1",
                reason_code="VLM_CONNECTION_LOST",
            )

        run = run_system_trial(
            self.trial, CallableSystemAdapter(interrupted)
        )
        self.assertEqual(
            run.result.analytical_classification,
            AnalyticalClassification.INVALID_RUN,
        )
        self.assertTrue(run.retry_required)
        self.assertEqual(run.result.invalidity.reason_code, "VLM_CONNECTION_LOST")
        self.assertFalse(run.result.enters_capability_denominator)

    def test_failure_to_contain_planned_outage_is_capability_fail(self) -> None:
        outage_trial = next(
            trial
            for trial in expand_system_trials(include_workflow_ladder=False)
            if trial.case_definition.case_id == "E2E-R-10"
        )

        def uncontained(_):
            raise ExternalServiceInterruption(
                "registered outage escaped the runtime",
                service_id="gemma-hri-planner-monitor-validator-v1",
                planned=True,
            )

        run = run_system_trial(
            outage_trial, CallableSystemAdapter(uncontained)
        )
        self.assertEqual(
            run.result.analytical_classification,
            AnalyticalClassification.CAPABILITY_FAIL,
        )
        self.assertFalse(run.retry_required)
        self.assertIsNone(run.result.invalidity)
        self.assertTrue(run.result.enters_capability_denominator)

    def test_execution_outcome_metrics_keep_strict_and_assisted_counts_separate(self) -> None:
        output = self._passing_output()
        output["executor_outcomes"] = [
            {
                "strict_system_result": "FAIL_GROUNDING",
                "oracle_fallback_used": True,
                "assisted_continuation_result": "PASS",
            },
            {
                "strict_system_result": "PASS",
                "oracle_fallback_used": False,
                "assisted_continuation_result": "NOT_APPLICABLE",
            },
        ]

        run = run_system_trial(
            self.trial,
            CallableSystemAdapter(lambda _: output),
        )

        self.assertEqual(
            run.result.analytical_classification,
            AnalyticalClassification.PASS,
        )
        summary = run.result.metrics["execution_outcomes"]
        self.assertEqual(
            summary["strict_counts"],
            {"FAIL_GROUNDING": 1, "PASS": 1},
        )
        self.assertEqual(summary["oracle_assisted_count"], 1)
        self.assertEqual(
            summary["assisted_continuation_counts"],
            {"NOT_APPLICABLE": 1, "PASS": 1},
        )
        self.assertEqual(
            summary["classification_authority"],
            "FROZEN_PRIMARY_ORACLE_ONLY",
        )

    def test_missing_execution_outcomes_is_reported_without_rescoring(self) -> None:
        self.assertEqual(
            summarize_execution_outcomes(self._passing_output()),
            {
                "reported": False,
                "publication_count": 0,
                "strict_counts": {},
                "oracle_assisted_count": 0,
                "assisted_continuation_counts": {},
                "unrecognized_count": 0,
                "classification_authority": "FROZEN_PRIMARY_ORACLE_ONLY",
            },
        )


if __name__ == "__main__":
    unittest.main()
