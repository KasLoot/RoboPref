from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest

from experiments_suite_v2.runners import e2e_development as e2e


class E2EDevelopmentTests(unittest.TestCase):
    def test_exact_case_and_row_registry(self) -> None:
        rows = e2e.build_rows()
        self.assertEqual(tuple(rows), e2e.CASE_ORDER)
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(map(len, rows.values())), 120)
        self.assertTrue(all(len(values) == 5 for values in rows.values()))
        self.assertEqual(
            Counter(row.block_id.value for values in rows.values() for row in values),
            {"E2E-N": 30, "E2E-A": 40, "E2E-R": 50},
        )
        self.assertEqual(
            len({row.trial_id for values in rows.values() for row in values}), 120
        )

    def test_action_contracts_cover_every_case(self) -> None:
        self.assertEqual(set(e2e.NOMINAL_ACTIONS), set(e2e.CASE_ORDER[:6]))
        self.assertEqual(set(e2e.AMBIGUITY_ACTIONS), set(e2e.CASE_ORDER[6:14]))
        self.assertEqual(set(e2e.RECOVERY_ACTIONS), set(e2e.CASE_ORDER[14:]))
        self.assertEqual(e2e.NOMINAL_ACTIONS["E2E-N-05"], ())
        self.assertEqual(len(e2e.NOMINAL_ACTIONS["E2E-N-04"]), 4)
        self.assertEqual(len(e2e.NOMINAL_ACTIONS["E2E-N-06"]), 3)

    def test_runtime_args_use_direct_third_person_and_oracle_continuation(self) -> None:
        args = e2e._runtime_args(Path("/tmp/e2e-test-row"), 17011)
        self.assertEqual(args.executor, "mujoco")
        self.assertFalse(args.simulation_viewer)
        self.assertEqual(args.simulation_viewer_camera, "prefmem")
        self.assertEqual(args.simulation_render_size, 768)
        self.assertEqual(args.sam_base_url, "http://127.0.0.1:9000")
        self.assertTrue(args.experiment_oracle_grounding_fallback)

    def test_route_is_derived_only_from_observed_evidence(self) -> None:
        empty = e2e._route("E2E-R-03", [], [], {"state": "EXECUTING", "execution_history": []})
        self.assertEqual(empty, [])
        trace = [{"stimulus_fired": True}]
        context = {
            "state": "EXECUTING",
            "execution_history": [
                {"publication_id": "old"},
                {"publication_id": "new"},
            ],
        }
        self.assertEqual(
            e2e._route("E2E-R-03", trace, [], context),
            ["SCENE_CHANGE", "PLANNER_FRESH_FRAME", "UPDATED_SOURCE_TASK"],
        )

    def test_contains_in_order_is_not_set_membership(self) -> None:
        self.assertTrue(e2e._contains_in_order(["a", "x", "b", "c"], ["a", "b", "c"]))
        self.assertFalse(e2e._contains_in_order(["b", "a", "c"], ["a", "b", "c"]))

    def test_case_summary_requires_five_valid_primary_passes(self) -> None:
        rows = [
            {
                "valid_trial": True,
                "formal_verdict": "PASS",
                "analytical_classification": "PASS",
                "strict_system_result": "PASS",
                "oracle_fallback_used": False,
                "assisted_continuation_result": "NOT_APPLICABLE",
                "physical_action_count": 1,
                "model_call_count": 4,
                "timed_out": False,
            }
            for _ in range(5)
        ]
        summary = e2e._summary("E2E-N-01", rows, {}, {})
        self.assertEqual(summary["formal_verdict"], "PASS")
        self.assertEqual(summary["strict_pass_count"], 5)
        rows[-1] = dict(rows[-1], formal_verdict="FAIL", analytical_classification="CAPABILITY_FAIL")
        summary = e2e._summary("E2E-N-01", rows, {}, {})
        self.assertEqual(summary["formal_verdict"], "FAIL")
        self.assertEqual(summary["analytical_classification"], "CAPABILITY_FAIL")

    def test_no_automatic_row_retry_contract(self) -> None:
        protocol = e2e.load_json(e2e.PROTOCOL_PATH)
        self.assertEqual(protocol["execution_policy"]["row_retry"], "NONE_RECORD_AND_ADVANCE")
        self.assertEqual(protocol["execution_policy"]["case_command_rows"], 5)
        self.assertEqual(protocol["execution_policy"]["camera_input"], "DIRECT_SIMULATION_THIRD_PERSON")


if __name__ == "__main__":
    unittest.main()
