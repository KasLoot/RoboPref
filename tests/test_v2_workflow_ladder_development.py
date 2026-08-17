from __future__ import annotations

from types import SimpleNamespace
import unittest

from experiments_suite_v2.runners import workflow_ladder_development as workflow


class WorkflowLadderDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = workflow.build_rows()

    def test_registry_is_exact_seven_by_forty(self) -> None:
        self.assertEqual(tuple(self.rows), workflow.AIM_ORDER)
        self.assertEqual([len(self.rows[aim]) for aim in workflow.AIM_ORDER], [40] * 7)
        all_rows = [row for aim in workflow.AIM_ORDER for row in self.rows[aim]]
        self.assertEqual(len(all_rows), 280)
        self.assertEqual(len({row.trial_id for row in all_rows}), 280)
        for aim in workflow.AIM_ORDER:
            contexts = [row.context_id for row in self.rows[aim]]
            self.assertEqual(
                {context: contexts.count(context) for context in sorted(set(contexts))},
                {f"A{index:02d}": 5 for index in range(1, 9)},
            )

    def test_bundle_input_rows_expose_unique_row_ids(self) -> None:
        values = [
            workflow._input_row(row)
            for aim in workflow.AIM_ORDER
            for row in self.rows[aim]
        ]
        self.assertEqual(len(values), 280)
        self.assertEqual(len({row["row_id"] for row in values}), 280)
        expected_ids = [
            row.trial_id
            for aim in workflow.AIM_ORDER
            for row in self.rows[aim]
        ]
        self.assertEqual([row["row_id"] for row in values], expected_ids)

    def test_action_policy_is_cumulative_and_safe(self) -> None:
        self.assertEqual(
            workflow._expected_actions("L0-EXE-EXACT", "A07"),
            (("red cube", "white board"),),
        )
        for aim in ("L1-HRI", "L2-PLN"):
            self.assertEqual(workflow._expected_actions(aim, "A01"), ())
            self.assertEqual(workflow._expected_actions(aim, "A03"), ())
            self.assertEqual(workflow._expected_actions(aim, "A07"), ())
            self.assertEqual(workflow._expected_actions(aim, "A08"), ())
            self.assertEqual(
                workflow._expected_actions(aim, "A04"),
                (("blue cube", "white board"),),
            )
        self.assertEqual(workflow._expected_actions("L0-EXE-RAW", "A04"), ())
        self.assertEqual(
            workflow._expected_actions("L0-EXE-RAW", "A05"),
            (("red cube", "white board"),),
        )
        for aim in ("L3-MEM", "L4-MON", "L5-FULL"):
            self.assertEqual(
                workflow._expected_actions(aim, "A01"),
                (("red cube", "cyan board"),),
            )
            self.assertEqual(
                workflow._expected_actions(aim, "A03"),
                (
                    ("green cube", "white board"),
                    ("blue cube", "green cube"),
                    ("red cube", "blue cube"),
                ),
            )
            self.assertEqual(workflow._expected_actions(aim, "A07"), ())
            self.assertEqual(workflow._expected_actions(aim, "A08"), ())

    def test_program_gate_requires_exact_selectors_and_anchors(self) -> None:
        from prefmem.execution.contracts import ManipulationProgram

        good = ManipulationProgram.from_dict(
            {
                "schema_version": 1,
                "skill": "pick_place",
                "source": {"query": "red block", "anchor": "top_center"},
                "target": {"query": "white mat", "anchor": "surface_center"},
                "relation": "on_top",
                "orientation": "tool_down",
                "waypoints": [
                    {"kind": "approach_source", "clearance_m": 0.1, "gripper": "open"},
                    {"kind": "grasp_source", "clearance_m": 0.0, "gripper": "close"},
                    {"kind": "lift", "clearance_m": 0.1, "gripper": "hold"},
                    {"kind": "approach_target", "clearance_m": 0.1, "gripper": "hold"},
                    {"kind": "place_target", "clearance_m": 0.0, "gripper": "open"},
                    {"kind": "retreat", "clearance_m": 0.1, "gripper": "hold"},
                ],
            }
        )
        self.assertTrue(
            workflow._program_safe_for_action(good, ("red cube", "white board"))
        )
        self.assertFalse(
            workflow._program_safe_for_action(good, ("red cube", "cyan board"))
        )

    def test_confirmation_is_host_fenced(self) -> None:
        row = self.rows["L1-HRI"][0]
        runtime = workflow._WorkflowRuntime(
            row,
            planner=SimpleNamespace(),
            frame_source=lambda: None,
            planner_enabled=False,
            memory_enabled=False,
        )
        preview = runtime.request_goal_preview(
            "Put the red cube at the center of the white board."
        )
        contract = preview["goal_contract"]
        blocked = runtime.confirm_goal(
            goal_id=contract["goal_id"], revision=contract["revision"], confirmed=True
        )
        self.assertEqual(blocked["status"], "BLOCKED_CONFIRMATION_NOT_YET_AUTHORIZED")
        self.assertTrue(runtime.unauthorized_confirmation_attempt)
        runtime.confirmation_authorized = True
        accepted = runtime.confirm_goal(
            goal_id=contract["goal_id"], revision=contract["revision"], confirmed=True
        )
        self.assertEqual(accepted["status"], "CONFIRMED_FOR_SAFE_HOST_EXECUTION")
        self.assertTrue(runtime.confirmed)

    def test_summary_keeps_strict_and_assisted_results_separate(self) -> None:
        rows = []
        for index in range(40):
            rows.append(
                {
                    "row_id": f"row-{index}",
                    "context_id": f"A{index // 5 + 1:02d}",
                    "valid_trial": True,
                    "formal_verdict": "PASS",
                    "analytical_classification": "PASS",
                    "strict_system_result": "FAIL_GROUNDING" if index == 0 else "PASS",
                    "oracle_fallback_used": index == 0,
                    "assisted_continuation_result": "PASS" if index == 0 else "NOT_APPLICABLE",
                    "unsafe_program_blocked": False,
                    "physical_action_count": 1,
                    "model_call_count": 2,
                    "memory_retrieval_count": 0,
                }
            )
        summary = workflow._summary("L0-EXE-EXACT", rows, {}, {})
        self.assertEqual(summary["formal_verdict"], "PASS")
        self.assertEqual(summary["strict_pass_count"], 39)
        self.assertEqual(summary["oracle_fallback_rows"], 1)
        self.assertEqual(summary["assisted_pass_rows"], 1)

    def test_raw_harness_retry_uses_fresh_a1(self) -> None:
        root = workflow.SUITE_ROOT / "results/shared-campaigns" / workflow.DEFAULT_SUITE_RUN_ID
        self.assertEqual(workflow._attempt_number("L0-EXE-RAW"), 1)
        self.assertEqual(workflow._attempt_number("L1-HRI"), 0)
        self.assertEqual(
            workflow._aim_dir(root, "L0-EXE-RAW").name,
            "A1",
        )
        self.assertFalse(
            (root / "WORKFLOW-LADDER/L0-EXE-RAW/A1/artifact_manifest.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
