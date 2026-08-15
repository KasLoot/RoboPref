from __future__ import annotations

import ast
import inspect
import unittest

from experiments_suite_v2.runners import cal_x03_context_diagnostic as diagnostic


class CalX03ContextDiagnosticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = diagnostic._input_rows()
        cls.inputs = diagnostic._inputs(cls.rows)
        cls.probes = diagnostic._probe_contracts(cls.rows)

    def test_grid_and_call_budget_are_exact(self) -> None:
        self.assertEqual(len(self.rows), 208)
        cells: dict[tuple[str, str], set[int]] = {}
        for row in self.rows:
            cells.setdefault(
                (str(row["object_id"]), str(row["position_id"])), set()
            ).add(int(row["repeat_index"]))
        self.assertEqual(len(cells), 104)
        self.assertTrue(all(repeats == {0, 1} for repeats in cells.values()))
        self.assertEqual(diagnostic.LIVE_CALL_COUNT, 416)
        self.assertEqual(self.inputs["detector"]["live_call_count"], 416)
        self.assertEqual(
            self.inputs["development_split"]["fresh_probe_contract_count"],
            416,
        )

    def test_416_new_request_records_are_condition_separated(self) -> None:
        self.assertEqual(len(self.probes), 416)
        probe_ids = [str(item["probe_row_id"]) for item in self.probes]
        request_paths = [str(item["initial_request_path"]) for item in self.probes]
        response_paths = [str(item["response_path"]) for item in self.probes]
        self.assertEqual(len(set(probe_ids)), 416)
        self.assertEqual(len(set(request_paths)), 416)
        self.assertEqual(len(set(response_paths)), 416)
        self.assertTrue(all(item["fresh_live_call"] is True for item in self.probes))
        self.assertEqual(
            sum(item["condition"] == "ISOLATED_A1" for item in self.probes),
            208,
        )
        self.assertEqual(
            sum(
                item["condition"] == "CONTEXT_PRESERVING_V00"
                for item in self.probes
            ),
            208,
        )
        self.assertTrue(all(float(item["threshold"]) == 0.50 for item in self.probes))

    def test_run_has_no_stage_one_response_reuse_path(self) -> None:
        tree = ast.parse(inspect.getsource(diagnostic.run))
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("_live_or_cached_detections", called_names)
        self.assertNotIn("_isolated_stage_one_response", called_names)
        self.assertFalse(hasattr(diagnostic, "_isolated_stage_one_response"))
        self.assertEqual(
            self.inputs["development_split"]["stage_one_response_reuse_count"],
            0,
        )
        self.assertTrue(
            self.inputs["conditions"]["ISOLATED_A1"]["fresh_live_sam_call"]
        )
        self.assertTrue(
            self.inputs["conditions"]["CONTEXT_PRESERVING_V00"][
                "fresh_live_sam_call"
            ]
        )

    def test_layouts_use_true_geom_aabbs_and_are_nonoverlapping(self) -> None:
        footprints = diagnostic._true_geom_footprints()
        self.assertEqual(set(footprints), set(diagnostic.OBJECT_TO_GEOM))
        for row in self.rows:
            layout = row["context_layout"]
            self.assertEqual(
                layout["geom_half_extents_source"],
                "MjModel.geom_aabb local AABB half extents from frozen unrotated box geoms",
            )
            positions = layout["positions_xy_m"]
            object_ids = sorted(positions)
            self.assertEqual(len(object_ids), 8)
            for index, left_id in enumerate(object_ids):
                for right_id in object_ids[index + 1 :]:
                    self.assertFalse(
                        diagnostic._overlap(
                            positions[left_id],
                            footprints[left_id],
                            positions[right_id],
                            footprints[right_id],
                        ),
                        f"{row['row_id']}: {left_id}/{right_id}",
                    )

    def test_occupancy_stack_conflict_and_supplement_are_frozen(self) -> None:
        scope = self.inputs["occupancy_and_stack_height_scope"]
        self.assertIn("no occupancy/stack-height factors", scope["known_design_schema_conflict"])
        supplement = scope["separate_supplementary_diagnostic_preregistration"]
        self.assertTrue(supplement["non_replacing"])
        self.assertTrue(supplement["must_be_frozen_before_canonical_rerun"])
        self.assertEqual(
            supplement["factors"]["board_occupancy"], ["EMPTY", "ONE_CUBE"]
        )
        self.assertEqual(supplement["factors"]["stack_height_cubes"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
