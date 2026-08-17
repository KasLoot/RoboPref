from __future__ import annotations

from pathlib import Path
import unittest

from experiments_suite_v2.runners.selector_pilot import (
    aggregate_selector_pilot,
    pilot_design,
)


class SelectorPilotTests(unittest.TestCase):
    def test_design_is_three_by_five_by_four(self) -> None:
        design = pilot_design()
        self.assertEqual(len(design), 60)
        for candidate in ("cyan", "pink", "beige"):
            rows = [row for row in design if row[0] == candidate]
            self.assertEqual(len(rows), 20)
            self.assertEqual({row[1] for row in rows}, set(range(5)))
            self.assertEqual(len({row[2] for row in rows}), 4)
            self.assertIn("EMPTY", {row[3] for row in rows})
            self.assertIn("ONE_CUBE", {row[3] for row in rows})
            self.assertIn("THREE_CUBE_TOWER", {row[3] for row in rows})

    def test_aggregate_enforces_cyan_readiness_and_comparative_rule(self) -> None:
        rows = []
        for candidate in ("cyan", "pink", "beige"):
            for index in range(20):
                rows.append(
                    {
                        "candidate": candidate,
                        "unique_intended_detection": not (candidate == "cyan" and index == 0),
                        "mask_iou": {"cyan": 0.92, "pink": 0.93, "beige": 0.91}[candidate],
                        "false_positive_area_px": 0,
                        "object_confusion": False,
                        "material_exterior_leak_area_px": 0,
                        "anchor_radial_xy_error_m": 0.003,
                    }
                )
        summary = aggregate_selector_pilot(rows)
        self.assertEqual(summary["candidates"]["cyan"]["unique_intended_count"], 19)
        self.assertTrue(summary["candidates"]["cyan"]["readiness_pass"])
        self.assertTrue(summary["candidates"]["cyan"]["not_materially_worse"])
        self.assertTrue(summary["freeze_cyan"])


if __name__ == "__main__":
    unittest.main()
