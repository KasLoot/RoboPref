from __future__ import annotations

import unittest

from experiments_suite_v2.runners import c_mon_development as target


class CMonitorDevelopmentTests(unittest.TestCase):
    def test_exact_registry_window_and_aim_order(self) -> None:
        rows = target._prefix_trials("TEST")
        self.assertEqual(len(rows), 200)
        self.assertEqual([row.order_index for row in rows], list(range(720, 920)))
        self.assertEqual(
            list(dict.fromkeys(row.case_definition.aim_id for row in rows)),
            list(target._AIM_ORDER),
        )
        for aim in target._AIM_ORDER:
            self.assertEqual(
                sum(row.case_definition.aim_id == aim for row in rows), 40
            )

    def test_frozen_visual_mappings_cover_all_registered_labels(self) -> None:
        rows = target._prefix_trials("TEST")
        labels = {
            aim: {
                target._label(row)
                for row in rows
                if row.case_definition.aim_id == aim
            }
            for aim in target._AIM_ORDER
        }
        self.assertEqual(labels["C-MON-CRIT"], set(target._CRIT_STATE))
        self.assertEqual(labels["C-MON-STATE"], set(target._STATE_EXPECTED))
        self.assertEqual(labels["C-MON-STABLE"], set(target._STABLE_SEQUENCE))
        self.assertEqual(labels["C-MON-PROG"], set(target._PROGRESS_SEQUENCE))

    def test_all_referenced_frames_are_hash_valid_third_person_frames(self) -> None:
        state_ids = {value[0] for value in target._CRIT_STATE.values()}
        state_ids.update(value[0] for value in target._STATE_EXPECTED.values())
        for sequence in target._STABLE_SEQUENCE.values():
            state_ids.update(sequence)
        for sequence in target._PROGRESS_SEQUENCE.values():
            state_ids.update(sequence)
        for index, state_id in enumerate(sorted(state_ids), start=1):
            frame = target._frame(
                state_id,
                observed_at=float(index),
                sequence=index,
            )
            self.assertEqual(frame.image_block["type"], "image_url")
            self.assertTrue(
                frame.image_block["image_url"]["url"].startswith(
                    "data:image/png;base64,"
                )
            )

    def test_local_fault_routes_use_production_monitor_error_kinds(self) -> None:
        self.assertEqual(
            target._service_fault_route("model_timeout")["observed"],
            "MODEL_ERROR_NEEDS_ATTENTION",
        )
        self.assertEqual(
            target._service_fault_route("endpoint_loss")["observed"],
            "MODEL_ERROR_NEEDS_ATTENTION",
        )
        self.assertEqual(
            target._service_fault_route("malformed_schema")["observed"],
            "OUTPUT_ERROR_NEEDS_ATTENTION",
        )
        self.assertEqual(
            target._service_fault_route("missing_frame")["observed"],
            "FRAME_ERROR_NEEDS_ATTENTION",
        )
        self.assertEqual(
            target._service_fault_route("emergency_flag")["observed"],
            "EMERGENCY_LATCHED",
        )

    def test_temporal_replay_constructs_the_production_controller(self) -> None:
        clock = target._Clock()
        controller, active, published = target._controller_for(
            target._tower_task(), clock
        )
        self.assertEqual(active.step_id, "tower")
        self.assertEqual(active.publication_id, published[0].publication_id)
        self.assertEqual(controller.snapshot.state.value, "EXECUTING")

    def test_setup_freeze_and_source_locks(self) -> None:
        result = target.validate_setup()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["row_count"], 200)
        self.assertEqual(result["physical_actions"], 0)
        self.assertEqual(result["external_model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
