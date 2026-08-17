from __future__ import annotations

import unittest

from experiments_suite_v2.registry import load_protocol_state, load_registry, planning_summary


class V2RegistryTests(unittest.TestCase):
    def test_registry_preserves_full_design_counts(self) -> None:
        registry = load_registry()
        summary = planning_summary(registry)
        self.assertEqual(summary["explicit_primary_valid_trials_minimum"], 22_600)
        self.assertEqual(summary["calibration_observations_or_placements"], 2_015)
        self.assertEqual(summary["workflow_ladder_if_all_conditions"], 280)
        grid = next(block for block in registry["ablations"] if block["block_id"] == "AB-MEM-Q")
        self.assertEqual(grid["parameter_cells"], 240)
        self.assertEqual(grid["frozen_needs"], 80)
        self.assertEqual(grid["planned_valid_total"], 19_200)
        semantic = next(block for block in registry["ablations"] if block["block_id"] == "AB-MEM-F")
        self.assertEqual(semantic["fixture_resolution"]["resolution_id"], "AB-MEM-F-NO-RELEVANT-EXCEPTION")
        self.assertEqual(semantic["two_turn_cleanup_audit"]["per_downstream_model"], 480)
        self.assertIn("small production-relevant VLM", semantic["confirmatory_gate"])

    def test_protocol_pins_approved_cameras_and_frozen_service_evidence(self) -> None:
        state = load_protocol_state()
        self.assertEqual(state.camera_approval["approval_status"], "APPROVED")
        self.assertTrue(state.camera_approval["approved_preview_hashes"]["sam_top_down"].startswith("dbff8a97"))
        self.assertFalse(state.confirmatory_authorized)
        services = {entry["service_id"]: entry for entry in state.services["services"]}
        gemma = services["gemma-hri-planner-monitor-validator-v1"]
        self.assertEqual(gemma["base_url"], "http://127.0.0.1:8000/v1")
        self.assertEqual(gemma["max_model_len"], 32768)
        self.assertEqual(gemma["system_fingerprint"], "vllm-0.27.1-f39dd7e9")
        self.assertEqual(gemma["probe"]["sentinel"], "SERVICE_OK")
        embedding = services["embeddinggemma-retriever-v1"]
        self.assertEqual(embedding["vector_dimension"], 768)
        sam = services["sam3.1-grounding-v1"]
        self.assertEqual(sam["probe"]["box_xyxy"], [195, 290, 341, 479])
        self.assertEqual(sam["probe"]["mask_area_px"], 27353)


if __name__ == "__main__":
    unittest.main()
