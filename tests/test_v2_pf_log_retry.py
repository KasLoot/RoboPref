from __future__ import annotations

from pathlib import Path
import unittest

from experiments_suite_v2.io import load_json, sha256_file
from experiments_suite_v2.registry import SUITE_ROOT
from experiments_suite_v2.runners.pf_log_retry import (
    EXPECTED_CAMERA_CONFIGURATION_SHA256,
    PF_LOG_A0_RELATIVE,
    PF_LOG_EVIDENCE_POLICY_RELATIVE,
    PF_LOG_ORPHAN_RELATIVE,
    PF_LOG_READINESS_RELATIVE,
    PF_LOG_SUITE_RUN_ID,
    PFLogRetryError,
    PFLogRetryInspection,
    _frozen_row,
    _verify_live_task_stream_gate,
)
from simulation.stacking import StackingEnvironment


class PFLogTaskStreamGateTests(unittest.TestCase):
    def _inspection(self) -> PFLogRetryInspection:
        suite_root = SUITE_ROOT.resolve()
        a0_dir = suite_root / PF_LOG_A0_RELATIVE
        manifest = load_json(a0_dir / "attempt_manifest.json")
        run_root = (
            suite_root
            / "results/PF/PF-LOG/evidence/campaigns"
            / PF_LOG_SUITE_RUN_ID
        )
        return PFLogRetryInspection(
            suite_root=suite_root,
            run_root=run_root,
            a0_dir=a0_dir,
            readiness_dir=suite_root / PF_LOG_READINESS_RELATIVE,
            orphan_dir=suite_root / PF_LOG_ORPHAN_RELATIVE,
            next_attempt_dir=run_root / "PF-LOG" / "unused-test-attempt",
            previous_invalid_dir=a0_dir,
            next_attempt_number=1,
            next_attempt_id="unused-test-attempt.A1",
            row=_frozen_row(manifest),
            a0_manifest=manifest,
            evidence_policy_sha256=sha256_file(
                suite_root / PF_LOG_EVIDENCE_POLICY_RELATIVE
            ),
            readiness_summary_sha256="0" * 64,
            source_hashes={},
        )

    def test_exact_768_synchronized_pair_passes_frozen_camera_gate(self) -> None:
        environment = StackingEnvironment(
            scene_path=SUITE_ROOT / "scenes" / "tb6c_v2.xml",
            start=True,
            viewer=False,
            realtime=True,
            width=768,
            height=768,
        )
        try:
            result = _verify_live_task_stream_gate(
                self._inspection(),
                environment,
            )
        finally:
            environment.close()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["approved_resolution"], [768, 768])
        self.assertTrue(result["synchronized_pair"])
        self.assertEqual(
            result["camera_configuration"]["sha256"],
            EXPECTED_CAMERA_CONFIGURATION_SHA256,
        )
        for name in ("sam_camera", "prefmem_camera"):
            self.assertEqual(result["streams"][name]["rgb_shape"], [768, 768, 3])
            self.assertEqual(result["streams"][name]["depth_shape"], [768, 768])

    def test_640_pair_is_rejected_before_motion(self) -> None:
        environment = StackingEnvironment(
            scene_path=SUITE_ROOT / "scenes" / "tb6c_v2.xml",
            start=True,
            viewer=False,
            realtime=True,
            width=640,
            height=640,
        )
        try:
            with self.assertRaisesRegex(PFLogRetryError, "768x768"):
                _verify_live_task_stream_gate(
                    self._inspection(),
                    environment,
                )
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
