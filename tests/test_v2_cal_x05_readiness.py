from __future__ import annotations

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

from experiments_suite_v2.registry import SUITE_ROOT
from experiments_suite_v2.runners.cal_x05_readiness import (
    CAL_X04_QUALIFICATION_PURPOSE,
    CAL_X04_X05_ORDERING_RESOLUTION_SHA256,
    CAL_X05_READINESS_FIELD,
    CAL_X05_READINESS_SCHEMA,
    EXPECTED_CAL_X05_MODEL_CONFIG_SHA256,
    EXPECTED_CAL_X05_PROMPT_TOOL_SCHEMA_SHA256,
    CalX05FullPathExecutor,
    CalX05ProgramContractCompiler,
    CalX05ProgramContractError,
    CalX05ReadinessError,
    CalX05ReadinessEvidence,
    CalX05ReadyCampaign,
    build_default_live_cal_x05_campaign,
    cal_x05_dry_plan,
    validate_cal_x05_frozen_configuration,
)
from experiments_suite_v2.runners.physical import (
    HiddenPlacementTruth,
    PhysicalProfileCallbacks,
    PhysicalRunError,
    build_cal_x05_rows,
    build_full_path_task,
    canonical_pick_place_program,
)
from experiments_suite_v2.runners.physical_campaign import (
    PhysicalCampaignError,
    build_default_live_physical_campaign,
)
from prefmem.execution.grounding import RGBDGrounder


class _Compiler:
    def __init__(self, program):
        self.program = program
        self.calls = 0
        self.callback = None

    def set_trace_callback(self, callback):
        self.callback = callback

    def compile(self, _task):
        self.calls += 1
        return self.program


class CalX05ReadinessTests(unittest.TestCase):
    def test_current_plan_is_read_only_exact_and_does_not_select_latest(self) -> None:
        plan = cal_x05_dry_plan(SUITE_ROOT)
        self.assertTrue(plan["dry_run"])
        self.assertFalse(plan["run_or_attempt_allocated"])
        self.assertFalse(plan["simulator_started"])
        self.assertEqual(plan["motion_commands"], 0)
        self.assertEqual(plan["external_calls"], 0)
        self.assertEqual(plan["grid"]["planned_valid_terminal_rows"], 300)
        self.assertEqual(
            plan["grid"]["row_payloads_canonical_sha256"],
            "d56a745ac553fcb70d5ee2a132b5d6be7c3aff5dd57184c759c3d384017f0a3e",
        )
        self.assertEqual(
            plan["grid"]["row_ids_canonical_sha256"],
            "c3150a38d69a605cc7f2d338b4403897fdd623708eb108fa90553a33afce2c47",
        )
        self.assertEqual(
            plan["core_blockers"],
            [
                "CAL_X05_READINESS_PROTOCOL_PIN_MISSING",
                "REPAIRED_CANONICAL_CAL_X03_NOT_DESIGNATED",
                "CAL_X04_QUALIFICATION_NOT_DESIGNATED",
            ],
        )
        self.assertNotIn("CENTRE_STACK_SUPPLEMENT_UNDEFINED", plan["blockers"])
        self.assertIn(
            "PHASE_P_600_BOTH_PER_AIM_PASS_REQUIRED_BEFORE_SUPPLEMENT",
            plan["blockers"],
        )
        self.assertEqual(plan["scope"]["supplement_rows"], 70)
        self.assertEqual(plan["scope"]["supplement_physical_actions"], 90)
        self.assertEqual(plan["scope"]["total_attempt_rows"], 970)
        self.assertEqual(plan["scope"]["total_physical_actions"], 990)
        self.assertEqual(
            plan["scope"]["supplement_status"],
            "DEFINED_FROZEN_NOT_ALLOCATED",
        )
        self.assertFalse(plan["ready_for_paired_600_campaign_allocation"])
        self.assertEqual(plan["scope"]["qualification_rows"], 300)
        self.assertEqual(plan["scope"]["paired_phase_core_rows"], 600)
        self.assertEqual(
            plan["execution_contract"]["periodic_task_stream_pause"],
            "PROHIBITED",
        )
        self.assertEqual(
            plan["execution_contract"]["phase_q_artifact_reuse_in_phase_p"],
            "PROHIBITED",
        )
        resolution = json.loads(
            (SUITE_ROOT / "protocol" / "cal_x04_x05_ordering_resolution_v1.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(resolution["counts"]["total_core_placements"], 900)
        self.assertFalse(
            resolution["no_reuse_policy"][
                "phase_q_rows_or_artifacts_reusable_in_phase_p"
            ]
        )

    def test_exact_endpoints_configuration_prompt_and_probe_are_frozen(self) -> None:
        record = validate_cal_x05_frozen_configuration(SUITE_ROOT)
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(
            record["model_config_sha256"], EXPECTED_CAL_X05_MODEL_CONFIG_SHA256
        )
        self.assertEqual(
            record["prompt_tool_schema_sha256"],
            EXPECTED_CAL_X05_PROMPT_TOOL_SCHEMA_SHA256,
        )

    def test_protocol_designation_requires_explicit_matching_bundle_path(self) -> None:
        zero = "0" * 64
        designation = {
            "schema_version": CAL_X05_READINESS_SCHEMA,
            "designation_id": "future-repaired-canonical-v1",
            "status": "CURRENT_FROZEN",
            "ordering_resolution": {
                "resolution_id": "CAL-X04-X05-ORDERING-v1",
                "path": "protocol/cal_x04_x05_ordering_resolution_v1.json",
                "sha256": CAL_X04_X05_ORDERING_RESOLUTION_SHA256,
            },
            "cal_x03": {
                "bundle_path": "preflight/CAL-X-02-CAL-X-03/future/A0",
                "result_sha256": zero,
                "attempt_manifest_sha256": zero,
                "artifact_manifest_sha256": zero,
                "checksums_sha256": zero,
            },
            "pf_selector": {
                "bundle_path": "preflight/PF-SELECTOR/future/A0",
                "result_sha256": zero,
                "attempt_manifest_sha256": zero,
                "artifact_manifest_sha256": zero,
                "checksums_sha256": zero,
            },
            "cal_x04_qualification": {
                "purpose": CAL_X04_QUALIFICATION_PURPOSE,
                "suite_run_id": "qualifier-run",
                "campaign_path": (
                    "results/shared-campaigns/future/campaigns/CAL-X-04"
                ),
                "run_manifest_sha256": zero,
                "run_checksums_sha256": zero,
                "campaign_manifest_sha256": zero,
                "progress_sha256": zero,
                "aggregate_sha256": zero,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "protocol").mkdir()
            (root / "protocol" / "cal_x04_x05_ordering_resolution_v1.json").write_bytes(
                (SUITE_ROOT / "protocol" / "cal_x04_x05_ordering_resolution_v1.json").read_bytes()
            )
            (root / "protocol" / "protocol_manifest.json").write_text(
                json.dumps({CAL_X05_READINESS_FIELD: designation}),
                encoding="utf-8",
            )
            evidence = CalX05ReadinessEvidence.from_protocol_manifest(root)
            with self.assertRaisesRegex(CalX05ReadinessError, "must be distinct"):
                evidence.assert_frozen_files_unchanged(
                    suite_root=root,
                    cal_x03_bundle_path=root / "preflight" / "wrong" / "A9",
                    cal_x04_qualification_dir=(
                        root
                        / "results/shared-campaigns/future/campaigns/CAL-X-04"
                    ),
                    pf_selector_bundle_path=(
                        root / "preflight" / "PF-SELECTOR" / "future" / "A0"
                    ),
                    paired_suite_run_id="qualifier-run",
                )
            with self.assertRaisesRegex(
                CalX05ReadinessError, "explicit cal_x03 bundle differs"
            ):
                evidence.assert_frozen_files_unchanged(
                    suite_root=root,
                    cal_x03_bundle_path=root / "preflight" / "wrong" / "A9",
                    cal_x04_qualification_dir=(
                        root
                        / "results/shared-campaigns/future/campaigns/CAL-X-04"
                    ),
                    pf_selector_bundle_path=root / "preflight" / "PF-SELECTOR" / "future" / "A0",
                    paired_suite_run_id="paired-run",
                )

    def test_program_contract_accepts_exact_row_and_rejects_selector_drift(self) -> None:
        row, other = build_cal_x05_rows()[:2]
        compiler = _Compiler(canonical_pick_place_program(row))
        contract = CalX05ProgramContractCompiler(compiler)
        contract.configure(row)
        program = contract.compile(build_full_path_task(row, published_at=1.0))
        contract.clear()
        self.assertEqual(program.source.query, row.cube_selector)
        self.assertEqual(compiler.calls, 1)

        compiler = _Compiler(canonical_pick_place_program(other))
        contract = CalX05ProgramContractCompiler(compiler)
        contract.configure(row)
        with self.assertRaises(CalX05ProgramContractError):
            contract.compile(build_full_path_task(row, published_at=1.0))
        contract.clear()

    def test_contract_fault_occurs_before_frame_capture_or_motion(self) -> None:
        row, other = build_cal_x05_rows()[:2]

        class Environment:
            periodic_rendering_paused = False

            def __init__(self):
                self.frame_calls = 0

            def sam_rgbd_frame(self):
                self.frame_calls += 1
                raise AssertionError("frame capture must not be reached")

        class Detector:
            def detect(self, *_args, **_kwargs):
                raise AssertionError("SAM must not be reached")

        class Controller:
            def __init__(self):
                self.motion_calls = 0

            def execute_pick_place(self, *_args, **_kwargs):
                self.motion_calls += 1

            def safe_hold(self):
                pass

        environment = Environment()
        controller = Controller()
        executor = CalX05FullPathExecutor(
            environment,
            compiler=_Compiler(canonical_pick_place_program(other)),
            grounder=RGBDGrounder(Detector()),
            controller=controller,
            timeout_seconds=1.0,
            poll_interval_seconds=0.001,
        )
        target = np.asarray(row.target_surface_world_m, dtype=np.float64)
        truth = HiddenPlacementTruth(
            row_id=row.row_id,
            cube_body_name=row.cube_body_name,
            source_cube_center_world_m=np.array([0.4, 0.0, 0.05]),
            source_anchor_world_m=np.array([0.4, 0.0, 0.075]),
            target_surface_anchor_world_m=target,
            expected_settled_cube_center_world_m=target + [0.0, 0.0, 0.025],
        )
        with self.assertRaises(PhysicalRunError):
            executor.execute(row, truth, PhysicalProfileCallbacks())
        self.assertEqual(environment.frame_calls, 0)
        self.assertEqual(controller.motion_calls, 0)

    def test_pause_barrier_refuses_before_delegate_allocation(self) -> None:
        class Campaign:
            aim_id = "CAL-X-05"
            campaign_dir = Path("unused")
            store = types.SimpleNamespace(suite_root=SUITE_ROOT)

            def __init__(self):
                self.calls = 0

            def run_next_paired(self, _owner):
                self.calls += 1
                return None

        environment = types.SimpleNamespace(
            realtime=True,
            width=768,
            height=768,
            periodic_rendering_paused=True,
        )
        delegate = Campaign()
        guarded = CalX05ReadyCampaign(
            delegate,
            environment=environment,
            readiness=types.SimpleNamespace(),
            cal_x03_bundle_path=Path("unused"),
            cal_x04_qualification_dir=Path("unused"),
            pf_selector_bundle_path=Path("unused"),
            readiness_artifact_sha256="0" * 64,
        )
        with self.assertRaisesRegex(CalX05ReadinessError, "prohibits"):
            guarded.run_next_paired(object())
        self.assertEqual(delegate.calls, 0)

    def test_legacy_paired_builder_fails_before_constructing_dependencies(self) -> None:
        with self.assertRaisesRegex(PhysicalCampaignError, "disabled"):
            build_default_live_physical_campaign(
                environment=None,
                store=None,
                suite_run_id="not-allocated",
                preflight=None,
            )

    def test_cal_x05_builder_refuses_missing_designation_before_run_access(self) -> None:
        class Environment:
            realtime = True
            width = 768
            height = 768
            periodic_rendering_paused = False

        class Store:
            suite_root = SUITE_ROOT

            def run_path(self, _value):
                raise AssertionError("run allocation/access must not be reached")

        with mock.patch(
            "experiments_suite_v2.runners.cal_x05_readiness.StackingEnvironment",
            Environment,
        ):
            with self.assertRaisesRegex(CalX05ReadinessError, "no uniquely designated"):
                build_default_live_cal_x05_campaign(
                    environment=Environment(),
                    store=Store(),
                    suite_run_id="must-not-be-read",
                    preflight=None,
                    cal_x03_bundle_path=Path("unused"),
                    cal_x04_qualification_dir=Path("unused"),
                    pf_selector_bundle_path=Path("unused"),
                )


if __name__ == "__main__":
    unittest.main()
