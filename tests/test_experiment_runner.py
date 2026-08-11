import json
import hashlib
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from experiments.harness.campaign import (
    ScheduleCell,
    build_blocked_schedule,
    freeze_protocol,
)
from experiments.harness.engines import (
    LiveModelServicePilotEngine,
    ScriptedEvidencePilotEngine,
)
from experiments.harness.production_engine import (
    ProductionEngineConfig,
    ProductionMujocoEngine,
    _SynchronizedDriver,
)
from experiments.harness.recording import AttemptRecorder
from experiments.harness.runner import (
    AttemptContext,
    CampaignRunner,
    CellCapability,
    HarnessInvalid,
    InfrastructureFailureEvidence,
    InfrastructureInterruption,
    ModelExchangeResult,
    _recovery_operational_provenance,
)
from experiments.harness.scenarios import load_scenarios
from experiments.harness.video import VideoConfig
from experiments.run_campaign import main as run_campaign_main


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    "experiments/harness/runner.py",
    "experiments/harness/engines.py",
    "experiments/harness/production_engine.py",
    "experiments/harness/driver.py",
    "experiments/harness/recording.py",
    "experiments/harness/video.py",
)
FROZEN_PATHS = (
    "experiments/scenarios.json",
    "experiments/config/profiles.json",
)


def _engine_identity(engine_type: type, source_path: str, engine_id: str) -> dict[str, str]:
    return {
        "engine_id": engine_id,
        "module": engine_type.__module__,
        "class_name": engine_type.__name__,
        "source_path": source_path,
        "source_sha256": hashlib.sha256((REPO_ROOT / source_path).read_bytes()).hexdigest(),
    }


def _freeze(
    root: Path,
    *,
    mode: str,
    production: bool = False,
    production_config: dict[str, object] | None = None,
    repetitions: int = 1,
) -> None:
    scenario = load_scenarios().get("NM02")
    split = "locked_test" if mode == "locked" else "pilot"
    variant_split = "locked" if mode == "locked" else "pilot"
    schedule = build_blocked_schedule(
        [
            ScheduleCell(
                matrix_id="runner-pilot",
                scenario_id="NM02",
                scenario_variant_id=scenario.split_variants[variant_split].variant_id,
                profiles=(("T5", "B1") if mode == "locked" else ("T5",)),
                repetitions=repetitions,
                executor="mujoco" if production else "synthetic_event",
            )
        ],
        master_seed=991,
        split=split,
    )
    locked_runtime = {
        "production_engine_config": (
            production_config
            if production_config is not None
            else asdict(ProductionEngineConfig())
            if production
            else {"fixture": True}
        ),
        "video_config": asdict(VideoConfig()),
        "model_service_mapping": {"upper": "fixture-loopback"},
        "reported_model_ids": {"upper": "fixture-model"},
        "service_endpoints": {"upper": "http://127.0.0.1:8000/v1"},
        "controls": {"sequential_execution": True},
        "hashes": {"fixture_sha256": "0" * 64},
        "health_checks": {"passed": True},
        "tunnel_epochs": {"upper": "fixture-epoch"},
        "environment": {"runtime": "unit-test"},
        "known_deviations": [],
    }
    freeze_protocol(
        root,
        campaign_id=f"runner-{mode}",
        mode=mode,
        protocol_source="# Runner test protocol\n",
        schedule=schedule,
        repo_root=REPO_ROOT,
        frozen_paths=FROZEN_PATHS,
        source_roots=SOURCE_ROOTS,
        metadata=(
            {
                "execution_engine": (
                    _engine_identity(
                        ProductionMujocoEngine,
                        "experiments/harness/production_engine.py",
                        ProductionMujocoEngine.engine_id,
                    )
                    if production
                    else _engine_identity(
                        ScriptedEvidencePilotEngine,
                        "experiments/harness/engines.py",
                        ScriptedEvidencePilotEngine.engine_id,
                    )
                ),
                **locked_runtime,
            }
            if mode == "locked" or production
            else None
        ),
    )


class CampaignRunnerTests(unittest.TestCase):
    def test_nonvisual_exchange_has_model_ledger_without_frame_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = AttemptRecorder.create(
                Path(directory) / "campaign",
                schedule_id="S-NONVISUAL",
                attempt_number=1,
            )
            driver = SimpleNamespace(
                emit=lambda kind, **payload: recorder.events.append(kind, **payload)
            )
            context = SimpleNamespace(recorder=recorder, driver=driver)

            result = AttemptContext.invoke_nonvisual_model_exchange(
                context,
                logical_agent="executor",
                request_id="REQ-NONVISUAL",
                call_id="CALL-NONVISUAL",
                model_id="executor-model",
                service_label="execution-compiler-openai",
                request={"messages": [{"role": "user", "content": "compile"}]},
                invoke=lambda request: ModelExchangeResult(
                    response={"program": "ok", "request": request},
                    response_id="RESP-NONVISUAL",
                    transport_metadata={"protocol": "unit-test"},
                ),
            )

            self.assertEqual(result.response_id, "RESP-NONVISUAL")
            self.assertEqual(
                (recorder.attempt_dir / "frame_requests.jsonl").read_text(), ""
            )
            call_records = [
                json.loads(line)
                for line in (recorder.attempt_dir / "model_calls.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [record["record_type"] for record in call_records],
                ["model_call_start", "model_call_end"],
            )
            self.assertEqual(call_records[0]["frame_request_ids"], [])
            event_records = [
                json.loads(line)
                for line in (recorder.attempt_dir / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [record["event_type"] for record in event_records],
                ["model_request", "model_response"],
            )

    def test_retry_setup_binds_recovery_report_and_active_tunnel_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "environment" / "recovery" / "retry.json"
            report_path.parent.mkdir(parents=True)
            payload = json.dumps(
                {"schema_version": 1, "tunnel_epoch": "gemma-forward-recovered-a2"},
                sort_keys=True,
            ).encode("utf-8")
            report_path.write_bytes(payload)
            provenance = _recovery_operational_provenance(
                root,
                {
                    "path": "environment/recovery/retry.json",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                {"gemma": "gemma-forward-initial-a1"},
            )
            self.assertEqual(
                provenance["active_recovery_tunnel_epoch"],
                "gemma-forward-recovered-a2",
            )
            self.assertEqual(
                provenance["manifest_tunnel_epochs_role"], "initial_freeze_epochs"
            )
            self.assertEqual(
                provenance["recovery_report"]["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )

    def test_live_service_transport_records_success_contract(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            return httpx.Response(
                200,
                json={"id": "live-response", "choices": [{"message": {"content": "OK"}}]},
            )

        engine = LiveModelServicePilotEngine(
            transport=httpx.MockTransport(handler)
        )
        result = engine._transport({"model": "fixture", "messages": []})
        self.assertEqual(result.response_id, "live-response")
        self.assertEqual(result.transport_metadata["status_code"], 200)

    def test_live_service_transport_needs_failed_health_probe_for_infra(self) -> None:
        def unavailable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("controlled disconnect", request=request)

        engine = LiveModelServicePilotEngine(
            transport=httpx.MockTransport(unavailable)
        )
        with self.assertRaises(InfrastructureInterruption) as caught:
            engine._transport({"model": "fixture", "messages": []})
        self.assertEqual(
            caught.exception.evidence.failure_kind,
            "transport_and_health_probe_failed",
        )

    def test_live_service_partial_infra_attempt_has_exact_error_frame_link(self) -> None:
        def unavailable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("controlled disconnect", request=request)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="pilot")
            report = CampaignRunner(
                root,
                engine=LiveModelServicePilotEngine(
                    transport=httpx.MockTransport(unavailable)
                ),
                verify_source=False,
            ).run_next()

            self.assertIsNotNone(report)
            assert report is not None
            self.assertEqual(report.run_status.value, "INFRA_INTERRUPTED")
            self.assertTrue(report.artifact_audit_passed, report.errors)
            self.assertFalse((report.attempt_path / "final_state.json").exists())

            frame_records = [
                json.loads(line)
                for line in (report.attempt_path / "frame_requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [record["record_type"] for record in frame_records],
                ["frame_request", "frame_error_link"],
            )
            error_link = frame_records[1]
            self.assertEqual(error_link["request_id"], "REQ-000001")
            self.assertEqual(error_link["call_id"], "CALL-000001")

            call_records = [
                json.loads(line)
                for line in (report.attempt_path / "model_calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(call_records[-1]["record_type"], "model_call_error")
            self.assertIsNone(call_records[-1]["response_id"])
            self.assertEqual(
                error_link["event_ids"], [call_records[-1]["event_id"]]
            )
            events = {
                event["event_id"]: event
                for event in (
                    json.loads(line)
                    for line in (report.attempt_path / "events.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            }
            error_event = events[error_link["event_ids"][0]]
            self.assertEqual(error_event["event_type"], "model_error")
            self.assertEqual(error_event["request_id"], error_link["request_id"])
            self.assertEqual(error_event["call_id"], error_link["call_id"])

    def test_live_call_failure_with_healthy_service_is_not_infra_excluded(self) -> None:
        def mixed(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": []})
            raise httpx.ConnectError("controlled call failure", request=request)

        engine = LiveModelServicePilotEngine(transport=httpx.MockTransport(mixed))
        with self.assertRaisesRegex(HarnessInvalid, "health probe passed"):
            engine._transport({"model": "fixture", "messages": []})

    def test_scripted_pilot_builds_auditable_attempt_and_seals_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="pilot")
            report = CampaignRunner(
                root,
                engine=ScriptedEvidencePilotEngine(),
                verify_source=False,
            ).run_next()

            self.assertIsNotNone(report)
            assert report is not None
            self.assertEqual(report.run_status.value, "VALID_PASS")
            self.assertEqual(report.oracle_verdict, "PASS")
            self.assertTrue(report.artifact_audit_passed, report.errors)
            result = json.loads(
                (report.attempt_path / "results.json").read_text(encoding="utf-8")
            )
            setup = json.loads(
                (report.attempt_path / "setup.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["run_status"], "VALID_PASS")
            self.assertNotIn("--setup-metadata", setup["reproduction_command"])
            self.assertIn("--no-source-check", setup["reproduction_command"])
            self.assertEqual(
                setup["engine"]["configuration_source"],
                "CAMPAIGN_MANIFEST.json#metadata",
            )
            self.assertTrue(setup["cell_capability"]["supported"])
            self.assertEqual(
                setup["cell_capability"]["scenario_variant_id"],
                setup["assignment"]["scenario_variant_id"],
            )
            self.assertEqual(
                setup["cell_capability"]["model_backbone"],
                setup["assignment"]["model_backbone"],
            )
            self.assertTrue((report.attempt_path / "video.mp4").is_file())
            self.assertTrue((report.attempt_path / "checksums.sha256").is_file())

    def test_slow_production_like_proxy_uses_one_origin_and_exact_video_mapping(self) -> None:
        class SlowProductionLikeEngine(ScriptedEvidencePilotEngine):
            def topology_manifest(self, profile, executor):
                # Longer than the configured pre-roll: video must already be
                # anchored before production-style topology construction.
                time.sleep(0.15)
                return super().topology_manifest(profile, executor)

            def execute(self, context):
                original = context.driver
                context.driver = _SynchronizedDriver(
                    original,
                    origin=context.recorder.origin,
                )
                try:
                    return super().execute(context)
                finally:
                    context.driver = original

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="pilot")
            report = CampaignRunner(
                root,
                engine=SlowProductionLikeEngine(),
                video_config=VideoConfig(
                    pre_roll_seconds=0,
                    final_hold_seconds=0.1,
                ),
                verify_source=False,
            ).run_next()

            self.assertIsNotNone(report)
            assert report is not None
            self.assertTrue(report.artifact_audit_passed, report.errors)
            setup = json.loads((report.attempt_path / "setup.json").read_text())
            metadata = json.loads(
                (report.attempt_path / "video_metadata.json").read_text()
            )
            self.assertEqual(
                setup["monotonic_clock_origin"], metadata["clock_origin"]
            )
            source_start = metadata["timeline"]["source_start_elapsed_seconds"]
            self.assertEqual(source_start, 0.0)
            events = [
                json.loads(line)
                for line in (report.attempt_path / "events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertTrue(events)
            for event in events:
                self.assertAlmostEqual(
                    event["video_time_seconds"],
                    event["elapsed_seconds"] - source_start,
                    places=9,
                )

    def test_cell_capability_is_mandatory_exact_and_precedes_reservation(self) -> None:
        class MissingCapabilityEngine(ScriptedEvidencePilotEngine):
            cell_capability = None

        class RefusingCapabilityEngine(ScriptedEvidencePilotEngine):
            def cell_capability(self, assignment, scenario, profile):
                del scenario, profile
                return CellCapability.for_assignment(
                    engine_id=self.engine_id,
                    assignment=assignment,
                    supported=False,
                    reason="controlled unsupported backbone",
                )

        class MismatchedCapabilityEngine(ScriptedEvidencePilotEngine):
            def cell_capability(self, assignment, scenario, profile):
                capability = super().cell_capability(assignment, scenario, profile)
                return replace(capability, model_backbone="wrong-backbone")

        cases = (
            (MissingCapabilityEngine, "mandatory cell_capability"),
            (RefusingCapabilityEngine, "does not support frozen cell"),
            (MismatchedCapabilityEngine, "does not exactly bind"),
        )
        for engine_type, message in cases:
            with self.subTest(engine=engine_type.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "campaign"
                _freeze(root, mode="pilot")
                with self.assertRaisesRegex(HarnessInvalid, message):
                    CampaignRunner(
                        root,
                        engine=engine_type(),
                        verify_source=False,
                    ).run_next()
                state = json.loads((root / "RUN_STATE.json").read_text(encoding="utf-8"))
                self.assertTrue(
                    all(not item["attempts"] for item in state["entries"].values())
                )
                self.assertEqual(
                    list((root / "episodes").rglob("attempt_*")),
                    [],
                )

    def test_out_of_order_cell_is_refused_before_engine_capability_binding(self) -> None:
        class CountingCapabilityEngine(ScriptedEvidencePilotEngine):
            def __init__(self):
                super().__init__()
                self.capability_calls = []

            def cell_capability(self, assignment, scenario, profile):
                self.capability_calls.append(assignment.schedule_id)
                return super().cell_capability(assignment, scenario, profile)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="pilot", repetitions=2)
            state = json.loads((root / "RUN_STATE.json").read_text(encoding="utf-8"))
            ordered = sorted(
                state["entries"].values(),
                key=lambda item: item["assignment"]["schedule_position"],
            )
            later_schedule_id = ordered[1]["assignment"]["schedule_id"]
            engine = CountingCapabilityEngine()
            with self.assertRaisesRegex(HarnessInvalid, "frozen order requires"):
                CampaignRunner(
                    root,
                    engine=engine,
                    verify_source=False,
                ).run_schedule_id(later_schedule_id)
            self.assertEqual(engine.capability_calls, [])
            unchanged = json.loads((root / "RUN_STATE.json").read_text(encoding="utf-8"))
            self.assertTrue(
                all(not item["attempts"] for item in unchanged["entries"].values())
            )
            self.assertEqual(
                list((root / "episodes").rglob("attempt_*")),
                [],
            )

    def test_instrumentation_engine_is_refused_for_locked_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="locked")
            with self.assertRaisesRegex(HarnessInvalid, "not authorized"):
                CampaignRunner(
                    root,
                    engine=ScriptedEvidencePilotEngine(),
                )

    def test_locked_runner_defaults_to_matching_frozen_instance_and_video_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="locked", production=True)
            runner = CampaignRunner(root, engine=ProductionMujocoEngine())
            try:
                self.assertEqual(runner.setup_metadata, runner.manifest["metadata"])
                self.assertEqual(asdict(runner.video_config), asdict(VideoConfig()))
                self.assertEqual(
                    asdict(runner.engine.config),
                    runner.manifest["metadata"]["production_engine_config"],
                )
            finally:
                runner.close()

    def test_locked_runner_refuses_live_engine_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="locked", production=True)
            with self.assertRaisesRegex(HarnessInvalid, "engine config differs"):
                CampaignRunner(
                    root,
                    engine=ProductionMujocoEngine(
                        ProductionEngineConfig(sam_threshold=0.4)
                    ),
                )

    def test_locked_runner_refuses_video_and_setup_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="locked", production=True)
            with self.assertRaisesRegex(HarnessInvalid, "video config differs"):
                CampaignRunner(
                    root,
                    engine=ProductionMujocoEngine(),
                    video_config=VideoConfig(crf=19),
                )
            with self.assertRaisesRegex(HarnessInvalid, "setup metadata differs"):
                CampaignRunner(
                    root,
                    engine=ProductionMujocoEngine(),
                    setup_metadata={"environment": {"injected": True}},
                )

    def test_locked_runner_rechecks_binding_before_registry_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="locked", production=True)
            runner = CampaignRunner(root, engine=ProductionMujocoEngine())
            runner.engine.config = ProductionEngineConfig(sam_threshold=0.4)
            with self.assertRaisesRegex(HarnessInvalid, "changed after locked"):
                runner.run_next()

    def test_locked_production_cli_stops_before_topology_without_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="locked", production=True)
            with (
                patch.object(
                    ProductionMujocoEngine,
                    "topology_manifest",
                    side_effect=AssertionError("must not construct production topology"),
                ) as topology,
                patch.object(
                    ProductionMujocoEngine,
                    "close",
                    autospec=True,
                ) as close,
            ):
                with self.assertRaisesRegex(Exception, "locked execution refused"):
                    run_campaign_main(
                        [
                            str(root),
                            "execute",
                            "--engine",
                            "production-mujoco",
                        ]
                    )
                topology.assert_not_called()
                self.assertTrue(close.called)

    def test_locked_cli_refuses_manual_attempt_state_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="locked", production=True)
            commands = (
                ["begin"],
                ["seal", "S000001", "1", "INVALID_HARNESS"],
                ["reconcile", "--reason", "fixture"],
            )
            for command in commands:
                with self.subTest(command=command):
                    with self.assertRaisesRegex(SystemExit, "manual locked"):
                        run_campaign_main([str(root), *command])

    def test_pilot_setup_defaults_to_manifest_without_mutable_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="pilot")
            runner = CampaignRunner(
                root,
                engine=ScriptedEvidencePilotEngine(),
                verify_source=False,
            )
            self.assertEqual(
                runner.setup_metadata["analysis_spec"],
                runner.manifest["metadata"]["analysis_spec"],
            )
            self.assertIn(
                "without checking live source",
                " ".join(runner.setup_metadata["known_deviations"]),
            )

    def test_production_cli_refuses_non_loopback_frozen_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            config = asdict(ProductionEngineConfig())
            config["upper_model_base_url"] = "http://10.0.0.7:8000/v1"
            _freeze(
                root,
                mode="pilot",
                production=True,
                production_config=config,
            )
            with self.assertRaisesRegex(SystemExit, "loopback"):
                run_campaign_main(
                    [
                        str(root),
                        "execute",
                        "--engine",
                        "production-mujoco",
                    ]
                )

    def test_infrastructure_interruption_is_preserved_with_auditable_video(self) -> None:
        class InterruptedEngine(ScriptedEvidencePilotEngine):
            engine_id = "controlled_infrastructure_interruption_v1"

            def execute(self, context):
                third = np.full((640, 640, 3), (40, 80, 120), dtype=np.uint8)
                robot = np.full((640, 640, 3), (120, 80, 40), dtype=np.uint8)
                context.capture(
                    third_person=third,
                    robot_camera=robot,
                    public_state={"state": "INITIALIZED", "active_agent": "hri"},
                    phase="PRE_TASK",
                    initial=True,
                )
                raise InfrastructureInterruption(
                    "controlled transport loss",
                    evidence=InfrastructureFailureEvidence(
                        service_label="controlled-test-service",
                        endpoint_label="loopback-test-endpoint",
                        failure_kind="connection_error",
                        transport_error_type="ControlledConnectionError",
                        safe_stop_confirmed=True,
                        health_observations=(
                            {"ok": False, "probe": "health", "status": "unreachable"},
                        ),
                    ),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            _freeze(root, mode="pilot")
            report = CampaignRunner(
                root,
                engine=InterruptedEngine(),
                verify_source=False,
            ).run_next()

            self.assertIsNotNone(report)
            assert report is not None
            self.assertEqual(report.run_status.value, "INFRA_INTERRUPTED")
            self.assertIsNone(report.oracle_verdict)
            self.assertTrue(report.artifact_audit_passed, report.errors)
            self.assertTrue((report.attempt_path / "video.mp4").is_file())
            self.assertTrue((report.attempt_path / "error.json").is_file())
            evidence = json.loads(
                (report.attempt_path / "infrastructure_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            observation = evidence["health_observations"][0]
            self.assertEqual(observation["service_id"], "controlled-test-service")
            self.assertEqual(observation["check_id"], "health-probe-001")
            self.assertEqual(observation["endpoint_label"], "loopback-test-endpoint")
            self.assertFalse(observation["passed"])
            self.assertTrue(observation["observed_utc"].endswith("Z"))


if __name__ == "__main__":
    unittest.main()
