from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import httpx
import numpy as np

from experiments_suite_v2.io import atomic_write_json
from experiments_suite_v2.runners import cal_x03_image_predictor_diagnostic as diagnostic
from experiments_suite_v2.runners import sam31_image_predictor_adapter as adapter
from experiments_suite_v2.runners import sam31_diagnostic_evidence as evidence


class _Response:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content


def _build_identity(mode: str) -> dict[str, object]:
    static = {
        "upstream_commit": diagnostic.UPSTREAM_COMMIT,
        "git_status_porcelain": (
            " M sam3/model/sam3_base_predictor.py\n?? sam31_server.py"
        ),
        "git_status_pass": True,
        "source_records": [
            {"path": path, "sha256": digest, "size_bytes": 1}
            for path, digest in diagnostic.ADAPTER_SOURCE_PINS.items()
        ],
        "source_records_sha256": "1" * 64,
        "imported_sam_python_source_closure": {
            "records_sha256": "2" * 64,
            "file_count": 100,
            "records": [
                {"path": path, "sha256": digest}
                for path, digest in diagnostic.ADAPTER_SOURCE_PINS.items()
                if path.startswith("sam3/")
            ],
        },
        "imported_sam_modules": {
            "records_sha256": "4" * 64,
            "records": [
                {
                    "module": "sam3.model_builder",
                    "path": "sam3/model_builder.py",
                    "sha256": diagnostic.ADAPTER_SOURCE_PINS["sam3/model_builder.py"],
                },
                {
                    "module": (
                        "sam3.model.sam3_image"
                        if mode == "image"
                        else "sam3.model.sam3_base_predictor"
                    ),
                    "path": (
                        "sam3/model/sam3_image.py"
                        if mode == "image"
                        else "sam3/model/sam3_base_predictor.py"
                    ),
                    "sha256": diagnostic.ADAPTER_SOURCE_PINS[
                        (
                            "sam3/model/sam3_image.py"
                            if mode == "image"
                            else "sam3/model/sam3_base_predictor.py"
                        )
                    ],
                },
                *(
                    [
                        {
                            "module": "sam3.model.sam3_image_processor",
                            "path": "sam3/model/sam3_image_processor.py",
                            "sha256": diagnostic.ADAPTER_SOURCE_PINS[
                                "sam3/model/sam3_image_processor.py"
                            ],
                        }
                    ]
                    if mode == "image"
                    else []
                ),
            ],
        },
        "checkpoint_sha256": diagnostic.CHECKPOINT_SHA256,
        "checkpoint_size_bytes": diagnostic.CHECKPOINT_SIZE_BYTES,
        "package_inventory": {
            "records_sha256": "3" * 64,
            "required_versions_pass": True,
            "records": [],
        },
        "python_version": "3.12.3",
        "adapter_source_sha256": diagnostic.ADAPTER_SHA256,
    }
    options = (
        {
            "builder": "build_sam3_predictor",
            "version": "sam3.1",
            "checkpoint_path_explicit": True,
            "max_num_objects": 16,
            "multiplex_count": 16,
            "use_fa3": False,
            "use_rope_real": False,
            "compile": False,
            "warm_up": False,
            "async_loading_frames": False,
        }
        if mode == "multiplex"
        else {
            "builder": "build_sam3_image_model",
            "checkpoint_path_explicit": True,
            "load_from_HF": False,
            "device": "cuda",
            "eval_mode": True,
            "enable_segmentation": True,
            "enable_inst_interactivity": False,
            "compile": False,
            "processor": "Sam3Processor",
            "processor_resolution": 1008,
            "processor_confidence_threshold_initial": 0.5,
        }
    )
    return {
        "schema_version": "fixture.build",
        "condition_id": (
            diagnostic.CONDITIONS[0] if mode == "multiplex" else diagnostic.CONDITIONS[1]
        ),
        "mode": mode,
        "upstream_and_environment": static,
        "model_options": options,
        "direct_image_checkpoint_coverage": (
            None
            if mode == "multiplex"
            else {
                "pass": True,
                "raw_selected_detector_key_count": 100,
                "mapped_detector_key_count": 100,
                "image_model_state_key_count": 100,
                "missing_key_count": 0,
                "unexpected_key_count": 0,
                "shape_mismatch_count": 0,
                "mapping_collision_count": 0,
            }
        ),
        "no_detect_calls_before_health": True,
    }


def _health(mode: str, *, pid: int) -> dict[str, object]:
    condition = (
        diagnostic.CONDITIONS[0] if mode == "multiplex" else diagnostic.CONDITIONS[1]
    )
    build = _build_identity(mode)
    return {
        "status": "ok",
        "development_only": True,
        "condition_id": condition,
        "mode": mode,
        "boot_id": f"boot-{mode}",
        "process_id": pid,
        "model_loaded": True,
        "detect_request_count": 0,
        "successful_detect_response_count": 0,
        "failed_detect_request_count": 0,
        "build_identity": build,
        "build_identity_sha256": diagnostic.canonical_sha256(build),
        "gpu": {
            "device_index": 0,
            "name": "NVIDIA GeForce RTX 4090",
            "gpu_uuid": "GPU-fixture",
            "driver_version": "580.159.04",
            "torch_version": "2.10.0+cu128",
            "torch_cuda_version": "12.8",
            "cuda_available": True,
            "cuda_device_capability": [8, 9],
            "total_memory_bytes": 24 * 1024**3,
            "free_memory_bytes": 8 * 1024**3,
            "allocated_memory_bytes": 6 * 1024**3,
            "reserved_memory_bytes": 6 * 1024**3,
            "cuda_visible_devices": "0",
            "nvidia_visible_devices": None,
            "cuda_device_order": "PCI_BUS_ID",
        },
    }


_SERVER_PATH = "/workspace/sam3/sam31_server.py"
_CHECKPOINT_PATH = "/workspace/checkpoints/sam3.1_multiplex.pt"
_SERVER_SHA = "9682104b0239e81726af15a3e470a2c3770fda6db2d27aed1016e750394a158e"


def _receipt(
    command: str,
    stdout: str = "",
    *,
    captured_at: str = "2026-08-12T20:00:00+00:00",
) -> dict[str, object]:
    return {
        "command": command,
        "stdout": stdout,
        "stderr": "",
        "exit_code": 0,
        "captured_at": captured_at,
    }


def _production_receipts(*, free_mib: int = 8192) -> dict[str, dict[str, object]]:
    stat_fields = ["S", *("0" for _ in range(18)), "12345"]
    return {
        "socket": _receipt(
            "ss -ltnp sport = :9000",
            'LISTEN 0 2048 127.0.0.1:9000 0.0.0.0:* users:(("python",pid=900,fd=3))\n',
        ),
        "proc_stat": _receipt(
            "cat /proc/900/stat", "900 (python) " + " ".join(stat_fields) + "\n"
        ),
        "proc_cmdline": _receipt(
            "cat /proc/900/cmdline",
            f"python uvicorn {_SERVER_PATH} --port 9000\n",
        ),
        "proc_cwd": _receipt(
            "readlink -f /proc/900/cwd", "/workspace/sam3\n"
        ),
        "server_sha256": _receipt(
            f"sha256sum {_SERVER_PATH}", f"{_SERVER_SHA}  {_SERVER_PATH}\n"
        ),
        "checkpoint_sha256": _receipt(
            f"sha256sum {_CHECKPOINT_PATH}",
            f"{diagnostic.CHECKPOINT_SHA256}  {_CHECKPOINT_PATH}\n",
        ),
        "source_sha256": _receipt(
            f"sha256sum {_SERVER_PATH}", f"{_SERVER_SHA}  {_SERVER_PATH}\n"
        ),
        "git_commit": _receipt(
            "git rev-parse HEAD", diagnostic.UPSTREAM_COMMIT + "\n"
        ),
        "git_status": _receipt(
            "git status --porcelain=v1",
            " M sam3/model/sam3_base_predictor.py\n?? sam31_server.py\n",
        ),
        "python_version": _receipt("python --version", "Python 3.12.3\n"),
        "package_inventory": _receipt(
            "pip freeze", "torch==2.10.0\nfastapi==0.141.1\n"
        ),
        "gpu": _receipt(
            "nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,memory.free",
            f"GPU-fixture, NVIDIA GeForce RTX 4090, 580.159.04, 24564, {free_mib}\n",
        ),
        "compute_apps": _receipt(
            "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory",
            "900, GPU-fixture, 1750\n",
        ),
        "health": _receipt(
            "curl http://127.0.0.1:9000/health",
            '{"status":"ok"}\n__HTTP_STATUS__:200\n',
        ),
    }


def _production_evidence() -> dict[str, object]:
    return evidence.derive_production_evidence(
        _production_receipts(),
        expected_port=9000,
        expected_server_path=_SERVER_PATH,
        expected_checkpoint_path=_CHECKPOINT_PATH,
        expected_generated_server_sha256=_SERVER_SHA,
        expected_checkpoint_sha256=diagnostic.CHECKPOINT_SHA256,
        expected_upstream_commit=diagnostic.UPSTREAM_COMMIT,
        expected_gpu_uuid="GPU-fixture",
    )


def _shutdown_evidence() -> dict[str, object]:
    before = "2026-08-12T20:01:00+00:00"
    terminated = "2026-08-12T20:01:01+00:00"
    exited = "2026-08-12T20:01:02+00:00"
    after = "2026-08-12T20:01:03+00:00"
    receipts = {
        "sockets_after": _receipt(
            "ss -H -ltnp",
            'LISTEN 0 2048 127.0.0.1:9000 0.0.0.0:* users:(("python",pid=900,fd=3))\n',
            captured_at=after,
        ),
        "process_table_after": _receipt(
            "ps -p 900,901,902 -o pid=,args=",
            "900 production\n",
            captured_at=after,
        ),
        "gpu_before": _receipt(
            "nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits",
            "GPU-fixture, NVIDIA GeForce RTX 4090, 580.159.04, 24564, 4096\n",
            captured_at=before,
        ),
        "gpu_after": _receipt(
            "nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits",
            "GPU-fixture, NVIDIA GeForce RTX 4090, 580.159.04, 24564, 12288\n",
            captured_at=after,
        ),
        "compute_apps_after": _receipt(
            "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits",
            "900, GPU-fixture, 1750\n",
            captured_at=after,
        ),
        "sigterm_901": _receipt("kill -TERM 901", captured_at=terminated),
        "exit_901": _receipt("test ! -e /proc/901", captured_at=exited),
        "sigterm_902": _receipt("kill -TERM 902", captured_at=terminated),
        "exit_902": _receipt("test ! -e /proc/902", captured_at=exited),
    }
    return evidence.derive_adapter_shutdown_evidence(
        receipts,
        adapter_pids=[901, 902],
        production_pre=_production_evidence(),
        adapter_ports=[9001, 9002],
    )


class _FakeClient:
    def __init__(self, health: dict[str, dict[str, object]]) -> None:
        self.health = health
        self.posts: list[dict[str, object]] = []
        mask = np.zeros((diagnostic.FRAME_SIZE, diagnostic.FRAME_SIZE), dtype=np.uint8)
        mask[100:110, 200:220] = 255
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            raise AssertionError("fixture mask encoding failed")
        self.mask_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    def get(self, url: str) -> _Response:
        condition = (
            diagnostic.CONDITIONS[0] if ":9001/" in url else diagnostic.CONDITIONS[1]
        )
        return _Response(200, json.dumps(self.health[condition]).encode("utf-8"))

    def post(self, url: str, *, data, files) -> _Response:
        condition = (
            diagnostic.CONDITIONS[0] if ":9001/" in url else diagnostic.CONDITIONS[1]
        )
        health = self.health[condition]
        health["detect_request_count"] = int(health["detect_request_count"]) + 1
        health["successful_detect_response_count"] = int(
            health["successful_detect_response_count"]
        ) + 1
        self.posts.append({"url": url, "data": dict(data), "files": files})
        body = {
            "prompt": data["prompt"],
            "threshold": float(data["threshold"]),
            "image": {"width": diagnostic.FRAME_SIZE, "height": diagnostic.FRAME_SIZE},
            "count": 1,
            "detections": [
                {
                    "object_id": 7,
                    "box_xyxy": [200, 100, 220, 110],
                    "mask_area": 200,
                    "mask_png_base64": self.mask_base64,
                    "score": 0.875,
                }
            ],
            "diagnostic_trace": {
                "request_id": data["request_id"],
                "condition_id": condition,
                "boot_id": health["boot_id"],
                "build_identity_sha256": health["build_identity_sha256"],
                "transmitted_jpeg_sha256": data["payload_sha256"],
                "server_decoded_rgb_array_sha256": "a" * 64,
                "predictor_resource_file_sha256": (
                    "b" * 64 if condition == diagnostic.CONDITIONS[0] else None
                ),
                "predictor_resource_decoded_rgb_array_sha256": "c" * 64,
                "predictor_resource_dimensions_wh": [
                    diagnostic.FRAME_SIZE,
                    diagnostic.FRAME_SIZE,
                ],
                "prompt_utf8_sha256": diagnostic.hashlib.sha256(
                    data["prompt"].encode("utf-8")
                ).hexdigest(),
                "uploaded_dimensions_wh": [diagnostic.FRAME_SIZE, diagnostic.FRAME_SIZE],
                "threshold": float(data["threshold"]),
                "response_box_policy": "exclusive XYXY derived from final boolean mask",
                "preprocessor": (
                    {
                        "measurement_basis": "fixture",
                        "wrapper_reencode": "PIL JPEG quality=100",
                        "model_image_size": 1008,
                        "image_mean": [0.5, 0.5, 0.5],
                        "image_std": [0.5, 0.5, 0.5],
                    }
                    if condition == diagnostic.CONDITIONS[0]
                    else {
                        "measurement_basis": "fixture",
                        "wrapper_reencode": None,
                        "resize_hw": [1008, 1008],
                        "padding": None,
                        "image_mean": [0.5, 0.5, 0.5],
                        "image_std": [0.5, 0.5, 0.5],
                        "boxes_scaled_to_original_dimensions": True,
                        "masks_interpolated_to_original_dimensions": True,
                    }
                ),
                "precision": (
                    {
                        "policy": "predictor-managed autocast from pinned source",
                        "outer_cuda_autocast_enabled_before_request": False,
                        "model_parameter_dtypes": ["torch.float32"],
                    }
                    if condition == diagnostic.CONDITIONS[0]
                    else {
                        "policy": "verbatim official Sam3Processor inference path",
                        "outer_cuda_autocast_enabled_before_processor": False,
                        "model_parameter_dtypes": ["torch.float32"],
                        "state_tensor_dtypes": ["torch.float32"],
                        "state_tensor_trace": {
                            "records": [
                                {
                                    "path": "$/nested/features",
                                    "dtype": "torch.float32",
                                    "shape": [1, 4],
                                }
                            ],
                            "record_count": 1,
                            "records_sha256": diagnostic.canonical_sha256(
                                [
                                    {
                                        "path": "$/nested/features",
                                        "dtype": "torch.float32",
                                        "shape": [1, 4],
                                    }
                                ]
                            ),
                            "container_count": 2,
                            "non_tensor_leaf_count": 0,
                            "traversal": (
                                "recursive mappings/sequences; deterministic "
                                "JSON-pointer paths"
                            ),
                        },
                    }
                ),
                "raw_model_output": {
                    "mask_shape": [1, diagnostic.FRAME_SIZE, diagnostic.FRAME_SIZE],
                    "mask_dtype": "bool",
                    "score_shape": [1],
                    "score_dtype": "float32",
                    "coordinate_space": "fixture original pixels",
                    **(
                        {"object_id_shape": [1]}
                        if condition == diagnostic.CONDITIONS[0]
                        else {"processor_boxes_xyxy": [[200, 100, 220, 110]]}
                    ),
                },
                "production_postprocess_audit": (
                    {"pass": True, "anomalies": []}
                    if condition == diagnostic.CONDITIONS[0]
                    else None
                ),
                "session_close_failure_policy": (
                    "suppressed after response materialization, matching production"
                    if condition == diagnostic.CONDITIONS[0]
                    else None
                ),
                "postprocessed_detection_count": 1,
            },
        }
        return _Response(200, json.dumps(body).encode("utf-8"))


def _fake_scored_row(value, _observation, _masks, condition, _detections, _responses, **_kwargs):
    intended = condition == diagnostic.CONDITIONS[1] or value["cohort"] == "STABLE_CONTROL"
    grounded = intended
    return {
        "diagnostic_row_id": value["diagnostic_row_id"],
        "condition": condition,
        "cohort": value["cohort"],
        "repeat_index": value["repeat_index"],
        "production_grounding_succeeded": grounded,
        "production_selected_intended_iou_pass": intended,
        "sam_anchor_error_mm": [0.0, 0.0, 0.0] if grounded else None,
        "sam_radial_xy_error_mm": 0.0 if grounded else None,
        "any_candidate_cross_object_collision": False,
        "selected_candidate_cross_object_collision": False,
        "ambiguous_selection_failure": False,
    }


class CalX03ImagePredictorDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = diagnostic._input_rows()

    def _attestation(self, root: Path):
        health = {
            diagnostic.CONDITIONS[0]: _health("multiplex", pid=901),
            diagnostic.CONDITIONS[1]: _health("image", pid=902),
        }
        value = {
            "captured_at": "fixture",
            "plan_id": diagnostic.PLAN_ID,
            "health": health,
            "production_9000": _production_evidence(),
        }
        path = root / "deployment_attestation.json"
        diagnostic.write_deployment_attestation(path, value)
        return path, health

    def _lifecycle_paths(self, root: Path) -> tuple[Path, Path]:
        production_post = root / "production_post.json"
        atomic_write_json(production_post, _production_evidence(), overwrite=False)
        shutdown = root / "shutdown.json"
        atomic_write_json(shutdown, _shutdown_evidence(), overwrite=False)
        return production_post, shutdown

    def _allocated_precommit(self, root: Path) -> tuple[Path, dict[str, dict[str, object]]]:
        attestation, health = self._attestation(root)
        output = root / "A0"
        diagnostic.allocate(output, attestation, execution_authorized=True)
        diagnostic.precommit_payloads(output)
        return output, health

    def _run_fake_live(
        self, output: Path, health: dict[str, dict[str, object]]
    ) -> tuple[_FakeClient, dict[str, object]]:
        client = _FakeClient(health)
        dummy = SimpleNamespace()
        with mock.patch.object(
            diagnostic, "_load_observation_and_masks", return_value=(dummy, {})
        ), mock.patch.object(
            diagnostic, "_existing_condition_result", side_effect=_fake_scored_row
        ):
            live = diagnostic.run_live(output, client=client)
        return client, live

    def test_exact_registered_schedule_and_dependence(self) -> None:
        contracts = diagnostic._call_contracts(self.rows, attempt_id="FIXTURE-A0")
        self.assertEqual(len(self.rows), 36)
        self.assertEqual(len({row["unique_stimulus_id"] for row in self.rows}), 18)
        self.assertEqual(len(contracts), 72)
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in self.rows:
            grouped.setdefault(str(row["unique_stimulus_id"]), []).append(row)
        for values in grouped.values():
            self.assertEqual({int(item["repeat_index"]) for item in values}, {0, 1})
            self.assertEqual(len({item["selector"] for item in values}), 1)
            self.assertEqual(len({item["object_id"] for item in values}), 1)
            self.assertEqual(len({item["position_id"] for item in values}), 1)
            self.assertEqual(
                abs(int(values[0]["pair_index"]) - int(values[1]["pair_index"])), 1
            )
        self.assertEqual([item["dispatch_ordinal"] for item in contracts], list(range(72)))
        for index in range(0, 72, 2):
            self.assertEqual(
                contracts[index]["diagnostic_row_id"],
                contracts[index + 1]["diagnostic_row_id"],
            )
            self.assertNotEqual(contracts[index]["condition"], contracts[index + 1]["condition"])
        first = [item for item in contracts if item["within_frame_ordinal"] == 0]
        for condition in diagnostic.CONDITIONS:
            self.assertEqual(sum(item["condition"] == condition for item in first), 18)
            for cohort in ("RESIDUAL", "STABLE_CONTROL"):
                self.assertEqual(
                    sum(
                        item["condition"] == condition and item["cohort"] == cohort
                        for item in first
                    ),
                    9,
                )

    def test_allocation_requires_prior_attestation_and_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(diagnostic.AttestationError):
                diagnostic.allocate(root / "A0", root / "missing.json", execution_authorized=True)
            path, _health_map = self._attestation(root)
            with self.assertRaises(diagnostic.AttestationError):
                diagnostic.allocate(root / "A0", path, execution_authorized=False)
            result = diagnostic.allocate(root / "A0", path, execution_authorized=True)
            self.assertEqual(result["live_detect_calls"], 0)
            committed = diagnostic.precommit_payloads(root / "A0")
            self.assertEqual(committed["payload_count"], 72)
            self.assertEqual(committed["unique_payload_sha256_count"], 18)
            self.assertEqual(committed["response_count"], 0)

    def test_postallocation_snapshot_failure_retains_embedded_rollback_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _health_map = self._attestation(root)
            output = root / "A0"
            with mock.patch.object(
                diagnostic,
                "_immutable_bytes",
                side_effect=diagnostic.DiagnosticError("injected snapshot failure"),
            ):
                with self.assertRaises(diagnostic.DiagnosticError):
                    diagnostic.allocate(output, path, execution_authorized=True)
            inputs = diagnostic.load_json(output / "inputs.json")
            embedded = inputs["deployment_attestation"]["document"]
            self.assertEqual(
                embedded["health"][diagnostic.CONDITIONS[0]]["process_id"], 901
            )
            failure = diagnostic.load_json(output / "failure.json")
            self.assertEqual(failure["classification"], "INVALID_SETUP")
            self.assertEqual(failure["completed_physical_post_attempts"], 0)
            self.assertEqual(failure["confirmed_physical_post_attempts"], 0)
            self.assertEqual(failure["unknown_physical_post_outcomes"], 0)
            self.assertFalse((output / "artifact_manifest.json").exists())
            with self.assertRaises(diagnostic.AttemptInvalidity):
                diagnostic.allocate(output, path, execution_authorized=True)
            with self.assertRaises(diagnostic.AttemptInvalidity):
                diagnostic.precommit_payloads(output)
            rejected_client = _FakeClient(_health_map)
            with self.assertRaises(diagnostic.AttemptInvalidity):
                diagnostic.run_live(output, client=rejected_client)
            self.assertEqual(rejected_client.posts, [])

    def test_invalidity_classification_no_replacement(self) -> None:
        self.assertEqual(
            diagnostic.classify_call_outcome(
                "MALFORMED_RESPONSE", endpoint_healthy_same_boot=True
            )["classification"],
            "CAPABILITY_FAIL",
        )
        transport = diagnostic.classify_call_outcome(
            "TRANSPORT", endpoint_healthy_same_boot=False
        )
        self.assertEqual(transport["classification"], "INVALID_RUN")
        self.assertFalse(transport["same_attempt_resume"])
        self.assertFalse(transport["prior_response_reuse"])
        self.assertEqual(
            diagnostic.classify_call_outcome(
                None, endpoint_healthy_same_boot=True, harness_or_adapter_defect=True
            )["classification"],
            "INVALID_SETUP",
        )

    def test_duplicate_stimulus_stability_score_tolerance(self) -> None:
        values = []
        for condition in diagnostic.CONDITIONS:
            for index in range(18):
                for repeat in range(2):
                    values.append(
                        {
                            "condition": condition,
                            "unique_stimulus_id": f"stimulus-{index}",
                            "capability_failure": False,
                            "detections": [
                                {
                                    "raw_detection_index": 0,
                                    "object_id": 1,
                                    "box_xyxy": [1, 2, 3, 4],
                                    "mask_area": 2,
                                    "logical_mask_sha256": "a" * 64,
                                    "score": 0.5 + repeat * 5e-7,
                                }
                            ],
                        }
                    )
        self.assertTrue(diagnostic.duplicate_stimulus_stability(values)["pass"])
        values[-1]["detections"][0]["score"] = 0.6
        self.assertFalse(diagnostic.duplicate_stimulus_stability(values)["pass"])

    def test_repeat_outcome_instability_is_valid_negative_not_exception(self) -> None:
        rows = []
        for value in self.rows:
            for condition in diagnostic.CONDITIONS:
                intended = condition == diagnostic.CONDITIONS[1] or value[
                    "cohort"
                ] == "STABLE_CONTROL"
                if (
                    condition == diagnostic.CONDITIONS[1]
                    and value["repeat_index"] == 1
                    and value["unique_stimulus_id"]
                    == self.rows[0]["unique_stimulus_id"]
                ):
                    intended = False
                rows.append(
                    {
                        **_fake_scored_row(
                            value, None, {}, condition, (), (),
                        ),
                        "unique_stimulus_id": value["unique_stimulus_id"],
                        "usable_response": True,
                        "capability_failure": False,
                        "trace_audit_pass": True,
                        "coordinate_audit_pass": True,
                        "production_selected_intended_iou_pass": intended,
                    }
                )
        summary = diagnostic.aggregate_scored_rows(rows)
        image_interval = summary["intended_success_wilson_intervals"][
            diagnostic.CONDITIONS[1]
        ]
        self.assertFalse(image_interval["computed"])
        self.assertFalse(summary["serving_inference_path_bundle_candidate_eligible"])

    def test_full_fake_end_to_end_dispatch_scoring_and_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, health = self._allocated_precommit(root)
            client, live = self._run_fake_live(output, health)
            self.assertEqual(live["completed_physical_post_attempts"], 72)
            self.assertEqual(live["confirmed_physical_post_attempts"], 72)
            self.assertEqual(live["unknown_physical_post_outcomes"], 0)
            self.assertEqual(len(client.posts), 72)
            self.assertTrue(live["duplicate_stimulus_stability_pass"])
            self.assertTrue(live["paired_trace_boundary_audit_pass"])
            for ordinal, posted in enumerate(client.posts):
                self.assertIn("request_id", posted["data"])
                self.assertEqual(
                    diagnostic.sha256_file(
                        output
                        / diagnostic.load_json(output / "request_index.json")["records"][ordinal][
                            "payload_path"
                        ]
                    ),
                    posted["data"]["payload_sha256"],
                )
            production_post, shutdown = self._lifecycle_paths(root)
            dummy = SimpleNamespace()
            with mock.patch.object(
                diagnostic, "_load_observation_and_masks", return_value=(dummy, {})
            ), mock.patch.object(
                diagnostic, "_existing_condition_result", side_effect=_fake_scored_row
            ):
                summary = diagnostic.finalize_run(output, production_post, shutdown)
            self.assertTrue(summary["multiplex_exact_prior_full_768_profile_reproduced"])
            self.assertTrue(summary["direct_image_all_or_none_gate_pass"])
            self.assertTrue(summary["serving_inference_path_bundle_candidate_eligible"])
            self.assertEqual(
                summary["paired_intended_success_risk_difference"]["paired_unit_count"],
                18,
            )
            self.assertEqual(
                summary["recommendation"], "SERVING_INFERENCE_PATH_BUNDLE_CANDIDATE"
            )
            self.assertEqual(diagnostic.verify_row_bundle(output), [])
            replay = diagnostic.finalize_run(output, production_post, shutdown)
            self.assertEqual(replay["recommendation"], summary["recommendation"])
            self.assertTrue((output / "artifact_manifest.json").is_file())

    def test_interrupted_intent_terminalizes_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, health = self._allocated_precommit(root)
            atomic_write_json(
                output / "request_attempts" / "000-interrupted" / "intent.json",
                {"physical_post_attempted": True},
                overwrite=False,
            )
            client = _FakeClient(health)
            with self.assertRaises(diagnostic.AttemptInvalidity):
                diagnostic.run_live(output, client=client)
            self.assertEqual(client.posts, [])
            failure = diagnostic.load_json(output / "failure.json")
            self.assertEqual(failure["classification"], "INVALID_RUN")
            self.assertEqual(failure["completed_physical_post_attempts"], 0)
            self.assertEqual(failure["confirmed_physical_post_attempts"], 0)
            self.assertEqual(failure["unknown_physical_post_outcomes"], 1)
            self.assertEqual(failure["maximum_possible_physical_post_attempts"], 1)
            self.assertFalse((output / "artifact_manifest.json").exists())

    def test_predispatch_restarted_boot_is_invalid_run_zero_posts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, health = self._allocated_precommit(root)
            health[diagnostic.CONDITIONS[0]]["boot_id"] = "restarted-boot"
            client = _FakeClient(health)
            with self.assertRaises(diagnostic.AttemptInvalidity):
                diagnostic.run_live(output, client=client)
            self.assertEqual(client.posts, [])
            failure = diagnostic.load_json(output / "failure.json")
            self.assertEqual(failure["classification"], "INVALID_RUN")
            self.assertEqual(failure["completed_physical_post_attempts"], 0)
            self.assertEqual(failure["confirmed_physical_post_attempts"], 0)
            self.assertEqual(failure["unknown_physical_post_outcomes"], 0)

    def test_raw_receipt_tamper_blocks_finalization_and_awaits_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, health = self._allocated_precommit(root)
            _client, _live = self._run_fake_live(output, health)
            receipt = next(output.glob("responses/*/http_receipt.json"))
            value = diagnostic.load_json(receipt)
            value["raw_body_base64"] = base64.b64encode(b"{}").decode("ascii")
            atomic_write_json(receipt, value)
            production_post, shutdown = self._lifecycle_paths(root)
            with self.assertRaises(diagnostic.AttemptInvalidity):
                diagnostic.finalize_run(output, production_post, shutdown)
            failure = diagnostic.load_json(output / "failure.json")
            self.assertEqual(failure["classification"], "INVALID_SETUP")
            self.assertTrue((output / "finalization_failure.json").is_file())
            self.assertFalse((output / "artifact_manifest.json").exists())

    def test_seal_failure_becomes_invalid_and_can_finalize_after_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, health = self._allocated_precommit(root)
            self._run_fake_live(output, health)
            production_post, shutdown = self._lifecycle_paths(root)
            dummy = SimpleNamespace()
            with mock.patch.object(
                diagnostic, "_load_observation_and_masks", return_value=(dummy, {})
            ), mock.patch.object(
                diagnostic, "_existing_condition_result", side_effect=_fake_scored_row
            ), mock.patch.object(
                diagnostic,
                "finalize_bundle_files",
                side_effect=diagnostic.DiagnosticError("injected seal failure"),
            ):
                with self.assertRaises(diagnostic.DiagnosticError):
                    diagnostic.finalize_run(output, production_post, shutdown)
            self.assertEqual(
                diagnostic.load_json(output / "failure.json")["classification"],
                "INVALID_SETUP",
            )
            self.assertTrue((output / "failed_seal" / "results_candidate.json").is_file())
            self.assertFalse((output / "artifact_manifest.json").exists())
            finalized = diagnostic.finalize_invalid_after_rollback(
                output, production_post, shutdown
            )
            self.assertEqual(finalized["analytical_verdict"], "INVALID_SETUP")
            self.assertEqual(diagnostic.verify_row_bundle(output), [])
            replay = diagnostic.finalize_invalid_after_rollback(
                output, production_post, shutdown
            )
            self.assertEqual(replay["status"], "SEALED")

    def test_finalizing_phase_can_resume_successful_offline_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, health = self._allocated_precommit(root)
            self._run_fake_live(output, health)
            production_post, shutdown = self._lifecycle_paths(root)
            dummy = SimpleNamespace()
            original_finalize = diagnostic.finalize_bundle_files
            first = True
            def fail_once(*args, **kwargs):
                nonlocal first
                if first:
                    first = False
                    raise KeyboardInterrupt("injected process interruption")
                return original_finalize(*args, **kwargs)
            with mock.patch.object(
                diagnostic, "_load_observation_and_masks", return_value=(dummy, {})
            ), mock.patch.object(
                diagnostic, "_existing_condition_result", side_effect=_fake_scored_row
            ), mock.patch.object(
                diagnostic, "finalize_bundle_files", side_effect=fail_once
            ):
                with self.assertRaises(KeyboardInterrupt):
                    diagnostic.finalize_run(output, production_post, shutdown)
                self.assertEqual(
                    diagnostic.load_json(output / "attempt_manifest.json")["status"],
                    "FINALIZING",
                )
                summary = diagnostic.finalize_run(output, production_post, shutdown)
            self.assertTrue(summary["serving_inference_path_bundle_candidate_eligible"])
            self.assertEqual(diagnostic.verify_row_bundle(output), [])

    def test_retry_lineage_requires_fresh_boot_and_attempt_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, health = self._allocated_precommit(root)
            client = _FakeClient(health)
            client.post = mock.Mock(side_effect=httpx.ConnectError("offline"))
            with self.assertRaises(diagnostic.AttemptInvalidity):
                diagnostic.run_live(output, client=client)
            production_post, shutdown = self._lifecycle_paths(root)
            diagnostic.finalize_invalid_after_rollback(
                output, production_post, shutdown
            )
            new_attestation, new_health = self._attestation(root / "new")
            with self.assertRaises(diagnostic.AttestationError):
                diagnostic.allocate(
                    root / "A1-bad",
                    new_attestation,
                    execution_authorized=True,
                    attempt_id="CAL-X03-IMAGE-PREDICTOR-v1-A0",
                    attempt_number=1,
                    retry_predecessor_dir=output,
                )
            for condition in diagnostic.CONDITIONS:
                new_health[condition]["boot_id"] = f"fresh-{condition}"
                build = new_health[condition]["build_identity"]
                new_health[condition]["build_identity_sha256"] = diagnostic.canonical_sha256(build)
            diagnostic.write_deployment_attestation(
                root / "fresh_attestation.json",
                {
                    "captured_at": "fresh",
                    "plan_id": diagnostic.PLAN_ID,
                    "health": new_health,
                    "production_9000": _production_evidence(),
                },
            )
            allocated = diagnostic.allocate(
                root / "A1",
                root / "fresh_attestation.json",
                execution_authorized=True,
                attempt_id="CAL-X03-IMAGE-PREDICTOR-v1-A1",
                attempt_number=1,
                retry_predecessor_dir=output,
            )
            self.assertEqual(allocated["live_detect_calls"], 0)

    def test_adapter_checkpoint_mask_and_porcelain_contracts(self) -> None:
        tensor = np.zeros((2, 3), dtype=np.float32)
        valid = adapter.checkpoint_coverage_report(
            {"model": {"detector.weight": tensor}}, {"weight": tensor.copy()}
        )
        self.assertTrue(valid["pass"])
        collision = adapter.checkpoint_coverage_report(
            {"detector.weight": tensor, "detector.detector.weight": tensor},
            {"weight": tensor.copy()},
        )
        self.assertFalse(collision["pass"])
        self.assertEqual(collision["mapping_collision_count"], 1)
        bad_shape = adapter.checkpoint_coverage_report(
            {"detector.weight": tensor}, {"weight": np.zeros((3, 2))}
        )
        self.assertEqual(bad_shape["shape_mismatch_count"], 1)

        fake_pil = types.ModuleType("PIL")
        fake_image = types.ModuleType("PIL.Image")
        class _ImageFactory:
            @staticmethod
            def fromarray(_value, mode=None):
                return SimpleNamespace(save=lambda buffer, **_kwargs: buffer.write(b"PNG"))
        fake_image.fromarray = _ImageFactory.fromarray
        fake_pil.Image = fake_image
        with mock.patch.dict(sys.modules, {"PIL": fake_pil, "PIL.Image": fake_image}):
            masks = np.zeros((1, 4, 4), dtype=bool)
            masks[0, 1:3, 1:4] = True
            records = adapter._mask_records(
                masks,
                np.asarray([0.5]),
                object_ids=np.asarray([7]),
                expected_height=4,
                expected_width=4,
            )
            self.assertEqual(records[0]["box_xyxy"], [1, 1, 4, 3])
            production_records, audit = adapter._production_mask_records(
                masks, np.asarray([0.5]), np.asarray([7])
            )
            self.assertTrue(audit["pass"])
            self.assertEqual(production_records[0]["object_id"], 7)
            fallback_records, fallback_audit = adapter._production_mask_records(
                masks, np.asarray([]), np.asarray([])
            )
            self.assertEqual(fallback_records[0]["object_id"], 0)
            self.assertFalse(fallback_audit["pass"])
            with self.assertRaises(adapter.AdapterSetupError):
                adapter._mask_records(
                    masks.astype(np.float32) * 0.2,
                    np.asarray([0.5]),
                    expected_height=4,
                    expected_width=4,
                )
        completed = SimpleNamespace(stdout=" M tracked.py\n?? untracked.py\n")
        with mock.patch.object(adapter.subprocess, "run", return_value=completed):
            status = adapter._git_output(
                Path("/repo"), "status", preserve_leading_columns=True
            )
        self.assertEqual(status.splitlines()[0], " M tracked.py")

    def test_recursive_processor_state_tensor_trace_and_validation(self) -> None:
        class _TensorLike:
            def __init__(self, dtype: str, shape: tuple[int, ...]) -> None:
                self.dtype = dtype
                self.shape = shape

        trace = adapter._processor_state_tensor_trace(
            {
                "top": _TensorLike("torch.float32", (1, 2)),
                "nested": [
                    {"mask": _TensorLike("torch.bool", (1, 3, 4))},
                    "metadata",
                ],
            }
        )
        self.assertEqual(
            [record["path"] for record in trace["records"]],
            ["$/nested/0/mask", "$/top"],
        )
        self.assertEqual(trace["record_count"], 2)
        self.assertEqual(
            trace["records_sha256"], diagnostic.canonical_sha256(trace["records"])
        )
        with self.assertRaises(adapter.AdapterSetupError):
            adapter._processor_state_tensor_trace({"metadata": "only"})

        prompt = "the red object"
        condition = diagnostic.CONDITIONS[1]
        health = _health("image", pid=902)
        fake_client = _FakeClient({condition: health})
        payload = json.loads(
            fake_client.post(
                diagnostic.DEFAULT_ENDPOINTS[condition] + "/detect",
                data={
                "request_id": "trace-request",
                "payload_sha256": "a" * 64,
                "prompt": prompt,
                "threshold": diagnostic.THRESHOLD,
                },
                files={"image": ("frame.jpg", b"fixture", "image/jpeg")},
            ).content
        )
        contract = {
            "request_id": "trace-request",
            "condition": condition,
            "payload_sha256": "a" * 64,
            "prompt_utf8_sha256": diagnostic.hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest(),
        }
        passed, _reason = diagnostic._validate_trace(
            payload, contract, expected_health=health
        )
        self.assertTrue(passed)
        payload["diagnostic_trace"]["precision"]["state_tensor_trace"][
            "records"
        ][0]["dtype"] = "torch.float16"
        passed, reason = diagnostic._validate_trace(
            payload, contract, expected_health=health
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "direct-image trace contract is incomplete")


if __name__ == "__main__":
    unittest.main()
