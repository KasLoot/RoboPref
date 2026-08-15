from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from experiments_suite_v2.io import atomic_write_json, load_json
from experiments_suite_v2.runners import cal_x03_multipart_encoding_diagnostic as diagnostic
from experiments_suite_v2.runners import sam31_diagnostic_evidence as evidence


class _Response:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": "application/json"}


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


_SERVER_PATH = "/workspace/sam3/sam31_server.py"
_CHECKPOINT_PATH = "/workspace/models/sam3.1/sam3.1_multiplex.pt"
_UPSTREAM_COMMIT = "a" * 40


def _production_evidence(
    *, captured_at: str = "2026-08-12T20:00:00+00:00"
) -> dict[str, object]:
    stat_fields = ["S", *("0" for _ in range(18)), "12345"]
    receipts = {
        "socket": _receipt(
            "ss -ltnp sport = :9000",
            'LISTEN 0 2048 127.0.0.1:9000 0.0.0.0:* users:(("python",pid=900,fd=3))\n',
            captured_at=captured_at,
        ),
        "proc_stat": _receipt(
            "cat /proc/900/stat", "900 (python) " + " ".join(stat_fields) + "\n",
            captured_at=captured_at,
        ),
        "proc_cmdline": _receipt(
            "cat /proc/900/cmdline",
            f"python uvicorn {_SERVER_PATH} --port 9000\n",
            captured_at=captured_at,
        ),
        "proc_cwd": _receipt(
            "readlink -f /proc/900/cwd", "/workspace/sam3\n",
            captured_at=captured_at,
        ),
        "server_sha256": _receipt(
            f"sha256sum {_SERVER_PATH}",
            f"{diagnostic.GENERATED_SERVER_SHA256}  {_SERVER_PATH}\n",
            captured_at=captured_at,
        ),
        "checkpoint_sha256": _receipt(
            f"sha256sum {_CHECKPOINT_PATH}",
            f"{diagnostic.CHECKPOINT_SHA256}  {_CHECKPOINT_PATH}\n",
            captured_at=captured_at,
        ),
        "source_sha256": _receipt(
            f"sha256sum {_SERVER_PATH}",
            f"{diagnostic.GENERATED_SERVER_SHA256}  {_SERVER_PATH}\n",
            captured_at=captured_at,
        ),
        "git_commit": _receipt(
            "git rev-parse HEAD", _UPSTREAM_COMMIT + "\n",
            captured_at=captured_at,
        ),
        "git_status": _receipt(
            "git status --porcelain=v1", "", captured_at=captured_at
        ),
        "python_version": _receipt(
            "python --version", "Python 3.12.3\n", captured_at=captured_at
        ),
        "package_inventory": _receipt(
            "pip freeze", "Pillow==11.3.0\ntorch==2.10.0\nfastapi==0.141.1\n",
            captured_at=captured_at,
        ),
        "gpu": _receipt(
            "nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,memory.free",
            "GPU-fixture, NVIDIA RTX, 580.1, 24564, 8192\n",
            captured_at=captured_at,
        ),
        "compute_apps": _receipt(
            "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory",
            "900, GPU-fixture, 1750\n",
            captured_at=captured_at,
        ),
        "health": _receipt(
            "curl http://127.0.0.1:9000/health",
            '{"status":"ok"}\n__HTTP_STATUS__:200\n',
            captured_at=captured_at,
        ),
    }
    return evidence.derive_production_evidence(
        receipts,
        expected_port=9000,
        expected_server_path=_SERVER_PATH,
        expected_checkpoint_path=_CHECKPOINT_PATH,
        expected_generated_server_sha256=diagnostic.GENERATED_SERVER_SHA256,
        expected_checkpoint_sha256=diagnostic.CHECKPOINT_SHA256,
        expected_upstream_commit=_UPSTREAM_COMMIT,
        expected_gpu_uuid="GPU-fixture",
    )


class _FrozenOutcomeClient:
    """Return the preregistered exact baseline and full lossless recovery."""

    def __init__(self, output_dir: Path, *, outage_at: int | None = None) -> None:
        self.output_dir = output_dir
        self.contracts = load_json(output_dir / "request_index.json")["records"]
        self.rows = {
            str(row["diagnostic_row_id"]): row
            for row in diagnostic.iter_jsonl(output_dir / "input_rows.jsonl")
        }
        self.outage_at = outage_at
        self.posts: list[dict[str, object]] = []

    def get(self, url: str) -> _Response:
        self.assertEqual(url, f"{diagnostic.BASE_URL}/health")
        return _Response(200, b'{"status":"ok"}')

    @staticmethod
    def assertEqual(left, right) -> None:
        if left != right:
            raise AssertionError(f"{left!r} != {right!r}")

    def post(self, url: str, *, data, files) -> _Response:
        ordinal = len(self.posts)
        if self.outage_at is not None and ordinal == self.outage_at:
            raise TimeoutError("fixture transport outage after invocation")
        contract = self.contracts[ordinal]
        filename, payload, mime_type = files["image"]
        self.assertEqual(url, f"{diagnostic.BASE_URL}/detect")
        self.assertEqual(data, {"prompt": contract["prompt"], "threshold": "0.5"})
        self.assertEqual(filename, contract["filename"])
        self.assertEqual(mime_type, contract["mime_type"])
        self.assertEqual(
            diagnostic.hashlib.sha256(payload).hexdigest(), contract["payload_sha256"]
        )
        self.posts.append(
            {
                "request_id": contract["request_id"],
                "condition": contract["condition"],
                "filename": filename,
                "mime_type": mime_type,
            }
        )
        row = self.rows[str(contract["diagnostic_row_id"])]
        success = not (
            contract["condition"] == diagnostic.CURRENT
            and row["cohort"] == "RESIDUAL"
        )
        detections: list[dict[str, object]] = []
        if success:
            mask_record = row["scorer_oracle_masks"][row["object_id"]]
            mask = cv2.imread(
                str(diagnostic.SUITE_ROOT / mask_record["path"]), cv2.IMREAD_GRAYSCALE
            )
            if mask is None:
                raise AssertionError("fixture oracle mask is missing")
            binary = mask > 0
            ys, xs = np.nonzero(binary)
            ok, encoded = cv2.imencode(".png", binary.astype(np.uint8) * 255)
            if not ok:
                raise AssertionError("fixture mask encoding failed")
            detections.append(
                {
                    "object_id": 7,
                    "box_xyxy": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()) + 1,
                        int(ys.max()) + 1,
                    ],
                    "mask_area": int(binary.sum()),
                    "mask_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                    "score": 0.875,
                }
            )
        body = {
            "prompt": data["prompt"],
            "threshold": 0.5,
            "image": {"width": diagnostic.FRAME_SIZE, "height": diagnostic.FRAME_SIZE},
            "count": len(detections),
            "detections": detections,
        }
        return _Response(200, json.dumps(body, separators=(",", ":")).encode("utf-8"))


class MultipartEncodingDiagnosticTests(unittest.TestCase):
    def _allocated(self, root: Path) -> tuple[Path, Path]:
        evidence_path = root / "production-pre.json"
        atomic_write_json(evidence_path, _production_evidence())
        output = root / "A0"
        diagnostic.allocate(
            output,
            evidence_path,
            execution_authorized=True,
        )
        diagnostic.precommit_payloads(output)
        return output, evidence_path

    def test_full_72_call_mocked_run_and_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, evidence_path = self._allocated(Path(temporary))
            client = _FrozenOutcomeClient(output)
            live = diagnostic.run_live(output, client=client)
            self.assertEqual(live["completed_raw_calls"], 72)
            self.assertEqual(len(client.posts), 72)
            self.assertEqual(
                sum(item["condition"] == diagnostic.CURRENT for item in client.posts), 36
            )
            self.assertEqual(
                sum(item["filename"] == "frame.png" for item in client.posts), 36
            )
            post_path = Path(temporary) / "production-post.json"
            atomic_write_json(
                post_path,
                _production_evidence(captured_at="2099-01-01T00:00:00+00:00"),
            )
            summary = diagnostic.finalize_run(output, post_path)
            self.assertEqual(
                summary["recommendation"], "LOSSLESS_MULTIPART_ENCODING_CANDIDATE"
            )
            self.assertTrue(summary["transport_candidate_eligible"])
            self.assertEqual(
                summary["conditions"][diagnostic.CURRENT][
                    "residual_safe_intended_success_count"
                ],
                0,
            )
            self.assertEqual(
                summary["conditions"][diagnostic.LOSSLESS][
                    "production_selected_safe_intended_grounding_count"
                ],
                36,
            )
            self.assertEqual(len(list((output / "post_intents").glob("*.json"))), 72)
            self.assertEqual(len(list((output / "post_receipts").glob("*.json"))), 72)
            self.assertFalse(diagnostic.verify_row_bundle(output))

    def test_finalizer_rejects_reused_or_out_of_order_post_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, evidence_path = self._allocated(Path(temporary))
            diagnostic.run_live(output, client=_FrozenOutcomeClient(output))
            with self.assertRaisesRegex(
                diagnostic.AttemptInvalidity, "reuses production-pre"
            ):
                diagnostic.finalize_run(output, evidence_path)

        with tempfile.TemporaryDirectory() as temporary:
            output, _ = self._allocated(Path(temporary))
            diagnostic.run_live(output, client=_FrozenOutcomeClient(output))
            stale = Path(temporary) / "production-post-stale.json"
            atomic_write_json(
                stale,
                _production_evidence(captured_at="2026-08-12T20:00:01+00:00"),
            )
            with self.assertRaisesRegex(
                diagnostic.AttemptInvalidity, "not newly captured"
            ):
                diagnostic.finalize_run(output, stale)

    def test_payload_tamper_is_rejected_before_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, _ = self._allocated(Path(temporary))
            record = load_json(output / "request_index.json")["records"][0]
            payload = output / record["payload_path"]
            payload.write_bytes(payload.read_bytes() + b"tamper")
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "payload differs"):
                diagnostic.validate_precommit(output)

    def test_unknown_outage_is_terminal_and_cannot_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output, _ = self._allocated(Path(temporary))
            client = _FrozenOutcomeClient(output, outage_at=0)
            with self.assertRaises(diagnostic.AttemptInvalidity):
                diagnostic.run_live(output, client=client)
            failure = load_json(output / "failure.json")
            self.assertEqual(failure["classification"], "INVALID_RUN")
            self.assertEqual(failure["confirmed_physical_post_attempts"], 0)
            self.assertEqual(failure["unknown_physical_post_outcomes"], 1)
            self.assertFalse(failure["same_attempt_resume"])
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "allocate fresh"):
                diagnostic.run_live(output, client=_FrozenOutcomeClient(output))

    def test_eligibility_is_exact_and_transport_scoped(self) -> None:
        rows: list[dict[str, object]] = []
        for row_index in range(36):
            cohort = "RESIDUAL" if row_index % 2 == 0 else "STABLE_CONTROL"
            for condition in diagnostic.CONDITIONS:
                success = condition == diagnostic.LOSSLESS or cohort == "STABLE_CONTROL"
                rows.append(
                    {
                        "diagnostic_row_id": f"MSDEV-{row_index:03d}",
                        "condition": condition,
                        "unique_image_prompt_cell_id": f"CELL-{row_index // 2:02d}",
                        "repeat_index": row_index % 2,
                        "cohort": cohort,
                        "usable_response": True,
                        "capability_failure": False,
                        "production_selected_safe_intended_grounding": success,
                        "ambiguous_selection_failure": False,
                        "any_candidate_cross_object_collision": False,
                        "selected_candidate_cross_object_collision": False,
                        "sam_anchor_error_mm": [0.0, 0.0, 0.0] if success else None,
                        "sam_radial_xy_error_mm": 0.0 if success else None,
                    }
                )
        stability = {"pass": True}
        summary = diagnostic.aggregate_scored_rows(rows, duplicate_stability=stability)
        self.assertTrue(summary["transport_candidate_eligible"])
        self.assertFalse(summary["mime_vs_payload_bytes_isolated"])

        unstable = [dict(row) for row in rows]
        residual_current = next(
            row
            for row in unstable
            if row["cohort"] == "RESIDUAL" and row["condition"] == diagnostic.CURRENT
        )
        residual_current["production_selected_safe_intended_grounding"] = True
        residual_current["sam_anchor_error_mm"] = [0.0, 0.0, 0.0]
        residual_current["sam_radial_xy_error_mm"] = 0.0
        self.assertEqual(
            diagnostic.aggregate_scored_rows(
                unstable, duplicate_stability=stability
            )["recommendation"],
            "INSTABILITY_OBSERVED_NO_CANDIDATE",
        )

        collision = [dict(row) for row in rows]
        candidate = next(row for row in collision if row["condition"] == diagnostic.LOSSLESS)
        candidate["any_candidate_cross_object_collision"] = True
        self.assertEqual(
            diagnostic.aggregate_scored_rows(
                collision, duplicate_stability=stability
            )["recommendation"],
            "NO_MULTIPART_ENCODING_CANDIDATE",
        )

    def test_allocation_requires_authorization_and_receipt_derived_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence_path = root / "production.json"
            atomic_write_json(evidence_path, _production_evidence())
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "execution_authorized"):
                diagnostic.allocate(root / "unauthorized", evidence_path)
            health_only = root / "health-only.json"
            atomic_write_json(health_only, {"status": "ok"})
            with self.assertRaisesRegex(diagnostic.DiagnosticError, "raw-receipt-derived"):
                diagnostic.allocate(
                    root / "health-only",
                    health_only,
                    execution_authorized=True,
                )


if __name__ == "__main__":
    unittest.main()
