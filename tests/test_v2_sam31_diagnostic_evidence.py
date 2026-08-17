from __future__ import annotations

from copy import deepcopy
import unittest

from experiments_suite_v2.runners import sam31_diagnostic_evidence as evidence


STAMP = "2026-08-12T21:00:00+00:00"
BEFORE_STAMP = "2026-08-12T21:01:00+00:00"
TERM_STAMP = "2026-08-12T21:01:01+00:00"
EXIT_STAMP = "2026-08-12T21:01:02+00:00"
AFTER_STAMP = "2026-08-12T21:01:03+00:00"
SERVER = "/workspace/sam3/sam31_server.py"
CHECKPOINT = "/workspace/models/sam3.1/sam3.1_multiplex.pt"
SERVER_SHA = "a" * 64
CHECKPOINT_SHA = "b" * 64
COMMIT = "c" * 40
GPU = "GPU-fixture"


def receipt(
    command: str,
    stdout: str = "",
    *,
    exit_code: int = 0,
    captured_at: str = STAMP,
) -> dict:
    return {
        "command": command,
        "stdout": stdout,
        "stderr": "",
        "exit_code": exit_code,
        "captured_at": captured_at,
    }


def production_receipts() -> dict[str, dict]:
    stat = "900 (python worker) S " + " ".join(str(i) for i in range(1, 19)) + " 123456\n"
    health = '{"status":"ok","model":"sam3.1"}\n__HTTP_STATUS__:200\n'
    return {
        "socket": receipt(
            "ss -H -ltnp 'sport = :9000'",
            'LISTEN 0 2048 0.0.0.0:9000 0.0.0.0:* users:(("python",pid=900,fd=7))\n',
        ),
        "proc_stat": receipt("sed -n 1p /proc/900/stat", stat),
        "proc_cmdline": receipt(
            "tr '\\0' ' ' < /proc/900/cmdline",
            "uvicorn sam31_server:app --host 0.0.0.0 --port 9000\n",
        ),
        "proc_cwd": receipt(
            "readlink -f /proc/900/cwd", "/workspace/sam3\n"
        ),
        "server_sha256": receipt(
            f"sha256sum {SERVER}", f"{SERVER_SHA}  {SERVER}\n"
        ),
        "checkpoint_sha256": receipt(
            f"sha256sum {CHECKPOINT}", f"{CHECKPOINT_SHA}  {CHECKPOINT}\n"
        ),
        "source_sha256": receipt(
            f"sha256sum {SERVER} /workspace/sam3/sam3/model_builder.py",
            f"{SERVER_SHA}  {SERVER}\n{'d' * 64}  /workspace/sam3/sam3/model_builder.py\n",
        ),
        "git_commit": receipt(
            "git -C /workspace/sam3 rev-parse HEAD", COMMIT + "\n"
        ),
        "git_status": receipt(
            "git -C /workspace/sam3 status --porcelain=v1 --untracked-files=all",
            " M sam3/model/sam3_base_predictor.py\n?? sam31_server.py\n",
        ),
        "python_version": receipt("python --version", "Python 3.12.3\n"),
        "package_inventory": receipt(
            "python -m pip freeze --all", "torch==2.10.0\nfastapi==0.116.1\n"
        ),
        "gpu": receipt(
            "nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits",
            f"{GPU}, NVIDIA GeForce RTX 4090, 580.159.04, 24564, 8000\n",
        ),
        "compute_apps": receipt(
            "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits",
            f"900, {GPU}, 6682\n1578, {GPU}, 1750\n",
        ),
        "health": receipt(
            "curl -fsS -w '\\n__HTTP_STATUS__:%{http_code}\\n' http://127.0.0.1:9000/health",
            health,
        ),
    }


def production() -> dict:
    return evidence.derive_production_evidence(
        production_receipts(),
        expected_port=9000,
        expected_server_path=SERVER,
        expected_checkpoint_path=CHECKPOINT,
        expected_generated_server_sha256=SERVER_SHA,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_upstream_commit=COMMIT,
        expected_gpu_uuid=GPU,
    )


def shutdown_receipts() -> dict[str, dict]:
    return {
        "sockets_after": receipt(
            "ss -H -ltnp",
            'LISTEN 0 2048 127.0.0.1:9000 0.0.0.0:* users:(("python",pid=900,fd=7))\n',
            captured_at=AFTER_STAMP,
        ),
        "process_table_after": receipt(
            "ps -p 900,901,902 -o pid=,args=",
            f"900 python {SERVER} --port 9000\n",
            captured_at=AFTER_STAMP,
        ),
        "gpu_before": receipt(
            "nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits",
            f"{GPU}, NVIDIA GeForce RTX 4090, 580.159.04, 24564, 4000\n",
            captured_at=BEFORE_STAMP,
        ),
        "gpu_after": receipt(
            "nvidia-smi --query-gpu=uuid,name,driver_version,memory.total,memory.free --format=csv,noheader,nounits",
            f"{GPU}, NVIDIA GeForce RTX 4090, 580.159.04, 24564, 12000\n",
            captured_at=AFTER_STAMP,
        ),
        "compute_apps_after": receipt(
            "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits",
            f"900, {GPU}, 6682\n1578, {GPU}, 1750\n",
            captured_at=AFTER_STAMP,
        ),
        "sigterm_901": receipt("kill -TERM 901", captured_at=TERM_STAMP),
        "exit_901": receipt("test ! -e /proc/901", captured_at=EXIT_STAMP),
        "sigterm_902": receipt("kill -TERM 902", captured_at=TERM_STAMP),
        "exit_902": receipt("test ! -e /proc/902", captured_at=EXIT_STAMP),
    }


class Sam31DiagnosticEvidenceTests(unittest.TestCase):
    def test_production_evidence_is_recomputed_from_raw_receipts(self) -> None:
        value = production()
        self.assertEqual(value["pid"], 900)
        self.assertEqual(value["process_start_time"], "123456")
        self.assertEqual(value["checkpoint_sha256"], CHECKPOINT_SHA)
        self.assertEqual(value["process_cwd"], "/workspace/sam3")
        self.assertEqual(value["production_process_gpu_record"]["gpu_uuid"], GPU)
        self.assertEqual(evidence.validate_production_evidence(value), value)

    def test_production_evidence_rejects_socket_hash_health_and_detect_tamper(self) -> None:
        values = production_receipts()
        values["socket"]["stdout"] = values["socket"]["stdout"].replace("pid=900", "pid=901")
        with self.assertRaises(evidence.EvidenceValidationError):
            evidence.derive_production_evidence(
                values,
                expected_server_path=SERVER,
                expected_checkpoint_path=CHECKPOINT,
                expected_generated_server_sha256=SERVER_SHA,
                expected_checkpoint_sha256=CHECKPOINT_SHA,
                expected_upstream_commit=COMMIT,
                expected_gpu_uuid=GPU,
            )
        values = production_receipts()
        values["server_sha256"]["stdout"] = values["server_sha256"]["stdout"].replace(SERVER_SHA, "e" * 64)
        with self.assertRaises(evidence.EvidenceValidationError):
            evidence.derive_production_evidence(
                values,
                expected_server_path=SERVER,
                expected_checkpoint_path=CHECKPOINT,
                expected_generated_server_sha256=SERVER_SHA,
                expected_checkpoint_sha256=CHECKPOINT_SHA,
                expected_upstream_commit=COMMIT,
            )
        values = production_receipts()
        values["health"]["command"] = values["health"]["command"].replace("/health", "/detect")
        with self.assertRaises(evidence.EvidenceValidationError):
            evidence.derive_production_evidence(
                values,
                expected_server_path=SERVER,
                expected_checkpoint_path=CHECKPOINT,
                expected_generated_server_sha256=SERVER_SHA,
                expected_checkpoint_sha256=CHECKPOINT_SHA,
                expected_upstream_commit=COMMIT,
            )
        values = production_receipts()
        values["proc_cwd"]["stdout"] = "/tmp\n"
        with self.assertRaises(evidence.EvidenceValidationError):
            evidence.derive_production_evidence(
                values,
                expected_server_path=SERVER,
                expected_checkpoint_path=CHECKPOINT,
                expected_generated_server_sha256=SERVER_SHA,
                expected_checkpoint_sha256=CHECKPOINT_SHA,
                expected_upstream_commit=COMMIT,
            )

    def test_derived_production_assertion_tamper_is_rejected(self) -> None:
        value = production()
        value["status"] = "PASS"
        with self.assertRaises(evidence.EvidenceValidationError):
            evidence.validate_production_evidence(value)

    def test_shutdown_evidence_is_derived_and_recomputed(self) -> None:
        pre = production()
        value = evidence.derive_adapter_shutdown_evidence(
            shutdown_receipts(), adapter_pids=(901, 902), production_pre=pre
        )
        self.assertEqual(value["terminated_pids"], [901, 902])
        self.assertEqual(value["ports_closed"], [9001, 9002])
        self.assertGreater(
            value["free_memory_bytes_after"], value["free_memory_bytes_before"]
        )
        self.assertEqual(
            evidence.validate_adapter_shutdown_evidence(value, production_pre=pre),
            value,
        )

    def test_shutdown_rejects_remaining_pid_socket_gpu_and_forged_pass(self) -> None:
        pre = production()
        for name, mutation in (
            (
                "process",
                lambda values: values["process_table_after"].update(
                    stdout=values["process_table_after"]["stdout"] + "901 python adapter.py\n"
                ),
            ),
            (
                "socket",
                lambda values: values["sockets_after"].update(
                    stdout=values["sockets_after"]["stdout"]
                    + 'LISTEN 0 2048 127.0.0.1:9001 0.0.0.0:* users:(("python",pid=901,fd=8))\n'
                ),
            ),
            (
                "gpu",
                lambda values: values["compute_apps_after"].update(
                    stdout=values["compute_apps_after"]["stdout"]
                    + f"902, {GPU}, 6000\n"
                ),
            ),
        ):
            with self.subTest(name=name):
                values = shutdown_receipts()
                mutation(values)
                with self.assertRaises(evidence.EvidenceValidationError):
                    evidence.derive_adapter_shutdown_evidence(
                        values, adapter_pids=(901, 902), production_pre=pre
                    )
        value = evidence.derive_adapter_shutdown_evidence(
            shutdown_receipts(), adapter_pids=(901, 902), production_pre=pre
        )
        value["free_memory_bytes_after"] = value["free_memory_bytes_before"] + 1
        with self.assertRaises(evidence.EvidenceValidationError):
            evidence.validate_adapter_shutdown_evidence(value, production_pre=pre)

    def test_shutdown_rejects_wrong_commands_and_swapped_phase_times(self) -> None:
        pre = production()
        values = shutdown_receipts()
        values["gpu_after"]["command"] = "printf not-nvidia-smi"
        with self.assertRaises(evidence.EvidenceValidationError):
            evidence.derive_adapter_shutdown_evidence(
                values, adapter_pids=(901, 902), production_pre=pre
            )
        values = shutdown_receipts()
        values["exit_901"]["command"] = "test -e /proc/901"
        with self.assertRaises(evidence.EvidenceValidationError):
            evidence.derive_adapter_shutdown_evidence(
                values, adapter_pids=(901, 902), production_pre=pre
            )
        values = shutdown_receipts()
        values["sigterm_901"]["command"] = "kill -TERM 900 901"
        with self.assertRaises(evidence.EvidenceValidationError):
            evidence.derive_adapter_shutdown_evidence(
                values, adapter_pids=(901, 902), production_pre=pre
            )
        values = shutdown_receipts()
        values["gpu_before"]["captured_at"] = AFTER_STAMP
        values["gpu_after"]["captured_at"] = BEFORE_STAMP
        with self.assertRaises(evidence.EvidenceValidationError):
            evidence.derive_adapter_shutdown_evidence(
                values, adapter_pids=(901, 902), production_pre=pre
            )

    def test_receipt_schema_and_failed_command_are_rejected(self) -> None:
        values = production_receipts()
        values["socket"]["trusted"] = True
        with self.assertRaises(evidence.EvidenceValidationError):
            production_value = evidence.derive_production_evidence(
                values,
                expected_server_path=SERVER,
                expected_checkpoint_path=CHECKPOINT,
                expected_generated_server_sha256=SERVER_SHA,
                expected_checkpoint_sha256=CHECKPOINT_SHA,
                expected_upstream_commit=COMMIT,
            )
            self.fail(production_value)
        values = production_receipts()
        values["git_commit"]["exit_code"] = 128
        values["git_commit"]["stderr"] = "fatal"
        with self.assertRaises(evidence.EvidenceValidationError):
            evidence.derive_production_evidence(
                values,
                expected_server_path=SERVER,
                expected_checkpoint_path=CHECKPOINT,
                expected_generated_server_sha256=SERVER_SHA,
                expected_checkpoint_sha256=CHECKPOINT_SHA,
                expected_upstream_commit=COMMIT,
            )


if __name__ == "__main__":
    unittest.main()
