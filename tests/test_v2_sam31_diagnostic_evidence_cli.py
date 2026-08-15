import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments_suite_v2.runners import sam31_diagnostic_evidence_cli as cli


class Sam31DiagnosticEvidenceCliTests(unittest.TestCase):
    def test_receipt_and_assembly_are_offline_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = root / "stdout.bin"
            stdout.write_bytes(b"value\x00with-null\n")
            receipt = root / "socket.json"
            with mock.patch("socket.socket", side_effect=AssertionError("network")):
                self.assertEqual(
                    cli.main(
                        [
                            "receipt",
                            "--command",
                            "ss -ltnp '( sport = :9000 )'",
                            "--stdout-file",
                            str(stdout),
                            "--stderr",
                            "",
                            "--exit-code",
                            "0",
                            "--captured-at",
                            "2026-08-12T20:00:00.000001Z",
                            "--output",
                            str(receipt),
                        ]
                    ),
                    0,
                )
            self.assertEqual(json.loads(receipt.read_text())["stdout"], "value\x00with-null\n")
            receipt_map = root / "receipts.json"
            self.assertEqual(
                cli.main(
                    [
                        "assemble-receipts",
                        "--entry",
                        f"socket={receipt}",
                        "--output",
                        str(receipt_map),
                    ]
                ),
                0,
            )
            self.assertEqual(set(json.loads(receipt_map.read_text())), {"socket"})
            with self.assertRaises(FileExistsError):
                cli.main(
                    [
                        "assemble-receipts",
                        "--entry",
                        f"socket={receipt}",
                        "--output",
                        str(receipt_map),
                    ]
                )

    def test_derive_and_validate_delegate_to_frozen_collector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = root / "receipts.json"
            receipts.write_text("{}\n", encoding="utf-8")
            output = root / "production.json"
            derived = {"schema_version": "derived"}
            with mock.patch.object(
                cli.evidence, "derive_production_evidence", return_value=derived
            ) as derive:
                cli.main(
                    [
                        "derive-production",
                        "--receipts",
                        str(receipts),
                        "--expected-server-path",
                        "/workspace/sam3/sam31_server.py",
                        "--expected-checkpoint-path",
                        "/workspace/models/sam3.1/sam3.1_multiplex.pt",
                        "--expected-generated-server-sha256",
                        "a" * 64,
                        "--expected-checkpoint-sha256",
                        "b" * 64,
                        "--expected-upstream-commit",
                        "c" * 40,
                        "--expected-gpu-uuid",
                        "GPU-test",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(json.loads(output.read_text()), derived)
            self.assertEqual(derive.call_args.kwargs["expected_port"], 9000)
            with mock.patch.object(
                cli.evidence, "validate_production_evidence", return_value=derived
            ) as validate:
                cli.main(["validate-production", "--document", str(output)])
            validate.assert_called_once_with(derived)

    def test_shutdown_requires_explicit_receipts_and_delegates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = root / "receipts.json"
            production = root / "production.json"
            receipts.write_text("{}\n", encoding="utf-8")
            production.write_text('{"pid":2091}\n', encoding="utf-8")
            output = root / "shutdown.json"
            derived = {"status": "PASS"}
            with mock.patch.object(
                cli.evidence, "derive_adapter_shutdown_evidence", return_value=derived
            ) as derive:
                cli.main(
                    [
                        "derive-shutdown",
                        "--receipts",
                        str(receipts),
                        "--production-pre",
                        str(production),
                        "--adapter-pid",
                        "901",
                        "--adapter-pid",
                        "902",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(derive.call_args.kwargs["adapter_pids"], [901, 902])
            self.assertEqual(derive.call_args.kwargs["adapter_ports"], [9001, 9002])
            with mock.patch.object(
                cli.evidence, "validate_adapter_shutdown_evidence", return_value=derived
            ) as validate:
                cli.main(
                    [
                        "validate-shutdown",
                        "--document",
                        str(output),
                        "--production-pre",
                        str(production),
                    ]
                )
            validate.assert_called_once_with(derived, production_pre={"pid": 2091})

    def test_assembly_rejects_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(
                json.dumps(
                    {
                        "command": "true",
                        "stdout": "",
                        "stderr": "",
                        "exit_code": 0,
                        "captured_at": "2026-08-12T20:00:00Z",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                cli.assemble_receipts([f"x={path}", f"x={path}"])


if __name__ == "__main__":
    unittest.main()
