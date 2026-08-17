from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments_suite_v2.io import load_json
from experiments_suite_v2.runners import bundles as bundle_module
from experiments_suite_v2.runners.bundles import (
    allocate_bundle_attempt,
    verify_row_bundle,
    write_row_bundle,
)


class RowBundleTests(unittest.TestCase):
    def test_bundle_is_closed_and_hash_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            manifest = write_row_bundle(
                output,
                schema_version="fixture.v2",
                inputs={"aim": "fixture"},
                result_rows=({"row_id": "R1"}, {"row_id": "R2"}),
                summary={"status": "PASS"},
            )
            self.assertEqual(manifest["file_count"], 3)
            self.assertEqual(verify_row_bundle(output), [])
            (output / "summary.json").write_text("{}\n", encoding="utf-8")
            self.assertIn("hash mismatch summary.json", verify_row_bundle(output))

    def test_bundle_verifier_rejects_unlisted_post_finalization_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            write_row_bundle(
                output,
                schema_version="fixture.v2",
                inputs={"aim": "fixture"},
                result_rows=({"row_id": "R1"},),
                summary={"status": "PASS"},
            )
            (output / "late.txt").write_text("not immutable\n", encoding="utf-8")
            self.assertIn("unlisted artifact late.txt", verify_row_bundle(output))

    def test_allocation_resumes_if_process_stops_after_first_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bundle"
            original_write = bundle_module.atomic_write_json

            def stop_before_inputs(path, value, *, overwrite=True):
                if Path(path).name == "inputs.json":
                    raise RuntimeError("fixture allocation crash")
                return original_write(path, value, overwrite=overwrite)

            with mock.patch.object(
                bundle_module,
                "atomic_write_json",
                side_effect=stop_before_inputs,
            ):
                with self.assertRaisesRegex(RuntimeError, "allocation crash"):
                    allocate_bundle_attempt(
                        output,
                        schema_version="fixture.v2",
                        attempt_id="fixture__A0",
                        inputs={"factor": "frozen"},
                        input_rows=({"row_id": "R1"},),
                    )
            self.assertEqual(
                [path.name for path in output.iterdir()],
                ["attempt_manifest.json"],
            )
            resumed = allocate_bundle_attempt(
                output,
                schema_version="fixture.v2",
                attempt_id="fixture__A0",
                inputs={"factor": "frozen"},
                input_rows=({"row_id": "R1"},),
            )
            self.assertTrue(resumed.resumed)
            self.assertEqual(load_json(output / "attempt_manifest.json")["status"], "RUNNING")
            self.assertTrue((output / "inputs.json").is_file())
            self.assertTrue((output / "input_rows.jsonl").is_file())
            self.assertTrue((output / "events.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
