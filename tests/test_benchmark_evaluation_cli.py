from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from simulation.benchmark.__main__ import main
from simulation.benchmark.catalog import build_catalog, build_control_catalog
from simulation.benchmark.generator import generate_benchmark
from simulation.benchmark.protocols import (
    build_memory_protocols,
    memory_protocol_fixtures,
)


class BenchmarkEvaluationCliTests(unittest.TestCase):
    def test_cold_dry_run_selects_packets_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "evaluation"
            generate_benchmark(
                benchmark,
                families=["block_stack"],
                seeds=[1],
                backend="synthetic",
            )

            stream = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "simulation.benchmark",
                        "evaluate",
                        str(benchmark),
                        "--output",
                        str(output),
                        "--outcomes",
                        "success",
                        "--max-scenarios",
                        "1",
                        "--repetitions",
                        "2",
                        "--dry-run",
                    ],
                ),
                redirect_stdout(stream),
            ):
                code = main()

            payload = json.loads(stream.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["selected_scenarios"], 1)
            self.assertEqual(payload["planned_trials"], 2)
            self.assertEqual(payload["model_calls"], 0)
            self.assertFalse(output.exists())

    def test_memory_dry_run_validates_every_selector_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            output = root / "memory-evaluation"
            generate_benchmark(
                benchmark,
                families=["block_stack"],
                seeds=[1],
                backend="synthetic",
                include_controls=True,
            )
            scenarios = [
                *build_catalog(families=["block_stack"], seeds=[1]),
                *build_control_catalog(families=["block_stack"], seeds=[1]),
            ]
            (benchmark / "protocols.json").write_text(
                json.dumps(
                    {
                        "schema_version": "robopref.memory-protocols.v1",
                        "fixtures": memory_protocol_fixtures(),
                        "protocols": build_memory_protocols(scenarios),
                    }
                ),
                encoding="utf-8",
            )

            stream = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "simulation.benchmark",
                        "evaluate-memory",
                        str(benchmark),
                        "--output",
                        str(output),
                        "--repetitions",
                        "2",
                        "--dry-run",
                    ],
                ),
                redirect_stdout(stream),
            ):
                code = main()

            payload = json.loads(stream.getvalue())
            self.assertEqual(code, 0)
            self.assertGreater(payload["selected_protocols"], 0)
            self.assertGreater(payload["steps_per_repetition"], 0)
            self.assertEqual(
                payload["planned_protocol_runs"],
                payload["selected_protocols"] * 2,
            )
            self.assertEqual(
                payload["preflighted_selection_plans"],
                payload["planned_protocol_runs"],
            )
            self.assertEqual(len(payload["selection_plan_sha256"]), 64)
            self.assertEqual(payload["model_calls"], 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
