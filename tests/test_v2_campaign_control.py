from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments_suite_v2.campaigns import (
    REGISTRY_PATHS,
    SYSTEM_EXECUTION_DEPENDENCY_FLAGS,
    SYSTEM_EXECUTION_PHASE_ORDER,
    campaign_dry_plan,
)
from experiments_suite_v2.cli import main
from experiments_suite_v2.io import load_json, sha256_file
from experiments_suite_v2.registry import SUITE_ROOT
from experiments_suite_v2.storage import RunStore


class CampaignControlTests(unittest.TestCase):
    def _portable_protocol_root(self, parent: Path) -> Path:
        root = parent / "suite"
        shutil.copytree(SUITE_ROOT / "protocol", root / "protocol")
        return root

    def test_dry_plan_has_exact_counts_hashes_and_no_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._portable_protocol_root(Path(temporary))
            plan = campaign_dry_plan(root)
            self.assertTrue(plan["dry_run"])
            self.assertEqual(plan["external_calls"], 0)
            self.assertFalse(plan["run_or_attempt_allocated"])
            self.assertFalse((root / "results/shared-campaigns").exists())
            self.assertEqual(plan["campaigns"]["component"]["planned_rows"], 1_600)
            self.assertEqual(plan["campaigns"]["memory"]["planned_rows"], 19_200)
            self.assertEqual(plan["campaigns"]["replay"]["planned_rows"], 20)
            self.assertEqual(plan["campaigns"]["system"]["primary_rows"], 680)
            self.assertEqual(
                plan["campaigns"]["system"]["supplementary_rows"], 280
            )
            self.assertFalse(plan["calibration_readiness"]["ready"])
            system = plan["campaigns"]["system"]
            self.assertEqual(
                system["execution_phase_order"],
                list(SYSTEM_EXECUTION_PHASE_ORDER),
            )
            self.assertTrue(system["order_gate_passed"])
            self.assertEqual(
                system["execution_dependencies"]["required_flags"],
                list(SYSTEM_EXECUTION_DEPENDENCY_FLAGS),
            )
            self.assertFalse(
                system["execution_dependencies"][
                    "resolved_or_constructed_during_plan"
                ]
            )
            hashes = system["schedule_hashes"]
            self.assertEqual(len(hashes), 5)
            self.assertTrue(
                all(len(value) == 64 for value in hashes.values())
            )
            self.assertNotEqual(
                hashes["registry_expansion_trial_tuple_order_sha256"],
                hashes["execution_order_trial_tuple_sha256"],
            )
            self.assertFalse(plan["confirmatory_execution_authorized"])
            self.assertTrue(
                plan["registry_pins"]["all_required_registries_pinned"]
            )

    def test_all_runner_registries_are_hash_pinned(self) -> None:
        manifest = load_json(SUITE_ROOT / "protocol" / "protocol_manifest.json")
        pins = {
            item["path"]: item["sha256"]
            for item in manifest["candidate_freeze_inputs"]
        }
        for relative in {
            path for paths in REGISTRY_PATHS.values() for path in paths
        }:
            with self.subTest(path=relative):
                self.assertIn(relative, pins)
                self.assertEqual(pins[relative], sha256_file(SUITE_ROOT / relative))

    def test_start_refuses_before_run_allocation_or_factory_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._portable_protocol_root(Path(temporary))
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(
                    [
                        "campaign-start",
                        "component",
                        "must-not-exist",
                        "--suite-root",
                        str(root),
                        "--source-commit",
                        "abc",
                        "--adapter-factory",
                        "module_that_must_not_be_imported:factory",
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("campaign is not ready", stderr.getvalue())
            self.assertFalse((root / "results/shared-campaigns").exists())

    def test_read_only_campaign_resume_does_not_require_execution_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._portable_protocol_root(Path(temporary))
            store = RunStore(root)
            store.create_run("inspect", source_commit="abc", dirty_state={})
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "campaign-resume",
                        "system",
                        "inspect",
                        "--suite-root",
                        str(root),
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            value = json.loads(stdout.getvalue())
            self.assertEqual(value["mode"], "READ_ONLY_CAMPAIGN_RESUME_PLAN")
            self.assertEqual(
                value["execution_dependencies"]["required_flags"],
                list(SYSTEM_EXECUTION_DEPENDENCY_FLAGS),
            )
            self.assertFalse(
                value["execution_dependencies"]["supplied_or_resolved"]
            )
            self.assertFalse(value["external_models_or_simulator_invoked"])

    def test_system_start_requires_all_six_dependencies_without_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._portable_protocol_root(Path(temporary))
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = main(
                    [
                        "campaign-start",
                        "system",
                        "must-not-exist",
                        "--suite-root",
                        str(root),
                        "--source-commit",
                        "abc",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("all six explicit dependencies", stderr.getvalue())
            self.assertFalse((root / "results/shared-campaigns").exists())

    def test_system_dependencies_resolve_only_after_run_manifest(self) -> None:
        class Health:
            def check(self, service_ids, trial):  # pragma: no cover - shape only
                raise AssertionError("health gate must not run")

        class Reset:
            def reset_and_verify(self, trial):  # pragma: no cover - shape only
                raise AssertionError("reset must not run")

        class Hooks:
            def begin(self, *args):  # pragma: no cover - shape only
                raise AssertionError("evidence must not run")

            def after_verified_reset(self, *args):  # pragma: no cover
                raise AssertionError("evidence must not run")

            def finish(self, *args):  # pragma: no cover - shape only
                raise AssertionError("evidence must not run")

        runtime_calls: list[str] = []

        def runtime_factory(*args):
            runtime_calls.append("factory")

        def runtime_configurator(*args):
            runtime_calls.append("configurator")

        def runtime_driver(*args):
            runtime_calls.append("driver")

        dependencies = {
            "fake.runtime:factory": runtime_factory,
            "fake.runtime:configurator": runtime_configurator,
            "fake.runtime:driver": runtime_driver,
            "fake.runtime:health": Health(),
            "fake.runtime:reset": Reset(),
            "fake.runtime:evidence": Hooks(),
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = self._portable_protocol_root(Path(temporary))
            resolution_order: list[str] = []

            def resolve(specification, *, label):
                self.assertTrue(
                    (
                        root
                        / "results/shared-campaigns/system-cli/run_manifest.json"
                    ).is_file()
                )
                resolution_order.append(specification)
                return dependencies[specification]

            class NoExecutionCampaign:
                def __init__(self, **kwargs):
                    self.assertions = kwargs

                def run_serial(self, **kwargs):
                    return ()

                def status(self):
                    return {"complete": False, "attempt_count": 0}

            arguments = [
                "campaign-start",
                "system",
                "system-cli",
                "--suite-root",
                str(root),
                "--source-commit",
                "abc",
                "--system-runtime-factory",
                "fake.runtime:factory",
                "--system-runtime-configurator",
                "fake.runtime:configurator",
                "--system-runtime-driver",
                "fake.runtime:driver",
                "--system-health-gate",
                "fake.runtime:health",
                "--system-reset-verifier",
                "fake.runtime:reset",
                "--system-evidence-hooks",
                "fake.runtime:evidence",
            ]
            stdout = io.StringIO()
            with (
                patch(
                    "experiments_suite_v2.cli._campaign_readiness",
                    return_value={
                        "campaign": "system",
                        "order_gate_passed": True,
                        "confirmatory_execution_authorized": False,
                    },
                ),
                patch(
                    "experiments_suite_v2.campaigns.import_dependency",
                    side_effect=resolve,
                ),
                patch(
                    "experiments_suite_v2.runners.system_campaign.SystemCampaign",
                    NoExecutionCampaign,
                ),
                redirect_stdout(stdout),
            ):
                code = main(arguments)
            self.assertEqual(code, 0, stdout.getvalue())
            self.assertEqual(resolution_order, list(dependencies))
            self.assertEqual(runtime_calls, [])
            self.assertFalse(
                any(
                    (root / "results/shared-campaigns/system-cli").glob(
                        "*/*/attempt_manifest.json"
                    )
                )
            )
            manifest = load_json(
                root
                / "results/shared-campaigns/system-cli/run_manifest.json"
            )
            self.assertEqual(
                set(manifest["runtime_options"]["system_dependency_specs"]),
                {
                    "runtime_factory",
                    "runtime_configurator",
                    "runtime_driver",
                    "health_gate",
                    "reset_verifier",
                    "evidence_hooks",
                },
            )
            self.assertFalse(
                manifest["protocol"]["confirmatory_execution_authorized"]
            )

    def test_system_and_campaign_apis_are_exported(self) -> None:
        import experiments_suite_v2.runners as runners

        for name in (
            "ComponentCampaign",
            "MemoryCampaign",
            "ReplayCampaign",
            "SystemBundleOrchestrator",
            "SystemCampaign",
            "expand_system_trials",
            "run_system_trial",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(runners, name))


if __name__ == "__main__":
    unittest.main()
