from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.harness.profiles import (
    PROFILE_IDS,
    ProfileSchemaError,
    UnsafeExecutorError,
    assert_executor_allowed,
    load_profiles,
)


class ExperimentProfileTests(unittest.TestCase):
    def test_complete_profile_catalogue_and_topology_instance_counts(self) -> None:
        catalogue = load_profiles()

        self.assertEqual(set(catalogue.profiles), PROFILE_IDS)
        self.assertEqual(
            {
                profile_id: catalogue.get(profile_id).unique_agent_count
                for profile_id in ("T5", "T4", "T3", "T2", "T1")
            },
            {"T5": 5, "T4": 4, "T3": 3, "T2": 2, "T1": 1},
        )
        self.assertEqual(
            catalogue.get("T4").logical_to_instance["hri"],
            catalogue.get("T4").logical_to_instance["memory"],
        )
        self.assertEqual(
            catalogue.get("T3").logical_to_instance["monitor"],
            catalogue.get("T3").logical_to_instance["validator"],
        )
        self.assertEqual(len(set(catalogue.get("T1").logical_to_instance.values())), 1)

    def test_ablation_role_removal_and_mechanism_modes_are_explicit(self) -> None:
        catalogue = load_profiles()

        self.assertIsNone(catalogue.get("A-Memory").logical_to_instance["memory"])
        self.assertEqual(
            catalogue.get("A-Memory").mechanism_modes["memory"],
            "removed_empty_isolated_store",
        )
        self.assertIsNone(catalogue.get("A-HRI").logical_to_instance["hri"])
        self.assertIsNone(catalogue.get("A-Monitor").logical_to_instance["monitor"])
        self.assertIsNone(
            catalogue.get("A-Validator").logical_to_instance["validator"]
        )
        self.assertEqual(
            catalogue.get("A-OpenLoop").mechanism_modes["planning"],
            "frozen_complete_queue",
        )
        self.assertEqual(
            catalogue.get("A-SingleEvidence").mechanism_modes["evidence"],
            "single_frame_zero_stability",
        )

    def test_executor_policy_fails_closed_for_unsafe_profiles(self) -> None:
        catalogue = load_profiles()

        assert_executor_allowed(catalogue.get("A-HostFence"), "synthetic_event")
        with self.assertRaises(UnsafeExecutorError):
            assert_executor_allowed(catalogue.get("A-HostFence"), "mujoco")

        for profile_id in (
            "A-HRI",
            "A-Monitor",
            "A-Validator",
            "A-Confirmation",
            "A-SingleEvidence",
            "A-DynamicGuard",
        ):
            with self.subTest(profile_id=profile_id):
                with self.assertRaises(UnsafeExecutorError):
                    assert_executor_allowed(catalogue.get(profile_id), "real_robot")

        with self.assertRaisesRegex(UnsafeExecutorError, "safety screen"):
            assert_executor_allowed(catalogue.get("B2"), "real_robot")
        assert_executor_allowed(
            catalogue.get("B2"), "real_robot", safety_screened=True
        )
        with self.assertRaises(UnsafeExecutorError):
            assert_executor_allowed(catalogue.get("B1"), "real_robot")

    def test_scenario_policy_is_intersected_with_profile_policy(self) -> None:
        profile = load_profiles().get("T5")
        with self.assertRaises(UnsafeExecutorError):
            assert_executor_allowed(
                profile,
                "real_robot",
                scenario_allowed=frozenset({"mujoco"}),
                safety_screened=True,
            )

    def test_human_executor_requires_ethics_approval(self) -> None:
        profile = load_profiles().get("T5")
        with self.assertRaisesRegex(UnsafeExecutorError, "ethics approval"):
            assert_executor_allowed(profile, "human_executor")
        assert_executor_allowed(profile, "human_executor", ethics_approved=True)

    def test_unknown_profile_manifest_key_fails(self) -> None:
        source_path = Path(__file__).parents[1] / "experiments/config/profiles.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["profiles"][0]["silent_override"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ProfileSchemaError, "unknown"):
                load_profiles(path)


if __name__ == "__main__":
    unittest.main()

