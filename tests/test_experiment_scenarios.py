from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from experiments.harness.scenarios import (
    ABLATION_MARKERS,
    FAMILY_COUNTS,
    ScenarioSchemaError,
    load_scenarios,
)


class ExperimentScenarioTests(unittest.TestCase):
    def test_catalogue_has_exact_frozen_family_and_subset_counts(self) -> None:
        catalogue = load_scenarios()

        self.assertEqual(len(catalogue.scenarios), 62)
        self.assertEqual(
            Counter(item.family for item in catalogue.scenarios), Counter(FAMILY_COUNTS)
        )
        self.assertEqual(len(catalogue.subset("topology24")), 24)
        self.assertEqual(len(catalogue.subset("sensitivity20")), 20)
        self.assertEqual(len(catalogue.subset("real12")), 12)

        topology_families = Counter(
            item.family for item in catalogue.subset("topology24")
        )
        self.assertEqual(topology_families, {family: 4 for family in FAMILY_COUNTS})

    def test_real_subset_matches_preregistered_strata(self) -> None:
        real_subset = load_scenarios().subset("real12")
        families = Counter(item.family for item in real_subset)

        self.assertEqual(families["nominal_manipulation"], 4)
        self.assertEqual(
            families["intent_confirmation"] + families["preference_memory"], 4
        )
        self.assertEqual(families["perturbation_recovery"], 2)
        self.assertEqual(families["evidence_validation"], 2)
        self.assertTrue(all(item.safety.real_system_eligible for item in real_subset))

    def test_split_variants_are_template_level_separated(self) -> None:
        catalogue = load_scenarios()
        variant_ids: set[str] = set()

        for scenario in catalogue.scenarios:
            variants = list(scenario.split_variants.values())
            self.assertEqual(len({item.separation_signature for item in variants}), 3)
            for item in variants:
                self.assertNotIn(item.variant_id, variant_ids)
                variant_ids.add(item.variant_id)

        self.assertEqual(len(variant_ids), 62 * 3)

    def test_claim_relevant_ablation_markers_are_explicit_and_applicable(self) -> None:
        catalogue = load_scenarios()

        for marker in ABLATION_MARKERS:
            with self.subTest(marker=marker):
                selected = catalogue.subset(marker)
                profile_id = marker.removeprefix("ablation:")
                self.assertTrue(selected)
                self.assertTrue(
                    all(profile_id in scenario.applicable_profiles for scenario in selected)
                )

    def test_perturbation_contracts_have_order_and_recovery_oracles(self) -> None:
        perturbations = [
            item
            for item in load_scenarios().scenarios
            if item.family == "perturbation_recovery"
        ]

        self.assertEqual(len(perturbations), 14)
        for scenario in perturbations:
            self.assertTrue(all(trigger.firing_count > 0 for trigger in scenario.triggers))
            self.assertTrue(
                all(trigger.max_response_events > 0 for trigger in scenario.triggers)
            )
            types = {oracle.oracle_type for oracle in scenario.oracles}
            self.assertGreaterEqual(types, {"event_order", "hidden_state", "safety"})
            self.assertTrue(scenario.preflight.calibration_required)

    def test_unknown_nested_scenario_key_fails(self) -> None:
        source_path = Path(__file__).parents[1] / "experiments/scenarios.json"
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source["scenarios"][0]["setup"]["unregistered_pose_hint"] = "leak"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenarios.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ScenarioSchemaError, "unknown"):
                load_scenarios(path)


if __name__ == "__main__":
    unittest.main()

