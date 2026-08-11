from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
from scipy.special import expit

from experiments.harness.mixed_effects import (
    DEFAULT_REPETITION_CANDIDATES,
    MixedEffectsError,
    fit_binary_mixed_effects,
    main,
    simulate_binary_mixed_effects_power,
)


def _simulated_rows() -> list[dict[str, object]]:
    """A fixed, recoverable two-level GLMM fixture."""

    generator = np.random.default_rng(151)
    rows: list[dict[str, object]] = []
    for template_number in range(12):
        template_effect = generator.normal(0.0, 0.7)
        family = "manipulation" if template_number % 2 else "memory"
        for instance_number in range(8):
            instance_effect = generator.normal(0.0, 0.45)
            for system, system_effect in (("B1", 0.0), ("T5", 1.0)):
                linear_predictor = (
                    -0.4
                    + system_effect
                    + (0.35 if family == "memory" else 0.0)
                    + template_effect
                    + instance_effect
                )
                rows.append(
                    {
                        "contract_success": bool(
                            generator.random() < expit(linear_predictor)
                        ),
                        "profile_id": system,
                        "scenario_template": f"template-{template_number:02d}",
                        "generated_instance": f"instance-{instance_number:02d}",
                        "scenario_family": family,
                    }
                )
    return rows


def _fit(rows, **overrides):
    options = {
        "reference_system": "B1",
        "scenario_factors": ("scenario_family",),
        "factor_references": {"scenario_family": "manipulation"},
    }
    options.update(overrides)
    return fit_binary_mixed_effects(rows, **options)


class BinaryMixedEffectsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _simulated_rows()
        cls.result = _fit(cls.rows)

    def test_recovers_simulated_system_effect_and_emits_audit_contract(self) -> None:
        result = self.result
        self.assertTrue(result["estimable"], result.get("reason"))
        coefficients = {
            item["term"]: item for item in result["fixed_effects"]
        }
        recovered = coefficients["profile_id[T5]"]["estimate_log_odds"]
        self.assertGreater(recovered, 0.7)
        self.assertLess(recovered, 2.0)
        self.assertEqual(len(result["random_effects"]), 2)
        self.assertTrue(
            all(item["standard_deviation"] > 0.02 for item in result["random_effects"])
        )
        contrast = result["system_contrasts"][0]
        self.assertGreater(contrast["marginal_absolute_difference"], 0.0)
        self.assertGreater(contrast["conditional_odds_ratio"], 1.0)
        self.assertGreaterEqual(contrast["wald_p_two_sided"], 0.0)
        self.assertLessEqual(contrast["wald_p_two_sided"], 1.0)

        specification = result["model_specification"]
        self.assertIn("(1 | scenario_template)", specification["formula"])
        self.assertIn(
            "(1 | scenario_template:generated_instance)", specification["formula"]
        )
        self.assertEqual(
            specification["reference_levels"],
            {"profile_id": "B1", "scenario_family": "manipulation"},
        )
        self.assertIn("scipy", result["software"])
        self.assertEqual(len(result["per_template_raw_outcomes"]), 12)
        self.assertIn(
            "template_random_intercept_qq",
            result["diagnostics"]["diagnostic_plot_data"],
        )
        self.assertEqual(
            result["input_sha256"],
            result["raw_regeneration"]["canonical_input_sha256"],
        )

    def test_raw_regeneration_is_exact_and_input_order_invariant(self) -> None:
        shuffled = copy.deepcopy(self.rows)
        random.Random(991).shuffle(shuffled)
        regenerated = _fit(shuffled)
        self.assertEqual(self.result, regenerated)

    def test_missing_episode_fails_closed_without_p_values(self) -> None:
        rows = copy.deepcopy(self.rows)
        rows[0].pop("contract_success")
        result = _fit(rows)
        self.assertFalse(result["estimable"])
        self.assertIn("unresolved", result["reason"])
        self.assertIsNone(result["inference"]["p_values"])

    def test_empty_input_returns_not_estimable_without_exception(self) -> None:
        result = fit_binary_mixed_effects([], reference_system="B1")
        self.assertFalse(result["estimable"])
        self.assertEqual(result["reason"], "no resolved episodes were supplied")
        self.assertEqual(result["input_summary"]["episodes_supplied"], 0)
        self.assertIsNone(result["inference"]["p_values"])

    def test_complete_separation_is_not_estimable(self) -> None:
        rows = []
        for template in range(6):
            for instance in range(3):
                rows.extend(
                    (
                        {
                            "contract_success": False,
                            "profile_id": "B1",
                            "scenario_template": f"t{template}",
                            "generated_instance": f"i{instance}",
                        },
                        {
                            "contract_success": True,
                            "profile_id": "T5",
                            "scenario_template": f"t{template}",
                            "generated_instance": f"i{instance}",
                        },
                    )
                )
        result = fit_binary_mixed_effects(rows, reference_system="B1")
        self.assertFalse(result["estimable"])
        self.assertIn("separation", result["reason"])
        self.assertEqual(result["diagnostics"]["separation"]["kind"], "complete")
        self.assertIsNone(result["inference"]["p_values"])

    def test_insufficient_random_groups_and_nonconvergence_fail_closed(self) -> None:
        too_few = [
            row
            for row in self.rows
            if row["scenario_template"] in {"template-00", "template-01"}
        ]
        insufficient = _fit(too_few)
        self.assertFalse(insufficient["estimable"])
        self.assertIn("insufficient", insufficient["reason"])

        stopped = _fit(self.rows, optimizer_maximum_iterations=1)
        self.assertFalse(stopped["estimable"])
        self.assertIn("did not converge", stopped["reason"])
        self.assertIsNone(stopped["inference"]["p_values"])

    def test_singular_random_effect_is_not_silently_removed(self) -> None:
        rows = []
        for template in range(8):
            for instance in range(8):
                success = (template + instance) % 2 == 0
                for system in ("B1", "T5"):
                    rows.append(
                        {
                            "contract_success": success,
                            "profile_id": system,
                            "scenario_template": f"t{template}",
                            "generated_instance": f"i{instance}",
                        }
                    )
        result = fit_binary_mixed_effects(rows, reference_system="B1")
        self.assertFalse(result["estimable"])
        self.assertTrue(
            "singular" in result["reason"]
            or "Hessian" in result["reason"]
            or "converge" in result["reason"]
        )
        self.assertIsNone(result["inference"]["p_values"])

    def test_references_are_mandatory_and_exact(self) -> None:
        with self.assertRaisesRegex(MixedEffectsError, "factor_references"):
            fit_binary_mixed_effects(
                self.rows,
                reference_system="B1",
                scenario_factors=("scenario_family",),
            )

        one_family = [
            row for row in self.rows if row["scenario_family"] == "memory"
        ]
        result = _fit(
            one_family,
            factor_references={"scenario_family": "memory"},
        )
        self.assertFalse(result["estimable"])
        self.assertIn("fewer than two", result["reason"])


class MixedEffectsPowerTests(unittest.TestCase):
    def test_candidates_are_frozen_and_preliminary_power_regenerates(self) -> None:
        self.assertEqual(DEFAULT_REPETITION_CANDIDATES, (10, 15, 20, 25))
        rows = _simulated_rows()
        arguments = {
            "comparison_system": "T5",
            "reference_system": "B1",
            "minimum_absolute_difference": 0.10,
            "scenario_factors": ("scenario_family",),
            "factor_references": {"scenario_family": "manipulation"},
            "repetition_candidates": (10,),
            "simulations": 2,
            "seed": 7081,
        }
        first = simulate_binary_mixed_effects_power(rows, **arguments)
        second = simulate_binary_mixed_effects_power(reversed(rows), **arguments)
        self.assertEqual(first, second)
        self.assertTrue(first["estimable"], first.get("reason"))
        self.assertEqual(first["candidate_repetitions"], [10])
        self.assertEqual(first["power_curve"][0]["repetitions"], 10)
        self.assertFalse(
            first["power_curve"][0]["publication_ready_monte_carlo_size"]
        )
        self.assertIsNone(first["selected_repetitions"])
        self.assertAlmostEqual(
            first["alternative"]["achieved_marginal_absolute_difference"],
            0.10,
            places=10,
        )

        with self.assertRaisesRegex(MixedEffectsError, "10, 15, 20, 25"):
            simulate_binary_mixed_effects_power(
                rows,
                comparison_system="T5",
                reference_system="B1",
                minimum_absolute_difference=0.10,
                repetition_candidates=(10,),
                simulations=1000,
            )

    def test_cli_binds_inputs_source_and_checksum_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = root / "rows.json"
            config = root / "config.json"
            output = root / "fit.json"
            rows.write_text("[]\n", encoding="utf-8")
            config.write_text('{"reference_system":"B1"}\n', encoding="utf-8")
            exit_code = main(
                [
                    "fit",
                    "--rows",
                    str(rows),
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 2)
            checksum = Path(f"{output}.sha256")
            self.assertTrue(checksum.is_file())
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            self.assertEqual(checksum.read_text().split()[0], digest)
            envelope = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(envelope["analysis_kind"], "fit")
            self.assertFalse(envelope["report"]["estimable"])
            with self.assertRaisesRegex(MixedEffectsError, "overwrite"):
                main(
                    [
                        "fit",
                        "--rows",
                        str(rows),
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
