from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from simulation.benchmark.conversation_cases import (
    build_conversation_cases,
    OPPOSITE_TARGETS,
    case_matrix,
    load_episode_catalog,
)
from simulation.benchmark.conversation_models import ConversationEvaluationConfig


ROOT = Path(__file__).resolve().parents[1] / "dataset" / "sim_datasets"


class ConversationCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_episode_catalog(ROOT)

    def config(self, **changes: object) -> ConversationEvaluationConfig:
        values = {
            "benchmark_root": ROOT,
            "output_dir": ROOT / "unused-evaluation-output",
            "suites": ("endpoint", "dialogue", "memory", "recovery", "safety"),
            "shuffle_seed": 0,
        }
        values.update(changes)
        return ConversationEvaluationConfig(**values)  # type: ignore[arg-type]

    def test_full_matrix_covers_every_behavior_suite(self) -> None:
        matrix = case_matrix(build_conversation_cases(self.catalog, self.config()))
        self.assertEqual(
            matrix["by_suite"],
            {
                "dialogue": 126,
                "endpoint": 234,
                "memory": 54,
                "recovery": 144,
                "safety": 36,
            },
        )
        self.assertEqual(matrix["cases"], 594)
        self.assertEqual(matrix["commands"], 810)

    def test_memory_matrix_is_family_complete_and_dependencies_are_explicit(
        self,
    ) -> None:
        cases = build_conversation_cases(
            self.catalog,
            self.config(suites=("memory",)),
        )
        by_id = {item.scenario_id: item for item in self.catalog}
        self.assertEqual(len(cases), 54)
        self.assertEqual(sum(len(case.commands) for case in cases), 126)
        self.assertEqual(
            Counter(case.metadata["family"] for case in cases),
            {
                "block_stack": 18,
                "category_sort": 18,
                "place_setting": 18,
            },
        )
        self.assertEqual(
            Counter(case.profile for case in cases),
            {
                "learn-consent-reuse-twice": 18,
                "approved-preference-one-off-override": 18,
                "cross-user-isolation": 18,
            },
        )
        for case in cases:
            dependencies = {
                by_id[command.episode_id].initial_frame_sha256
                for command in case.commands
            }
            scenarios = {command.episode_id for command in case.commands}
            self.assertEqual(
                case.metadata["dependency_cluster_ids"],
                sorted(dependencies),
            )
            self.assertEqual(
                case.metadata["dependency_scenario_ids"],
                sorted(scenarios),
            )
            if case.profile == "learn-consent-reuse-twice":
                first = by_id[case.commands[0].episode_id]
                transfer = by_id[case.commands[2].episode_id]
                self.assertNotEqual(
                    first.initial_frame_sha256,
                    transfer.initial_frame_sha256,
                )
                self.assertIsNone(case.fixture)
            elif case.profile == "approved-preference-one-off-override":
                opposite = by_id[case.commands[0].episode_id]
                primary = by_id[case.commands[1].episode_id]
                self.assertEqual(
                    opposite.target_id,
                    OPPOSITE_TARGETS[primary.target_id],
                )
                self.assertEqual(
                    opposite.initial_frame_sha256,
                    primary.initial_frame_sha256,
                )
                self.assertIsNotNone(case.fixture)
            else:
                self.assertIsNotNone(case.fixture)

    def test_single_opposite_success_filter_keeps_real_transfer_semantics(
        self,
    ) -> None:
        opposite = next(
            item
            for item in self.catalog
            if item.outcome == "success"
            and item.control_kind is None
            and item.target_id
            in {
                "bgr_bottom_to_top",
                "electronics_left",
                "left_handed",
            }
        )
        cases = build_conversation_cases(
            self.catalog,
            self.config(
                suites=("memory",),
                scenario_ids=(opposite.scenario_id,),
            ),
        )
        self.assertEqual(len(cases), 3)
        learn = next(
            case
            for case in cases
            if case.profile == "learn-consent-reuse-twice"
        )
        by_id = {item.scenario_id: item for item in self.catalog}
        self.assertNotEqual(
            by_id[learn.commands[0].episode_id].initial_frame_sha256,
            by_id[learn.commands[2].episode_id].initial_frame_sha256,
        )
        self.assertEqual(len(learn.metadata["dependency_cluster_ids"]), 2)

    def test_max_cases_is_deterministically_stratified(self) -> None:
        full = build_conversation_cases(self.catalog, self.config())
        first = build_conversation_cases(
            self.catalog,
            self.config(max_cases=102, shuffle_seed=91),
        )
        repeated = build_conversation_cases(
            self.catalog,
            self.config(max_cases=102, shuffle_seed=91),
        )
        different = build_conversation_cases(
            self.catalog,
            self.config(max_cases=102, shuffle_seed=92),
        )

        def stratum(case):
            return (
                case.suite,
                case.profile,
                case.metadata.get("family"),
                case.metadata.get("outcome"),
            )

        self.assertEqual(
            [case.case_id for case in first],
            [case.case_id for case in repeated],
        )
        self.assertNotEqual(
            [case.case_id for case in first],
            [case.case_id for case in different],
        )
        self.assertEqual(
            {stratum(case) for case in first},
            {stratum(case) for case in full},
        )
        self.assertEqual(set(Counter(map(stratum, first)).values()), {2})

    def test_failure_only_filter_still_finds_private_success_recovery_sibling(self) -> None:
        cases = build_conversation_cases(
            self.catalog,
            self.config(suites=("recovery",), outcomes=("partial",)),
        )
        self.assertEqual(len(cases), 36)
        self.assertTrue(all(case.commands[0].recovery_episode_id for case in cases))

    def test_public_queries_do_not_expose_private_oracle_fields(self) -> None:
        cases = build_conversation_cases(self.catalog, self.config(max_cases=40))
        forbidden = (
            "scenario_id",
            "target_id",
            "expected_outcome",
            "control_kind",
            "goal_predicates",
            "ep-",
        )
        for case in cases:
            for command in case.commands:
                with self.subTest(case=case.case_id, command=command.command_id):
                    lowered = command.query.casefold()
                    self.assertFalse(any(value in lowered for value in forbidden))


if __name__ == "__main__":
    unittest.main()
