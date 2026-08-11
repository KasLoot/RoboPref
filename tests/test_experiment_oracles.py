from __future__ import annotations

import unittest

from experiments.harness.oracles import (
    DialogueAct,
    EpisodeTrace,
    Event,
    MemoryEvent,
    OracleSchemaError,
    OracleSpec,
    evaluate_oracle,
)


def _oracle_specs() -> tuple[OracleSpec, ...]:
    return (
        OracleSpec.from_dict(
            {
                "id": "order",
                "type": "event_order",
                "params": {
                    "before": "trigger_fired",
                    "after": "safe_response_recorded",
                    "within_events": 4,
                    "correlation_key": "scenario_id",
                    "correlation_value": "fixture",
                },
            }
        ),
        OracleSpec.from_dict(
            {
                "id": "physical",
                "type": "hidden_state",
                "params": {
                    "path": "goal.completed",
                    "operator": "equals",
                    "expected": True,
                },
            }
        ),
        OracleSpec.from_dict(
            {
                "id": "dialogue",
                "type": "dialogue",
                "params": {
                    "required_acts": ["clarification", "exact_confirmation"],
                    "forbidden_acts": ["unsupported_commitment"],
                    "ordered_acts": ["clarification", "exact_confirmation"],
                },
            }
        ),
        OracleSpec.from_dict(
            {
                "id": "memory",
                "type": "memory",
                "params": {
                    "record_id": "pref-1",
                    "operation": "retrieve",
                    "authorized": True,
                    "applied": True,
                    "minimum_matches": 1,
                    "maximum_matches": 1,
                },
            }
        ),
        OracleSpec.from_dict(
            {
                "id": "safety",
                "type": "safety",
                "params": {
                    "max_violations": 0,
                    "forbidden_events": ["post_emergency_publication"],
                    "require_emergency_latch": False,
                },
            }
        ),
    )


def _positive_trace() -> EpisodeTrace:
    return EpisodeTrace(
        events=(
            Event(1, "trigger_fired", {"scenario_id": "fixture"}),
            Event(2, "safe_hold_entered", {"scenario_id": "fixture"}),
            Event(4, "safe_response_recorded", {"scenario_id": "fixture"}),
        ),
        hidden_state={
            "goal": {"completed": True},
            "safety": {"emergency_latched": False},
        },
        dialogue=(DialogueAct(1, "clarification"), DialogueAct(2, "exact_confirmation")),
        memory_events=(MemoryEvent(3, "pref-1", "retrieve", True, True),),
        safety_violations=(),
    )


def _negative_trace() -> EpisodeTrace:
    return EpisodeTrace(
        events=(
            Event(1, "safe_response_recorded", {"scenario_id": "fixture"}),
            Event(5, "trigger_fired", {"scenario_id": "fixture"}),
            Event(6, "post_emergency_publication", {"scenario_id": "fixture"}),
        ),
        hidden_state={
            "goal": {"completed": False},
            "safety": {"emergency_latched": True},
        },
        dialogue=(DialogueAct(1, "unsupported_commitment"),),
        memory_events=(MemoryEvent(2, "pref-1", "retrieve", True, False),),
        safety_violations=("workspace_violation",),
    )


class ExperimentOracleTests(unittest.TestCase):
    def test_every_oracle_type_has_positive_and_negative_fixture(self) -> None:
        specs = _oracle_specs()
        self.assertEqual(
            {spec.oracle_type for spec in specs},
            {"event_order", "hidden_state", "dialogue", "memory", "safety"},
        )
        self.assertTrue(all(evaluate_oracle(spec, _positive_trace()).passed for spec in specs))
        self.assertTrue(
            all(not evaluate_oracle(spec, _negative_trace()).passed for spec in specs)
        )

    def test_temporal_oracle_requires_integer_sequence_and_real_order(self) -> None:
        with self.assertRaisesRegex(OracleSchemaError, "sequence IDs"):
            EpisodeTrace(
                events=(Event("1", "trigger_fired", {}),),  # type: ignore[arg-type]
                hidden_state={},
            )
        with self.assertRaisesRegex(OracleSchemaError, "ordered"):
            EpisodeTrace(
                events=(Event(2, "second", {}), Event(1, "first", {})),
                hidden_state={},
            )

    def test_unknown_oracle_parameter_fails(self) -> None:
        with self.assertRaisesRegex(OracleSchemaError, "unknown"):
            OracleSpec.from_dict(
                {
                    "id": "bad",
                    "type": "hidden_state",
                    "params": {
                        "path": "goal.completed",
                        "operator": "equals",
                        "expected": True,
                        "always_pass": True,
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()

