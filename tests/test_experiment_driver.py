from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from experiments.harness.driver import (
    EvaluationCapability,
    ScenarioDriver,
    ScenarioDriverError,
    jsonable,
)
from experiments.harness.recording import AttemptRecorder
from experiments.harness.scenarios import TriggerSpec, load_scenarios


class ScenarioDriverTests(unittest.TestCase):
    def test_jsonable_orders_sets_for_reproducible_hashing(self) -> None:
        self.assertEqual(jsonable(frozenset({"z", "a", "m"})), ["a", "m", "z"])

    def _attempt(self, root: Path) -> AttemptRecorder:
        return AttemptRecorder.create(
            root, schedule_id="S000001", attempt_number=1
        )

    def _controlled_driver(self, scenario, attempt, *, handlers=None):
        return ScenarioDriver.with_evaluation_capability(
            scenario,
            attempt,
            trigger_handlers=handlers,
        )

    def test_trigger_and_temporal_oracle_use_recorder_event_ids(self) -> None:
        scenario = load_scenarios().get("EV01")
        with tempfile.TemporaryDirectory() as directory:
            attempt = self._attempt(Path(directory))

            def handler(trigger, driver):
                driver.record_trigger_response(
                    trigger_id=trigger.trigger_id,
                    event_kind="safe_response_recorded",
                    response="held until evidence recovered",
                )

            driver, capability = self._controlled_driver(
                scenario,
                attempt,
                handlers={scenario.triggers[0].trigger_id: handler},
            )
            driver.emit(
                "episode_initialized", boundary=scenario.triggers[0].boundary
            )
            driver.set_hidden("goal.completed", True, evidence="fixture")
            driver.set_hidden(
                "safety.emergency_latched", False, evidence="fixture"
            )
            evaluation = driver.evaluate(capability)

            self.assertTrue(evaluation.passed, evaluation.to_dict())
            trigger = evaluation.trigger_results[0]
            self.assertEqual(len(trigger.predicate_event_ids), 1)
            self.assertEqual(len(trigger.response_event_ids), 1)
            self.assertLessEqual(
                trigger.response_event_deltas[0], trigger.max_response_events
            )
            sequences = [event.seq for event in driver.trace().events]
            self.assertEqual(sequences, sorted(sequences))
            self.assertEqual(len(sequences), len(set(sequences)))

    def test_unfired_trigger_fails_separately_from_oracle(self) -> None:
        scenario = load_scenarios().get("NM02")
        with tempfile.TemporaryDirectory() as directory:
            attempt = self._attempt(Path(directory))
            driver, capability = self._controlled_driver(
                scenario,
                attempt,
                handlers={scenario.triggers[0].trigger_id: lambda *_: None},
            )
            driver.set_hidden("goal.completed", True, evidence="fixture")
            driver.set_hidden(
                "safety.emergency_latched", False, evidence="fixture"
            )
            evaluation = driver.evaluate(capability)

            self.assertTrue(all(item.passed for item in evaluation.oracle_results))
            self.assertFalse(evaluation.trigger_results[0].passed)
            self.assertFalse(evaluation.passed)

    def test_missing_frozen_action_handler_fails_closed(self) -> None:
        scenario = load_scenarios().get("NM02")
        with tempfile.TemporaryDirectory() as directory:
            attempt = self._attempt(Path(directory))
            driver = ScenarioDriver(scenario, attempt)
            with self.assertRaisesRegex(ScenarioDriverError, "no action handler"):
                driver.emit(
                    "episode_initialized", boundary=scenario.triggers[0].boundary
                )

    def test_handler_must_record_one_response_evidence_event(self) -> None:
        scenario = load_scenarios().get("NM02")
        with tempfile.TemporaryDirectory() as directory:
            attempt = self._attempt(Path(directory))
            driver, _ = self._controlled_driver(
                scenario,
                attempt,
                handlers={scenario.triggers[0].trigger_id: lambda *_: None},
            )
            with self.assertRaisesRegex(
                ScenarioDriverError, "without exactly one recorded response"
            ):
                driver.emit(
                    "episode_initialized", boundary=scenario.triggers[0].boundary
                )

    def test_trigger_boundary_is_required_and_exact(self) -> None:
        scenario = load_scenarios().get("NM02")

        def handler(trigger, driver):
            driver.record_trigger_response(
                trigger.trigger_id,
                event_kind="scripted_input_delivered",
            )

        for observed in (None, "after_goal_proposal"):
            with self.subTest(observed=observed), tempfile.TemporaryDirectory() as directory:
                attempt = self._attempt(Path(directory))
                driver, _ = self._controlled_driver(
                    scenario,
                    attempt,
                    handlers={scenario.triggers[0].trigger_id: handler},
                )
                attributes = {} if observed is None else {"boundary": observed}
                with self.assertRaisesRegex(ScenarioDriverError, "expected boundary"):
                    driver.emit("episode_initialized", **attributes)

    def test_trigger_response_must_fit_frozen_event_bound(self) -> None:
        scenario = load_scenarios().get("NM02")
        trigger = scenario.triggers[0]

        def late_handler(active, driver):
            for index in range(active.max_response_events):
                driver.emit("handler_progress", index=index)
            driver.record_trigger_response(
                active.trigger_id,
                event_kind="scripted_input_delivered",
            )

        with tempfile.TemporaryDirectory() as directory:
            attempt = self._attempt(Path(directory))
            driver, _ = self._controlled_driver(
                scenario,
                attempt,
                handlers={trigger.trigger_id: late_handler},
            )
            with self.assertRaisesRegex(ScenarioDriverError, "exceeds frozen limit"):
                driver.emit("episode_initialized", boundary=trigger.boundary)

    def test_nested_trigger_predicates_are_queued_not_lost(self) -> None:
        base = load_scenarios().get("NM02")
        first = replace(base.triggers[0], trigger_id="nested-1", action="emit-next")
        second = TriggerSpec(
            trigger_id="nested-2",
            kind="scripted_input",
            boundary="after_first_response",
            firing_count=1,
            predicate_event="next_boundary_reached",
            predicate_occurrence=1,
            action="finish",
            max_response_events=3,
        )
        scenario = replace(base, triggers=(first, second))
        handled: list[str] = []

        def first_handler(trigger, driver):
            handled.append(trigger.trigger_id)
            driver.record_trigger_response(
                trigger.trigger_id, event_kind="first_response_recorded"
            )
            driver.emit(
                "next_boundary_reached", boundary=second.boundary
            )

        def second_handler(trigger, driver):
            handled.append(trigger.trigger_id)
            driver.record_trigger_response(
                trigger.trigger_id, event_kind="second_response_recorded"
            )

        with tempfile.TemporaryDirectory() as directory:
            attempt = self._attempt(Path(directory))
            driver, capability = self._controlled_driver(
                scenario,
                attempt,
                handlers={"nested-1": first_handler, "nested-2": second_handler},
            )
            driver.emit("episode_initialized", boundary=first.boundary)
            driver.set_hidden("goal.completed", True, evidence="fixture")
            driver.set_hidden("safety.emergency_latched", False, evidence="fixture")
            evaluation = driver.evaluate(capability)

            self.assertEqual(handled, ["nested-1", "nested-2"])
            self.assertTrue(evaluation.passed, evaluation.to_dict())

    def test_hidden_state_is_a_deep_copy_and_cannot_bypass_journal(self) -> None:
        scenario = load_scenarios().get("NM02")
        with tempfile.TemporaryDirectory() as directory:
            attempt = self._attempt(Path(directory))
            driver = ScenarioDriver(scenario, attempt)
            driver.replace_hidden(
                {"goal": {"completed": False, "history": ["initial"]}},
                evidence="fixture",
            )
            journal_before = (attempt.attempt_dir / "hidden_states.jsonl").read_text(
                encoding="utf-8"
            )
            exposed = driver.hidden_state
            exposed["goal"]["completed"] = True  # type: ignore[index]
            exposed["goal"]["history"].append("mutated")  # type: ignore[index,union-attr]

            self.assertFalse(driver.trace().hidden_state["goal"]["completed"])
            self.assertEqual(driver.trace().hidden_state["goal"]["history"], ["initial"])
            self.assertEqual(
                (attempt.attempt_dir / "hidden_states.jsonl").read_text(encoding="utf-8"),
                journal_before,
            )

    def test_evaluation_capability_is_driver_specific_and_one_shot(self) -> None:
        scenario = load_scenarios().get("NM02")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_attempt = self._attempt(root / "first")
            second_attempt = self._attempt(root / "second")
            first, first_capability = self._controlled_driver(scenario, first_attempt)
            _, second_capability = self._controlled_driver(scenario, second_attempt)
            first.set_hidden("goal.completed", True, evidence="fixture")
            first.set_hidden("safety.emergency_latched", False, evidence="fixture")

            with self.assertRaisesRegex(ScenarioDriverError, "missing or invalid"):
                first.evaluate(second_capability)
            evaluation = first.evaluate(first_capability)
            self.assertFalse(evaluation.passed)  # the frozen trigger never fired
            with self.assertRaisesRegex(ScenarioDriverError, "one-shot"):
                first.evaluate(first_capability)

            with self.assertRaises(TypeError):
                EvaluationCapability(object(), object())


if __name__ == "__main__":
    unittest.main()
