from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
import unittest

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from prefmem.agents.monitor import CapturedFrame
from prefmem.agents.validator import (
    DEFAULT_VALIDATOR_PROMPT,
    THINKING_DISABLED_OPTIONS,
    ValidatorAgent,
    ValidatorErrorKind,
    ValidatorOutputError,
    ValidatorService,
    parse_validator_json_object,
)
from prefmem.contracts import (
    GoalContract,
    ObservationCriterion,
    PublishedTask,
    TaskPhase,
    ValidationChecklistDraft,
    freeze_validation_contract,
)
from prefmem.emergency import EmergencyStopCoordinator


def goal() -> GoalContract:
    return GoalContract(
        goal_id="goal-1",
        revision=1,
        goal="Put the blue block in the bin and clear the table.",
        final_expected_observation=(
            "The blue block is in the bin.",
            "The table is clear.",
        ),
        constraints=("Do not move the red block.",),
        nominal_tasks=("Move the blue block.",),
    )


def validation_contract():
    draft = ValidationChecklistDraft.from_model_output(
        {
            "broad_items": [
                {
                    "broad_index": 0,
                    "detailed_criteria": [
                        "The blue block is visibly inside the bin boundary."
                    ],
                },
                {
                    "broad_index": 1,
                    "detailed_criteria": [
                        "No movable object other than the red block is on the table."
                    ],
                },
            ]
        }
    )
    return freeze_validation_contract(
        goal(),
        draft,
        validation_id="goal-1:r1:validation",
    )


def final_task(
    publication_id: str = "goal-1:validation:attempt-1",
    *,
    frame_sequence: int = 4,
    published_at: float = 1.0,
) -> PublishedTask:
    return PublishedTask(
        plan_id="goal-1",
        revision=1,
        step_id="goal-1:validation",
        phase=TaskPhase.FINAL_VALIDATION,
        instruction="Hold the scene still for final validation.",
        expected_observation=(
            ObservationCriterion(
                criterion_id="goal-1:validation:summary",
                description="The frozen high-level goal is complete.",
            ),
        ),
        known_failure_conditions=(),
        published_at=published_at,
        publication_id=publication_id,
        cycle_id=3,
        frame_sequence=frame_sequence,
    )


def criterion_ids(contract) -> tuple[str, ...]:
    return tuple(
        criterion.criterion_id
        for broad in contract.broad_items
        for criterion in broad.detailed_criteria
    )


def assessment_output(
    contract,
    states: tuple[str, str],
    evidence: tuple[str, str],
) -> str:
    return json.dumps(
        {
            "emergency_stop": False,
            "emergency_reason": None,
            "criteria": [
                {"id": identifier, "state": state, "evidence": sentence}
                for identifier, state, sentence in zip(
                    criterion_ids(contract),
                    states,
                    evidence,
                    strict=True,
                )
            ],
            "observation": "The current frame was checked against every criterion.",
        }
    )


class SequenceModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[list[object], dict[str, object]]] = []
        self._lock = threading.Lock()

    def invoke(self, messages, **kwargs):
        with self._lock:
            index = min(len(self.calls), len(self.outputs) - 1)
            self.calls.append((messages, kwargs))
            output = self.outputs[index]
        return SimpleNamespace(content=output)


class SequenceFrames:
    def __init__(self, sequences: list[int], *, first_time: float = 10.0) -> None:
        self.sequences = list(sequences)
        self.first_time = first_time
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self) -> CapturedFrame:
        with self._lock:
            index = min(self.calls, len(self.sequences) - 1)
            sequence = self.sequences[index]
            self.calls += 1
            observed_at = self.first_time + self.calls
        return CapturedFrame(
            image_block={
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/"},
            },
            observed_at=observed_at,
            sequence=sequence,
        )


class ValidatorAgentTests(unittest.TestCase):
    def test_compilation_repairs_once_freezes_labels_and_caches(self) -> None:
        valid = json.dumps(
            {
                "broad_items": [
                    {
                        "broad_index": 0,
                        "detailed_criteria": [
                            "The blue block is visibly within the bin boundary."
                        ],
                    },
                    {
                        "broad_index": 1,
                        "detailed_criteria": [
                            "No movable object other than the red block is on the table."
                        ],
                    },
                ]
            }
        )
        model = SequenceModel(
            ['{"broad_items":[]}', f"```json\n{valid}\n```"]
        )
        agent = ValidatorAgent(
            model=model,
            system_prompt="Validator system contract.",
        )
        frame = SequenceFrames([1])()

        first = agent.compile_contract(goal(), frame)
        second = agent.compile_contract(goal(), frame)

        self.assertIs(first, second)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(
            tuple(item.label for item in first.broad_items),
            goal().final_expected_observation,
        )
        for messages, kwargs in model.calls:
            self.assertIsInstance(messages[0], SystemMessage)
            self.assertIsInstance(messages[1], HumanMessage)
            self.assertEqual(messages[1].content[0]["type"], "image_url")
            self.assertEqual(messages[1].content[1]["type"], "text")
            self.assertEqual(kwargs, THINKING_DISABLED_OPTIONS)
        retry_request = json.loads(model.calls[1][0][1].content[1]["text"])
        self.assertIn("schema_correction", retry_request)

    def test_duplicate_indices_receive_goal_specific_repair_guidance(self) -> None:
        duplicate = json.dumps(
            {
                "broad_items": [
                    {
                        "broad_index": 0,
                        "detailed_criteria": [
                            "The blue block is visibly within the bin boundary."
                        ],
                    },
                    {
                        "broad_index": 0,
                        "detailed_criteria": [
                            "No movable object other than the red block is on the table."
                        ],
                    },
                ]
            }
        )
        corrected = json.dumps(
            {
                "broad_items": [
                    {
                        "broad_index": 1,
                        "detailed_criteria": [
                            "No movable object other than the red block is on the table."
                        ],
                    },
                    {
                        "broad_index": 0,
                        "detailed_criteria": [
                            "The blue block is visibly within the bin boundary."
                        ],
                    },
                ]
            }
        )
        model = SequenceModel([duplicate, corrected])
        agent = ValidatorAgent(
            model=model,
            system_prompt="Validator system contract.",
        )

        contract = agent.compile_contract(goal(), SequenceFrames([1])())

        self.assertEqual(len(model.calls), 2)
        request = json.loads(model.calls[0][0][1].content[1]["text"])
        self.assertEqual(request["required_broad_item_count"], 2)
        self.assertEqual(request["required_broad_index_sequence"], [0, 1])
        retry = json.loads(model.calls[1][0][1].content[1]["text"])
        self.assertIn("duplicate broad_index", retry["schema_correction"])
        self.assertIn("exactly [0, 1]", retry["schema_correction"])
        self.assertEqual(
            contract.broad_items[0].detailed_criteria[0].description,
            "The blue block is visibly within the bin boundary.",
        )
        self.assertEqual(
            contract.broad_items[1].detailed_criteria[0].description,
            "No movable object other than the red block is on the table.",
        )

    def test_repeated_duplicate_indices_use_frozen_goal_fallback(self) -> None:
        clear_table_goal = GoalContract(
            goal_id="goal-clear-table",
            revision=1,
            goal=(
                "Put the battery, cable, microphone, scissors, and pencil "
                "into the box, leaving the mat on the table."
            ),
            final_expected_observation=(
                (
                    "the battery, cable, microphone, scissors, and pencil "
                    "are all inside the box"
                ),
                "the black mat remains on the table",
            ),
            constraints=("Leave the mat on the table.",),
        )
        duplicate = json.dumps(
            {
                "broad_items": [
                    {
                        "broad_index": 0,
                        "detailed_criteria": [
                            "Every named loose object is visible inside the box."
                        ],
                    },
                    {
                        "broad_index": 0,
                        "detailed_criteria": [
                            "The black mat is visible on the table."
                        ],
                    },
                ]
            }
        )
        model = SequenceModel([duplicate, duplicate])
        agent = ValidatorAgent(
            model=model,
            system_prompt="Validator system contract.",
        )

        with self.assertLogs("prefmem.agents.validator", level="WARNING"):
            first = agent.compile_contract(
                clear_table_goal,
                SequenceFrames([1])(),
            )
        second = agent.compile_contract(
            clear_table_goal,
            SequenceFrames([2])(),
        )

        self.assertIs(first, second)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(
            tuple(item.label for item in first.broad_items),
            clear_table_goal.final_expected_observation,
        )
        self.assertEqual(
            tuple(
                item.detailed_criteria[0].description
                for item in first.broad_items
            ),
            clear_table_goal.final_expected_observation,
        )
        self.assertEqual(
            tuple(item.broad_index for item in first.broad_items),
            (0, 1),
        )
        self.assertEqual(
            tuple(item.detailed_criteria[0].criterion_id for item in first.broad_items),
            (
                "goal-clear-table:r1:validation:b1:c1",
                "goal-clear-table:r1:validation:b2:c1",
            ),
        )

    def test_unrecoverable_non_mapping_schema_still_fails_closed(self) -> None:
        model = SequenceModel(['{"broad_items":[]}', '{"broad_items":[]}'])
        agent = ValidatorAgent(
            model=model,
            system_prompt="Validator system contract.",
        )

        with self.assertRaisesRegex(
            ValidatorOutputError,
            "failed its JSON contract twice",
        ):
            agent.compile_contract(goal(), SequenceFrames([1])())

        self.assertEqual(len(model.calls), 2)

    def test_compilation_accepts_gemma_thinking_and_json_fence(self) -> None:
        valid = json.dumps(
            {
                "broad_items": [
                    {
                        "broad_index": 0,
                        "detailed_criteria": [
                            "The blue block is visibly within the bin boundary."
                        ],
                    },
                    {
                        "broad_index": 1,
                        "detailed_criteria": [
                            "No movable object other than the red block is on the table."
                        ],
                    },
                ]
            }
        )
        model = SequenceModel(
            [
                "<think>Prepare the checklist.</think>\n"
                f"```json\n{valid}\n```"
            ]
        )
        agent = ValidatorAgent(
            model=model,
            system_prompt="Validator system contract.",
        )

        contract = agent.compile_contract(goal(), SequenceFrames([1])())

        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(contract.broad_items), 2)

    def test_assessment_accepts_wrapped_json(self) -> None:
        contract = validation_contract()
        output = assessment_output(
            contract,
            ("MET", "UNKNOWN"),
            (
                "The block is visibly inside the bin.",
                "The far table edge is outside the current view.",
            ),
        )
        agent = ValidatorAgent(
            model=SequenceModel([f"```json\n{output}\n```"]),
            system_prompt="Validator system contract.",
        )

        assessment = agent.assess_contract(
            contract,
            SequenceFrames([5])(),
            publication_id="final-publication",
        )

        self.assertEqual(
            tuple(item.state.value for item in assessment.criteria),
            ("MET", "UNKNOWN"),
        )


class ValidatorJsonTransportTests(unittest.TestCase):
    def test_prompt_uses_unfenced_valid_json_examples(self) -> None:
        prompt = DEFAULT_VALIDATOR_PROMPT.read_text(encoding="utf-8")

        self.assertNotIn("```", prompt)
        self.assertNotIn("MET | NOT_MET | UNKNOWN", prompt)
        self.assertIn('"state":"UNKNOWN"', prompt)
        self.assertIn('"broad_index":1', prompt)
        self.assertIn("Never repeat its `broad_index`", prompt)

    def test_content_blocks_ignore_reasoning_and_parse_visible_fence(self) -> None:
        response = AIMessage(
            content=[
                {"type": "reasoning", "text": "Private transport reasoning."},
                {"type": "text", "text": '```json\n{"answer":1}\n```'},
            ]
        )

        self.assertEqual(
            parse_validator_json_object(response),
            {"answer": 1},
        )

    def test_wrapper_normalization_preserves_strict_rejections(self) -> None:
        invalid = (
            ('```json\n{"answer":1}\n```\nextra', "outside"),
            (
                '```json\n{"answer":1}\n```\n```json\n{"answer":2}\n```',
                "multiple",
            ),
            ('```json\n{"answer":1,"answer":2}\n```', "duplicate"),
            ('```json\n{"answer":NaN}\n```', "non-standard"),
        )
        for response, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValidatorOutputError, message):
                    parse_validator_json_object(response)


class ValidatorServiceTests(unittest.TestCase):
    def test_replacement_fences_in_flight_result_and_remains_single_flight(self) -> None:
        contract = validation_contract()
        first_started = threading.Event()
        release_first = threading.Event()
        finished = threading.Event()
        active = 0
        max_active = 0
        lock = threading.Lock()

        class BlockingModel:
            def __init__(self) -> None:
                self.calls = 0

            def invoke(inner_self, messages, **kwargs):
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                    inner_self.calls += 1
                    call = inner_self.calls
                try:
                    if call == 1:
                        first_started.set()
                        release_first.wait(2.0)
                    return SimpleNamespace(
                        content=assessment_output(
                            contract,
                            ("MET", "MET"),
                            (
                                "The block is visibly inside the bin.",
                                "The table is visibly clear.",
                            ),
                        )
                    )
                finally:
                    with lock:
                        active -= 1

        model = BlockingModel()
        received = []
        holder = {}

        def accept(assessment) -> None:
            received.append(assessment)
            holder["service"].retire()
            finished.set()

        service = ValidatorService(
            accept,
            model=model,
            frame_source=SequenceFrames([5, 6, 7]),
            system_prompt="Validator system contract.",
            min_interval_seconds=0,
        )
        holder["service"] = service
        try:
            service.publish(final_task("old-publication"), contract)
            self.assertTrue(first_started.wait(2.0))
            service.publish(final_task("new-publication"), contract)
            release_first.set()
            self.assertTrue(finished.wait(2.0))
        finally:
            release_first.set()
            service.stop()

        self.assertEqual(max_active, 1)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].publication_id, "new-publication")

    def test_accumulates_decisive_evidence_across_views(self) -> None:
        contract = validation_contract()
        first_evidence = (
            "The blue block is visibly inside the bin.",
            "The far side of the table is outside this camera view.",
        )
        second_evidence = (
            "The bin is occluded in this camera view.",
            "The swept table view shows no other movable objects.",
        )
        third_evidence = (
            "The blue block is now visibly outside the bin.",
            "The table is outside this camera view.",
        )
        model = SequenceModel(
            [
                assessment_output(
                    contract,
                    ("MET", "UNKNOWN"),
                    first_evidence,
                ),
                assessment_output(
                    contract,
                    ("UNKNOWN", "MET"),
                    second_evidence,
                ),
                assessment_output(
                    contract,
                    ("NOT_MET", "UNKNOWN"),
                    third_evidence,
                ),
            ]
        )
        received = []
        finished = threading.Event()
        holder = {}

        def accept(assessment) -> None:
            received.append(assessment)
            if len(received) == 3:
                holder["service"].retire()
                finished.set()

        service = ValidatorService(
            accept,
            model=model,
            frame_source=SequenceFrames([5, 6, 7]),
            system_prompt="Validator system contract.",
            min_interval_seconds=0,
        )
        holder["service"] = service
        try:
            service.publish(final_task(), contract)
            self.assertTrue(finished.wait(2.0))
        finally:
            service.stop()

        final = received[-1]
        self.assertEqual(received[1].criteria[0].evidence, first_evidence[0])
        self.assertEqual(
            tuple(item.state.value for item in final.criteria),
            ("NOT_MET", "MET"),
        )
        self.assertEqual(final.criteria[0].evidence, third_evidence[0])
        self.assertEqual(final.criteria[1].evidence, second_evidence[1])
        self.assertEqual(final.frame_sequence, 7)

    def test_duplicate_frame_cannot_replace_accumulated_evidence(self) -> None:
        contract = validation_contract()
        model = SequenceModel(
            [
                assessment_output(
                    contract,
                    ("MET", "MET"),
                    (
                        "The block is visibly inside the bin.",
                        "The table is visibly clear.",
                    ),
                ),
                assessment_output(
                    contract,
                    ("NOT_MET", "NOT_MET"),
                    (
                        "A duplicate frame appears to contradict the bin state.",
                        "A duplicate frame appears to contradict the table state.",
                    ),
                ),
                assessment_output(
                    contract,
                    ("UNKNOWN", "UNKNOWN"),
                    (
                        "The bin is outside this newer view.",
                        "The table is outside this newer view.",
                    ),
                ),
            ]
        )
        received = []
        finished = threading.Event()
        holder = {}

        def accept(assessment) -> None:
            received.append(assessment)
            if len(received) == 2:
                holder["service"].retire()
                finished.set()

        service = ValidatorService(
            accept,
            model=model,
            frame_source=SequenceFrames([5, 5, 6]),
            system_prompt="Validator system contract.",
            min_interval_seconds=0,
        )
        holder["service"] = service
        try:
            service.publish(final_task(frame_sequence=4), contract)
            self.assertTrue(finished.wait(2.0))
        finally:
            service.stop()

        self.assertEqual(len(model.calls), 3)
        self.assertEqual(
            tuple(item.state.value for item in received[-1].criteria),
            ("MET", "MET"),
        )
        self.assertEqual(received[-1].frame_sequence, 6)

    def test_new_publication_resets_accumulated_evidence(self) -> None:
        contract = validation_contract()
        model = SequenceModel(
            [
                assessment_output(
                    contract,
                    ("MET", "UNKNOWN"),
                    (
                        "The blue block is visibly inside the bin.",
                        "The table edge is occluded.",
                    ),
                ),
                assessment_output(
                    contract,
                    ("UNKNOWN", "UNKNOWN"),
                    (
                        "The bin is outside this view.",
                        "The table is outside this view.",
                    ),
                ),
            ]
        )
        received = []
        finished = threading.Event()
        holder = {}

        def accept(assessment) -> None:
            received.append(assessment)
            if len(received) == 1:
                holder["service"].publish(
                    final_task("goal-1:validation:attempt-2"),
                    contract,
                )
            else:
                holder["service"].retire()
                finished.set()

        service = ValidatorService(
            accept,
            model=model,
            frame_source=SequenceFrames([5, 6]),
            system_prompt="Validator system contract.",
            min_interval_seconds=0,
        )
        holder["service"] = service
        try:
            service.publish(final_task(), contract)
            self.assertTrue(finished.wait(2.0))
        finally:
            service.stop()

        self.assertEqual(
            tuple(item.state.value for item in received[1].criteria),
            ("UNKNOWN", "UNKNOWN"),
        )

    def test_bad_live_output_emits_output_error_and_retires(self) -> None:
        contract = validation_contract()
        errors = []
        finished = threading.Event()

        def report(event) -> None:
            errors.append(event)
            finished.set()

        model = SequenceModel(["not json"])
        service = ValidatorService(
            lambda assessment: self.fail("assessment must not be delivered"),
            on_error=report,
            model=model,
            frame_source=SequenceFrames([5]),
            system_prompt="Validator system contract.",
            min_interval_seconds=0,
        )
        try:
            service.publish(final_task(), contract)
            self.assertTrue(finished.wait(2.0))
        finally:
            service.stop()

        self.assertEqual(len(errors), 1)
        self.assertIs(errors[0].kind, ValidatorErrorKind.OUTPUT)
        self.assertIsNone(service.active_publication_id)
        self.assertEqual(len(model.calls), 1)

    def test_emergency_bypasses_invalid_ordinary_assessment(self) -> None:
        contract = validation_contract()
        output = json.dumps(
            {
                "emergency_stop": True,
                "emergency_reason": "A person is inside the robot workspace.",
                "criteria": "invalid ordinary payload",
                "observation": None,
            }
        )
        stops = []
        emergency = EmergencyStopCoordinator(stops.append)
        received = []
        errors = []
        service = ValidatorService(
            received.append,
            on_error=errors.append,
            emergency=emergency,
            model=SequenceModel([output]),
            frame_source=SequenceFrames([5]),
            system_prompt="Validator system contract.",
            min_interval_seconds=0,
        )
        try:
            service.publish(final_task(), contract)
            self.assertTrue(service.shutdown_event.wait(2.0))
        finally:
            service.stop()

        self.assertEqual(stops, ["A person is inside the robot workspace."])
        self.assertEqual(received, [])
        self.assertEqual(errors, [])

    def test_frames_at_or_before_publication_are_not_inferred(self) -> None:
        contract = validation_contract()
        model = SequenceModel(
            [
                assessment_output(
                    contract,
                    ("MET", "MET"),
                    (
                        "The block is visibly inside the bin.",
                        "The table is visibly clear.",
                    ),
                )
            ]
        )
        frames = SequenceFrames([4, 5])
        finished = threading.Event()
        holder = {}

        def accept(assessment) -> None:
            holder["service"].retire()
            finished.set()

        service = ValidatorService(
            accept,
            model=model,
            frame_source=frames,
            system_prompt="Validator system contract.",
            min_interval_seconds=0,
        )
        holder["service"] = service
        try:
            service.publish(final_task(frame_sequence=4), contract)
            self.assertTrue(finished.wait(2.0))
        finally:
            service.stop()

        self.assertGreaterEqual(frames.calls, 2)
        self.assertEqual(len(model.calls), 1)


if __name__ == "__main__":
    unittest.main()
