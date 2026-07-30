from __future__ import annotations

import unittest
from datetime import UTC, datetime

from prefmem.agents.contracts import (
    HRIDecision,
    HRIDecisionKind,
    HRIInteraction,
    HRIMemoryAction,
    InteractionKind,
    MemoryActionKind,
    MemoryContext,
    MemoryStatus,
    Observation,
    PreferenceRecord,
    TaskPhase,
)
from prefmem.controller import PrefMemController
from prefmem.observations import ObservationEnvelope


class FakeObservationSource:
    def __init__(self) -> None:
        self.sequence = 0

    def capture(self, *, purpose: str) -> ObservationEnvelope:
        self.sequence += 1
        return ObservationEnvelope(
            observation=Observation(
                observation_id=f"obs-{self.sequence}",
                captured_at=datetime.now(UTC),
                sequence=self.sequence,
                image_block={
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,frame"},
                },
            ),
            frame_path=None,
            source=purpose,
        )


class SequenceHRI:
    def __init__(self, decisions: list[HRIDecision]) -> None:
        self.decisions = decisions
        self.pending_contexts: list[dict | None] = []

    def decide(self, **kwargs) -> HRIDecision:
        self.pending_contexts.append(kwargs["pending_memory_consent"])
        return self.decisions.pop(0)


class FakeMemory:
    def __init__(self) -> None:
        self.remember_calls: list[dict] = []

    def remember_preference(self, **kwargs) -> PreferenceRecord:
        self.remember_calls.append(kwargs)
        return PreferenceRecord(
            id="pref-1",
            username=kwargs["username"],
            statement=kwargs["statement"],
            scope=kwargs["scope"],
            applicability=kwargs["applicability"],
            structured_value=kwargs["structured_value"],
        )


class UnusedReasoner:
    def __getattr__(self, name):
        raise AssertionError(f"{name} should not be called")


def consent_question() -> HRIDecision:
    return HRIDecision(
        decision=HRIDecisionKind.ASK_USER,
        reply_to_user="Remember the blue mug as your default?",
        interaction=HRIInteraction(
            kind=InteractionKind.MEMORY_CONSENT,
            proposed_value={
                "action": "REMEMBER",
                "statement": "Use the blue mug by default.",
                "scope": {"kind": "contextual"},
                "structured_value": {"mug": "blue"},
                "target_record_id": None,
            },
        ),
        reason_code="ASK_MEMORY_CONSENT",
    )


def consent_response(*, statement: str) -> HRIDecision:
    return HRIDecision(
        decision=HRIDecisionKind.RESPOND,
        reply_to_user="Okay.",
        memory_action=HRIMemoryAction(
            action=MemoryActionKind.REMEMBER,
            statement=statement,
            scope={"kind": "contextual"},
            structured_value={"mug": "blue"},
        ),
        reason_code="MEMORY_CONSENT_RESPONSE",
    )


class MemoryConsentBindingTests(unittest.TestCase):
    def make_controller(self, hri: SequenceHRI, memory: FakeMemory):
        return PrefMemController(
            username="alice",
            observation_source=FakeObservationSource(),
            hri=hri,
            planner=UnusedReasoner(),
            monitor=UnusedReasoner(),
            validator=UnusedReasoner(),
            memory=memory,
            plan_only=True,
        )

    def test_affirmation_cannot_substitute_a_different_preference(self) -> None:
        hri = SequenceHRI(
            [
                consent_question(),
                consent_response(statement="Use the red mug by default."),
            ]
        )
        memory = FakeMemory()
        controller = self.make_controller(hri, memory)

        first = controller.handle_user("I often pick the same mug.")
        second = controller.handle_user("yes")

        self.assertEqual(first.phase, TaskPhase.WAITING_FOR_USER)
        self.assertEqual(second.phase, TaskPhase.COMPLETE)
        self.assertEqual(memory.remember_calls, [])
        self.assertIn("not committed", second.text)
        self.assertEqual(
            hri.pending_contexts[1]["pending_question"],
            "Remember the blue mug as your default?",
        )

    def test_affirmation_commits_only_the_exact_pending_proposal(self) -> None:
        hri = SequenceHRI(
            [
                consent_question(),
                consent_response(statement="Use the blue mug by default."),
            ]
        )
        memory = FakeMemory()
        controller = self.make_controller(hri, memory)

        controller.handle_user("I often pick the same mug.")
        result = controller.handle_user("yes")

        self.assertEqual(result.phase, TaskPhase.COMPLETE)
        self.assertEqual(len(memory.remember_calls), 1)
        self.assertEqual(
            memory.remember_calls[0]["statement"],
            "Use the blue mug by default.",
        )
        self.assertIn("Saved preference", result.text)


if __name__ == "__main__":
    unittest.main()
