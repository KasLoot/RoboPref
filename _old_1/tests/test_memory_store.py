import json
import tempfile
import unittest
from pathlib import Path

from memory.store import PreferenceStore


RGB = ["red", "green", "blue"]
BGR = ["blue", "green", "red"]


def stack_preference(value=None):
    return {
        "task_type": "stack_blocks",
        "key": "bottom_to_top_color_order",
        "value": value or RGB,
        "context": {"object_type": "cube"},
        "scope": "contextual",
        "confidence": 0.75,
    }


def upsert(
    scope_marker="unmarked",
    value=None,
    *,
    origin="open_preference_answer",
    quote="Red, green, then blue.",
    independent=True,
):
    return {
        "action": "UPSERT",
        "preference": stack_preference(value),
        "evidence": {
            "quote": quote,
            "reason": "Test evidence.",
            "scope_marker": scope_marker,
            "origin": origin,
            "independent": independent,
        },
    }


class PreferenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = PreferenceStore(Path(self.temp_dir.name) / "preferences.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_one_off_evidence_is_not_persisted(self):
        applied = self.store.apply_operations(
            [upsert("explicit_one_off", quote="Use RGB just this time.")]
        )
        self.assertEqual(applied, [])
        self.assertEqual(self.store.list_preferences(), [])

    def test_fully_specified_current_task_is_not_preference_evidence(self):
        applied = self.store.apply_operations(
            [upsert(origin="current_task", quote="Stack red, green, then blue.")]
        )
        self.assertEqual(applied, [])
        self.assertEqual(self.store.list_preferences(), [])

    def test_yes_to_action_proposal_is_not_preference_evidence(self):
        applied = self.store.apply_operations(
            [upsert(origin="assistant_plan_confirmation", quote="Yes.")]
        )
        self.assertEqual(applied, [])
        self.assertEqual(self.store.list_preferences(), [])

    def test_missing_evidence_origin_fails_closed(self):
        operation = upsert()
        operation["evidence"].pop("origin")
        applied = self.store.apply_operations([operation])
        self.assertEqual(applied, [])
        self.assertEqual(self.store.list_preferences(), [])

    def test_unmarked_open_answer_starts_as_candidate(self):
        applied = self.store.apply_operations([upsert()])
        preference = self.store.list_preferences()[0]
        self.assertEqual(applied[0]["action"], "ADDED")
        self.assertEqual(preference["status"], "candidate")
        self.assertEqual(preference["independent_evidence_count"], 1)

    def test_repeated_evidence_requests_consent_but_does_not_auto_promote(self):
        self.store.apply_operations([upsert()], session_id="session-1")
        applied = self.store.apply_operations([upsert()], session_id="session-2")
        preference = self.store.list_preferences()[0]
        self.assertEqual(applied[0]["action"], "CONFIRM_REQUIRED")
        self.assertEqual(preference["status"], "candidate")
        self.assertEqual(preference["independent_evidence_count"], 2)
        self.assertTrue(preference["confirmation_pending"])

    def test_dedicated_memory_yes_promotes_candidate(self):
        self.store.apply_operations([upsert()])
        applied = self.store.apply_operations([upsert()])
        preference_id = applied[0]["preference_id"]
        result = self.store.confirm_candidate(preference_id, quote="Yes, remember it.")
        self.assertEqual(result["action"], "CONFIRMED")
        self.assertEqual(self.store.list_preferences()[0]["status"], "durable")

    def test_declining_memory_question_keeps_candidate(self):
        self.store.apply_operations([upsert()])
        prompt = self.store.apply_operations([upsert()])[0]
        self.store.decline_candidate(prompt["preference_id"], quote="No.")
        preference = self.store.list_preferences()[0]
        self.assertEqual(preference["status"], "candidate")
        self.assertFalse(preference["confirmation_pending"])

        # New independent evidence may justify asking again; the decline itself does not.
        reapplied = self.store.apply_operations([upsert()])
        self.assertEqual(reapplied[0]["action"], "CONFIRM_REQUIRED")
        self.assertEqual(self.store.list_preferences()[0]["status"], "candidate")

    def test_explicit_future_preference_is_immediately_durable(self):
        applied = self.store.apply_operations(
            [
                upsert(
                    "explicit_durable",
                    origin="explicit_preference",
                    quote="From now on, stack red, green, then blue.",
                )
            ]
        )
        self.assertEqual(applied[0]["status"], "durable")

    def test_present_task_correction_cannot_overwrite_durable_default(self):
        self.store.apply_operations(
            [
                upsert(
                    "explicit_durable",
                    origin="explicit_preference",
                    quote="Remember RGB as my default.",
                )
            ]
        )
        result = self.store.apply_operations(
            [
                upsert(
                    "correction",
                    BGR,
                    origin="user_initiated_override",
                    quote="No, the opposite order.",
                )
            ]
        )[0]
        preferences = self.store.list_preferences()
        durable = next(item for item in preferences if item["status"] == "durable")
        self.assertEqual(durable["value"], RGB)
        self.assertEqual(result["action"], "CONFIRM_REQUIRED")
        self.assertEqual(result["reason"], "conflicts_with_durable")

    def test_explicit_future_correction_replaces_durable_default(self):
        self.store.apply_operations(
            [
                upsert(
                    "explicit_durable",
                    origin="explicit_preference",
                    quote="Remember RGB as my default.",
                )
            ]
        )
        self.store.apply_operations(
            [
                upsert(
                    "correction",
                    BGR,
                    origin="explicit_preference",
                    quote="I want the opposite order in the future; replace my default.",
                )
            ]
        )
        active = [item for item in self.store.list_preferences() if item["status"] in {"candidate", "durable"}]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["status"], "durable")
        self.assertEqual(active[0]["value"], BGR)

    def test_retrieve_returns_matching_active_preferences(self):
        self.store.apply_operations(
            [
                upsert(
                    "explicit_durable",
                    origin="explicit_preference",
                    quote="I prefer RGB for block stacks.",
                )
            ]
        )
        matches = self.store.retrieve("Stack the blocks")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["key"], "bottom_to_top_color_order")

    def test_legacy_auto_promoted_record_is_read_as_candidate(self):
        legacy = stack_preference()
        legacy.update(
            {
                "id": "pref-legacy",
                "user_id": "default",
                "status": "durable",
                "evidence_count": 2,
                "evidence": [
                    {
                        "quote": "RGB.",
                        "scope_marker": "unmarked",
                        "reason": "First choice.",
                    },
                    {
                        "quote": "Yes.",
                        "scope_marker": "unmarked",
                        "reason": "Confirmed an action proposal.",
                    },
                ],
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-02T00:00:00+00:00",
            }
        )
        self.store.path.write_text(
            json.dumps({"schema_version": 1, "preferences": [legacy]}),
            encoding="utf-8",
        )

        preference = self.store.list_preferences()[0]

        self.assertEqual(preference["status"], "candidate")


if __name__ == "__main__":
    unittest.main()
