import tempfile
import unittest
from pathlib import Path

from memory.store import PreferenceStore


def stack_preference(value=None):
    return {
        "task_type": "stack_blocks",
        "key": "bottom_to_top_color_order",
        "value": value or ["red", "green", "blue"],
        "context": {"object_type": "cube"},
        "scope": "contextual",
        "confidence": 0.55,
    }


def upsert(scope_marker, value=None):
    return {
        "action": "UPSERT",
        "preference": stack_preference(value),
        "evidence": {
            "quote": "First red, then green, then blue.",
            "reason": "The user selected an order after an ambiguous request.",
            "scope_marker": scope_marker,
        },
    }


class PreferenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = PreferenceStore(Path(self.temp_dir.name) / "preferences.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_one_off_evidence_is_not_persisted(self):
        applied = self.store.apply_operations([upsert("explicit_one_off")])

        self.assertEqual(applied, [])
        self.assertEqual(self.store.list_preferences(), [])

    def test_unmarked_evidence_starts_as_candidate(self):
        applied = self.store.apply_operations([upsert("unmarked")])

        preferences = self.store.list_preferences()
        self.assertEqual(applied[0]["status"], "candidate")
        self.assertEqual(preferences[0]["status"], "candidate")
        self.assertEqual(preferences[0]["evidence_count"], 1)

    def test_repeated_matching_evidence_promotes_candidate(self):
        self.store.apply_operations([upsert("unmarked")])
        applied = self.store.apply_operations([upsert("unmarked")])

        preferences = self.store.list_preferences()
        self.assertEqual(applied[0]["action"], "REINFORCED")
        self.assertEqual(preferences[0]["status"], "durable")
        self.assertEqual(preferences[0]["evidence_count"], 2)

    def test_explicit_durable_evidence_is_immediately_durable(self):
        self.store.apply_operations([upsert("explicit_durable")])

        self.assertEqual(self.store.list_preferences()[0]["status"], "durable")

    def test_repeated_conflicting_value_reinforces_its_own_candidate(self):
        alternate_order = ["blue", "green", "red"]
        self.store.apply_operations([upsert("unmarked")])
        self.store.apply_operations([upsert("unmarked", alternate_order)])
        self.store.apply_operations([upsert("unmarked", alternate_order)])

        preferences = self.store.list_preferences()
        alternate = next(item for item in preferences if item["value"] == alternate_order)
        self.assertEqual(len(preferences), 2)
        self.assertEqual(alternate["status"], "durable")
        self.assertEqual(alternate["evidence_count"], 2)

    def test_retrieve_returns_matching_active_preferences(self):
        self.store.apply_operations([upsert("explicit_durable")])

        matches = self.store.retrieve("Stack the blocks")

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["key"], "bottom_to_top_color_order")


if __name__ == "__main__":
    unittest.main()