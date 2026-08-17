from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments_suite_v2.runners import ab_mem_f_development as runner


class _Response:
    def __init__(self, status_code: int, body: dict | str) -> None:
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return self._body if isinstance(self._body, dict) else json.loads(self._body)


class _Client:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls = 0

    def post(self, url, json):
        del url, json
        self.calls += 1
        return self.response


class ABMemoryFilterDevelopmentTests(unittest.TestCase):
    def test_exact_registered_row_grid(self) -> None:
        rows = runner.build_rows()
        self.assertEqual(len(rows), 1410)
        self.assertEqual(len({row["row_id"] for row in rows}), 1410)
        self.assertEqual(
            {phase: sum(row["phase"] == phase for row in rows) for phase in {"PRIMARY", "STOCHASTIC", "CLEANUP"}},
            {"PRIMARY": 480, "STOCHASTIC": 450, "CLEANUP": 480},
        )
        self.assertEqual(
            {condition: sum(row["phase"] == "PRIMARY" and row["condition"] == condition for row in rows) for condition in runner.FILTER_CONDITIONS},
            {"RAW": 160, "SEMANTIC_FILTER": 160, "ORACLE_FILTER": 160},
        )

    def test_model_request_does_not_expose_scoring_labels(self) -> None:
        row = runner.build_rows()[0]
        serialized = json.dumps(runner._first_messages(row), sort_keys=True)
        self.assertNotIn("correct_decision_scoring_only", serialized)
        self.assertNotIn("relevant_ids_scoring_only", serialized)
        self.assertNotIn('"relevant"', serialized)
        self.assertNotIn("pollution_kind", serialized)

    def test_semantic_and_cleanup_scorers(self) -> None:
        row = next(
            item
            for item in runner.build_rows()
            if item["condition"] == "SEMANTIC_FILTER" and item["relevant_ids_scoring_only"]
        )
        first = runner._score_first(
            row,
            {
                "kept_ids": row["relevant_ids_scoring_only"],
                "decision": row["correct_decision_scoring_only"],
                "unsupported_memory_premise": False,
                "unauthorized_action": False,
            },
        )
        self.assertTrue(first["passed"])
        self.assertTrue(
            runner._score_cleanup(
                {"visible_boards": ["white", "cyan"], "used_memory": False, "unauthorized_action": False}
            )["passed"]
        )

    def test_healthy_malformed_json_is_capability_not_transport_invalidity(self) -> None:
        response = _Response(
            200,
            {"choices": [{"message": {"content": "not-json"}}]},
        )
        with tempfile.TemporaryDirectory() as directory:
            outcome = runner._call(
                _Client(response),
                Path(directory),
                call_id="malformed",
                ordinal=0,
                payload={"model": runner.MODEL_ID, "messages": []},
            )
        self.assertFalse(outcome["usable"])
        self.assertFalse(outcome["invalid"])

    def test_non_200_is_invalid_and_never_retried(self) -> None:
        client = _Client(_Response(503, "service unavailable"))
        with tempfile.TemporaryDirectory() as directory:
            outcome = runner._call(
                client,
                Path(directory),
                call_id="service-fail",
                ordinal=0,
                payload={"model": runner.MODEL_ID, "messages": []},
            )
        self.assertFalse(outcome["usable"])
        self.assertTrue(outcome["invalid"])
        self.assertEqual(client.calls, 1)


if __name__ == "__main__":
    unittest.main()
