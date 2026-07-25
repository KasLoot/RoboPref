from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.model import OllamaJsonModel


class OllamaTelemetryTests(unittest.TestCase):
    def test_success_emits_metadata_without_raw_prompt_or_response(self) -> None:
        records: list[dict[str, object]] = []
        response = SimpleNamespace(
            message=SimpleNamespace(content='{"mode":"REPORT"}'),
            eval_count=7,
            prompt_eval_count=11,
            total_duration=123,
        )
        requests: list[dict[str, object]] = []

        def chat(**kwargs: object) -> object:
            requests.append(kwargs)
            return response

        client = SimpleNamespace(chat=chat)
        model = OllamaJsonModel(
            "model-under-test",
            seed=42,
            agent_name="HRI Agent",
            telemetry_observer=records.append,
        )

        with patch("ollama.Client", return_value=client):
            result = model.generate(
                purpose="resolve_hri_turn",
                system_prompt="private prompt",
                payload={"user_message": "private request"},
                images=[b"private-image"],
            )

        self.assertEqual(result, {"mode": "REPORT"})
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertTrue(record["success"])
        self.assertEqual(record["agent"], "HRI Agent")
        self.assertEqual(record["model"], "model-under-test")
        self.assertEqual(record["seed"], 42)
        self.assertEqual(record["image_count"], 1)
        self.assertEqual(requests[0]["options"], {"temperature": 0.0, "seed": 42})
        self.assertEqual(
            record["response_metadata"],
            {
                "total_duration": 123,
                "prompt_eval_count": 11,
                "eval_count": 7,
            },
        )
        serialized = repr(record)
        self.assertNotIn("private prompt", serialized)
        self.assertNotIn("private request", serialized)
        self.assertNotIn("private-image", serialized)

    def test_failure_is_observed_and_reraised(self) -> None:
        records: list[dict[str, object]] = []
        client = SimpleNamespace(
            chat=lambda **_kwargs: (_ for _ in ()).throw(TimeoutError("late"))
        )
        model = OllamaJsonModel(
            "model-under-test",
            telemetry_observer=records.append,
        )

        with patch("ollama.Client", return_value=client):
            with self.assertRaisesRegex(TimeoutError, "late"):
                model.generate(
                    purpose="plan_task",
                    system_prompt="prompt",
                    payload={},
                )

        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["success"])
        self.assertEqual(records[0]["error_type"], "TimeoutError")

    def test_observer_failure_does_not_change_model_result(self) -> None:
        response = SimpleNamespace(
            message=SimpleNamespace(content='{"outcome":"SUCCESS"}')
        )
        client = SimpleNamespace(chat=lambda **_kwargs: response)

        def broken_observer(_record: dict[str, object]) -> None:
            raise OSError("telemetry unavailable")

        model = OllamaJsonModel(
            "model-under-test",
            telemetry_observer=broken_observer,
        )
        with patch("ollama.Client", return_value=client):
            result = model.generate(
                purpose="validate_task",
                system_prompt="prompt",
                payload={},
            )

        self.assertEqual(result, {"outcome": "SUCCESS"})


if __name__ == "__main__":
    unittest.main()
