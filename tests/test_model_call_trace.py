from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from prefmem.agents.metrics import TurnMetrics


class ModelCallTraceTests(unittest.TestCase):
    def test_turn_metrics_emits_exact_prompt_response_and_usage(self) -> None:
        traces: list[dict] = []
        metrics = TurnMetrics(None, trace_sink=traces.append)
        response = AIMessage(
            content="done",
            usage_metadata={
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            },
        )

        metrics.record(
            agent="HRI Agent",
            prompt_messages=[
                SystemMessage(content="system"),
                HumanMessage(content="question"),
            ],
            response=response,
            elapsed_seconds=0.25,
        )

        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(trace["kind"], "MODEL_CALL")
        self.assertEqual(trace["agent"], "HRI Agent")
        self.assertEqual(trace["prompt_messages"][1]["content"], "question")
        self.assertEqual(trace["raw_response"]["content"], "done")
        self.assertEqual(trace["usage"]["total_tokens"], 6)
        self.assertFalse(metrics.trace_errors)


if __name__ == "__main__":
    unittest.main()
