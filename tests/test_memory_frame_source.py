from __future__ import annotations

import unittest
from unittest.mock import patch

from langchain.messages import HumanMessage

from prefmem.agents.memory import Memory_Agent


FRAME = {
    "type": "image_url",
    "image_url": {"url": "data:image/png;base64,simulation"},
}


class MemoryFrameSourceTests(unittest.TestCase):
    def test_explicit_runtime_frame_bypasses_webcam_fetch(self) -> None:
        agent = object.__new__(Memory_Agent)
        reset_calls = []
        received_frames = []
        agent._reset_request_state = reset_calls.append
        agent.invoke_agent = lambda messages, current_frame: (
            received_frames.append(current_frame) or {"messages": messages}
        )

        with patch(
            "prefmem.agents.memory.get_live_frame",
            side_effect=AssertionError("unexpected webcam fetch"),
        ):
            result = agent.run(
                [HumanMessage(content="RETRIEVE REQUEST: block order")],
                current_frame=FRAME,
            )

        self.assertEqual(reset_calls, ["RETRIEVE"])
        self.assertEqual(received_frames, [FRAME])
        self.assertEqual(len(result["messages"]), 1)


if __name__ == "__main__":
    unittest.main()
