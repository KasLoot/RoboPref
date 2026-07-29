from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from urllib.request import Request, urlopen

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)


def message_text(content: object) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
        )

    return ""


class VLLMTokenCounter:
    """Count raw text tokens with the tokenizer used by the vLLM server."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
    ) -> None:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        self.url = f"{base_url}/tokenize"
        self.model = model
        self.api_key = api_key
        self.failed = False

    @lru_cache(maxsize=512)
    def count(self, text: str) -> int | None:
        if not text:
            return 0

        if self.failed:
            return None

        payload = json.dumps(
            {
                "model": self.model,
                "prompt": text,
                "add_special_tokens": False,
            }
        ).encode("utf-8")

        request = Request(
            self.url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urlopen(request, timeout=5) as response:
                return int(json.load(response)["count"])
        except (OSError, ValueError, KeyError):
            # Exact API totals remain available even if attribution fails.
            self.failed = True
            return None


@dataclass(slots=True)
class LLMCallMetric:
    agent: str
    system_texts: tuple[str, ...]
    query_texts: tuple[str, ...]
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    elapsed_seconds: float
    server_tokens_per_second: float | None
    generated_tool_call: bool


def optional_sum(values: list[int | None]) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


class TurnMetrics:
    def __init__(self, token_counter: VLLMTokenCounter | None) -> None:
        self.token_counter = token_counter
        self.calls: list[LLMCallMetric] = []

    def reset(self) -> None:
        self.calls.clear()

    def record(
        self,
        *,
        agent: str,
        prompt_messages: list[BaseMessage],
        response: AIMessage,
        elapsed_seconds: float,
    ) -> None:
        usage = response.usage_metadata or {}

        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")

        server_metrics = response.response_metadata.get("vllm_metrics") or {}
        server_tps = server_metrics.get("tokens_per_second")

        self.calls.append(
            LLMCallMetric(
                agent=agent,
                system_texts=tuple(
                    text
                    for message in prompt_messages
                    if isinstance(message, SystemMessage)
                    if (text := message_text(message.content))
                ),
                query_texts=tuple(
                    text
                    for message in prompt_messages
                    if isinstance(message, HumanMessage)
                    if (text := message_text(message.content))
                ),
                input_tokens=(
                    int(input_tokens) if input_tokens is not None else None
                ),
                output_tokens=(
                    int(output_tokens) if output_tokens is not None else None
                ),
                total_tokens=(
                    int(total_tokens) if total_tokens is not None else None
                ),
                elapsed_seconds=elapsed_seconds,
                server_tokens_per_second=(
                    float(server_tps) if server_tps is not None else None
                ),
                generated_tool_call=bool(response.tool_calls),
            )
        )

    def _text_tokens(
        self,
        calls: list[LLMCallMetric],
        attribute: str,
    ) -> int | None:
        if self.token_counter is None:
            return None

        total = 0
        for call in calls:
            for text in getattr(call, attribute):
                count = self.token_counter.count(text)
                if count is None:
                    return None
                total += count
        return total

    @staticmethod
    def _throughput(
        calls: list[LLMCallMetric],
    ) -> tuple[float | None, str]:
        output_tokens = optional_sum(
            [call.output_tokens for call in calls]
        )
        if output_tokens is None:
            return None, "unavailable"

        generated_calls = [
            call for call in calls if (call.output_tokens or 0) > 0
        ]
        if not generated_calls:
            return 0.0, "none"

        if all(
            call.server_tokens_per_second is not None
            and call.server_tokens_per_second > 0
            for call in generated_calls
        ):
            inference_seconds = sum(
                call.output_tokens / call.server_tokens_per_second
                for call in generated_calls
                if (
                    call.output_tokens is not None
                    and call.server_tokens_per_second is not None
                )
            )
            return output_tokens / inference_seconds, "vllm"

        client_seconds = sum(
            call.elapsed_seconds for call in generated_calls
        )
        return output_tokens / client_seconds, "client"

    def _agent_summary(
        self,
        calls: list[LLMCallMetric],
    ) -> dict:
        input_tokens = optional_sum(
            [call.input_tokens for call in calls]
        )
        output_tokens = optional_sum(
            [call.output_tokens for call in calls]
        )
        total_tokens = optional_sum(
            [call.total_tokens for call in calls]
        )

        system_tokens = self._text_tokens(calls, "system_texts")
        query_tokens = self._text_tokens(calls, "query_texts")

        other_input_tokens = None
        if (
            input_tokens is not None
            and system_tokens is not None
            and query_tokens is not None
        ):
            other_input_tokens = max(
                0,
                input_tokens - system_tokens - query_tokens,
            )

        throughput, throughput_source = self._throughput(calls)

        return {
            "calls": len(calls),
            "system_prompt_tokens": system_tokens,
            "query_prompt_tokens": query_tokens,
            "other_input_tokens": other_input_tokens,
            "input_tokens": input_tokens,
            "generated_tokens": output_tokens,
            "total_tokens": total_tokens,
            "throughput_tokens_per_second": throughput,
            "throughput_source": throughput_source,
        }

    def summary(self, *, turn_seconds: float) -> dict:
        hri_calls = [
            call for call in self.calls if call.agent == "HRI Agent"
        ]
        internal_calls = [
            call for call in self.calls if call.agent != "HRI Agent"
        ]

        input_tokens = optional_sum(
            [call.input_tokens for call in self.calls]
        )
        output_tokens = optional_sum(
            [call.output_tokens for call in self.calls]
        )
        total_tokens = optional_sum(
            [call.total_tokens for call in self.calls]
        )

        system_tokens = self._text_tokens(
            self.calls,
            "system_texts",
        )
        external_query_tokens = self._text_tokens(
            hri_calls,
            "query_texts",
        )
        internal_query_tokens = self._text_tokens(
            internal_calls,
            "query_texts",
        )

        other_input_tokens = None
        if (
            input_tokens is not None
            and system_tokens is not None
            and external_query_tokens is not None
            and internal_query_tokens is not None
        ):
            other_input_tokens = max(
                0,
                input_tokens
                - system_tokens
                - external_query_tokens
                - internal_query_tokens,
            )

        internal_generated = optional_sum(
            [call.output_tokens for call in internal_calls]
        )
        hri_generated = optional_sum(
            [
                call.output_tokens
                for call in hri_calls
                if not call.generated_tool_call
            ]
        )
        internal_query_generated = optional_sum(
            [
                call.output_tokens
                for call in hri_calls
                if call.generated_tool_call
            ]
        )

        throughput, throughput_source = self._throughput(self.calls)

        agent_names = dict.fromkeys(call.agent for call in self.calls)

        return {
            "total": {
                "system_prompt_tokens": system_tokens,
                "query_prompt_tokens": external_query_tokens,
                "internal_query_tokens": internal_query_tokens,
                "internal_query_generated_tokens": (
                    internal_query_generated
                ),
                "internal_generated_tokens": internal_generated,
                "hri_generated_tokens": hri_generated,
                "other_input_tokens": other_input_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "throughput_tokens_per_second": throughput,
                "throughput_source": throughput_source,
                "turn_wall_seconds": turn_seconds,
                "turn_output_tokens_per_second": (
                    output_tokens / turn_seconds
                    if output_tokens is not None and turn_seconds > 0
                    else None
                ),
            },
            "agents": {
                agent: self._agent_summary(
                    [
                        call
                        for call in self.calls
                        if call.agent == agent
                    ]
                )
                for agent in agent_names
            },
        }
