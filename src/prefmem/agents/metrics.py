from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
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


def reasoning_texts(message: AIMessage) -> tuple[str, ...]:
    """Return reasoning text exposed separately from assistant content."""

    texts = []
    for key in ("reasoning", "reasoning_content"):
        value = message.additional_kwargs.get(key)
        if isinstance(value, str) and value:
            texts.append(value)

    if isinstance(message.content, list):
        for block in message.content:
            if not isinstance(block, dict):
                continue
            if block.get("type") not in {
                "reasoning",
                "reasoning_content",
                "thinking",
            }:
                continue
            text = block.get("text") or block.get("reasoning")
            if isinstance(text, str) and text:
                texts.append(text)

    return tuple(texts)


def tool_call_texts(message: AIMessage) -> tuple[str, ...]:
    """Return canonical text for the semantic payload of each tool call."""

    texts = []
    for call in [*message.tool_calls, *message.invalid_tool_calls]:
        arguments = call.get("args", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        texts.append(
            json.dumps(
                {
                    "name": call.get("name"),
                    "arguments": arguments,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        )
    return tuple(texts)


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
    reasoning_texts: tuple[str, ...]
    response_texts: tuple[str, ...]
    tool_call_texts: tuple[str, ...]
    provider_reasoning_tokens: int | None
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
    def __init__(
        self,
        token_counter: VLLMTokenCounter | None,
        *,
        trace_sink: Callable[[dict], None] | None = None,
    ) -> None:
        self.token_counter = token_counter
        self.calls: list[LLMCallMetric] = []
        self.trace_sink = trace_sink
        self.trace_errors: list[str] = []

    def set_trace_sink(self, sink: Callable[[dict], None] | None) -> None:
        if sink is not None and not callable(sink):
            raise TypeError("trace sink must be callable or None")
        self.trace_sink = sink

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
        output_token_details = usage.get("output_token_details") or {}
        provider_reasoning_tokens = output_token_details.get("reasoning")

        if provider_reasoning_tokens is None:
            raw_usage = (
                response.response_metadata.get("token_usage") or {}
            )
            raw_output_details = (
                raw_usage.get("completion_tokens_details") or {}
            )
            provider_reasoning_tokens = raw_output_details.get(
                "reasoning_tokens"
            )

        server_metrics = response.response_metadata.get("vllm_metrics") or {}
        server_tps = server_metrics.get("tokens_per_second")

        metric = LLMCallMetric(
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
                reasoning_texts=reasoning_texts(response),
                response_texts=(
                    (text,)
                    if (text := message_text(response.content))
                    else ()
                ),
                tool_call_texts=tool_call_texts(response),
                provider_reasoning_tokens=(
                    int(provider_reasoning_tokens)
                    if provider_reasoning_tokens is not None
                    else None
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
        self.calls.append(metric)

        sink = self.trace_sink
        if sink is not None:
            event = {
                "schema_version": 1,
                "kind": "MODEL_CALL",
                "call_id": uuid.uuid4().hex,
                "agent": agent,
                "recorded_at_unix": time.time(),
                "elapsed_seconds": elapsed_seconds,
                "prompt_messages": [
                    self._message_payload(message)
                    for message in prompt_messages
                ],
                "raw_response": self._message_payload(response),
                "usage": {
                    "input_tokens": metric.input_tokens,
                    "output_tokens": metric.output_tokens,
                    "total_tokens": metric.total_tokens,
                    "provider_reasoning_tokens": (
                        metric.provider_reasoning_tokens
                    ),
                    "server_tokens_per_second": (
                        metric.server_tokens_per_second
                    ),
                },
            }
            try:
                sink(event)
            except Exception as error:
                self.trace_errors.append(
                    f"{type(error).__name__}: {error}"
                )

    @staticmethod
    def _message_payload(message: BaseMessage) -> dict:
        try:
            payload = message.model_dump(mode="json")
        except (AttributeError, TypeError, ValueError):
            payload = {
                "type": type(message).__name__,
                "content": str(getattr(message, "content", message)),
            }
        return payload

    def _raw_text_tokens(self, texts: tuple[str, ...]) -> int | None:
        if not texts:
            return 0
        if self.token_counter is None:
            return None

        total = 0
        for text in texts:
            count = self.token_counter.count(text)
            if count is None:
                return None
            total += count
        return total

    def _call_output_breakdown(
        self,
        call: LLMCallMetric,
    ) -> tuple[int | None, int | None, int | None, int | None]:
        """Split a completion into thinking, tool, response, and other tokens.

        Providers report the exact completion total and may report exact
        reasoning tokens. When reasoning details are absent, separately exposed
        reasoning text is counted with the server tokenizer. The remainder of a
        completion is assigned to its visible output channel. This preserves
        the provider total without treating parsed tool-call JSON as an exact
        reproduction of the model's wire-format control tokens.
        """

        if call.provider_reasoning_tokens is not None:
            thinking_tokens = call.provider_reasoning_tokens
        else:
            thinking_tokens = self._raw_text_tokens(call.reasoning_texts)

        if call.output_tokens is None:
            return (
                thinking_tokens,
                self._raw_text_tokens(call.tool_call_texts),
                self._raw_text_tokens(call.response_texts),
                None,
            )

        if thinking_tokens is None:
            return None, None, None, None

        thinking_tokens = min(
            max(0, thinking_tokens),
            call.output_tokens,
        )
        remaining_tokens = call.output_tokens - thinking_tokens
        has_tool_calls = bool(call.tool_call_texts)
        has_response = bool(call.response_texts)

        if has_tool_calls and not has_response:
            return thinking_tokens, remaining_tokens, 0, 0
        if has_response and not has_tool_calls:
            return thinking_tokens, 0, remaining_tokens, 0
        if not has_tool_calls and not has_response:
            return thinking_tokens, 0, 0, remaining_tokens

        tool_estimate = self._raw_text_tokens(call.tool_call_texts)
        response_estimate = self._raw_text_tokens(call.response_texts)
        if tool_estimate is None or response_estimate is None:
            return thinking_tokens, None, None, None

        estimated_tokens = tool_estimate + response_estimate
        if estimated_tokens <= remaining_tokens:
            return (
                thinking_tokens,
                tool_estimate,
                response_estimate,
                remaining_tokens - estimated_tokens,
            )
        if estimated_tokens == 0:
            return thinking_tokens, 0, 0, remaining_tokens

        # A response containing both text and tool calls is unusual. Preserve
        # the exact provider total while using tokenizer counts as the ratio.
        tool_tokens = round(
            remaining_tokens * tool_estimate / estimated_tokens
        )
        return (
            thinking_tokens,
            tool_tokens,
            remaining_tokens - tool_tokens,
            0,
        )

    def _output_breakdown(self, calls: list[LLMCallMetric]) -> dict:
        breakdowns = [
            self._call_output_breakdown(call) for call in calls
        ]
        breakdown = {
            "thinking_output_tokens": optional_sum(
                [breakdown[0] for breakdown in breakdowns]
            ),
            "tool_call_output_tokens": optional_sum(
                [breakdown[1] for breakdown in breakdowns]
            ),
            "response_output_tokens": optional_sum(
                [breakdown[2] for breakdown in breakdowns]
            ),
            "other_output_tokens": optional_sum(
                [breakdown[3] for breakdown in breakdowns]
            ),
        }
        if not calls:
            source = "none"
        elif any(value is None for value in breakdown.values()):
            source = "unavailable"
        else:
            used_tokenizer = any(
                (
                    call.provider_reasoning_tokens is None
                    and bool(call.reasoning_texts)
                )
                or (
                    bool(call.tool_call_texts)
                    and bool(call.response_texts)
                )
                or call.output_tokens is None
                for call in calls
            )
            used_provider = any(
                call.output_tokens is not None
                or call.provider_reasoning_tokens is not None
                for call in calls
            )
            if used_provider and used_tokenizer:
                source = "provider_and_tokenizer"
            elif used_provider:
                source = "provider"
            elif used_tokenizer:
                source = "tokenizer"
            else:
                source = "none"

        return {
            **breakdown,
            "output_token_breakdown_source": source,
        }

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
            **self._output_breakdown(calls),
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
                **self._output_breakdown(self.calls),
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
