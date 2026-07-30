"""Typed model boundary shared by PrefMem's reasoning agents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from prefmem.recording import ExperimentRecorder, NullRecorder


OutputT = TypeVar("OutputT", bound=BaseModel)


class StructuredModelError(RuntimeError):
    """Raised when a model request does not yield the promised typed result."""


@dataclass(frozen=True, slots=True)
class ModelEndpoint:
    model: str
    base_url: str
    api_key: str = "EMPTY"
    timeout_seconds: float = 300.0
    max_tokens: int = 16384
    temperature: float = 0.0
    reasoning_effort: str | None = None


class StructuredAgentClient:
    """Invoke an OpenAI-compatible multimodal model with a Pydantic schema."""

    def __init__(
        self,
        endpoint: ModelEndpoint,
        *,
        recorder: ExperimentRecorder | None = None,
        model: Any | None = None,
        structured_method: str = "json_schema",
    ) -> None:
        self.endpoint = endpoint
        self.recorder = recorder or NullRecorder()
        self.structured_method = structured_method
        self.model = model or ChatOpenAI(
            model=endpoint.model,
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            max_completion_tokens=endpoint.max_tokens,
            temperature=endpoint.temperature,
            timeout=endpoint.timeout_seconds,
            max_retries=1,
            streaming=False,
            reasoning_effort=endpoint.reasoning_effort,
        )

    @staticmethod
    def build_messages(
        *,
        system_prompt: str,
        text: str,
        frame: Mapping[str, Any] | None = None,
        visual_blocks: Sequence[Mapping[str, Any]] = (),
        prior_messages: Sequence[Any] = (),
    ) -> list[Any]:
        content: str | list[dict[str, Any]]
        blocks = [dict(block) for block in visual_blocks]
        if frame is not None:
            blocks.insert(0, dict(frame))
        if not blocks:
            content = text
        else:
            content = [
                *blocks,
                {"type": "text", "text": text},
            ]
        return [
            SystemMessage(content=system_prompt),
            *prior_messages,
            HumanMessage(content=content),
        ]

    def invoke(
        self,
        *,
        agent: str,
        schema: type[OutputT],
        system_prompt: str,
        text: str,
        frame: Mapping[str, Any] | None = None,
        visual_blocks: Sequence[Mapping[str, Any]] = (),
        prior_messages: Sequence[Any] = (),
        frame_paths: Sequence[Path] = (),
        metadata: Mapping[str, Any] | None = None,
        structured_method: str | None = None,
    ) -> OutputT:
        messages = self.build_messages(
            system_prompt=system_prompt,
            text=text,
            frame=frame,
            visual_blocks=visual_blocks,
            prior_messages=prior_messages,
        )

        try:
            method = structured_method or self.structured_method
            runnable = self.model.with_structured_output(
                schema,
                method=method,
                include_raw=True,
                strict=True if method == "json_schema" else None,
            )
            result = runnable.invoke(messages)
        except Exception as exc:
            self.recorder.record_model_exchange(
                agent=agent,
                messages=messages,
                response={"error": f"{type(exc).__name__}: {exc}"},
                frame_paths=frame_paths,
                metadata=metadata,
            )
            raise StructuredModelError(
                f"{agent} request failed: {type(exc).__name__}: {exc}"
            ) from exc

        if isinstance(result, schema):
            parsed = result
            raw: Any = None
            parsing_error: Any = None
        elif isinstance(result, Mapping):
            parsed = result.get("parsed")
            raw = result.get("raw")
            parsing_error = result.get("parsing_error")
        else:
            parsed = None
            raw = result
            parsing_error = None

        self.recorder.record_model_exchange(
            agent=agent,
            messages=messages,
            response={
                "parsed": parsed,
                "raw": raw,
                "parsing_error": parsing_error,
            },
            frame_paths=frame_paths,
            metadata=metadata,
        )

        if isinstance(parsed, schema):
            return parsed
        if isinstance(parsed, Mapping):
            try:
                return schema.model_validate(parsed)
            except Exception as exc:
                raise StructuredModelError(
                    f"{agent} returned data that failed {schema.__name__}: {exc}"
                ) from exc

        detail = f": {parsing_error}" if parsing_error else ""
        raise StructuredModelError(
            f"{agent} did not return a valid {schema.__name__}{detail}"
        )
