"""Gemma instruction compiler for the constrained pick-and-place executor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any, Protocol

from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from prefmem.contracts import PublishedTask, TaskPhase
from prefmem.execution.contracts import ManipulationProgram


DEFAULT_EXECUTION_MODEL = "/workspace/models/gemma-4-26B-A4B-it"
DEFAULT_EXECUTION_BASE_URL = "http://localhost:8000/v1"
THINKING_DISABLED_OPTIONS = {
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
}


class ExecutionCompilerError(ValueError):
    """Gemma did not produce a valid, safely constrained program."""


class ExecutionModel(Protocol):
    def invoke(self, messages: list[Any], **kwargs: Any) -> Any: ...


_THINKING_BLOCK = re.compile(
    r"<(?P<tag>think|thinking|analysis|reasoning)>.*?</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_JSON_FENCE = re.compile(
    r"```(?:json)?\s*(?P<body>.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _visible_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") not in {
                "reasoning",
                "thinking",
                "analysis",
            }:
                value = block.get("text", block.get("content"))
                if isinstance(value, str):
                    parts.append(value)
        if "".join(parts).strip():
            return "".join(parts)
    raise ExecutionCompilerError("Gemma returned no visible compiler output")


def parse_execution_json(response: object) -> dict[str, Any]:
    """Accept one JSON object with only known Gemma transport wrappers."""

    text = _THINKING_BLOCK.sub("", _visible_text(response)).strip()
    fences = list(_JSON_FENCE.finditer(text))
    if fences:
        if len(fences) != 1 or _JSON_FENCE.sub("", text).strip():
            raise ExecutionCompilerError(
                "Gemma compiler output must contain exactly one JSON object"
            )
        text = fences[0].group("body").strip()

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ExecutionCompilerError(
                    f"Gemma compiler output repeats field {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ExecutionCompilerError(
                    f"Gemma compiler output contains invalid constant {value}"
                )
            ),
        )
    except ExecutionCompilerError:
        raise
    except json.JSONDecodeError as error:
        raise ExecutionCompilerError(
            f"Gemma compiler output is not strict JSON: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise ExecutionCompilerError("Gemma compiler output must be a JSON object")
    return payload


SYSTEM_PROMPT = (
    "You compile one tabletop robot instruction into a constrained symbolic "
    "pick-and-place program.\n"
    "Return one strict JSON object only. Never return actuator values, joint "
    "angles, pixels, bounding boxes, or guessed world coordinates. Preserve "
    "the object descriptions from the instruction as concise open-vocabulary "
    "detector queries. The host will ground both queries with synchronized "
    "RGB-D perception.\n\n"
    "Only the `pick_place` skill is allowed. The source anchor is `top_center`. "
    "Use target anchor `surface_center` for a mat/table/area and `top_center` "
    "when placing on another object. The relation is always `on_top`, "
    "orientation is always `tool_down`, and the six waypoint kinds and "
    "gripper commands must be copied in exactly this order. Approach, lift, "
    "and retreat clearances must be in [0.05, 0.20] metres; contact stages "
    "must use zero. Use 0.10 unless the instruction clearly requires a safer "
    "value.\n\nOutput shape:\n"
    '{"schema_version":1,"skill":"pick_place","source":{"query":"red block",'
    '"anchor":"top_center"},"target":{"query":"white mat","anchor":'
    '"surface_center"},"relation":"on_top","orientation":"tool_down",'
    '"waypoints":[{"kind":"approach_source","clearance_m":0.10,"gripper":'
    '"open"},{"kind":"grasp_source","clearance_m":0.0,"gripper":"close"},'
    '{"kind":"lift","clearance_m":0.10,"gripper":"hold"},{"kind":'
    '"approach_target","clearance_m":0.10,"gripper":"hold"},{"kind":'
    '"place_target","clearance_m":0.0,"gripper":"open"},{"kind":"retreat",'
    '"clearance_m":0.10,"gripper":"hold"}]}\n'
)


class GemmaExecutionCompiler:
    """Compile a published natural-language task with one correction retry."""

    def __init__(
        self,
        *,
        model: ExecutionModel | None = None,
        model_name: str = DEFAULT_EXECUTION_MODEL,
        model_base_url: str = DEFAULT_EXECUTION_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        if model is not None:
            self.model = model
        else:
            self.model = ChatOpenAI(
                model=model_name,
                api_key="EMPTY",
                base_url=model_base_url,
                max_tokens=1024,
                temperature=0,
                timeout=timeout,
            )

    def compile(self, task: PublishedTask) -> ManipulationProgram:
        if not isinstance(task, PublishedTask):
            raise TypeError("task must be a PublishedTask")
        if task.phase is not TaskPhase.STEP:
            raise ValueError("only STEP publications can be compiled for motion")
        request: dict[str, Any] = {
            "instruction": task.instruction,
            "expected_observation": [
                item.description for item in task.expected_observation
            ],
        }
        last_error: Exception | None = None
        invalid_output = ""
        for attempt in range(2):
            payload: dict[str, Any] = {"request": request}
            if attempt:
                payload["schema_correction"] = {
                    "error": str(last_error),
                    "invalid_output": invalid_output,
                    "instruction": (
                        "Return one corrected strict JSON object using the exact "
                        "six-stage schema. Do not change the task meaning."
                    ),
                }
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
            ]
            response = self.model.invoke(messages, **THINKING_DISABLED_OPTIONS)
            try:
                return ManipulationProgram.from_dict(parse_execution_json(response))
            except (TypeError, ValueError) as error:
                last_error = error
                try:
                    invalid_output = _visible_text(response)
                except ExecutionCompilerError:
                    invalid_output = "<no visible output>"
        raise ExecutionCompilerError(
            "Gemma returned an invalid manipulation program after one correction: "
            f"{last_error}"
        ) from last_error


__all__ = [
    "ExecutionCompilerError",
    "GemmaExecutionCompiler",
    "parse_execution_json",
]
