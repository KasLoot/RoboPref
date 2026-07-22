from __future__ import annotations

import json
from typing import Any, Protocol, Sequence


class JsonModelError(RuntimeError):
    pass


class JsonModel(Protocol):
    def generate(
        self,
        *,
        purpose: str,
        system_prompt: str,
        payload: dict[str, Any],
        images: Sequence[bytes] = (),
    ) -> dict[str, Any]: ...


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise JsonModelError("Model output must be a JSON object or string.")
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise JsonModelError("Model output did not contain a JSON object.")
    try:
        parsed = json.loads(value[start : end + 1])
    except json.JSONDecodeError as error:
        raise JsonModelError("Model output was not valid JSON.") from error
    if not isinstance(parsed, dict):
        raise JsonModelError("Model output JSON must be an object.")
    return parsed


class OllamaJsonModel:
    """The only production adapter that calls Ollama directly."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        *,
        host: str | None = None,
        timeout_seconds: float = 120.0,
    ):
        self.model = model
        self.temperature = temperature
        self.host = host
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        *,
        purpose: str,
        system_prompt: str,
        payload: dict[str, Any],
        images: Sequence[bytes] = (),
    ) -> dict[str, Any]:
        from ollama import Client

        user_message: dict[str, Any] = {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        }
        if images:
            user_message["images"] = list(images)
        client = Client(host=self.host, timeout=self.timeout_seconds)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            think=False,
            stream=False,
            format="json",
            options={"temperature": self.temperature},
        )
        try:
            content = response.message.content
        except AttributeError as error:
            raise JsonModelError(f"Malformed Ollama response for {purpose}.") from error
        return parse_json_object(content)
