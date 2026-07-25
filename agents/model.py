from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence


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
        seed: int | None = None,
        agent_name: str | None = None,
        telemetry_observer: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.host = host
        self.timeout_seconds = timeout_seconds
        self.seed = seed
        self.agent_name = agent_name
        self.telemetry_observer = telemetry_observer

    def generate(
        self,
        *,
        purpose: str,
        system_prompt: str,
        payload: dict[str, Any],
        images: Sequence[bytes] = (),
    ) -> dict[str, Any]:
        from ollama import Client

        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        call_id = f"model-call-{uuid.uuid4().hex}"
        user_message: dict[str, Any] = {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        }
        if images:
            user_message["images"] = list(images)
        client = Client(host=self.host, timeout=self.timeout_seconds)
        try:
            options: dict[str, Any] = {"temperature": self.temperature}
            if self.seed is not None:
                options["seed"] = self.seed
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    user_message,
                ],
                think=False,
                stream=False,
                format="json",
                options=options,
            )
            try:
                content = response.message.content
            except AttributeError as error:
                raise JsonModelError(
                    f"Malformed Ollama response for {purpose}."
                ) from error
            parsed = parse_json_object(content)
        except Exception as error:
            self._emit_telemetry(
                {
                    **self._telemetry_base(
                        call_id=call_id,
                        purpose=purpose,
                        system_prompt=system_prompt,
                        payload=payload,
                        images=images,
                        started_at=started_at,
                        duration_seconds=time.perf_counter() - started,
                    ),
                    "success": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise
        self._emit_telemetry(
            {
                **self._telemetry_base(
                    call_id=call_id,
                    purpose=purpose,
                    system_prompt=system_prompt,
                    payload=payload,
                    images=images,
                    started_at=started_at,
                    duration_seconds=time.perf_counter() - started,
                ),
                "success": True,
                "response_sha256": self._digest(parsed),
                "response_metadata": self._response_metadata(response),
            }
        )
        return parsed

    def _emit_telemetry(self, record: dict[str, Any]) -> None:
        if self.telemetry_observer is None:
            return
        try:
            self.telemetry_observer(record)
        except Exception:
            # Evaluation logging must never alter the agent decision path.
            return

    def _telemetry_base(
        self,
        *,
        call_id: str,
        purpose: str,
        system_prompt: str,
        payload: dict[str, Any],
        images: Sequence[bytes],
        started_at: str,
        duration_seconds: float,
    ) -> dict[str, Any]:
        return {
            "call_id": call_id,
            "agent": self.agent_name,
            "purpose": purpose,
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "started_at": started_at,
            "duration_seconds": duration_seconds,
            "system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "payload_sha256": self._digest(payload),
            "image_count": len(images),
            "image_sha256": [
                hashlib.sha256(bytes(image)).hexdigest() for image in images
            ],
        }

    @staticmethod
    def _digest(value: Any) -> str:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _response_metadata(response: Any) -> dict[str, Any]:
        names = (
            "created_at",
            "done",
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )
        metadata: dict[str, Any] = {}
        for name in names:
            value = (
                response.get(name)
                if isinstance(response, Mapping)
                else getattr(response, name, None)
            )
            if value is not None:
                metadata[name] = (
                    value
                    if isinstance(value, (str, int, float, bool))
                    else str(value)
                )
        return metadata
