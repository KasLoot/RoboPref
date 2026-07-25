from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from agents.configs import AgentModelConfig


VLLM_DEFAULT_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY_ENV = "VLLM_API_KEY"


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
    """Production JSON adapter for Ollama."""

    provider = "ollama"

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
            "provider": self.provider,
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


class VLLMJsonModel(OllamaJsonModel):
    """JSON adapter for a vLLM OpenAI-compatible Chat Completions server."""

    provider = "vllm"

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        *,
        base_url: str = VLLM_DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        timeout_seconds: float = 120.0,
        seed: int | None = None,
        agent_name: str | None = None,
        telemetry_observer: Callable[[dict[str, Any]], None] | None = None,
    ):
        super().__init__(
            model,
            temperature,
            timeout_seconds=timeout_seconds,
            seed=seed,
            agent_name=agent_name,
            telemetry_observer=telemetry_observer,
        )
        from agents.configs import normalize_model_base_url
        from openai import OpenAI

        self.base_url = normalize_model_base_url(base_url)
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=self.timeout_seconds,
            max_retries=0,
        )

    def generate(
        self,
        *,
        purpose: str,
        system_prompt: str,
        payload: dict[str, Any],
        images: Sequence[bytes] = (),
    ) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        call_id = f"model-call-{uuid.uuid4().hex}"
        try:
            request: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": self._user_content(payload, images),
                    },
                ],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            }
            if self.seed is not None:
                request["seed"] = self.seed
            response = self._client.chat.completions.create(**request)
            try:
                choice = response.choices[0]
                content = choice.message.content
            except (AttributeError, IndexError, TypeError) as error:
                raise JsonModelError(
                    f"Malformed vLLM response for {purpose}."
                ) from error
            if getattr(choice, "finish_reason", None) == "length":
                raise JsonModelError(
                    f"vLLM response for {purpose} was truncated at its token limit."
                )
            if not isinstance(content, str) or not content.strip():
                raise JsonModelError(
                    f"vLLM returned empty content for {purpose}."
                )
            parsed = parse_json_object(content)
        except Exception as error:
            if isinstance(error, JsonModelError):
                safe_error = error
            else:
                action = (
                    "timed out"
                    if "timeout" in type(error).__name__.casefold()
                    else "failed"
                )
                safe_error = JsonModelError(
                    f"vLLM request {action} for {purpose} "
                    f"({type(error).__name__})."
                )
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
                    "error": str(safe_error),
                }
            )
            if safe_error is error:
                raise
            # OpenAI-compatible servers may echo request content in error bodies.
            # Suppress the original exception context so downstream tracebacks and
            # durable benchmark artifacts contain only this sanitized message.
            raise safe_error from None
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

    @classmethod
    def _user_content(
        cls,
        payload: dict[str, Any],
        images: Sequence[bytes],
    ) -> str | list[dict[str, Any]]:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if not images:
            return text
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": cls._image_data_url(bytes(image))},
            }
            for image in images
        )
        return content

    @staticmethod
    def _image_data_url(image: bytes) -> str:
        if image.startswith(b"\x89PNG\r\n\x1a\n"):
            media_type = "image/png"
        elif image.startswith(b"\xff\xd8\xff"):
            media_type = "image/jpeg"
        elif image.startswith((b"GIF87a", b"GIF89a")):
            media_type = "image/gif"
        elif image.startswith(b"RIFF") and image[8:12] == b"WEBP":
            media_type = "image/webp"
        elif image.startswith(b"BM"):
            media_type = "image/bmp"
        elif image.startswith((b"II*\x00", b"MM\x00*")):
            media_type = "image/tiff"
        else:
            raise JsonModelError(
                "vLLM image input must be PNG, JPEG, GIF, WebP, BMP, or TIFF."
            )
        encoded = base64.b64encode(image).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    @staticmethod
    def _response_metadata(response: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for source_name, output_name in (
            ("id", "id"),
            ("created", "created"),
            ("model", "response_model"),
            ("system_fingerprint", "system_fingerprint"),
            ("service_tier", "service_tier"),
            ("_request_id", "request_id"),
        ):
            value = getattr(response, source_name, None)
            if isinstance(value, (str, int, float, bool)):
                metadata[output_name] = value
        try:
            finish_reason = response.choices[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            finish_reason = None
        if isinstance(finish_reason, str):
            metadata["finish_reason"] = finish_reason
        usage = getattr(response, "usage", None)
        if usage is not None:
            token_usage: dict[str, int] = {}
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = (
                    usage.get(name)
                    if isinstance(usage, Mapping)
                    else getattr(usage, name, None)
                )
                if isinstance(value, int) and not isinstance(value, bool):
                    token_usage[name] = value
            if token_usage:
                metadata["usage"] = token_usage
        return metadata


def effective_model_base_url(config: AgentModelConfig) -> str | None:
    """Return the sanitized endpoint that the configured adapter will use."""

    from agents.configs import normalize_model_base_url

    provider = config.provider.strip().casefold()
    if provider == "ollama":
        if config.base_url is not None:
            raise ValueError("Agent model base URL requires provider='vllm'.")
        return None
    if provider == "vllm":
        if config.host is not None:
            raise ValueError("Agent Ollama host requires provider='ollama'.")
        return normalize_model_base_url(config.base_url or VLLM_DEFAULT_BASE_URL)
    raise ValueError(
        f"Unsupported model provider {config.provider!r}; expected 'ollama' or 'vllm'."
    )


def build_json_model(
    config: AgentModelConfig,
    *,
    agent_name: str | None = None,
    telemetry_observer: Callable[[dict[str, Any]], None] | None = None,
) -> JsonModel:
    """Construct the configured production model adapter."""

    provider = config.provider.strip().casefold()
    base_url = effective_model_base_url(config)
    common: dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "timeout_seconds": config.timeout_seconds,
        "seed": config.seed,
        "agent_name": agent_name,
        "telemetry_observer": telemetry_observer,
    }
    if provider == "ollama":
        return OllamaJsonModel(host=config.host, **common)
    if provider == "vllm":
        api_key = os.environ.get(VLLM_API_KEY_ENV, "").strip() or "EMPTY"
        return VLLMJsonModel(
            base_url=base_url or VLLM_DEFAULT_BASE_URL,
            api_key=api_key,
            **common,
        )
    raise ValueError(
        f"Unsupported model provider {config.provider!r}; expected 'ollama' or 'vllm'."
    )


__all__ = [
    "JsonModel",
    "JsonModelError",
    "OllamaJsonModel",
    "VLLMJsonModel",
    "VLLM_API_KEY_ENV",
    "VLLM_DEFAULT_BASE_URL",
    "build_json_model",
    "effective_model_base_url",
    "parse_json_object",
]
