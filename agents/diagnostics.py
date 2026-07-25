from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from agents.model import JsonModel


_HIDDEN_REASONING_KEYS = {
    "chain_of_thought",
    "reasoning",
    "thinking",
}


class AgentOutputDisplay:
    """Opt-in terminal display for agent outputs; never used as agent context."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.enabled = enabled
        self.observer = observer

    def emit(self, agent: str, stage: str, output: Any) -> None:
        safe_output = self._safe_value(output)
        if self.observer is not None:
            try:
                self.observer(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "agent": str(agent),
                        "stage": str(stage),
                        "output": safe_output,
                    }
                )
            except Exception:
                # Diagnostics must never alter orchestration behavior.
                pass
        if not self.enabled:
            return
        print(f"\n[Display All] {agent} — {stage}")
        print(
            json.dumps(
                safe_output,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    @classmethod
    def _safe_value(cls, value: Any) -> Any:
        if isinstance(value, bytes):
            return {
                "byte_count": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
                "raw_bytes_omitted": True,
            }
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value) and not isinstance(value, type):
            return cls._safe_value(asdict(value))
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, item in value.items():
                name = str(key)
                if name.lower() in _HIDDEN_REASONING_KEYS:
                    cleaned[name] = "[omitted]"
                else:
                    cleaned[name] = cls._safe_value(item)
            return cleaned
        if isinstance(value, (list, tuple)):
            return [cls._safe_value(item) for item in value]
        return copy.deepcopy(value)


class DisplayingJsonModel:
    """JsonModel decorator that exposes each raw agent response when enabled."""

    def __init__(
        self,
        model: JsonModel,
        display: AgentOutputDisplay,
        agent_name: str,
    ) -> None:
        self.model = model
        self.display = display
        self.agent_name = agent_name

    def generate(
        self,
        *,
        purpose: str,
        system_prompt: str,
        payload: dict[str, Any],
        images: Sequence[bytes] = (),
    ) -> dict[str, Any]:
        try:
            output = self.model.generate(
                purpose=purpose,
                system_prompt=system_prompt,
                payload=payload,
                images=images,
            )
        except Exception as error:
            self.display.emit(
                self.agent_name,
                f"{purpose} error",
                {"error_type": type(error).__name__, "error": str(error)},
            )
            raise
        self.display.emit(self.agent_name, f"{purpose} raw output", output)
        return output


__all__ = ["AgentOutputDisplay", "DisplayingJsonModel"]
