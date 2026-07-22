from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from colorama import Fore, Style
from ollama import chat

from agents.configs import Memory_Agent_Config



class MemoryAgentError(ValueError):
    pass


class Memory_Agent:
    """Extract preference evidence without owning persistence policy."""

    def __init__(self, config: Memory_Agent_Config):
        self.config = config
        self.host = config.host
        self.base_url = config.base_url
        self.model = config.model
        with open(config.system_prompt_path, "r", encoding="utf-8") as handle:
            self.system_prompt = handle.read()

    def propose_updates(
        self,
        interaction: list[dict[str, str]],
        existing_preferences: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        payload = {
            "interaction": interaction,
            "existing_preferences": existing_preferences,
        }
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]
        response = chat(
            model=self.model,
            messages=messages,
            think=False,
            stream=False,
            format="json",
            options={"temperature": 0.0},
        )

        content = response.message.content
        parsed = self._parse_output(content)
        print(Fore.BLUE + "\nMemory Curator:\n" + json.dumps(parsed, indent=2) + Style.RESET_ALL)
        return parsed["operations"]

    @staticmethod
    def _parse_output(content: str) -> dict[str, Any]:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            raise MemoryAgentError("Memory curator output did not contain a JSON object.")
        try:
            parsed = json.loads(content[start:end + 1])
        except json.JSONDecodeError as error:
            raise MemoryAgentError("Memory curator output was not valid JSON.") from error
        if not isinstance(parsed, dict) or not isinstance(parsed.get("operations"), list):
            raise MemoryAgentError("Memory curator output requires an operations list.")
        return parsed