from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from agents.configs import AgentModelConfig, Planner_Agent_Config, VisionConfig
from agents.contracts import PlanResult, ValidationSpec
from agents.diagnostics import AgentOutputDisplay, DisplayingJsonModel
from agents.model import JsonModel, OllamaJsonModel
from agents.vision import prepare_vision_image


class PlannerAgentError(RuntimeError):
    pass


class PlannerAgent:
    def __init__(
        self,
        config: AgentModelConfig,
        *,
        vision: VisionConfig | None = None,
        model: JsonModel | None = None,
        output_display: AgentOutputDisplay | None = None,
    ):
        self.config = config
        self.vision = vision or VisionConfig()
        base_model = model or OllamaJsonModel(
            config.model,
            config.temperature,
            host=config.host,
            timeout_seconds=config.timeout_seconds,
        )
        self.model = (
            DisplayingJsonModel(base_model, output_display, "Planner Agent")
            if output_display is not None
            else base_model
        )
        self.system_prompt = Path(config.system_prompt_path).read_text(encoding="utf-8")

    def plan(
        self,
        task_contract: dict[str, Any] | str,
        image_path: str,
        *,
        recovery_context: dict[str, Any] | None = None,
    ) -> PlanResult:
        if isinstance(task_contract, str):
            contract = {"confirmed_intent": task_contract}
        elif isinstance(task_contract, dict):
            contract = copy.deepcopy(task_contract)
        else:
            raise PlannerAgentError("Task contract must be an object or string.")
        confirmed_intent = str(contract.get("confirmed_intent", "")).strip()
        if not confirmed_intent:
            raise PlannerAgentError("Task contract requires confirmed_intent.")
        image = prepare_vision_image(
            image_path,
            resize=self.vision.resize_images,
            width=self.vision.image_width,
            height=self.vision.image_height,
            jpeg_quality=self.vision.image_jpeg_quality,
        )
        raw = self.model.generate(
            purpose="plan_task",
            system_prompt=self.system_prompt,
            payload={
                "task_contract": contract,
                "recovery_context": recovery_context,
            },
            images=[image],
        )
        status = str(raw.get("planning_status", "UNKNOWN")).upper()
        subtasks = raw.get("subtasks", [])
        if not isinstance(subtasks, list):
            raise PlannerAgentError("Planner subtasks must be a list.")
        if status == "READY" and any(
            not isinstance(subtask, dict)
            or not str(subtask.get("task_instruction", "")).strip()
            for subtask in subtasks
        ):
            raise PlannerAgentError(
                "Every READY subtask must be an object with a non-empty task_instruction."
            )
        preconditions = raw.get("preconditions", [])
        if not isinstance(preconditions, list):
            raise PlannerAgentError("Planner preconditions must be a list.")
        validation_spec = None
        if status in {"READY", "ALREADY_SATISFIED"}:
            validation_spec = ValidationSpec.from_plan(raw, confirmed_intent)
            raw["validation_spec"] = validation_spec.to_dict()
        return PlanResult(
            status=status,
            subtasks=copy.deepcopy(subtasks),
            validation_spec=validation_spec,
            preconditions=copy.deepcopy(preconditions),
            failure=copy.deepcopy(raw.get("failure")),
            confidence=self._confidence(raw.get("planner_confidence")),
            raw=copy.deepcopy(raw),
        )

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0


Planner_Agent = PlannerAgent

__all__ = ["PlannerAgent", "Planner_Agent", "Planner_Agent_Config", "PlannerAgentError"]
