from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from agents.configs import AgentModelConfig, Validator_Agent_Config, VisionConfig
from agents.contracts import ExecutionResult, ValidationResult, ValidationSpec
from agents.diagnostics import AgentOutputDisplay, DisplayingJsonModel
from agents.model import JsonModel, OllamaJsonModel
from agents.vision import prepare_vision_image


class ValidatorAgentError(RuntimeError):
    pass


class ValidatorAgent:
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
            DisplayingJsonModel(base_model, output_display, "Validator Agent")
            if output_display is not None
            else base_model
        )
        self.system_prompt = Path(config.system_prompt_path).read_text(encoding="utf-8")

    def validate(
        self,
        validation_spec: ValidationSpec,
        execution: ExecutionResult,
    ) -> ValidationResult:
        image = prepare_vision_image(
            execution.final_observation,
            resize=self.vision.resize_images,
            width=self.vision.image_width,
            height=self.vision.image_height,
            jpeg_quality=self.vision.image_jpeg_quality,
        )
        spec_payload = validation_spec.to_dict()
        raw = self.model.generate(
            purpose="validate_task",
            system_prompt=self.system_prompt,
            payload={
                "validation_spec": spec_payload,
                "execution_evidence": execution.to_dict(),
            },
            images=[image],
        )
        returned_spec_id = str(raw.get("spec_id", ""))
        if returned_spec_id != validation_spec.spec_id:
            raise ValidatorAgentError(
                f"Validator spec mismatch: expected {validation_spec.spec_id}, got {returned_spec_id or '<missing>'}."
            )
        checks = raw.get("goal_checks", [])
        if not isinstance(checks, list):
            raise ValidatorAgentError("Validator goal_checks must be a list.")
        expected_ids = {condition.id for condition in validation_spec.goal_conditions}
        returned_ids = {
            str(check.get("goal_id"))
            for check in checks
            if isinstance(check, dict) and check.get("goal_id") is not None
        }
        if returned_ids != expected_ids or len(checks) != len(expected_ids):
            missing = sorted(expected_ids - returned_ids)
            extra = sorted(returned_ids - expected_ids)
            raise ValidatorAgentError(
                f"Validator must check every frozen goal exactly once; missing={missing}, extra={extra}."
            )
        outcome = str(raw.get("outcome", "UNKNOWN")).upper()
        if outcome not in {"SUCCESS", "PARTIAL", "FAILURE", "UNKNOWN", "UNSAFE"}:
            raise ValidatorAgentError(f"Unsupported validator outcome: {outcome}")
        task_complete = raw.get("task_complete") is True
        required_ids = {
            condition.id for condition in validation_spec.goal_conditions if condition.required
        }
        checks_by_id = {str(check.get("goal_id")): check for check in checks}
        if outcome == "SUCCESS" and (
            not task_complete
            or any(checks_by_id[goal_id].get("satisfied") is not True for goal_id in required_ids)
        ):
            raise ValidatorAgentError("SUCCESS requires task_complete and every goal satisfied.")
        if outcome != "SUCCESS" and task_complete:
            raise ValidatorAgentError("Only SUCCESS may set task_complete true.")
        discrepancies = raw.get("discrepancies", [])
        if not isinstance(discrepancies, list):
            raise ValidatorAgentError("Validator discrepancies must be a list.")
        failure = copy.deepcopy(raw.get("failure"))
        user_message = str(
            raw.get("user_message")
            or (failure or {}).get("user_message")
            or ("Task complete." if task_complete else "The task outcome could not be confirmed.")
        )
        recoverability = str(
            raw.get("recoverability") or (failure or {}).get("recoverability") or "REOBSERVE"
        ).upper()
        if recoverability not in {
            "NONE",
            "REOBSERVE",
            "AUTO_LOCAL",
            "REPLAN",
            "USER_ASSIST",
            "ABORT_SAFETY",
        }:
            raise ValidatorAgentError(
                f"Unsupported validator recoverability: {recoverability}"
            )
        return ValidationResult(
            outcome=outcome,
            task_complete=task_complete,
            goal_checks=copy.deepcopy(checks),
            discrepancies=list(map(str, discrepancies)),
            confidence=self._confidence(raw.get("validator_confidence")),
            recoverability=recoverability,
            user_message=user_message,
            failure=failure,
            raw=copy.deepcopy(raw),
        )

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return 0.0


Validator_Agent = ValidatorAgent

__all__ = [
    "ValidatorAgent",
    "Validator_Agent",
    "Validator_Agent_Config",
    "ValidatorAgentError",
]
