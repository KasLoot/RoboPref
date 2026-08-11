"""Deterministic construction preflight for every T/A/B system profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.harness.assembler import SystemAssembler
from experiments.harness.profiles import ExperimentProfile, load_profiles


class ProfilePreflightError(RuntimeError):
    """A declared profile did not instantiate or expose its mechanism modes."""


class _ProbeBackend:
    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id

    def invoke(self, payload: Any, **kwargs: Any) -> Mapping[str, Any]:
        return {
            "probe": True,
            "instance_id": self.instance_id,
            "payload_type": type(payload).__name__,
            "kwargs": sorted(kwargs),
        }

    def stream(self, payload: Any, **kwargs: Any):
        yield self.invoke(payload, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ProbeBackend":
        del tools, kwargs
        return self


@dataclass(frozen=True, slots=True)
class ProfilePreflightResult:
    profile_id: str
    executor: str
    passed: bool
    topology_manifest: Mapping[str, Any]
    mechanism_adapters: Mapping[str, str]
    role_probe_calls: int
    assertions: Mapping[str, bool]
    errors: tuple[str, ...] = ()


def _executor(profile: ExperimentProfile) -> str:
    return "mujoco" if "mujoco" in profile.allowed_executors else "synthetic_event"


def _adapter_names(variant: Any) -> dict[str, str]:
    return {
        name: type(getattr(variant, name)).__name__
        for name in (
            "memory",
            "interaction",
            "planning",
            "monitor",
            "validator",
            "confirmation",
            "evidence",
            "dynamic_guard",
            "host_fence",
        )
    } | {"personalization_enabled": str(variant.personalization_enabled)}


def preflight_profile(profile: ExperimentProfile) -> ProfilePreflightResult:
    executor = _executor(profile)
    errors: list[str] = []
    system = SystemAssembler(
        profile,
        executor=executor,
        backend_factory=lambda request: _ProbeBackend(request.model_instance_id),
        scenario_allowed_executors=frozenset({executor}),
    ).assemble()
    manifest = system.topology_manifest.to_dict()
    responses = []
    for role, routed in system.routed_models.items():
        if routed is None:
            continue
        response = routed.invoke(
            {"profile_id": profile.profile_id, "logical_role": role},
            _experiment_phase="PROFILE_PREFLIGHT",
        )
        responses.append((role, response))
        if response.get("instance_id") != profile.logical_to_instance[role]:
            errors.append(f"{role} routed to the wrong actual instance")

    actual_sessions = {
        role: (None if routed is None else routed.session_id)
        for role, routed in system.routed_models.items()
    }
    sharing_matches = True
    roles = tuple(profile.logical_to_instance)
    for left in roles:
        for right in roles:
            declared_same = (
                profile.logical_to_instance[left] is not None
                and profile.logical_to_instance[left]
                == profile.logical_to_instance[right]
            )
            actual_same = (
                actual_sessions[left] is not None
                and actual_sessions[left] == actual_sessions[right]
            )
            sharing_matches &= declared_same == actual_same
    if not sharing_matches:
        errors.append("actual session identity does not match the declared topology")

    variant = system.variant
    ready = variant.confirmation.authorize(
        preview_status="READY",
        staged_goal_id="goal",
        staged_revision=1,
        confirmed_goal_id="goal",
        confirmed_revision=1,
        user_confirmed=True,
        oracle_authorized=True,
    )
    confirmation_executable = bool(ready)
    if not confirmation_executable:
        errors.append("confirmation adapter rejected its own valid fixture")

    evidence_mode = variant.evidence.mode
    if evidence_mode == "temporally_repeated":
        evidence_executable = (
            not variant.evidence.terminal(
                "MET", consecutive_count=1, stable_seconds=3.0
            )
            and variant.evidence.terminal(
                "MET", consecutive_count=2, stable_seconds=2.0
            )
        )
    elif evidence_mode == "oracle":
        evidence_executable = variant.evidence.terminal(
            "UNKNOWN",
            consecutive_count=0,
            stable_seconds=0.0,
            oracle_terminal=True,
        )
    elif not variant.evidence.enabled:
        evidence_executable = not variant.evidence.terminal(
            "MET", consecutive_count=999, stable_seconds=999.0
        )
    else:
        evidence_executable = variant.evidence.terminal(
            "MET", consecutive_count=1, stable_seconds=0.0
        )
    if not evidence_executable:
        errors.append("evidence adapter did not implement its declared threshold mode")

    guard = variant.dynamic_guard.evaluate(
        confirmed_members=frozenset({"red"}),
        current_members=frozenset({"red", "green"}),
        motion_active=True,
        oracle_cancel=True,
    )
    dynamic_guard_executable = guard.mode == variant.dynamic_guard.mode
    if not dynamic_guard_executable:
        errors.append("dynamic guard returned a different mode")

    assertions = {
        "unique_instance_count_matches": (
            manifest["assembled_unique_agent_count"] == profile.unique_agent_count
        ),
        "session_sharing_matches_mapping": sharing_matches,
        "all_active_roles_invoked": len(responses)
        == sum(value is not None for value in profile.logical_to_instance.values()),
        "confirmation_adapter_executable": confirmation_executable,
        "evidence_adapter_executable": evidence_executable,
        "dynamic_guard_adapter_executable": dynamic_guard_executable,
        "manifest_hash_present": len(str(manifest.get("manifest_sha256", ""))) == 64,
    }
    errors.extend(name for name, passed in assertions.items() if not passed)
    return ProfilePreflightResult(
        profile_id=profile.profile_id,
        executor=executor,
        passed=not errors,
        topology_manifest=manifest,
        mechanism_adapters=_adapter_names(variant),
        role_probe_calls=len(responses),
        assertions=assertions,
        errors=tuple(errors),
    )


def run_profile_preflights() -> tuple[ProfilePreflightResult, ...]:
    catalogue = load_profiles()
    results = tuple(
        preflight_profile(catalogue.get(profile_id))
        for profile_id in sorted(catalogue.profiles)
    )
    if set(item.profile_id for item in results) != set(catalogue.profiles):
        raise ProfilePreflightError("profile preflight coverage is incomplete")
    return results


def write_profile_preflights(path: str | Path) -> str:
    results = run_profile_preflights()
    payload = {
        "schema_version": 1,
        "scope": "construction_and_mechanism_activation_no_remote_inference",
        "passed": all(item.passed for item in results),
        "profile_count": len(results),
        "profiles": [asdict(item) for item in results],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


__all__ = [
    "ProfilePreflightError",
    "ProfilePreflightResult",
    "preflight_profile",
    "run_profile_preflights",
    "write_profile_preflights",
]
