"""Strict declarative experiment profiles and executor safety policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


LOGICAL_ROLES = ("hri", "memory", "planner", "monitor", "validator")
PROFILE_IDS = frozenset(
    {
        "T5",
        "T4",
        "T3",
        "T2",
        "T1",
        "A-Memory",
        "A-HRI",
        "A-OpenLoop",
        "A-Monitor",
        "A-Validator",
        "A-Confirmation",
        "A-SingleEvidence",
        "A-DynamicGuard",
        "A-HostFence",
        "B1",
        "B2",
        "B3",
        "B4",
    }
)
PROFILE_KINDS = frozenset({"topology", "ablation", "baseline"})
EXECUTORS = frozenset({"synthetic_event", "mujoco", "human_executor", "real_robot"})
SAFETY_TIERS = frozenset({"synthetic_only", "simulation_only", "safety_screened"})
MECHANISM_KEYS = frozenset(
    {
        "memory",
        "hri",
        "planning",
        "monitor",
        "validator",
        "confirmation",
        "evidence",
        "dynamic_guard",
        "host_fence",
        "personalization",
    }
)

_TIER_EXECUTOR_CEILINGS = {
    "synthetic_only": frozenset({"synthetic_event"}),
    "simulation_only": frozenset({"synthetic_event", "mujoco"}),
    "safety_screened": EXECUTORS,
}


class ProfileSchemaError(ValueError):
    """Raised when the frozen profile manifest violates its schema."""


class UnsafeExecutorError(RuntimeError):
    """Raised before a profile can reach a forbidden execution adapter."""


def _strict_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProfileSchemaError(
            f"{context}: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class ExperimentProfile:
    profile_id: str
    label: str
    kind: str
    base_profile: str | None
    logical_to_instance: Mapping[str, str | None]
    unique_agent_count: int
    prompt_refs: Mapping[str, str | None]
    schema_refs: Mapping[str, str | None]
    mechanism_modes: Mapping[str, str]
    safety_tier: str
    allowed_executors: frozenset[str]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentProfile":
        _strict_keys(
            value,
            {
                "id",
                "label",
                "kind",
                "base_profile",
                "logical_to_instance",
                "unique_agent_count",
                "prompt_refs",
                "schema_refs",
                "mechanism_modes",
                "safety_tier",
                "allowed_executors",
            },
            context="profile",
        )
        profile_id = value["id"]
        if profile_id not in PROFILE_IDS:
            raise ProfileSchemaError(f"unknown profile ID: {profile_id!r}")
        if not isinstance(value["label"], str) or not value["label"]:
            raise ProfileSchemaError(f"{profile_id}: label must be non-empty")
        if value["kind"] not in PROFILE_KINDS:
            raise ProfileSchemaError(f"{profile_id}: invalid kind")
        base_profile = value["base_profile"]
        if base_profile is not None and base_profile not in PROFILE_IDS:
            raise ProfileSchemaError(f"{profile_id}: invalid base_profile")

        role_maps: dict[str, dict[str, str | None]] = {}
        for map_name in ("logical_to_instance", "prompt_refs", "schema_refs"):
            mapping = value[map_name]
            if not isinstance(mapping, Mapping):
                raise ProfileSchemaError(f"{profile_id}.{map_name} must be an object")
            _strict_keys(mapping, set(LOGICAL_ROLES), context=f"{profile_id}.{map_name}")
            normalized: dict[str, str | None] = {}
            for role, item in mapping.items():
                if item is not None and (not isinstance(item, str) or not item):
                    raise ProfileSchemaError(
                        f"{profile_id}.{map_name}.{role} must be non-empty or null"
                    )
                normalized[role] = item
            role_maps[map_name] = normalized

        instances = {
            instance
            for instance in role_maps["logical_to_instance"].values()
            if instance is not None
        }
        count = value["unique_agent_count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ProfileSchemaError(f"{profile_id}: unique_agent_count must be >= 0")
        if count != len(instances):
            raise ProfileSchemaError(
                f"{profile_id}: declared {count} agents but mapping has {len(instances)}"
            )
        for role, instance in role_maps["logical_to_instance"].items():
            for ref_name in ("prompt_refs", "schema_refs"):
                ref = role_maps[ref_name][role]
                if instance is None and ref is not None:
                    raise ProfileSchemaError(
                        f"{profile_id}: inactive role {role} has a {ref_name} reference"
                    )
                if instance is not None and ref is None:
                    raise ProfileSchemaError(
                        f"{profile_id}: active role {role} lacks a {ref_name} reference"
                    )

        mechanism_modes = value["mechanism_modes"]
        if not isinstance(mechanism_modes, Mapping):
            raise ProfileSchemaError(f"{profile_id}.mechanism_modes must be an object")
        _strict_keys(
            mechanism_modes, set(MECHANISM_KEYS), context=f"{profile_id}.mechanism_modes"
        )
        if not all(isinstance(mode, str) and mode for mode in mechanism_modes.values()):
            raise ProfileSchemaError(f"{profile_id}: mechanism modes must be strings")

        tier = value["safety_tier"]
        if tier not in SAFETY_TIERS:
            raise ProfileSchemaError(f"{profile_id}: invalid safety_tier")
        executors = value["allowed_executors"]
        if not isinstance(executors, list) or not executors:
            raise ProfileSchemaError(f"{profile_id}: allowed_executors must be non-empty")
        if len(executors) != len(set(executors)) or not set(executors) <= EXECUTORS:
            raise ProfileSchemaError(f"{profile_id}: invalid or duplicate executor")
        executor_set = frozenset(executors)
        if not executor_set <= _TIER_EXECUTOR_CEILINGS[tier]:
            raise ProfileSchemaError(
                f"{profile_id}: executor exceeds safety tier {tier}"
            )
        if profile_id == "A-HostFence" and executor_set != {"synthetic_event"}:
            raise ProfileSchemaError("A-HostFence must be synthetic-event only")
        simulation_only = {
            "A-HRI",
            "A-Monitor",
            "A-Validator",
            "A-Confirmation",
            "A-SingleEvidence",
            "A-DynamicGuard",
        }
        if profile_id in simulation_only and not executor_set <= {
            "synthetic_event",
            "mujoco",
        }:
            raise ProfileSchemaError(f"{profile_id} is simulation-only")

        return cls(
            profile_id=profile_id,
            label=value["label"],
            kind=value["kind"],
            base_profile=base_profile,
            logical_to_instance=role_maps["logical_to_instance"],
            unique_agent_count=count,
            prompt_refs=role_maps["prompt_refs"],
            schema_refs=role_maps["schema_refs"],
            mechanism_modes=dict(mechanism_modes),
            safety_tier=tier,
            allowed_executors=executor_set,
        )


@dataclass(frozen=True)
class ProfileCatalogue:
    profiles: Mapping[str, ExperimentProfile]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if set(self.profiles) != PROFILE_IDS:
            raise ProfileSchemaError(
                f"profile catalogue mismatch: missing={sorted(PROFILE_IDS - set(self.profiles))}, "
                f"unknown={sorted(set(self.profiles) - PROFILE_IDS)}"
            )

    def get(self, profile_id: str) -> ExperimentProfile:
        try:
            return self.profiles[profile_id]
        except KeyError as exc:
            raise ProfileSchemaError(f"unknown profile ID: {profile_id}") from exc


def load_profiles(path: str | Path | None = None) -> ProfileCatalogue:
    """Load and validate the complete frozen profile manifest."""

    manifest_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / "config" / "profiles.json"
    )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ProfileSchemaError("profile manifest must be an object")
    _strict_keys(raw, {"schema_version", "logical_roles", "profiles"}, context="manifest")
    if raw["schema_version"] != 1:
        raise ProfileSchemaError("unsupported profile schema version")
    if raw["logical_roles"] != list(LOGICAL_ROLES):
        raise ProfileSchemaError("logical_roles must match the frozen role order")
    if not isinstance(raw["profiles"], list):
        raise ProfileSchemaError("profiles must be an array")
    profiles = [ExperimentProfile.from_dict(value) for value in raw["profiles"]]
    by_id = {profile.profile_id: profile for profile in profiles}
    if len(by_id) != len(profiles):
        raise ProfileSchemaError("duplicate profile ID")
    return ProfileCatalogue(profiles=by_id)


def assert_executor_allowed(
    profile: ExperimentProfile,
    executor: str,
    *,
    scenario_allowed: frozenset[str] | None = None,
    safety_screened: bool = False,
    ethics_approved: bool = False,
) -> None:
    """Fail closed before constructing an unsafe profile/executor combination."""

    if executor not in EXECUTORS:
        raise UnsafeExecutorError(f"unknown executor: {executor}")
    if executor not in profile.allowed_executors:
        raise UnsafeExecutorError(
            f"profile {profile.profile_id} forbids executor {executor}; "
            f"allowed={sorted(profile.allowed_executors)}"
        )
    if scenario_allowed is not None and executor not in scenario_allowed:
        raise UnsafeExecutorError(
            f"scenario safety policy forbids executor {executor}; "
            f"allowed={sorted(scenario_allowed)}"
        )
    if executor == "real_robot" and not safety_screened:
        raise UnsafeExecutorError(
            "real_robot requires an explicit, campaign-recorded safety screen"
        )
    if executor == "human_executor" and not ethics_approved:
        raise UnsafeExecutorError(
            "human_executor requires explicit ethics approval for the campaign"
        )
