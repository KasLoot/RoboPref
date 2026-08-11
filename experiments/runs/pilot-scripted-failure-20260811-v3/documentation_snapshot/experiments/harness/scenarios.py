"""Strict schema and validation for the 62-template experiment catalogue."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .oracles import ORACLE_TYPES, OracleSchemaError, OracleSpec
from .profiles import EXECUTORS, PROFILE_IDS


FAMILY_COUNTS = {
    "intent_confirmation": 8,
    "preference_memory": 12,
    "nominal_manipulation": 10,
    "perturbation_recovery": 14,
    "evidence_validation": 8,
    "safety_fault_capability": 10,
}
SPLITS = ("dev", "pilot", "locked")
CATALOGUES = frozenset({"mechanism_isolation", "end_to_end"})
SUITES = frozenset({"layer_c_simulation", "layer_d_screened", "synthetic_invariant"})
DIFFICULTIES = frozenset({"easy", "moderate", "hard", "stress"})
RISK_LEVELS = frozenset({"low", "medium", "high"})
OUTCOMES = frozenset(
    {
        "physical_goal_completion",
        "contract_success",
        "perturbation_recovery",
        "preference_satisfaction",
        "false_completion",
        "safety_compliance",
        "capability_honesty",
    }
)
TRIGGER_KINDS = frozenset(
    {
        "scripted_input",
        "scene_change",
        "evidence_change",
        "service_fault",
        "external_emergency",
        "schedule",
    }
)
PREFLIGHT_CHECKS = frozenset(
    {
        "artifact_path",
        "service_health",
        "scene_load",
        "target_detection",
        "identity",
        "reachability",
        "collision_free",
        "trigger_hook",
        "executor_capability",
    }
)
FROZEN_MARKERS = frozenset({"topology24", "sensitivity20", "real12"})
ABLATION_MARKERS = frozenset(
    {
        "ablation:A-Memory",
        "ablation:A-HRI",
        "ablation:A-OpenLoop",
        "ablation:A-Monitor",
        "ablation:A-Validator",
        "ablation:A-Confirmation",
        "ablation:A-SingleEvidence",
        "ablation:A-DynamicGuard",
        "ablation:A-HostFence",
    }
)
TOPOLOGY24_IDS = frozenset(
    {
        "IC02", "IC03", "IC05", "IC06",
        "PM01", "PM02", "PM07", "PM11",
        "NM02", "NM04", "NM06", "NM08",
        "PR02", "PR03", "PR09", "PR11",
        "EV01", "EV03", "EV07", "EV08",
        "SF01", "SF04", "SF07", "SF10",
    }
)
SENSITIVITY20_IDS = frozenset(
    {
        "IC01", "IC03", "IC07",
        "PM02", "PM06", "PM10",
        "NM02", "NM06", "NM08", "NM10",
        "PR02", "PR04", "PR09", "PR12",
        "EV01", "EV06", "EV08",
        "SF01", "SF07", "SF10",
    }
)
REAL12_IDS = frozenset(
    {
        "IC01", "IC05", "PM02", "PM06",
        "NM01", "NM02", "NM03", "NM04",
        "PR01", "PR11", "EV01", "EV03",
    }
)
ABLATION_SUBSET_IDS = {
    "ablation:A-Memory": frozenset(
        {"IC02", *(f"PM{index:02d}" for index in range(1, 13))}
    ),
    "ablation:A-HRI": frozenset(
        {"IC02", "IC03", "IC04", "IC05", "IC07", "IC08", "SF08"}
    ),
    "ablation:A-OpenLoop": frozenset(
        {"NM05", "NM06", *(f"PR{index:02d}" for index in range(1, 15))}
    ),
    "ablation:A-Monitor": frozenset(
        {"PR02", "PR03", "PR09", "PR10", "PR11", "PR12", "EV01", "EV02"}
    ),
    "ablation:A-Validator": frozenset(
        {"NM01", "PR08", "PR09", "EV03", "EV04", "EV08"}
    ),
    "ablation:A-Confirmation": frozenset(
        {"IC04", "IC05", "IC06", "IC07", "PM06", "PM09"}
    ),
    "ablation:A-SingleEvidence": frozenset({"EV01", "EV02", "EV06", "EV07"}),
    "ablation:A-DynamicGuard": frozenset({"NM08", "PR04", "PR05"}),
    "ablation:A-HostFence": frozenset({"EV05"}),
}


class ScenarioSchemaError(ValueError):
    """Raised when the scenario catalogue is not exactly reproducible."""


def _strict_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ScenarioSchemaError(
            f"{context}: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScenarioSchemaError(f"{context} must be a non-empty string")
    return value


def _string_tuple(value: Any, *, context: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ScenarioSchemaError(f"{context} must be a string array")
    if not allow_empty and not value:
        raise ScenarioSchemaError(f"{context} must not be empty")
    if len(value) != len(set(value)):
        raise ScenarioSchemaError(f"{context} contains duplicates")
    return tuple(value)


def _positive_int(value: Any, *, context: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ScenarioSchemaError(f"{context} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class SplitVariant:
    variant_id: str
    semantic_family: str
    layout_family: str
    language_family: str
    preference_history: str
    distractor_family: str
    perturbation_boundary: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, context: str) -> "SplitVariant":
        _strict_keys(
            value,
            {
                "id",
                "semantic_family",
                "layout_family",
                "language_family",
                "preference_history",
                "distractor_family",
                "perturbation_boundary",
            },
            context=context,
        )
        return cls(
            variant_id=_string(value["id"], context=f"{context}.id"),
            semantic_family=_string(
                value["semantic_family"], context=f"{context}.semantic_family"
            ),
            layout_family=_string(value["layout_family"], context=f"{context}.layout_family"),
            language_family=_string(
                value["language_family"], context=f"{context}.language_family"
            ),
            preference_history=_string(
                value["preference_history"], context=f"{context}.preference_history"
            ),
            distractor_family=_string(
                value["distractor_family"], context=f"{context}.distractor_family"
            ),
            perturbation_boundary=_string(
                value["perturbation_boundary"], context=f"{context}.perturbation_boundary"
            ),
        )

    @property
    def separation_signature(self) -> tuple[str, ...]:
        return (
            self.semantic_family,
            self.layout_family,
            self.language_family,
            self.preference_history,
            self.distractor_family,
            self.perturbation_boundary,
        )


@dataclass(frozen=True)
class ScenarioSafety:
    risk_level: str
    allowed_executors: frozenset[str]
    real_system_eligible: bool
    hazards: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, context: str) -> "ScenarioSafety":
        _strict_keys(
            value,
            {"risk_level", "allowed_executors", "real_system_eligible", "hazards"},
            context=context,
        )
        if value["risk_level"] not in RISK_LEVELS:
            raise ScenarioSchemaError(f"{context}.risk_level is invalid")
        allowed = _string_tuple(value["allowed_executors"], context=f"{context}.allowed_executors")
        if not set(allowed) <= EXECUTORS:
            raise ScenarioSchemaError(f"{context}.allowed_executors contains an unknown adapter")
        if not isinstance(value["real_system_eligible"], bool):
            raise ScenarioSchemaError(f"{context}.real_system_eligible must be boolean")
        if value["real_system_eligible"] and "real_robot" not in allowed:
            raise ScenarioSchemaError(f"{context}: real eligibility requires real_robot")
        hazards = _string_tuple(value["hazards"], context=f"{context}.hazards", allow_empty=True)
        return cls(value["risk_level"], frozenset(allowed), value["real_system_eligible"], hazards)


@dataclass(frozen=True)
class ScenarioSetup:
    scene: str
    executor: str
    objects: tuple[str, ...]
    initial_predicates: tuple[str, ...]
    goal_predicates: tuple[str, ...]
    protected_objects: tuple[str, ...]
    memory_records: tuple[str, ...]
    max_cycles: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, context: str) -> "ScenarioSetup":
        _strict_keys(
            value,
            {
                "scene",
                "executor",
                "objects",
                "initial_predicates",
                "goal_predicates",
                "protected_objects",
                "memory_records",
                "max_cycles",
            },
            context=context,
        )
        if value["executor"] not in EXECUTORS:
            raise ScenarioSchemaError(f"{context}.executor is invalid")
        return cls(
            scene=_string(value["scene"], context=f"{context}.scene"),
            executor=value["executor"],
            objects=_string_tuple(value["objects"], context=f"{context}.objects"),
            initial_predicates=_string_tuple(
                value["initial_predicates"], context=f"{context}.initial_predicates"
            ),
            goal_predicates=_string_tuple(
                value["goal_predicates"], context=f"{context}.goal_predicates"
            ),
            protected_objects=_string_tuple(
                value["protected_objects"],
                context=f"{context}.protected_objects",
                allow_empty=True,
            ),
            memory_records=_string_tuple(
                value["memory_records"], context=f"{context}.memory_records", allow_empty=True
            ),
            max_cycles=_positive_int(value["max_cycles"], context=f"{context}.max_cycles"),
        )


@dataclass(frozen=True)
class TriggerSpec:
    trigger_id: str
    kind: str
    boundary: str
    firing_count: int
    predicate_event: str
    predicate_occurrence: int
    action: str
    max_response_events: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, context: str) -> "TriggerSpec":
        _strict_keys(
            value,
            {
                "id",
                "kind",
                "boundary",
                "firing_count",
                "predicate_event",
                "predicate_occurrence",
                "action",
                "max_response_events",
            },
            context=context,
        )
        if value["kind"] not in TRIGGER_KINDS:
            raise ScenarioSchemaError(f"{context}.kind is invalid")
        return cls(
            trigger_id=_string(value["id"], context=f"{context}.id"),
            kind=value["kind"],
            boundary=_string(value["boundary"], context=f"{context}.boundary"),
            firing_count=_positive_int(value["firing_count"], context=f"{context}.firing_count"),
            predicate_event=_string(
                value["predicate_event"], context=f"{context}.predicate_event"
            ),
            predicate_occurrence=_positive_int(
                value["predicate_occurrence"], context=f"{context}.predicate_occurrence"
            ),
            action=_string(value["action"], context=f"{context}.action"),
            max_response_events=_positive_int(
                value["max_response_events"], context=f"{context}.max_response_events"
            ),
        )


@dataclass(frozen=True)
class PreflightSpec:
    calibration_required: bool
    checks: tuple[str, ...]
    expected_boundary: str
    capability_requirements: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, context: str) -> "PreflightSpec":
        _strict_keys(
            value,
            {"calibration_required", "checks", "expected_boundary", "capability_requirements"},
            context=context,
        )
        if not isinstance(value["calibration_required"], bool):
            raise ScenarioSchemaError(f"{context}.calibration_required must be boolean")
        checks = _string_tuple(value["checks"], context=f"{context}.checks")
        if not set(checks) <= PREFLIGHT_CHECKS:
            raise ScenarioSchemaError(f"{context}.checks contains an unknown check")
        return cls(
            calibration_required=value["calibration_required"],
            checks=checks,
            expected_boundary=_string(
                value["expected_boundary"], context=f"{context}.expected_boundary"
            ),
            capability_requirements=_string_tuple(
                value["capability_requirements"],
                context=f"{context}.capability_requirements",
            ),
        )


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    family: str
    title: str
    claim: str
    catalogue: str
    suite: str
    split_variants: Mapping[str, SplitVariant]
    difficulty: str
    safety: ScenarioSafety
    applicable_profiles: tuple[str, ...]
    outcomes: tuple[str, ...]
    seed_dimensions: tuple[str, ...]
    setup: ScenarioSetup
    user_script: tuple[str, ...]
    triggers: tuple[TriggerSpec, ...]
    oracles: tuple[OracleSpec, ...]
    preflight: PreflightSpec
    frozen_subsets: frozenset[str]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScenarioSpec":
        _strict_keys(
            value,
            {
                "id",
                "family",
                "title",
                "claim",
                "catalogue",
                "suite",
                "split_variants",
                "difficulty",
                "safety",
                "applicable_profiles",
                "outcomes",
                "seed_dimensions",
                "setup",
                "user_script",
                "triggers",
                "oracles",
                "preflight",
                "frozen_subsets",
            },
            context="scenario",
        )
        scenario_id = _string(value["id"], context="scenario.id")
        if value["family"] not in FAMILY_COUNTS:
            raise ScenarioSchemaError(f"{scenario_id}: invalid family")
        if value["catalogue"] not in CATALOGUES:
            raise ScenarioSchemaError(f"{scenario_id}: invalid catalogue")
        if value["suite"] not in SUITES:
            raise ScenarioSchemaError(f"{scenario_id}: invalid suite")
        if value["difficulty"] not in DIFFICULTIES:
            raise ScenarioSchemaError(f"{scenario_id}: invalid difficulty")

        split_data = value["split_variants"]
        if not isinstance(split_data, Mapping):
            raise ScenarioSchemaError(f"{scenario_id}.split_variants must be an object")
        _strict_keys(split_data, set(SPLITS), context=f"{scenario_id}.split_variants")
        variants = {
            split: SplitVariant.from_dict(
                split_data[split], context=f"{scenario_id}.split_variants.{split}"
            )
            for split in SPLITS
        }
        signatures = [variant.separation_signature for variant in variants.values()]
        if len(signatures) != len(set(signatures)):
            raise ScenarioSchemaError(f"{scenario_id}: split variants overlap")

        safety = ScenarioSafety.from_dict(value["safety"], context=f"{scenario_id}.safety")
        profiles = _string_tuple(
            value["applicable_profiles"], context=f"{scenario_id}.applicable_profiles"
        )
        if not set(profiles) <= PROFILE_IDS:
            raise ScenarioSchemaError(f"{scenario_id}: unknown applicable profile")
        outcomes = _string_tuple(value["outcomes"], context=f"{scenario_id}.outcomes")
        if not set(outcomes) <= OUTCOMES:
            raise ScenarioSchemaError(f"{scenario_id}: unknown outcome")
        seeds = _string_tuple(value["seed_dimensions"], context=f"{scenario_id}.seed_dimensions")
        required_seeds = {"layout_seed", "language_seed", "controller_seed"}
        if not required_seeds <= set(seeds):
            raise ScenarioSchemaError(f"{scenario_id}: base seed dimensions are incomplete")

        setup = ScenarioSetup.from_dict(value["setup"], context=f"{scenario_id}.setup")
        if setup.executor not in safety.allowed_executors:
            raise ScenarioSchemaError(f"{scenario_id}: setup executor violates scenario safety")
        user_script = _string_tuple(value["user_script"], context=f"{scenario_id}.user_script")
        trigger_data = value["triggers"]
        if not isinstance(trigger_data, list) or not trigger_data:
            raise ScenarioSchemaError(f"{scenario_id}.triggers must be a non-empty array")
        triggers = tuple(
            TriggerSpec.from_dict(item, context=f"{scenario_id}.triggers[{index}]")
            for index, item in enumerate(trigger_data)
        )
        if len({trigger.trigger_id for trigger in triggers}) != len(triggers):
            raise ScenarioSchemaError(f"{scenario_id}: duplicate trigger ID")

        oracle_data = value["oracles"]
        if not isinstance(oracle_data, list) or not oracle_data:
            raise ScenarioSchemaError(f"{scenario_id}.oracles must be a non-empty array")
        try:
            oracles = tuple(OracleSpec.from_dict(item) for item in oracle_data)
        except OracleSchemaError as exc:
            raise ScenarioSchemaError(f"{scenario_id}: {exc}") from exc
        if len({oracle.oracle_id for oracle in oracles}) != len(oracles):
            raise ScenarioSchemaError(f"{scenario_id}: duplicate oracle ID")
        if "safety" not in {oracle.oracle_type for oracle in oracles}:
            raise ScenarioSchemaError(f"{scenario_id}: safety oracle is mandatory")

        preflight = PreflightSpec.from_dict(
            value["preflight"], context=f"{scenario_id}.preflight"
        )
        if value["catalogue"] == "mechanism_isolation":
            required_checks = {
                "target_detection",
                "identity",
                "reachability",
                "collision_free",
                "trigger_hook",
                "executor_capability",
            }
            if not preflight.calibration_required or not required_checks <= set(preflight.checks):
                raise ScenarioSchemaError(
                    f"{scenario_id}: mechanism-isolation preflight is incomplete"
                )

        markers = _string_tuple(
            value["frozen_subsets"], context=f"{scenario_id}.frozen_subsets", allow_empty=True
        )
        allowed_markers = FROZEN_MARKERS | ABLATION_MARKERS
        if not set(markers) <= allowed_markers:
            raise ScenarioSchemaError(f"{scenario_id}: unknown frozen subset marker")
        for marker in markers:
            if marker.startswith("ablation:") and marker.removeprefix("ablation:") not in profiles:
                raise ScenarioSchemaError(
                    f"{scenario_id}: {marker} must also be an applicable profile"
                )
        if ("real12" in markers) != safety.real_system_eligible:
            raise ScenarioSchemaError(
                f"{scenario_id}: real12 marker and real-system eligibility disagree"
            )
        if "A-HostFence" in profiles and "synthetic_event" not in safety.allowed_executors:
            raise ScenarioSchemaError(f"{scenario_id}: A-HostFence needs synthetic-event support")

        if value["family"] == "perturbation_recovery":
            oracle_types = {oracle.oracle_type for oracle in oracles}
            if not {"event_order", "hidden_state"} <= oracle_types:
                raise ScenarioSchemaError(
                    f"{scenario_id}: perturbations require order and recovery-state oracles"
                )
            if "perturbation_recovery" not in outcomes:
                raise ScenarioSchemaError(
                    f"{scenario_id}: perturbation recovery outcome is mandatory"
                )

        return cls(
            scenario_id=scenario_id,
            family=value["family"],
            title=_string(value["title"], context=f"{scenario_id}.title"),
            claim=_string(value["claim"], context=f"{scenario_id}.claim"),
            catalogue=value["catalogue"],
            suite=value["suite"],
            split_variants=variants,
            difficulty=value["difficulty"],
            safety=safety,
            applicable_profiles=profiles,
            outcomes=outcomes,
            seed_dimensions=seeds,
            setup=setup,
            user_script=user_script,
            triggers=triggers,
            oracles=oracles,
            preflight=preflight,
            frozen_subsets=frozenset(markers),
        )


@dataclass(frozen=True)
class ScenarioCatalogue:
    scenarios: tuple[ScenarioSpec, ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if len(self.scenarios) != 62:
            raise ScenarioSchemaError(f"expected 62 scenarios, found {len(self.scenarios)}")
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ScenarioSchemaError("scenario IDs are not unique")
        observed_families = Counter(scenario.family for scenario in self.scenarios)
        if dict(observed_families) != FAMILY_COUNTS:
            raise ScenarioSchemaError(
                f"family counts mismatch: expected={FAMILY_COUNTS}, "
                f"actual={dict(observed_families)}"
            )
        variant_ids = [
            variant.variant_id
            for scenario in self.scenarios
            for variant in scenario.split_variants.values()
        ]
        if len(variant_ids) != len(set(variant_ids)):
            raise ScenarioSchemaError("split variant IDs are not globally unique")
        subset_expectations = {
            "topology24": TOPOLOGY24_IDS,
            "sensitivity20": SENSITIVITY20_IDS,
            "real12": REAL12_IDS,
        }
        for marker, expected_ids in subset_expectations.items():
            observed_ids = {
                scenario.scenario_id
                for scenario in self.scenarios
                if marker in scenario.frozen_subsets
            }
            if observed_ids != expected_ids:
                raise ScenarioSchemaError(
                    f"{marker} membership mismatch: "
                    f"missing={sorted(expected_ids - observed_ids)}, "
                    f"unexpected={sorted(observed_ids - expected_ids)}"
                )
        for marker, expected_ids in ABLATION_SUBSET_IDS.items():
            observed_ids = {
                scenario.scenario_id
                for scenario in self.scenarios
                if marker in scenario.frozen_subsets
            }
            if observed_ids != expected_ids:
                raise ScenarioSchemaError(
                    f"{marker} membership mismatch: "
                    f"missing={sorted(expected_ids - observed_ids)}, "
                    f"unexpected={sorted(observed_ids - expected_ids)}"
                )
        observed_oracles = {
            oracle.oracle_type for scenario in self.scenarios for oracle in scenario.oracles
        }
        if observed_oracles != ORACLE_TYPES:
            raise ScenarioSchemaError(
                f"catalogue oracle coverage mismatch: {sorted(observed_oracles)}"
            )

    def get(self, scenario_id: str) -> ScenarioSpec:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise ScenarioSchemaError(f"unknown scenario ID: {scenario_id}")

    def subset(self, marker: str) -> tuple[ScenarioSpec, ...]:
        if marker not in FROZEN_MARKERS | ABLATION_MARKERS:
            raise ScenarioSchemaError(f"unknown subset marker: {marker}")
        return tuple(
            scenario for scenario in self.scenarios if marker in scenario.frozen_subsets
        )


def load_scenarios(path: str | Path | None = None) -> ScenarioCatalogue:
    """Load the complete catalogue; unknown fields fail at every schema level."""

    catalogue_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / "scenarios.json"
    )
    raw = json.loads(catalogue_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ScenarioSchemaError("scenario catalogue must be an object")
    _strict_keys(
        raw,
        {"schema_version", "family_counts", "subset_counts", "scenarios"},
        context="catalogue",
    )
    if raw["schema_version"] != 2:
        raise ScenarioSchemaError("unsupported scenario schema version")
    if raw["family_counts"] != FAMILY_COUNTS:
        raise ScenarioSchemaError("declared family counts differ from the frozen contract")
    if raw["subset_counts"] != {"topology24": 24, "sensitivity20": 20, "real12": 12}:
        raise ScenarioSchemaError("declared frozen subset counts are invalid")
    if not isinstance(raw["scenarios"], list):
        raise ScenarioSchemaError("scenarios must be an array")
    return ScenarioCatalogue(tuple(ScenarioSpec.from_dict(item) for item in raw["scenarios"]))
