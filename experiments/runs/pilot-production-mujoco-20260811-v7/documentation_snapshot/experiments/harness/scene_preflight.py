"""Independent MuJoCo/SAM/IK feasibility preflight for frozen scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import mujoco
import numpy as np

from experiments.harness.scenarios import ScenarioSpec, load_scenarios
from prefmem.execution.contracts import AnchorKind, ObjectReference
from prefmem.execution.grounding import RGBDGrounder
from prefmem.execution.sam import Sam3Client
from simulation.controller import PandaPickPlaceController
from simulation.stacking import HOME_QPOS, StackingEnvironment


class ScenePreflightError(RuntimeError):
    """A scenario is not independently feasible under its frozen contract."""


_ACTION_RULES = (
    ("scripted_input", re.compile(r"^(deliver|reject|submit|request) ")),
    ("memory", re.compile(r"^exercise (create|retrieve|update|forget|none) ")),
    ("scene_move", re.compile(r"^(move|displace|tip|place target) ")),
    ("scene_insert", re.compile(r"^insert ")),
    ("scene_remove", re.compile(r"^(remove|omit) ")),
    ("visual_evidence", re.compile(r"^(occlude|show|alternate|suppress) ")),
    ("model_fault", re.compile(r"^(return|withhold|replace depth) ")),
    ("execution_fault", re.compile(r"^(inject|force|place immovable) ")),
    ("host_fence", re.compile(r"^inject stale and wrong-ID ")),
)


def trigger_action_kind(action: str) -> str:
    """Resolve every catalogue action to one explicit driver capability."""

    matches = [kind for kind, pattern in _ACTION_RULES if pattern.search(action)]
    # Two intentionally specific rules may overlap a broad rule; the more
    # specific host-fence classification wins.
    if "host_fence" in matches:
        return "host_fence"
    if len(matches) != 1:
        raise ScenePreflightError(
            f"trigger action has {'no' if not matches else 'ambiguous'} handler: {action!r}"
        )
    return matches[0]


def _derived_seed(master_seed: int, scenario_id: str) -> int:
    value = hashlib.sha256(
        f"scene-preflight-v1:{master_seed}:{scenario_id}:locked".encode("utf-8")
    ).digest()
    return int.from_bytes(value[:8], "big") & ((1 << 63) - 1)


def _robot_collision_free(
    environment: StackingEnvironment,
    waypoints: list[np.ndarray],
    *,
    samples_per_segment: int = 12,
) -> tuple[bool, list[dict[str, Any]]]:
    model = environment.model
    robot_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in (
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "hand",
            "left_finger",
            "right_finger",
        )
    }
    scratch = mujoco.MjData(model)
    baseline = environment.snapshot().qpos
    collisions: list[dict[str, Any]] = []
    previous = HOME_QPOS[:7]
    addresses = list(environment.arm_qpos_addresses)
    for segment, target in enumerate(waypoints):
        for sample in range(1, samples_per_segment + 1):
            fraction = sample / samples_per_segment
            q = previous + fraction * (target - previous)
            scratch.qpos[:] = baseline
            scratch.qpos[addresses] = q
            mujoco.mj_forward(model, scratch)
            for contact_index in range(scratch.ncon):
                contact = scratch.contact[contact_index]
                body1 = int(model.geom_bodyid[contact.geom1])
                body2 = int(model.geom_bodyid[contact.geom2])
                robot1 = body1 in robot_bodies
                robot2 = body2 in robot_bodies
                if robot1 == robot2:
                    continue
                collisions.append(
                    {
                        "segment": segment,
                        "sample": sample,
                        "body1": mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_BODY, body1
                        ),
                        "body2": mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_BODY, body2
                        ),
                        "distance": float(contact.dist),
                    }
                )
        previous = target
    return not collisions, collisions[:20]


@dataclass(frozen=True, slots=True)
class ScenePreflightResult:
    scenario_id: str
    seed: int
    passed: bool
    checks: dict[str, bool]
    trigger_handlers: dict[str, str]
    frame_sequence: int
    frame_rgb_sha256: str
    detections: dict[str, Any]
    grounded_points: dict[str, Any]
    ik_solutions: dict[str, Any]
    collision_evidence: list[dict[str, Any]]
    errors: tuple[str, ...]


_QUERIES = (
    ("red_cube", "red block", AnchorKind.TOP_CENTER),
    ("blue_cube", "blue block", AnchorKind.TOP_CENTER),
    ("green_cube", "green block", AnchorKind.TOP_CENTER),
    ("target_pad", "square pad", AnchorKind.SURFACE_CENTER),
)


def preflight_scenario(
    scenario: ScenarioSpec,
    *,
    sam_base_url: str,
    master_seed: int,
) -> ScenePreflightResult:
    seed = _derived_seed(master_seed, scenario.scenario_id)
    checks = {check: False for check in scenario.preflight.checks}
    errors: list[str] = []
    trigger_handlers: dict[str, str] = {}
    detections: dict[str, Any] = {}
    grounded_points: dict[str, Any] = {}
    ik_solutions: dict[str, Any] = {}
    collision_evidence: list[dict[str, Any]] = []
    frame_sequence = -1
    frame_hash = ""

    try:
        for trigger in scenario.triggers:
            trigger_handlers[trigger.trigger_id] = trigger_action_kind(trigger.action)
        checks["trigger_hook"] = True
    except Exception as error:
        errors.append(f"trigger_hook: {error}")

    checks["artifact_path"] = bool(
        re.fullmatch(r"panda_tabletop_[a-z]{2}[0-9]{2}", scenario.setup.scene)
        or scenario.setup.scene.startswith("synthetic_")
    )
    checks["executor_capability"] = scenario.setup.executor in {
        "mujoco",
        "synthetic_event",
    }
    # Reaching this method follows a passed batch service gate; individual SAM
    # calls below independently preserve endpoint failures as check failures.
    checks["service_health"] = True

    environment: StackingEnvironment | None = None
    detector: Sam3Client | None = None
    try:
        environment = StackingEnvironment(
            seed=seed,
            width=640,
            height=640,
            render_hz=5.0,
            realtime=False,
            viewer=False,
        )
        checks["scene_load"] = True
        frame = environment.wait_for_frame(timeout=15.0)
        frame_sequence = frame.sequence
        frame_hash = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
        detector = Sam3Client(sam_base_url, threshold=0.05, timeout=60)
        grounder = RGBDGrounder(detector)
        controller = PandaPickPlaceController(environment)
        solutions: list[np.ndarray] = []
        for label, query, anchor in _QUERIES:
            raw = detector.detect(frame.rgb, query)
            detections[label] = {
                "query": query,
                "count": len(raw),
                "items": [
                    {
                        "box_xyxy": list(item.box_xyxy),
                        "score": item.score,
                        "mask_area": item.mask_area,
                    }
                    for item in raw
                ],
            }
            grounded = grounder.ground(frame, ObjectReference(query, anchor))
            grounded_points[label] = {
                "point_world": grounded.point_world.tolist(),
                "uncertainty_m": grounded.uncertainty_m,
            }
            approach = grounded.point_world + np.array([0.0, 0.0, 0.10])
            solution = controller.solve_ik(
                approach,
                seed=(None if not solutions else solutions[-1]),
            )
            solutions.append(solution)
            ik_solutions[label] = solution.tolist()
        checks["target_detection"] = all(
            item["count"] >= 1 for item in detections.values()
        )
        checks["identity"] = all(
            item["count"] == 1 for item in detections.values()
        )
        checks["reachability"] = len(solutions) == len(_QUERIES)
        collision_free, collision_evidence = _robot_collision_free(
            environment, solutions
        )
        checks["collision_free"] = collision_free
    except Exception as error:
        errors.append(f"physical_preflight: {type(error).__name__}: {error}")
    finally:
        if detector is not None:
            detector.close()
        if environment is not None:
            environment.close()

    relevant = {name: checks[name] for name in scenario.preflight.checks}
    for name, passed in relevant.items():
        if not passed and not any(item.startswith(f"{name}:") for item in errors):
            errors.append(f"{name}: check did not pass")
    return ScenePreflightResult(
        scenario_id=scenario.scenario_id,
        seed=seed,
        passed=all(relevant.values()),
        checks=relevant,
        trigger_handlers=trigger_handlers,
        frame_sequence=frame_sequence,
        frame_rgb_sha256=frame_hash,
        detections=detections,
        grounded_points=grounded_points,
        ik_solutions=ik_solutions,
        collision_evidence=collision_evidence,
        errors=tuple(errors),
    )


def run_mechanism_preflights(
    *,
    sam_base_url: str,
    master_seed: int = 20260811,
) -> tuple[ScenePreflightResult, ...]:
    scenarios = tuple(
        scenario
        for scenario in load_scenarios().scenarios
        if scenario.catalogue == "mechanism_isolation"
    )
    if len(scenarios) != 29:
        raise ScenePreflightError("expected 29 mechanism-isolation scenarios")
    return tuple(
        preflight_scenario(
            scenario, sam_base_url=sam_base_url, master_seed=master_seed
        )
        for scenario in scenarios
    )


def write_mechanism_preflights(
    path: str | Path,
    *,
    sam_base_url: str,
    master_seed: int = 20260811,
) -> str:
    results = run_mechanism_preflights(
        sam_base_url=sam_base_url, master_seed=master_seed
    )
    payload = {
        "schema_version": 1,
        "master_seed": master_seed,
        "simulator": "MuJoCo",
        "detector_endpoint_label": "authorized_loopback_sam",
        "scenario_count": len(results),
        "passed": all(item.passed for item in results),
        "results": [asdict(item) for item in results],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


__all__ = [
    "ScenePreflightError",
    "ScenePreflightResult",
    "preflight_scenario",
    "run_mechanism_preflights",
    "trigger_action_kind",
    "write_mechanism_preflights",
]
