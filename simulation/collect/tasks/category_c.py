"""Category C -- stacking_blocks_decomposed (tasks 1-8), colors red/green/blue/yellow.

The stacking task cut into its two atomic skills:

* tasks 1-4, "T-MAT", one per color: put a named block on the cross. No stack;
  2-4 blocks on the black mat, always including the target.
* tasks 5-8, "T-TOP", one per color: put a named block on top of a stack that is
  already at the cross. Stack height h in {1,2,3}, round-robin so each height gets
  40 of the task's 120 episodes; the stack's colors are drawn without replacement
  from the three colors that are *not* the target, and 0-2 of whatever colors are
  left over join the target on the black mat.

With four colors, h = 3 consumes every non-target color, so a height-3 episode has
no distractors -- the number available is exactly ``3 - h``.
"""

import numpy as np

import config as C
import expert as E
import scene_builder as SB

CATEGORY = "c"
NAME = "stacking_blocks_decomposed"
BUDGET = {1: 60, 2: 60, 3: 60, 4: 60, 5: 120, 6: 120, 7: 120, 8: 120}

MAT_TASKS = (1, 2, 3, 4)
TOP_TASKS = (5, 6, 7, 8)
STACK_HEIGHTS = (1, 2, 3)

MAT_INSTRUCTION = "Place the {color} block on the white mat at the cross position."
TOP_INSTRUCTION = "Place the {color} block on top of the stack at the cross position."

MAT_BLOCKS_RANGE = (2, 4)  # inclusive
MAX_DISTRACTORS = 2


def target_color(task_id):
    if task_id in MAT_TASKS:
        return C.COLORS[task_id - 1]
    if task_id in TOP_TASKS:
        return C.COLORS[task_id - 5]
    raise ValueError(f"category C has tasks 1-8, got {task_id}")


def stack_height_for(episode_id):
    """Round-robin over {1,2,3}: 120 episodes -> 40 each."""
    return STACK_HEIGHTS[(episode_id - 1) % len(STACK_HEIGHTS)]


def _build_mat(task_id, episode_id, seed):
    from . import EpisodePlan, canonical

    rng = np.random.default_rng(seed)
    color = target_color(task_id)

    n_blocks = int(rng.integers(MAT_BLOCKS_RANGE[0], MAT_BLOCKS_RANGE[1] + 1))
    others = [c for c in C.COLORS if c != color]
    extras = list(rng.permutation(others)[: n_blocks - 1])
    present = canonical([color, *extras])

    cross = SB.sample_cross(rng)
    placements = SB.sample_mat_blocks(rng, present)
    scene = SB.build_scene(placements, cross, init_stack=(), mat_colors=present)

    return EpisodePlan(
        scene=scene,
        instruction=MAT_INSTRUCTION.format(color=color),
        order=[color],
        verify=lambda sc, d: E.verify_on_cross(sc, d, color),
        safe_z=E.safe_transit_z(0),
        init_stack=[],
        blocks_present=present,
    )


def _build_top(task_id, episode_id, seed):
    from . import EpisodePlan, canonical

    rng = np.random.default_rng(seed)
    color = target_color(task_id)
    height = stack_height_for(episode_id)

    others = [c for c in C.COLORS if c != color]
    shuffled = list(rng.permutation(others))
    stack = shuffled[:height]  # bottom -> top
    leftover = shuffled[height:]
    n_distractors = int(rng.integers(0, min(MAX_DISTRACTORS, len(leftover)) + 1))
    distractors = leftover[:n_distractors]

    cross = SB.sample_cross(rng)
    mat_colors = canonical([color, *distractors])
    placements = SB.stack_placements(cross, stack) + SB.sample_mat_blocks(rng, mat_colors)
    scene = SB.build_scene(placements, cross, init_stack=stack, mat_colors=mat_colors)

    return EpisodePlan(
        scene=scene,
        instruction=TOP_INSTRUCTION.format(color=color),
        order=[color],
        verify=lambda sc, d: E.verify_on_top(sc, d, color, stack),
        safe_z=E.safe_transit_z(height),
        init_stack=list(stack),
        blocks_present=canonical([color, *stack, *distractors]),
    )


def build(task_id, episode_id, seed):
    if task_id in MAT_TASKS:
        return _build_mat(task_id, episode_id, seed)
    if task_id in TOP_TASKS:
        return _build_top(task_id, episode_id, seed)
    raise ValueError(f"category C has tasks 1-8, got {task_id}")


def scene_yaml():
    from .category_a import _randomization, _seed_formula

    tasks = {}
    for t in MAT_TASKS:
        tasks[t] = {
            "kind": "T-MAT", "episodes": BUDGET[t], "target": target_color(t),
            "instruction": MAT_INSTRUCTION.format(color=target_color(t)),
            "init_stack": None,
            "blocks_on_black_mat": f"{MAT_BLOCKS_RANGE[0]}-{MAT_BLOCKS_RANGE[1]}, "
                                   f"random subset always including the target",
        }
    for t in TOP_TASKS:
        tasks[t] = {
            "kind": "T-TOP", "episodes": BUDGET[t], "target": target_color(t),
            "instruction": TOP_INSTRUCTION.format(color=target_color(t)),
            "stack_heights": list(STACK_HEIGHTS),
            "episodes_per_height": BUDGET[t] // len(STACK_HEIGHTS),
            "height_assignment": "STACK_HEIGHTS[(episode_id - 1) % 3]",
            "stack_colors": "sampled without replacement from the three non-target colors",
            "distractors": f"0-{MAX_DISTRACTORS}, limited to the 3 - h colors left over",
        }

    return {
        "schema_version": C.SCHEMA_VERSION,
        "category": CATEGORY,
        "name": NAME,
        "description": (
            "The stacking task decomposed into its two primitives. T-MAT places a "
            "named block on the cross; T-TOP places a named block onto a pre-built "
            "stack sitting on the cross. Four colors: red, green, blue, yellow."
        ),
        "instruction_templates": {"T-MAT": MAT_INSTRUCTION, "T-TOP": TOP_INSTRUCTION},
        "tasks": tasks,
        "prebuilt_stack": {
            "spawn": "blocks at the cross, yaw 0, settled for "
                     f"{C.STACK_SETTLE_TIME} s before the episode starts",
            "verification": "intact, upright, on the cross and at rest, else the "
                            "episode is resampled with a derived seed",
        },
        "randomization": _randomization(),
        "seed_formula": _seed_formula(),
        "split_rule": "episode_id % 10 == 0 -> val, else train",
    }
