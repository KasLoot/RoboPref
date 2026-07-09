"""Category A -- stacking_blocks_ambiguous (task 1, 240 episodes).

One instruction for every episode, and it never names an order. The order the
expert actually used lives only in the ``executed_order`` attribute, which is what
makes this set useful for studying preference under ambiguity.

40 episodes for each of the 6 permutations of (red, green, blue), interleaved by
``(episode_id - 1) % 6`` so that any prefix of a partial run stays balanced to
within one episode per order.
"""

import itertools

import numpy as np

import config as C
import expert as E
import scene_builder as SB

CATEGORY = "a"
NAME = "stacking_blocks_ambiguous"
BUDGET = {1: 240}

INSTRUCTION = "Stack the blocks from the black mat to the white mat at the cross position."
BLOCKS = ("red", "green", "blue")
ORDERS = [list(p) for p in itertools.permutations(BLOCKS)]  # fixed, documented order


def order_for(episode_id):
    return list(ORDERS[(episode_id - 1) % len(ORDERS)])


def build(task_id, episode_id, seed):
    from . import EpisodePlan, canonical

    if task_id != 1:
        raise ValueError(f"category A has only task 1, got {task_id}")
    rng = np.random.default_rng(seed)
    order = order_for(episode_id)

    cross = SB.sample_cross(rng)
    # canonical order, not `order`: the spawn must not reveal the stacking order
    placements = SB.sample_mat_blocks(rng, canonical(BLOCKS))
    scene = SB.build_scene(placements, cross, init_stack=(), mat_colors=canonical(BLOCKS))

    return EpisodePlan(
        scene=scene,
        instruction=INSTRUCTION,
        order=order,
        verify=lambda sc, d: E.verify_stack(sc, d, order),
        safe_z=E.safe_transit_z(len(order) - 1),
        init_stack=[],
        blocks_present=canonical(BLOCKS),
    )


def scene_yaml():
    return {
        "schema_version": C.SCHEMA_VERSION,
        "category": CATEGORY,
        "name": NAME,
        "description": (
            "ARX L5 arm, a white mat on the left and a black mat on the right of the "
            "third_person view. A magenta cross lies flush on the white mat. Three "
            "blocks (red, green, blue) start at random non-overlapping poses on the "
            "black mat and must be stacked on the cross. The instruction never states "
            "the order; the order used is recorded in the executed_order attribute."
        ),
        "instruction_templates": {"all_episodes": INSTRUCTION},
        "tasks": {1: {"episodes": BUDGET[1], "blocks": list(BLOCKS),
                      "orders": ["-".join(o) for o in ORDERS],
                      "episodes_per_order": BUDGET[1] // len(ORDERS)}},
        "order_assignment": "ORDERS[(episode_id - 1) % 6]; a partial run stays balanced",
        "randomization": _randomization(),
        "seed_formula": _seed_formula(),
        "split_rule": "episode_id % 10 == 0 -> val, else train",
    }


def _randomization():
    return {
        "cross_center": {
            "region": f"centred square of side {C.CROSS_REGION} m on the white mat",
            "white_mat_center": list(C.WHITE_MAT_CENTER),
            "distribution": "uniform",
        },
        "block_xy": {
            "region": "black mat, inset by SPAWN_MARGIN",
            "black_mat_center": list(C.BLACK_MAT_CENTER),
            "half_extent": list(C.SPAWN_HALF),
            "min_center_gap_m": C.SPAWN_MIN_GAP,
            "distribution": "uniform with rejection; whole set resampled if stranded",
            "sampled_in": "canonical color order (red, green, blue, yellow)",
        },
        "block_yaw_rad": list(C.BLOCK_YAW_RANGE),
    }


def _seed_formula():
    return ("seed = 100_000_000*category_code + 1_000_000*task_id + episode_id "
            "+ 10_000*attempt, category_code = {a:1, b:2, c:3}; "
            "rng = numpy.random.default_rng(seed)")
