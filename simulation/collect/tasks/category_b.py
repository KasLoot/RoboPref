"""Category B -- stacking_blocks_ordered (tasks 1-4, 60 episodes each).

Same scene as Category A, but the instruction names the order. Four of the six
permutations are trained; the other two are held out with zero episodes so a policy
can be tested on orders it was never shown. The held-out orders are recorded in
scene.yaml, not merely omitted.
"""

import numpy as np

import config as C
import expert as E
import scene_builder as SB

CATEGORY = "b"
NAME = "stacking_blocks_ordered"
BUDGET = {1: 60, 2: 60, 3: 60, 4: 60}

BLOCKS = ("red", "green", "blue")
TRAINED_ORDERS = {
    1: ["red", "green", "blue"],
    2: ["blue", "green", "red"],
    3: ["green", "red", "blue"],
    4: ["red", "blue", "green"],
}
HELD_OUT_ORDERS = [["blue", "red", "green"], ["green", "blue", "red"]]

INSTRUCTION_TEMPLATE = (
    "Stack the blocks from the black mat to the white mat at the cross position, "
    "with order from bottom to top of {c1}, {c2}, {c3} blocks.")


def instruction_for(order):
    return INSTRUCTION_TEMPLATE.format(c1=order[0], c2=order[1], c3=order[2])


def build(task_id, episode_id, seed):
    from . import EpisodePlan, canonical

    if task_id not in TRAINED_ORDERS:
        raise ValueError(f"category B has tasks 1-4, got {task_id}")
    rng = np.random.default_rng(seed)
    order = list(TRAINED_ORDERS[task_id])

    cross = SB.sample_cross(rng)
    placements = SB.sample_mat_blocks(rng, canonical(BLOCKS))
    scene = SB.build_scene(placements, cross, init_stack=(), mat_colors=canonical(BLOCKS))

    return EpisodePlan(
        scene=scene,
        instruction=instruction_for(order),
        order=order,
        verify=lambda sc, d: E.verify_stack(sc, d, order),
        safe_z=E.safe_transit_z(len(order) - 1),
        init_stack=[],
        blocks_present=canonical(BLOCKS),
    )


def scene_yaml():
    from .category_a import _randomization, _seed_formula

    return {
        "schema_version": C.SCHEMA_VERSION,
        "category": CATEGORY,
        "name": NAME,
        "description": (
            "Identical scene to Category A: three blocks (red, green, blue) on the "
            "black mat, a magenta cross on the white mat. The instruction states the "
            "bottom-to-top order explicitly."
        ),
        "instruction_templates": {"all_episodes": INSTRUCTION_TEMPLATE},
        "tasks": {t: {"episodes": BUDGET[t], "order": "-".join(o),
                      "instruction": instruction_for(o)}
                  for t, o in TRAINED_ORDERS.items()},
        "trained_orders": ["-".join(o) for o in TRAINED_ORDERS.values()],
        "held_out_orders": ["-".join(o) for o in HELD_OUT_ORDERS],
        "held_out_episodes": 0,
        "randomization": _randomization(),
        "seed_formula": _seed_formula(),
        "split_rule": "episode_id % 10 == 0 -> val, else train",
    }
