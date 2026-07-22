"""Task registry and the shared shape of an episode.

A task turns ``(task_id, episode_id, seed)`` into an :class:`EpisodePlan`: a compiled
scene, the verbatim instruction, the colors to place bottom -> top, and the ground
truth predicate that decides whether the attempt is kept.

One rule every category obeys: **block spawn positions are always sampled in the
canonical color order** (red, green, blue, yellow), never in the task's stacking
order. Rejection sampling gives the first color drawn a free choice and squeezes
later ones, so sampling in stacking order would correlate a block's position with
its rank -- and Category A's whole point is that the initial image does not reveal
the order.
"""

from dataclasses import dataclass, field

import config as C

from . import category_a, category_b, category_c

CATEGORIES = {
    category_a.CATEGORY: category_a,
    category_b.CATEGORY: category_b,
    category_c.CATEGORY: category_c,
}

__all__ = ["CATEGORIES", "EpisodePlan", "canonical", "category_a", "category_b",
           "category_c"]


def canonical(colors):
    """Colors in the fixed COLORS order, whatever order they were named in."""
    seen = set(colors)
    return [c for c in C.COLORS if c in seen]


@dataclass
class EpisodePlan:
    scene: object  # scene_builder.EpisodeScene
    instruction: str
    order: list  # colors to place, bottom -> top (a single color for Category C)
    verify: object  # (scene, data) -> (bool, reason)
    safe_z: float
    init_stack: list = field(default_factory=list)
    blocks_present: list = field(default_factory=list)

    @property
    def executed_order(self):
        return ",".join(self.order)

    def below(self, placed):
        """Colors already stacked under the next placement."""
        return list(self.init_stack) + list(placed)
