"""Embodiment conventions for fine-tuning / running pi05 on a specific robot.

Everything that ties the model to a particular robot lives here, in one frozen
spec per embodiment: which sim/dataset cameras feed which model image slots, the
state and action layouts, the prompt padding length, and the chunk timing. The
training pipeline (``data.py``/``norm_stats.py``/``train.py``) and the sim
runner (``simulation/collect/run_policy.py``) all import the same spec, which is
what keeps the train-time and inference-time observation pipelines identical.

The module is deliberately dependency-light (numpy only) so the sim-side replay
tooling can import it without pulling in torch.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class EmbodimentSpec:
    """How one robot maps onto the pi05 interface.

    ``image_obs_keys`` maps a model image slot (a ``Pi05Model.IMAGE_KEYS`` name)
    to the camera name used by the dataset and the simulator. Slots that are
    absent are simply not fed to the model -- masked-out tokens are provably
    inert (see ``make_attn_mask``), so dropping them is exact and faster.
    """

    name: str
    image_obs_keys: dict[str, str]
    state_dim: int
    action_dim: int
    # Prompt padding length. Shorter than Pi05Config.max_token_len is exact
    # (pad tokens are masked); it must merely fit the longest instruction plus
    # the state bins -- asserted by the dataset at index time.
    token_len: int
    control_hz: float  # rate of the recorded actions (physics rate)
    chunk_hz: float    # rate the model's action chunk is trained/executed at
    horizon: int = 50  # chunk rows; must match Pi05Config.action_horizon

    @property
    def control_dt(self) -> float:
        """Seconds per recorded control step (the sim timestep)."""
        return 1.0 / self.control_hz

    @property
    def stride(self) -> int:
        """Recorded control steps per chunk step (500 Hz / 50 Hz = 10)."""
        stride = self.control_hz / self.chunk_hz
        assert abs(stride - round(stride)) < 1e-9, "chunk_hz must divide control_hz"
        return int(round(stride))


def chunk_indices(start: int, n_steps: int, stride: int, horizon: int) -> np.ndarray:
    """Control-step indices of one action chunk starting at ``start``.

    Indices past the end of the episode clamp to the last recorded step, so the
    tail of the final chunk repeats the final action (the arm holds still).
    """
    return np.minimum(start + stride * np.arange(horizon), n_steps - 1)


# ARX L5 in the RoboPref MuJoCo sim: 6 arm joints + 1 gripper.
#   state  = measured joint_pos (6) + commanded gripper ctrl (1)
#            (the dataset stores no measured gripper state; the command is the
#             same quantity the runner can feed back at inference time)
#   action = q_cmd joint-position setpoints (6) + gripper ctrl (1),
#            exactly what the sim's position actuators consume via data.ctrl.
ARX_L5 = EmbodimentSpec(
    name="arx_l5",
    image_obs_keys={"base_0_rgb": "third_person", "left_wrist_0_rgb": "wrist_cam"},
    state_dim=7,
    action_dim=7,
    token_len=96,
    control_hz=500.0,
    chunk_hz=50.0,
)

# Three-camera variant for datasets recorded with the near-top-down top_cam
# (RoboPref_dataset_v3 onward). The extra view goes in the model's unused
# right_wrist_0_rgb slot -- pi05 has no camera-identity embeddings, so slot
# assignment is arbitrary; keeping the first two where arx_l5 had them makes
# the specs differ by exactly one added view. Same state/action/chunk layout.
ARX_L5_3CAM = EmbodimentSpec(
    name="arx_l5_3cam",
    image_obs_keys={"base_0_rgb": "third_person", "left_wrist_0_rgb": "wrist_cam",
                    "right_wrist_0_rgb": "top_cam"},
    state_dim=7,
    action_dim=7,
    token_len=96,
    control_hz=500.0,
    chunk_hz=50.0,
)

EMBODIMENTS = {ARX_L5.name: ARX_L5, ARX_L5_3CAM.name: ARX_L5_3CAM}
