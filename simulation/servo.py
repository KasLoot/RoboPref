"""Critical damping for the ARX L5's joint position servos.

MuJoCo's ``<position kp kv>`` actuator is a PD law,

    tau = kp * (ctrl - q) - kv * qvel,

so each joint behaves as a second order system with natural frequency
``sqrt(kp/M)`` and damping ratio ``zeta = kv / (2*sqrt(kp*M))``, where ``M`` is
the inertia that joint actually sees. The shipped gains leave the shoulder and
elbow underdamped (zeta = 0.81 and 0.61 at the home pose) and the wrist heavily
overdamped (zeta up to 7.9).

Both extremes hurt a differential IK loop, for the same reason. That loop
integrates a velocity command, ``q_cmd += dq*dt``, so it is an integrator: while
the servo lags, the integrator keeps accumulating, and the stored lead drives
the joints past the target once the task error reaches zero. Underdamping gives
the resulting overshoot a resonance to ring at; overdamping makes the servo lag
more and so stores more lead. Measured over engine.py's waypoint tour, the stock
gains overshoot by 11.6 mm, zeta=1.2 by 3.4 mm, and zeta=1.0 by 0.25 mm.

``M`` changes with configuration -- roughly 2x over that tour -- so the damping
is rescheduled from the current mass matrix every step rather than fixed once.
Setting ``dampratio="1"`` on the actuators in the XML is the static equivalent
(MuJoCo evaluates the inertia at the reference pose); it recovers most of the
benefit, 0.69 mm, with no runtime cost.
"""

import mujoco
import numpy as np

from ik import ARM_JOINTS


def _actuated_joints(model, joint_names):
    """Actuator index and mass-matrix address for each named joint."""
    act, madr = [], []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1:
            raise ValueError(f"unknown joint: {name}")
        driven = [
            a
            for a in range(model.nu)
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            and model.actuator_trnid[a, 0] == jid
        ]
        if not driven:
            raise ValueError(f"joint {name} has no actuator")
        act.append(driven[0])
        # data.qM is sparse, but the diagonal entry for dof i lives at dof_Madr[i].
        madr.append(model.dof_Madr[model.jnt_dofadr[jid]])
    return np.array(act), np.array(madr)


class CriticalDamper:
    """Reschedules each joint servo's kv to hold ``zeta`` as the inertia changes.

    Construct once, then call `apply` each step before ``mj_step``, with
    ``data.qM`` valid (any ``mj_forward``/``mj_step`` leaves it so).

    ``zeta=1.0`` is critical. Do not raise it above 1 hoping for a safety margin:
    extra damping is extra servo lag, which the IK integrator turns back into
    overshoot.
    """

    def __init__(self, model, joint_names=ARM_JOINTS, zeta=1.0):
        self.model = model
        self.zeta = zeta
        self.act, self.madr = _actuated_joints(model, joint_names)
        self.kp = model.actuator_gainprm[self.act, 0].copy()

    def apply(self, data):
        """Set kv = 2*zeta*sqrt(kp*M) against the current mass matrix diagonal.

        Ignores inter-joint coupling -- an approximation, but one that measures
        out well on this arm.
        """
        kv = 2.0 * self.zeta * np.sqrt(self.kp * data.qM[self.madr])
        # position actuator: biasprm = [0, -kp, -kv]
        self.model.actuator_biasprm[self.act, 2] = -kv


def damping_ratios(model, data, joint_names=ARM_JOINTS):
    """Current zeta per joint -- for inspecting how a set of gains is tuned."""
    act, madr = _actuated_joints(model, joint_names)
    kp = model.actuator_gainprm[act, 0]
    kv = -model.actuator_biasprm[act, 2]
    return kv / (2.0 * np.sqrt(kp * data.qM[madr]))
