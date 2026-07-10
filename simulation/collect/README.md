# RoboPref data collection

Scripted pick-and-place expert on the ARX L5 in MuJoCo, producing three VLA training
dataset categories with synchronized proprio, actions and RGB-D from two cameras.

```
uv run python simulation/collect/collect.py --category {a,b,c,all} \
    [--task N] [--episodes N] [--root PATH] [--preview] [--overwrite]

uv run python simulation/collect/validate.py [--root PATH] [--category a|b|c|all] \
    [--samples N] [--no-sheets]
```

Collection is resumable: an episode file that passes `recorder.integrity_check` is
skipped, so an interrupted run continues where it stopped. `--overwrite` redoes them.
Tasks are independent, so several `--task N` processes can run in parallel on
different tasks to use more cores.

An episode is written to `<name>.hdf5.partial` and renamed only on success, so a
discarded attempt never lands. A hard kill (SIGKILL, or a crash inside the cleanup)
can strand a partial; `collect_task` deletes orphaned partials **for the task it is
about to collect** before starting. It is scoped that way so sharded runs never delete
each other's work in progress, and it is skipped under `--preview`.

### `--preview`

Attaches the interactive viewer plus `camera.CameraCanvas`, overlays engine.py's target
frame and TCP markers, paces the loop to wall clock, and **writes nothing** — it is a
debugging view, not a collection mode. Defaults to one episode per task, since at
wall-clock speed the full budget would take hours; pass `--episodes N` for more. Close
either window to stop the run.

    uv run python simulation/collect/collect.py --category a --preview
    uv run python simulation/collect/collect.py --category c --task 7 --episodes 3 --preview

Two things make this work, and both are easy to get wrong:

* **GL backend.** `mujoco.viewer` always creates a GLFW context. An EGL
  `mujoco.Renderer` — which is what `CameraCanvas` builds — then cannot be made current
  on the same thread: it raises `Failed to make the EGL context current` and the process
  dies. So `collect.py` puts the whole process on `MUJOCO_GL=glfw` when it sees
  `--preview` in `sys.argv`, *before* importing `config` (which otherwise defaults to
  EGL). Headless collection is unaffected and still uses EGL.
* **Teardown order.** The viewer must be closed **before** the canvas. Any other order,
  or leaving either open, double-frees GLFW and the interpreter segfaults on exit
  (status 139). `Preview.close` closes the viewer first.

The preview costs ~0.65 ms of the 2 ms step budget, so episodes run at roughly 1x
real time (a 9.8 s Category C episode takes ~11 s of wall clock).

The default is headless (EGL) at full speed.

Dataset root defaults to `/data/datasets/RoboPref`, created if missing. One subfolder
per category (`stacking_blocks_ambiguous/`, `stacking_blocks_ordered/`,
`stacking_blocks_decomposed/`), one `scene.yaml` per category, one HDF5 per episode
named `task_{task_id}_episode_{episode_id}.hdf5` with episode ids starting at 1.

---

## Phase 0 — what the model actually says

Everything below was measured off the compiled model, not assumed. `config.py` holds
the constants and `config.assert_model_matches` re-checks the load-bearing ones at
runtime so the suite fails loudly if the scene drifts.

**Timestep.** `model.opt.timestep = 0.002 s` → 500 Hz. Proprio and actions are logged
at *every* physics step and the actual timestamps are stored; nothing is resampled.

**Gripper.** The one actuator not driving `joint1..joint6` is `gripper` (index 6). It
drives `joint7`, a slide joint with `ctrlrange = [0, 0.044]`; `joint8` is mirrored to
`-joint7` by an equality constraint, so the two fingers move symmetrically.

Sweeping the ctrl value and measuring the inner-face separation of the finger pads
along the `link6` y axis gives an exactly affine law:

```
gap(ctrl) = 0.2 mm + 2 * ctrl        # 0.2 mm shut, 87.8 mm wide open
```

A 35 mm cube therefore closes the fingers at `ctrl = 0.0174`. Commanding `ctrl = 0`
against the block leaves a steady squeeze of `kp * 0.0174 = 0.70 N` per finger, which
with a friction coefficient of 1.4 holds a 30 g block with a ~6x margin.

Derived commands: `GRIP_CLOSE = 0.0`, `GRIP_OPEN = 0.030`.

`GRIP_OPEN` is deliberately **not** the full 0.044. A fully open finger reaches 43.9 mm
from the tool axis, while a neighbouring block's near face is only 35 mm away at the
minimum spawn gap — the finger would strike it on the way down. At 0.030 the gap is
60.2 mm, giving 12.6 mm of clearance around a 35 mm block on each side.

**How far the gripper reaches below the TCP.** The long finger pad is centred exactly
on the TCP and is 40 mm tall, so with the tool pointing down the lowest point of the
gripper is `TCP_z - 0.0200 m`, independent of how far the fingers are open (measured,
not derived). This single number sets the grasp height and the transit height.

**Tool-down orientation.** `TCP_OFFSET` puts the tool axis along `+x` of `link6`. The
world quaternion that points it along `-z` is built directly:

```
x_axis = (0, 0, -1)                       # tool axis, pointing down
y_axis = (cos yaw, sin yaw, 0)            # finger closing axis
z_axis = x_axis x y_axis
quat   = mju_mat2Quat([x_axis y_axis z_axis])
```

`yaw = pi/2` (fingers closing along world y) is the minimal rotation from the home
pose — it leaves `joint6` at 0 — and gives `quat = (0.7071, 0, 0.7071, 0)`. Verified by
servoing home → tool-down at yaw ∈ {0°, 45°, 90°, 135°}: every one converges to
`pos_err < 0.52 mm`, `ori_err < 0.18°` in ~2.1 s. Blocks spawn with a yaw in ±45°, and
the grasp uses `yaw = block_yaw + pi/2` so the fingers square onto opposite cube faces.

**Workspace.** Probed by servoing to a grid of tool-down poses from home and bisecting
the reachable radius. The reach is radially symmetric about the base (`joint1` is a
z-hinge at the origin; checked at azimuths ±50°, all 0.510 m):

| TCP z (m) | 0.0215 | 0.0245 | 0.0285 | 0.10 | 0.14 | 0.18 | 0.20 | 0.22 |
|---|---|---|---|---|---|---|---|---|
| r_max (m) | 0.350 | 0.429 | **0.510** | 0.477 | 0.445 | 0.400 | 0.364 | 0.299 |

The collapse below z ≈ 0.026 is **not** kinematic — the arm is nowhere near a joint
limit. It is the finger pads fouling the floor: at `TCP_z = 0.0215` the pads sit 1.5 mm
above z = 0 and MuJoCo reports `floor~pad` contacts. Raising the TCP 7 mm restores the
full 0.510 m. `config.r_max(z)` interpolates this table and `expert.servo` refuses any
target outside it.

**Consequences for the layout.** Grasp with the TCP `GRASP_DZ = 7 mm` *above* the block
centre, not at it. Then the pads clear the mat by 4.5 mm and still grip 26.5 mm of the
block's 35 mm face. Below ~5.5 mm the pads foul the mat.

Nothing here made the spec impossible, so no questions were needed.

---

## Scene

| | value | why |
|---|---|---|
| block | 35 mm cube, 30 g, friction (1.4, 0.02, 0.002), `condim=4` | fits the gripper with margin; `condim=4` adds torsional friction so stacked blocks don't spin off |
| mats | 210 x 190 x 4 mm boxes | big enough for four blocks at the 70 mm spawn gap without the sampler stalling |
| white mat | centre (0.25, -0.135) | "left" |
| black mat | centre (0.25, +0.135) | "right" |
| cross | two 50 x 6 x 0.5 mm magenta bars, `contype=conaffinity=0` | flush on the mat, visual only, so it cannot perturb a block placed on it |
| robot | mid grey `(0.55, 0.55, 0.55)` | stock `0.1` is nearly the black mat's `0.07` |
| third_person camera | at `(0.85, 0, 0.55)`, aimed at `(0.27, 0, 0.06)` | stock pose framed the mats at the very bottom edge |
| cross centre | uniform in a 100 mm square centred on the white mat | |
| block spawn | uniform on the black mat, inset 28 mm, centres ≥ 70 mm apart | |
| block yaw | uniform in ±45° | |

**"Left" and "right" are as seen by the `third_person` camera** — the view a policy
consumes. That camera's right axis is world `+y`, so world `-y` is image-left. The
white mat is at `-y` (image left) and the black mat at `+y` (image right). Note this is
the mirror image of the *robot's* own left/right, which is why it is stated explicitly.

### Appearance overrides

Applied to the spec per episode by `scene_builder._restyle`, so the vendored robot asset
under `assets/robots/arx_l5` (which ships its own LICENSE and CHANGELOG) is never edited.
`engine.py` loads `scene.xml` directly and therefore still shows the stock look.

* **`ROBOT_RGBA = (0.55, 0.55, 0.55, 1)`.** Every visible robot geom shares the
  `black_mat` material; at its stock `0.1` grey the arm is nearly the same value as the
  black mat (`0.07`) and hard to segment against it. Mid grey separates cleanly from both
  mats. The two red strips on `link3` carry no material and stay red — they are the arm's
  own markings, not body colour.
* **`THIRD_PERSON_POS = (0.85, 0, 0.55)`, `THIRD_PERSON_AIM = (0.27, 0, 0.06)`.** The stock
  camera sits at `x = 1.05` with a fixed `xyaxes` whose centre ray meets the ground at
  `x = -0.29` — *behind* the base — so the mats sit at the very bottom of the frame.
  Moving it forward alone magnifies that framing and **crops the mats out**: at `x = 0.75`
  the white mat is nearly gone, at `x = 0.65` both are. So it is moved forward *and*
  re-aimed at the workspace, via `scene_builder.look_at_xyaxes`. The camera stays on the
  `y = 0` plane, so its right axis remains exactly world `+y` and the image-left
  convention above is unchanged. Verified that a 4-high tower — the tallest thing the
  suite ever builds — stays fully in frame.

  Both are one-line changes in `config.py`; `x = 0.95` is a gentler crop and `x = 0.75`
  a tighter one. Changing either invalidates already-collected episodes (their pixels and
  their `camera_third_person_pos` attr), so recollect with `--overwrite`.

Colours are chosen to be unambiguous under the scene lighting:
red `(0.85, 0.10, 0.10)`, green `(0.10, 0.70, 0.20)`, blue `(0.10, 0.25, 0.85)`,
yellow `(0.95, 0.85, 0.10)`; the cross is magenta `(0.90, 0.10, 0.75)`, distinct from
all four.

The spawn gap is **2.0 x block size**, not the 1.5 x floor the spec allows: at 1.5 x the
open fingers clear a neighbouring block by under 1 mm on descent.

Scenes are built with `MjSpec`: `scene.xml` is loaded, mats/cross/blocks are appended,
and the model is recompiled per episode (~17 ms). The robot XML is never touched.

> **A trap worth knowing.** Adding free-joint blocks grows `nq`, and MuJoCo will happily
> compile a model whose `key_qpos` is *shorter* than `nq` — it zero-pads. A stale `home`
> keyframe therefore teleports every block to the world origin on
> `mj_resetDataKeyframe`, silently. `scene_builder.build_scene` rewrites the keyframe to
> cover every new dof and asserts `len(key.qpos) == model.nq`.

---

## Expert

Primitives use `DiffIK` + `CriticalDamper`. Terminal convergence retains
`engine.servo_to`'s inner-loop shape:

```python
damper.apply(data)
q_cmd = ik.step(data, target_pos, target_quat, q_cmd)
data.ctrl[:6] = q_cmd
mujoco.mj_step(model, data)
```

The **command** is integrated and fed back; the setpoint is never rebuilt from measured
`qpos`. Grasp, place, and the final retreat settle on `POS_TOL`/`ORI_TOL` held for
`HOLD`, with an 8 s terminal timeout.

Intermediate safe-height, pregrasp, hover, and retreat poses are pass-through control
points of a continuous Cartesian trajectory. Short quadratic fillets round their
corners, while a quintic time law limits scalar and angular velocity, acceleration,
and jerk over the complete motion phase. Differential IK tracks the moving pose using its
trajectory twist as feed-forward plus proportional pose-error feedback. Joint command
acceleration receives a final per-step limit. This avoids the old pattern of braking,
holding for 0.1 s, and accelerating again at every transit waypoint. Orientation
changes are delayed until the open gripper has cleared a newly placed stack.

`pick(color)` → smoothly join a pending retreat (if any), safe transit, pregrasp, and
grasp; stop and close.
`place(xy, z)` → smoothly join vertical lift, safe transit, hover, and place; stop and
open. The vertical retreat is joined to the next pick, or executed by `finish()` after
the last placement.

Two corrections make stacking work, and both are load-bearing:

1. **Carry-offset compensation.** Closing the fingers nudges the block ~0.7 mm and it can
   settle slightly low. After the lift the expert measures `TCP - block_pos` and offsets
   the place target by it, so the *block* lands on the target rather than the tool.

2. **A safe transit height.** Every horizontal move happens at
   `expert.safe_transit_z(top_level)`, and the tool always leaves a placed block by rising
   straight up. Retreating only 45 mm and then moving diagonally toward the next block
   drags the open fingers across the top of the stack. This is what breaks a 4-high build:
   with the diagonal departure, 4-high stacking succeeded 5/12; with the vertical
   departure, **24/24**.

Place targets come from ground truth, re-read from `data` after every placement:
mat placement is the cross at `z = MAT_TOP + BLOCK_HALF`; stack placement is the
*measured* centre of the current top block, one block size up (`expert.stack_target`).

### Success predicates

Evaluated on ground truth after a 1.0 s settle: base block within 2 cm (xy) of the cross;
each stacked block within 2 cm (xy) of the block below and within 1 cm of its expected
height; every block's linear speed below 1 cm/s. Full-task success additionally requires
that the **executed** order — read back by sorting the blocks by resting height — equals
the intended order. For Category C T-TOP the pre-existing stack must also be undisturbed.

Any timeout or failed predicate discards the recording (the `.partial` file is deleted,
never renamed) and the episode retries with seed `seed + 10_000 * attempt`, up to 8
attempts. Discards are counted and printed. If a task's failure rate exceeds 20% after at
least 10 attempts the task is abandoned and reported rather than ground out.

### Measured reliability

| configuration | result |
|---|---|
| 3-high from the mat, all 6 orders | 24/24 |
| 4-high from the mat | 24/24 |
| T-TOP onto a pre-built stack, h = 0,1,2,3 | 24/24 |
| T-MAT with 2, 3, 4 blocks present | 18/18 |
| pre-built stack intact after a 0.5 s settle, h = 1,2,3 | 60/60 |

Block mass and friction turned out not to be delicate: 3-high stacking succeeded for
every combination tried in `mass ∈ [20, 50] g`, `friction ∈ [1.0, 2.0]`, `condim ∈ {3,4,6}`.
The chosen values sit in the middle of that range.

---

## Determinism

```
seed = 100_000_000 * category_code + 1_000_000 * task_id + episode_id + 10_000 * attempt
category_code = {a: 1, b: 2, c: 3}          rng = numpy.random.default_rng(seed)
```

Task blocks are `1e6` apart and attempts step by `1e4`, so with `episode_id ≤ 240` and
`attempt ≤ 8` no two `(task, episode, attempt)` triples can collide. Attempt 0 is the
first try; a discarded attempt never reuses a seed.

Block spawn positions are always sampled **in the canonical colour order**
(red, green, blue, yellow), never in the task's stacking order. Rejection sampling gives
the first colour drawn a free choice and squeezes later ones, so sampling in stacking
order would correlate a block's position with its rank — and Category A's whole point is
that the initial image does not reveal the order.

Physics and proprio are bitwise reproducible from the seed (verified: identical SHA-256
over the whole `qpos`/`q_cmd` trajectory across runs). RGB is reproducible to within
±1 LSB on <0.01% of pixels — that is GPU multisample-resolve nondeterminism, present even
between two renders from the same renderer. Depth is bit-identical.

`split` is `"val"` for every 10th episode of a task (`episode_id % 10 == 0`), else `"train"`.

---

## HDF5 schema (version 1.0)

All datasets chunked and gzip-4 compressed.

```
/proprio/t                  (T,)            f64  sim time, s
/proprio/joint_pos          (T,6)           f32  rad
/proprio/joint_vel          (T,6)           f32  rad/s
/proprio/joint_torque       (T,6)           f32  qfrc_actuator at the arm dofs
/proprio/ee_pos             (T,3)           f32  TCP world position (DiffIK.tool_pose)
/proprio/ee_quat_wxyz       (T,4)           f32
/actions/t                  (T,)            f64  identical to /proprio/t
/actions/q_cmd              (T,6)           f32  the setpoints written to data.ctrl
/actions/gripper            (T,)            f32  gripper ctrl value
/cameras/<name>/t           (K,)            f64  name in {third_person, wrist_cam}
/cameras/<name>/rgb         (K,480,640,3)   u8
/cameras/<name>/depth       (K,480,640)     u16  millimetres, 0 = invalid
```

Root attrs: `category`, `category_name`, `task_id`, `episode_id`, `instruction`
(the verbatim string), `seed`, `success` (always `True`), `executed_order`
(e.g. `"red,green,blue"` bottom→top; a single colour for Category C), `blocks_present`,
`init_stack` (bottom→top, empty string if none), `split`, `sim_timestep`, `tcp_offset`,
`quat_convention = "wxyz"`, `schema_version`, `cross_xy`, `block_size`,
`nominal_camera_fps`, `n_steps`, `n_camera_frames`, `attempts`, `discarded_attempts`,
and per-camera `camera_<name>_{fovy,resolution,native_resolution,sensorsize,focal,principal,pos,quat_wxyz,body_mounted}`.
The same per-camera values are attrs on each `/cameras/<name>` group.

### Timestamps and synchronization

`mj_step` runs its forward pass at the pre-step `qpos` and only then integrates, so
immediately after the call `data.xpos` and `data.qfrc_actuator` describe the state at the
step's *start* while `data.time` has already advanced. The recorder therefore caches
`qpos`/`qvel` before the step and reads torque, TCP pose and camera frames after it,
stamping all of them `t = data.time - timestep`. Every logged row is one consistent
snapshot, and the frozen control loop is untouched. (Verified: after `mj_step`,
`tool_pose(data)` equals the forward kinematics of the pre-step `qpos` exactly.)

Cameras are captured on the first physics step at or past the next `1/30 s` tick. Because
2 ms does not divide `1/30 s`, inter-frame gaps alternate between 32 and 34 ms; the mean
comes out at 30.000 Hz. **The stored timestamps are authoritative** — 30 Hz is nominal.
Every camera timestamp coincides exactly with a proprio timestamp, which `validate.py`
checks.

The episode clock is zeroed once the scene is known valid — after a pre-built stack has
settled and been verified — so `t` starts at 0 and camera extrinsics are captured at that
instant.

### Why uint16 millimetres for depth

A deliberate storage decision. Depth arrives as float32 metres; stored as uint16 mm it
halves the depth bytes, and 1 mm quantisation is far below this scene's geometric
fidelity (a 35 mm block, sub-mm servo tolerances). The far plane is `zfar * extent` ≈ 34.7 m,
so the whole range fits in 65535 mm with room to spare. Pixels with no geometry come back
at exactly the far plane, so anything at or beyond 99% of `zfar` is written as **0 =
invalid** (about 2.5% of a typical `third_person` frame — the sky).

`zfar` is recomputed per episode: `model.stat.extent` depends on the scene bounding box,
which the mats and blocks change.

### Cameras

Both are rendered headless at 640×480 with a single `mujoco.Renderer`, toggling
`enable_depth_rendering()` between the two passes (2.6 ms per two-camera tick). The Tk
canvas is never used during collection.

`wrist_cam` is defined in the robot XML with `resolution="1280 720"` and an explicit
`sensorsize`, so MuJoCo builds its frustum from the 1.82:1 sensor while we render into a
1.33:1 viewport — the wrist images are horizontally compressed relative to the physical
camera. This is inherited from the given robot model, not introduced here; `fovy`
(58.01°), `sensorsize`, `focal` and `native_resolution` are all recorded so the intrinsics
can be reconstructed. `third_person` is a plain 45° fovy camera, repositioned per episode
(see *Appearance overrides*); its recorded `camera_third_person_pos` / `_quat_wxyz` reflect
the override, not the XML. `wrist_cam` is body-mounted (`body_mounted = True`): its stored
extrinsics are the episode-start pose, and since `ee_pos`/`ee_quat_wxyz` are logged every
step its pose is recoverable at any time.

---

## Categories

**A — `stacking_blocks_ambiguous`** (task 1, 240 episodes). One instruction, always:
`"Stack the blocks from the black mat to the white mat at the cross position."`
Blocks red, green, blue. 40 episodes for each of the 6 stacking orders, assigned by
`ORDERS[(episode_id - 1) % 6]` so a partial run stays balanced to within one episode per
order. The order appears only in `executed_order`, never in the instruction.

**B — `stacking_blocks_ordered`** (tasks 1–4, 60 episodes each). Instruction:
`"Stack the blocks from the black mat to the white mat at the cross position, with order
from bottom to top of {c1}, {c2}, {c3} blocks."`
Trained: red-green-blue, blue-green-red, green-red-blue, red-blue-green.
Held out with zero episodes, recorded in `scene.yaml`: blue-red-green, green-blue-red.

**C — `stacking_blocks_decomposed`** (tasks 1–8), colours red/green/blue/yellow.
Tasks 1–4 (T-MAT, one per colour, 60 episodes each):
`"Place the {color} block on the white mat at the cross position."` No stack; 2–4 blocks
on the black mat, a random subset always including the target.
Tasks 5–8 (T-TOP, one per colour, 120 episodes each):
`"Place the {color} block on top of the stack at the cross position."` A pre-built stack
of height h ∈ {1,2,3} sits on the cross, assigned round-robin by `(episode_id - 1) % 3`
for 40 episodes per height; its colours are drawn without replacement from the three
non-target colours, and 0–2 of the leftovers join the target on the black mat. With four
colours, h = 3 consumes every non-target colour, so a height-3 episode necessarily has
**zero** distractors — the number available is exactly `3 - h`.

A pre-built stack is spawned at the cross with yaw 0, settled for 0.5 s and checked
(upright, on the cross, at rest) *before* the recorder is created; a stack that collapses
on spawn costs no frames and the episode is resampled with a derived seed.

---

## validate.py

Per task: episode count against budget, and the fast integrity check on every file.
On sampled episodes (`--samples`, default 3): dataset shapes, dtypes, chunking and
compression; strictly increasing proprio timestamps exactly one physics step apart;
`/actions/t == /proprio/t`; every camera timestamp within one physics step of a proprio
timestamp; `success == True`; `quat_convention == "wxyz"`; the number of gripper
close events equal to the number of placements in `executed_order`; and `executed_order`
a subset of `blocks_present`. Balance invariants: Category A 40 episodes per order,
Category C T-TOP 40 per stack height. Then three contact sheets per task (first / middle /
last frame, RGB and colorized depth for both cameras) under
`<category>/contact_sheets/`, and a summary table with per-episode sim duration, step and
frame counts, measured camera rate, and megabytes per episode.

Exits non-zero if anything fails.
