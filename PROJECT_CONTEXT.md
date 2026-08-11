# RoboPref / PrefMem: Compact Project Context (v2.2.1)

Use this document to orient a new AI session. See
[`PROJECT_CONTEXT_FULL.md`](PROJECT_CONTEXT_FULL.md) for the detailed narrative.
The workspace implementation is the source of truth if either document diverges
from the code.

## Purpose and scope

PrefMem is a preference-aware agentic layer above a physical-task executor. It
clarifies ambiguous intent, retrieves consented preferences, obtains exact goal
confirmation, exposes an inspectable task state, replans after terminal task
results, and validates the complete goal independently of local action success.
It is a closed-loop orchestration and visual-evidence system, **not** an online
self-supervised learner: it does not update model weights or learn a policy from
rollouts.

The default `--executor human` path publishes one instruction at a time to a
human-operated webcam page. `--executor mujoco` instead runs a simulated Franka
Emika Panda using Gemma 4 semantic compilation, SAM 3.1 segmentation,
synchronized RGB-D grounding, and deterministic motion control. This simulation
adapter is not a learned VLA. `--simulation-scene PATH` may substitute a
compatible XML scene, but it is not a generic robot interface: the scene must
preserve the Panda joints, actuators, gripper site, `task_camera`, and named-block
structure expected by the stacking environment and controller.

## Architecture

Five upper-level agents are active in both modes:

- **HRI Agent:** the only user-facing agent; clarifies intent, applies memory,
  proposes goals, obtains exact confirmation, and exposes safe runtime controls.
- **Planner Agent:** a stateless, scene-grounded VLM boundary; produces
  non-executable previews before confirmation and receding-horizon decisions
  during execution.
- **Memory Agent:** retrieves and explicitly mutates preferences stored as text
  in `preference.json` and normalized EmbeddingGemma vectors in
  `preference.npy`.
- **Monitor Agent:** evaluates the one active step from post-publication frames
  and reports `ONGOING`, `SUCCESS`, or `FAIL`, criterion evidence, and an
  emergency flag.
- **Validator Agent:** freezes a detailed visual checklist at confirmation and
  independently evaluates holistic goal completion, accumulating evidence
  across views.

MuJoCo adds a lower-level **Execution Agent**. It accepts only the current,
publication-fenced `STEP`; Gemma compiles the instruction to a host-validated
symbolic pick/place program, SAM and RGB-D ground it, and a deterministic Panda
controller executes it. Gemma never supplies joint angles, actuator values,
pixels, or absolute world coordinates. Monitor—not the executor—decides task
success or failure.

Three deterministic host components enforce the model boundaries:

- `PrefMemRuntime` composes services, captures fresh frames, routes publications
  and observations, emits bounded telemetry, and owns the shared emergency
  coordinator.
- `RecedingHorizonController` is the authoritative, model-free state machine. It
  owns the frozen goal, cycles, current publication, terminal history, evidence
  streaks, trusted IDs, attention state, and loop guards.
- `ExecutionService` owns the single pending/active MuJoCo job, cancellation,
  and safe hold. Its current skill is strict top-down pick/place with a fixed
  six-stage waypoint sequence.

Human mode uses the webcam server and single-slot `/api/task` display. MuJoCo
uses a synchronized RGB-D source and in-memory display slot. The terminal UI
supports MuJoCo; the NiceGUI console currently targets human/webcam mode.

## Authoritative workflow and invariants

1. HRI receives the request, a current frame, conversation context, and trusted
   runtime state. An explicit current instruction overrides stored preferences.
   Genuine ambiguity produces one focused clarification.
2. HRI requests a fresh-frame goal preview. Planner returns `READY`,
   `ALREADY_SATISFIED`, or `BLOCKED`, with a precise goal, constraints, and
   observable outcomes. A nominal outline is explanatory, never an executable
   queue.
3. Execution requires exact confirmation of the host-generated `goal_id` and
   `revision`. Rejection is not an execution failure; a replacement preview
   invalidates the old one.
4. Confirmation freezes an immutable `GoalContract`. Validator must compile and
   freeze its detailed `ValidationContract` before a newer frame starts the
   first planning cycle.
5. Each cycle yields `ACT`, `REQUEST_FINAL_VALIDATION`, `BLOCKED`, or
   `NEEDS_USER_INPUT`. `ACT` may contain up to three predictions, but the host
   publishes only the first and discards the tail. The execution horizon is
   always one; host code assigns all trusted cycle, step, criterion, and
   publication IDs.
6. Monitor accepts only newer, matching post-publication evidence. A terminal
   MuJoCo judgment waits for settled motion. Stable terminal results require
   repetition: by default, two success observations spanning at least two
   seconds, or two failure observations. Stable success or failure becomes
   immutable history and triggers fresh-frame replanning toward the same goal;
   failure never weakens the goal.
7. Open-category goals may use a frozen `DynamicObjectScope` with live
   membership. Confirmed new-object detection during motion requests
   cancellation, safe hold, and replanning. Fallen stacks and other scene
   changes use the ordinary Monitor/Validator evidence path.
8. Final validation is a separate publication routed only to Validator. Host
   aggregation maps detailed evidence to broad outcomes: all `MET` means
   `COMPLETE`; any `NOT_MET` means `INCOMPLETE`; otherwise an `UNKNOWN` means
   `NEEDS_EVIDENCE`. Evidence requests permit camera movement only. Stable
   incompleteness triggers replanning; stable completeness is the sole terminal
   success route.
9. Model, camera, schema, or service errors; planner blockers/questions;
   no-progress, repetition, failure, or cycle guards produce `NEEDS_ATTENTION`,
   not a fabricated task failure. The operator may resume the current task with
   a new publication attempt or request replanning with guidance.
10. Monitor/Validator emergency output or the GUI software stop latches the
    shared `EMERGENCY_STOPPED` state, invokes `emergency_stop(reason)` once,
    stops observation services, and prevents further publication.

## Preference-memory policy

There is no Summarising Agent or automatic `memory.md` update. HRI labels Memory
calls as `RETRIEVE REQUEST:` or `MUTATE REQUEST:`. Every mutation performs
semantic retrieval first and makes at most one justified remember, update, or
forget operation. Writes require direct future-facing consent (for example,
“remember this” or “make this my default”), an explicit update/forget request,
or affirmation of a dedicated memory-consent question. Goal confirmation
authorizes only the current run and never a persistent write. Retrieval and
mutation-target restrictions are enforced in code; future-facing consent is
currently an HRI policy/tool-use constraint, not a host-issued token.


## Boundaries and source of truth

The MuJoCo adapter preserves a clean `PublishedTask` boundary so a future
learned VLA can replace the compiler/grounder/controller without changing goal
contracts, planning, monitoring, validation, or emergency fencing. The current
`emergency_stop()` is a logging-only integration hook, and the GUI stop is not
safety-rated. Real deployment requires a robot-specific acknowledged stop API
and independent hardware safety.

Start with `README.md`, then consult:

- `src/prefmem/runtime.py`, `controller.py`, and `contracts.py` for orchestration,
  authoritative state transitions, and typed contracts;
- `src/prefmem/agents/` for HRI, Planner, Memory, Monitor, and Validator;
- `src/prefmem/execution/` for the fenced MuJoCo execution boundary;
- `src/simulation/stacking_scene.xml`, `stacking.py`, and `controller.py` for the
  simulated scene, camera, Panda IK, and motion control;
- `src/prefmem/cli.py`, `task_publisher.py`, and `stream_camera.py` for backend
  composition and human-mode publication;
- `src/ui/` for the operator console; and
- `experiments/scenarios.json` plus `experiments/harness/` for scenario and
  oracle definitions.

Do not describe the obsolete design with Summarising and VLA agents. Do not
assume automatic memory updates, a frozen sub-task queue, a learned VLA,
completion inferred from sub-task count, or a generic robot-model interface.
Use current code and typed runtime state as authority.
