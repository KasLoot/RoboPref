# RoboPref / PrefMem: Current Project Context

Use this document as the current project description when starting a new AI
session. The workspace implementation is the source of truth if this summary
and the code ever diverge.

## My background

I am a Master's student in Robotics and AI. My graduate thesis investigates an
upper-level agentic system that can understand a user's physical-task request,
resolve preference ambiguity, maintain an inspectable goal and execution state,
and coordinate a lower-level Vision-Language-Action (VLA) robot.

## Vanilla VLA baseline

The baseline robot is a conventional closed-loop VLA policy:

1. The user supplies a natural-language instruction.
2. The model processes that instruction together with the current camera frame.
   A vision-language backbone builds a multimodal representation, which is
   coupled through cross-attention to a diffusion-transformer (DiT) action head.
3. The DiT action head generates continuous robot actions or action chunks.
   New camera frames replace the previous observation as execution proceeds, so
   the policy repeatedly conditions its next actions on the latest view.

This is effective for direct, well-specified commands, but the policy alone does
not provide a strong upper-level mechanism for clarifying vague intent,
remembering user preferences with consent, obtaining confirmation of an exact
goal, exposing an auditable task state, recovering from task-level failures, or
validating the complete high-level outcome independently of local action
success.

## Thesis motivation

My project, PrefMem, adds a preference-aware agentic layer above the physical
executor. The research motivation is to make long-horizon human-robot
interaction more aligned, inspectable, and recoverable:

- Ambiguous instructions such as "tidy the table" should be clarified or
  grounded in an applicable saved preference rather than converted directly
  into motion.
- The user should confirm the exact physical goal, constraints, and expected
  final state before execution begins.
- Planning should respond to the current scene after every terminal task result
  instead of executing a stale, fully precomputed action list.
- Local sub-task success should not be treated as proof that the whole user goal
  is complete.
- Preference memory should be persistent but consent-controlled; accepting a
  task proposal must not silently create a permanent preference.
- Model outputs should be fenced by deterministic contracts, trusted IDs,
  state transitions, temporal evidence requirements, and bounded recovery
  loops.

PrefMem is currently a closed-loop task orchestration and visual-evidence
system. It should not be described as an online self-supervised learning system:
the present implementation does not update model weights or learn a policy from
its own rollouts.

## Current implemented architecture

The original concept had six agents, including a Summarising Agent and a VLA
Agent. The current implementation instead has five active model-facing agents:

1. **HRI Agent** — the only user-facing agent. It receives the user message, a
   current frame, conversation context, and authoritative runtime state. It
   clarifies intent, uses preference memory, presents a goal proposal, obtains
   exact confirmation, and exposes only safe runtime controls.
2. **Planner Agent** — a stateless, scene-grounded VLM boundary. Before
   confirmation it produces a non-executable goal preview. During execution it
   performs task-level receding-horizon planning from the frozen goal, a fresh
   frame, the planning trigger, terminal execution history, and optional
   operator guidance.
3. **Memory Agent** — retrieves and explicitly mutates persistent user
   preferences. Text records are stored in `preference.json`; normalized
   EmbeddingGemma vectors are stored in `preference.npy` for semantic retrieval.
4. **Monitor Agent** — continuously evaluates the single active step against
   its expected visual observations using post-publication frames. It reports
   `ONGOING`, `SUCCESS`, or `FAIL`, criterion states, visible observations, and
   an immediate emergency flag.
5. **Validator Agent** — compiles and freezes a detailed visual checklist at
   confirmation, then independently evaluates holistic goal completion when
   final validation is requested. It can accumulate evidence across multiple
   camera views.

Two deterministic host components are equally important but are not language
agents:

- `PrefMemRuntime` owns model/service composition, fresh-frame capture, task and
  display publication, observation-service routing, notifications, and the
  shared emergency coordinator.
- `RecedingHorizonController` is the authoritative state machine. It owns the
  frozen goal, cycle number, current publication, terminal history, evidence
  streaks, attention state, trusted identifiers, and loop guards. It performs
  no model calls or I/O.

The webcam server supplies the live stream and snapshots and exposes the
single-slot `/api/task` display API. PrefMem can be operated through a terminal
front end or a NiceGUI operator console.

## Implemented workflow

1. **Understand and clarify.** The HRI Agent receives the user's request, a
   current camera frame, and `PREFMEM_RUNTIME_STATE`. An explicit current
   instruction overrides stored preferences. The HRI may retrieve applicable
   preference memory. Any available conversation history is evidence for a
   question, not authorization for a persistent preference or a new task
   default. If the intended physical outcome or an important constraint is
   genuinely ambiguous, HRI asks one focused clarification.

2. **Create a goal preview.** Once the request is sufficiently clear, HRI calls
   `request_goal_preview`. The runtime captures a fresh frame and the Planner
   returns `READY`, `ALREADY_SATISFIED`, or `BLOCKED`. A non-blocked proposal
   contains the precise goal, explicit constraints, and observable final
   outcomes. `READY` also includes a short nominal task outline;
   `ALREADY_SATISFIED` includes no nominal tasks. Any outline explains the
   strategy but is not an executable queue.

3. **Obtain exact confirmation.** HRI presents the proposal and asks the user to
   confirm its host-generated `goal_id` and `revision`. A rejection does not
   count as an execution failure; the proposal remains available for discussion
   and a replacement preview invalidates it. No physical work is authorized
   before exact confirmation.

4. **Freeze validation and start a fresh cycle.** On confirmation, the runtime
   captures a confirmation frame. Validator expands every broad final outcome
   into detailed visible criteria and freezes a `ValidationContract`. Checklist
   compilation must succeed before execution starts. The runtime then captures
   a newer post-compilation frame and starts planning with the immutable
   `GoalContract`.

5. **Plan with an execution horizon of one.** On each cycle, Planner returns one
   of `ACT`, `REQUEST_FINAL_VALIDATION`, `BLOCKED`, or `NEEDS_USER_INPUT`. `ACT`
   may predict one to three candidate tasks, but the controller selects and
   publishes only `candidate_tasks[0]`; the remaining prediction tail is
   discarded. Host code assigns the cycle, step, criterion, and publication
   identities.

6. **Execute and monitor one step.** The camera page shows one current physical
   instruction and its expected observations. In the current prototype, a human
   watching that page performs the manipulation. Monitor consumes only newer,
   post-publication frames. The controller rejects stale or mismatched evidence
   and requires repeated terminal observations; by default, success needs two
   confirmations spanning at least two seconds and failure needs two
   confirmations.

7. **Replan after every terminal step.** A stable success or failure becomes an
   immutable execution-history record. Failure records preserve both the visible
   observation and reason. The runtime captures a fresh frame and calls Planner
   again toward the same frozen high-level goal. A task failure never weakens or
   rewrites the goal to make the run appear successful.

8. **Validate the entire goal.** Planner requests final validation only when the
   current scene appears ready. The runtime creates a `FINAL_VALIDATION`
   publication and routes it to Validator, not Monitor. Validator assesses the
   frozen detailed checklist and accumulates criterion evidence across views.
   Host code aggregates it to the original broad outcomes and deterministically
   derives the result:

   - all broad items `MET` -> `COMPLETE`;
   - any broad item `NOT_MET` -> `INCOMPLETE`;
   - otherwise, at least one `UNKNOWN` -> `NEEDS_EVIDENCE`.

   `NEEDS_EVIDENCE` asks the operator to move only the camera while leaving all
   scene objects unchanged. Stable `INCOMPLETE` is recorded as
   `FINAL_VALIDATION_FAIL` and triggers replanning. Stable `COMPLETE` is the only
   route to the terminal complete state. HRI reports one overall sentence and
   the full broad checklist; detailed internal criteria are not exposed as the
   ordinary user-facing report.

9. **Pause safely when automation cannot continue.** Camera/model/schema/service
   errors, no-progress timeouts, Planner blockers or questions, repeated-task
   guards, too many consecutive failures, and the maximum-cycle guard produce
   `NEEDS_ATTENTION`. These conditions are not silently converted into task
   failure. HRI reports the exact reason. If a current task exists, the operator
   may resume it with a new publication attempt; otherwise, or when a new route
   is wanted, HRI can request a fresh planning cycle with guidance. Replanning
   records an interrupted current attempt when applicable.

10. **Latch emergencies.** A valid `emergency_stop=true` output from Monitor or
    Validator, or the operator GUI's software-stop control, triggers one shared
    `EmergencyStopCoordinator`. It latches `EMERGENCY_STOPPED`, invokes
    `emergency_stop(reason)` once, stops the observation services, and prevents
    further publication in that process.

## Preference-memory policy

There is no Summarising Agent and no automatic `memory.md` update. HRI calls the
Memory Agent with a labeled `RETRIEVE REQUEST:` or `MUTATE REQUEST:`. Every
mutation first performs semantic retrieval and then makes at most one justified
remember, update, or forget operation. A write is permitted only when the user
uses direct future-facing language such as "remember this" or "make this my
default", explicitly requests an update/forget operation, or affirms a dedicated
memory-consent question. Confirming the current goal authorizes only that run
and never authorizes a persistent memory write. The retrieval-before-write and
mutation-target restrictions are enforced by Memory Agent code; the
future-facing consent rule is currently an HRI policy/tool-use constraint, not
a separate host-issued authorization token.

## Current boundary and intended VLA integration

The implemented workspace does **not yet call a lower-level VLA robot**. The
human following the camera-page instruction is the current physical executor
and provides an observable stand-in for that downstream component. The planned
research integration is to replace this human execution boundary with a VLA
adapter that accepts the controller's one current task while preserving the
same goal contracts, monitoring, replanning, validation, and emergency fencing.

The shared `emergency_stop()` function is currently a logging-only integration
hook. The GUI stop is a software latch, not a safety-rated or hardware emergency
stop. A real robot deployment requires a robot-specific, acknowledged stop API
and an independent hardware safety system.

## Useful source-of-truth files

- `README.md` — system diagrams, runtime behavior, and operating instructions.
- `src/prefmem/runtime.py` — orchestration and service routing.
- `src/prefmem/controller.py` — authoritative receding-horizon state machine.
- `src/prefmem/contracts.py` — typed goal, planning, publication, monitoring,
  and validation contracts.
- `src/prefmem/agents/hri.py` — user-facing graph and safe runtime tools.
- `src/prefmem/agents/planner.py` — preview and cycle model boundary.
- `src/prefmem/agents/memory.py` — preference retrieval and mutation store.
- `src/prefmem/agents/monitor.py` — per-task visual observation service.
- `src/prefmem/agents/validator.py` — checklist compilation and final validation.
- `src/prefmem/task_publisher.py` and `src/prefmem/stream_camera.py` — single
  current-task display contract and webcam service.
- `src/ui/service.py` and `src/ui/ui.py` — local operator console.

When helping with this project, do not assume the obsolete six-agent workflow,
an automatic summarisation/memory step, a frozen stack of sub-tasks, direct VLA
execution, or completion inferred from sub-task count. Use the current code and
typed runtime state as the authority.
