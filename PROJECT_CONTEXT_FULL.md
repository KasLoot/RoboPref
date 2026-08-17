# RoboPref / PrefMem: Full Project Context

Use this detailed document when a task needs the complete project description.
`PROJECT_CONTEXT.md` is the brief session primer. The workspace implementation
is the source of truth if either document and the code ever diverge.

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

## Current snapshot

- The installable project is RoboPref/PrefMem `2.2.1` and requires Python 3.13
  or newer.
- The production contribution is the preference-aware agentic wrapper and its
  deterministic control contracts, not a new VLA policy.
- The workspace also contains PrefMem Experiment Suite v2, a separate evaluation
  control plane with development evidence through 15 August 2026. Its protocol
  is still `PROPOSED_NOT_FROZEN`, so its completed development runs are not
  confirmatory evidence.
- `dataset/sim_datasets/` contains 234 generated two-frame MuJoCo endpoint
  packets across `block_stack`, `category_sort`, and `place_setting`. These are
  scripted counterfactual fixtures, not recorded robot rollouts; their manifests
  contain privileged scorer truth and must never be included in model prompts.

## Current implemented architecture

The production runtime has five active model-facing agents. It also has an
optional lower-level MuJoCo execution agent; this is a constrained Gemma
compiler plus deterministic perception/control pipeline, not a learned VLA.
The five upper-level agents share the configured conversational model and base
URL, while EmbeddingGemma and the execution compiler retain separately
configurable model endpoints:

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
   an immediate emergency flag. In MuJoCo terminal runs, a bounded read-only
   telemetry stream exposes each assessment, its controller disposition, and
   optional inference lifecycle/latency without participating in decisions.
5. **Validator Agent** — compiles and freezes a detailed visual checklist at
   confirmation, then independently evaluates holistic goal completion when
   final validation is requested. It can accumulate evidence across multiple
   camera views.

Three deterministic host components are equally important but are not language
agents:

- `PrefMemRuntime` owns model/service composition, fresh-frame capture, task and
  display publication, observation-service routing, notifications, bounded
  Monitor telemetry, and the shared emergency coordinator.
- `RecedingHorizonController` is the authoritative state machine. It owns the
  frozen goal, cycle number, current publication, terminal history, evidence
  streaks, attention state, trusted identifiers, and loop guards. It performs
  no model calls or I/O.
- `ExecutionService` is the optional, publication-fenced lower-level execution
  boundary. Gemma 4 converts only the current natural-language step into a
  strict symbolic pick/place program. SAM 3.1 supplies open-vocabulary masks;
  a dedicated synchronized MuJoCo RGB-D stream and calibrated camera geometry
  supply world-frame source and target points; a damped-least-squares IK and
  position controller drives the Franka Emika Panda through a fixed safe
  waypoint sequence.

In human mode, the webcam server supplies the live stream and snapshots and
exposes the single-slot `/api/task` display API. In MuJoCo mode, the simulation
atomically publishes synchronized `sam` and `prefmem` RGB-D streams plus an
in-memory display slot. Only `ExecutionService` and SAM consume the calibrated,
robot-hidden `sam` stream. HRI, Planner, Memory, Monitor, and Validator consume
the separate `prefmem` stream, which preserves the visible robot embodiment.
The canonical `TB6C-v2` experiment scene gives these streams distinct top-down
and third-person cameras; the legacy built-in stacking scene maps both names to
its single `task_camera` pose as a compatibility fallback while retaining
distinct render visibility. The interactive viewer is independent and defaults
to a freely navigable overview. The NiceGUI operator console currently targets
the human/webcam mode.

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

6. **Execute and monitor one step.** In `human` mode, the camera page shows one
   current physical instruction for the operator. In `mujoco` mode, the same
   `PublishedTask` is sent to `ExecutionService` and Monitor. The executor
   compiles, grounds, and controls one Panda pick/place action using the `sam`
   RGB-D stream, then parks the robot and waits for the manipulated object to
   settle. Monitor observes the separate `prefmem` stream and consumes only
   newer, post-publication frames; a terminal Monitor judgment is ignored until
   robot motion has settled. The controller rejects stale or mismatched evidence
   and requires repeated terminal observations; by default, success needs two
   confirmations spanning at least two seconds and failure needs two.

   Open-category goals can freeze a `DynamicObjectScope` while keeping its
   membership live. For “Stack the blocks,” every matching block in the robot
   workspace at final validation belongs to the goal, including blocks dragged
   from the out-of-view `items_area` after confirmation. A confirmed new-object
   detection during motion requests cancellation; the Panda reaches a safe hold
   before the runtime replans. A fallen stack is detected through ordinary
   Monitor/Validator evidence and likewise causes replanning.

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

9. **Apply the configured no-progress policy.** The default `AUTO_REPLAN` policy
   applies only to an ordinary `STEP`; when a lower-level executor exists, the
   runtime defers the visual timeout while that executor is still preparing or
   moving. On timeout, the runtime retires the exact publication, appends an
   `INTERRUPTED` history record with a typed `MONITOR_TIMEOUT` termination
   context and the last accepted evidence, captures a strictly newer frame, and
   starts another Planner cycle. Automatic timeout replans are bounded by
   consecutive and per-normalized-instruction guards, both defaulting to two.
   Final-validation timeouts, the `ATTENTION_GATE` ablation, exhausted timeout
   guards, or any failure while retiring/capturing/replanning enter
   `NEEDS_ATTENTION`; a timeout is never fabricated as a Monitor `FAIL`.

10. **Pause safely when automation cannot continue.** Camera/model/schema/service
    errors, Planner blockers or questions, repeated-task guards, too many
    consecutive failures, final-validation evidence timeouts, and cycle or
    timeout-replan guards produce `NEEDS_ATTENTION`. HRI reports the exact
    reason. If a current task exists, the operator may resume it with a new
    publication attempt; otherwise, or when a new route is wanted, HRI can
    request a fresh planning cycle with guidance. Operator-requested replanning
    records an interrupted current attempt when applicable.

11. **Latch emergencies.** A valid `emergency_stop=true` output from Monitor or
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

Retrieval is a two-stage pipeline. The host embeds each generated query,
retrieves the default top five candidates per query, deterministically ranks and
deduplicates them, and caps the merged set at ten. The Memory model then performs
the semantic applicability filter, but its validated response may only return
unchanged records from that capped set. Optional trace sinks expose query-level
scores, tie-breaking provenance, the pre-cap and post-cap sets, semantic labels,
final IDs, and timing without changing the Memory response or write policy.

## Experimentation and evidence status

PrefMem Experiment Suite v2 evaluates the wrapper rather than training a robot
policy. It fixes the `TB6C-v2` MuJoCo scene (Panda, white and cyan boards, and
six coloured cubes) and separates five layers of evidence: `PF` readiness,
`CAL-X` camera/grounding/control decomposition, isolated `C-*` components,
interfaces and the cumulative workflow ladder, and finally ablations plus
end-to-end cases. A trial's immutable identity includes the case, condition,
scene/reset hashes, language and context, model/decoding configuration,
prompts/tools, seed, source version, and artifact profile.

The current protocol metadata reports:

- suite version `PrefMem-Experiment-Suite-v2`;
- protocol and registry status `PROPOSED_NOT_FROZEN`;
- approved camera previews and frozen service identities;
- metadata preflight `PASS`, but `confirmatory_execution_authorized=false`;
- current campaign-readiness failures at `CAL-X-03` and `CAL-X-04`;
- downstream-continuation implementation-pin readiness `PASS` after flattening
  the suite to its intended repository-root package location;
- an additional protocol-freeze requirement to verify a distinct small,
  production-relevant VLM before `AB-MEM-F` begins;
- a preregistered minimum of 22,600 valid primary trials, excluding readiness
  attempts, 2,015 calibration observations/placements, conditional workflow
  rows, and separately declared semantic-filter audits.

The results catalogue has terminal development evidence for all 98 currently
registered experiment gates as of 15 August 2026: 28 `PASS` and 70
`CAPABILITY_FAIL`. That ratio is not a single model score because it mixes
readiness gates, deterministic grids, component contracts, ablations, and
all-five end-to-end case gates. The defensible current reading is that host
invariants, camera/reset/evidence infrastructure, several isolated contracts,
and selected recovery behaviours are strong, while Memory integration, final
validation/closure, preference robustness, the pause-before-replan seam, and
parts of the SAM/execution surrogate remain demonstrated limitations. These
runs remain development or otherwise non-confirmatory wherever their sealed
result metadata says so.

Experiment outcomes follow a strict audit policy. Only `PASS` and
`CAPABILITY_FAIL` enter the capability denominator. A valid capability failure
is retained and cannot be replaced by a passing retry; `INVALID_SETUP` and
`INVALID_RUN` are also retained but may receive an exact-tuple sibling attempt.
Artifacts are written under canonical attempt IDs, finalized with manifests and
checksums, and audited without rewriting sealed evidence.

MuJoCo experiments may explicitly enable
`--experiment-oracle-grounding-fallback`. The production SAM path is always
attempted first, and only a typed post-call `GroundingError` can use simulator
truth; service outages and malformed responses are not eligible. Even when the
assisted downstream motion succeeds, the publication retains strict
`FAIL_GROUNDING` and reports the assisted continuation separately. A plausible
but wrong non-empty SAM grounding is not silently corrected.

Optional observability seams support the suite: shared model metrics can emit
prompt/response/usage traces, Memory emits retrieval and semantic-filter traces,
and the execution compiler, grounding pipeline, IK/controller, contacts, poses,
and perception artifacts can be recorded. Logging callbacks are best-effort and
must not change controller decisions or interrupt physical control.

## Current boundary and intended VLA integration

The workspace has two execution adapters: the original human camera-page path
and a MuJoCo Franka Panda path. The MuJoCo adapter is deliberately modular and
does **not** claim to be a real VLA. It uses Gemma 4 for constrained semantic
compilation, SAM 3.1 plus RGB-D for grounding, and a hand-designed IK/position
controller for motion. This provides an executable research workaround while
preserving the clean `PublishedTask` boundary. A future learned VLA can replace
`ExecutionService`'s compiler/grounder/controller adapter without changing the
Planner's output, frozen goal contracts, monitoring, replanning, validation, or
emergency fencing. Simulator-oracle grounding is an experiment-only opt-in and
is disabled on the ordinary strict execution path; it is not part of the
intended production adapter.

The shared `emergency_stop()` function is currently a logging-only integration
hook. The GUI stop is a software latch, not a safety-rated or hardware emergency
stop. A real robot deployment requires a robot-specific, acknowledged stop API
and an independent hardware safety system.

## Repository and reproducibility caveats

- The local `experiments_suite_v2/` tree is currently excluded by the broad
  `.gitignore` rule `experiments*`; `git ls-files experiments_suite_v2` returns
  no files. Tracked `test_v2_*` modules import that package, so the current Git
  revision alone does not contain everything needed to reproduce those tests or
  the results catalogue. Do not delete or rewrite the local suite as generated
  output, but do not claim it is version-controlled either.
- The package now lives directly at `experiments_suite_v2/`, matching its
  repository-root path calculations. With `src/` importable (or RoboPref
  installed), the documented repository-root suite commands, asset preflight,
  and implementation-pin checks resolve correctly. Do not reintroduce the
  former extra `experiments_suite_v2/experiments_suite_v2/` nesting.
- `README.md` still describes the older single-camera routing and treats all
  no-progress timeouts as attention gates. For dual-camera routing,
  `AUTO_REPLAN`, experiment-only oracle assistance, and suite status, prefer the
  implementation and this context file.
- Benchmark `manifest.json` files contain targets, expected outcomes, simulator
  state, and scoring evidence. Agent-facing code must use
  `BenchmarkEpisode.model_context()`, the human instruction, and numbered image
  frames only; never pass the manifest, filesystem path, or opaque scenario ID
  into an agent prompt.

## Useful source-of-truth files

- `README.md` — production overview and operating instructions, subject to the
  caveats above.
- `pyproject.toml` and `src/prefmem/cli.py` — package version, dependencies,
  command-line defaults, model endpoints, timeout policy, and executor options.
- `src/prefmem/runtime.py` — orchestration and service routing.
- `src/prefmem/controller.py` — authoritative receding-horizon state machine.
- `src/prefmem/contracts.py` — typed goal, planning, publication, monitoring,
  and validation contracts.
- `src/prefmem/agents/hri.py` — user-facing graph and safe runtime tools.
- `src/prefmem/agents/planner.py` — preview and cycle model boundary.
- `src/prefmem/agents/memory.py` — preference retrieval and mutation store.
- `src/prefmem/agents/monitor.py` — per-task visual observation service.
- `src/prefmem/agents/validator.py` — checklist compilation and final validation.
- `src/prefmem/agents/metrics.py` — optional model-call metrics and trace sink.
- `src/prefmem/execution/` — strict Gemma compiler, SAM client, calibrated
  RGB-D grounding, explicit experiment fallback contracts, execution traces,
  and publication-fenced service.
- `src/simulation/stacking_scene.xml`, `stacking.py`, `controller.py`, and
  `oracle_grounding.py` — legacy block scene, synchronized dual-stream camera
  plumbing, Panda IK/control, and the experiment-only hidden-state provider.
- `src/simulation/robots/franka_emika_panda/` — canonical Panda XML, meshes,
  and licenses used by current simulation scenes.
- `src/prefmem/task_publisher.py` and `src/prefmem/stream_camera.py` — single
  current-task display contract and webcam service.
- `src/ui/service.py` and `src/ui/ui.py` — local operator console.
- `dataset/episode.py`, `dataset/benchmark.py`, and `dataset/sim_datasets/` —
  offline endpoint packets and the enforced model/oracle boundary.
- Local-only `experiments_suite_v2/README.md`, its
  `protocol/experiment_registry.json`, and its `results/EXPERIMENT_REPORT.md` —
  suite control plane, current freeze state, and development evidence catalogue.
- `tests/test_v2_*.py` and `tests/test_experiments_suite_v2_scene.py` — experiment
  infrastructure, runner, storage, scene, and integration contract coverage.

When helping with this project, do not assume the obsolete six-agent workflow,
an automatic summarisation/memory step, a frozen stack of sub-tasks, a learned
VLA policy, completion inferred from sub-task count, a single shared MuJoCo
camera image, or that every timeout immediately needs an operator. Do not count
an oracle-assisted continuation as a strict system pass, treat scripted endpoint
packets as robot rollouts, or describe the current experiment programme as
confirmatory/frozen. Use the current code, typed runtime state, and sealed
per-attempt metadata as the authority.
