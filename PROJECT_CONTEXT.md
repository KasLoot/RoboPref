# RoboPref / PrefMem: Brief Project Context

Use this as the session primer. See [PROJECT_CONTEXT_FULL.md](PROJECT_CONTEXT_FULL.md)
for the detailed architecture, policies, evidence status, and repository caveats.
The implementation and typed runtime state are authoritative if the documents
ever diverge from the code.

## Purpose and scope

I am a Master's Robotics and AI student. My thesis studies PrefMem, a
preference-aware layer that clarifies ambiguous requests, uses consent-controlled
memory, confirms exact physical goals, replans from observed outcomes, and
independently validates completion.

PrefMem orchestrates tasks and visual evidence; it does not learn from rollouts
or implement a learned VLA. Its optional MuJoCo surrogate uses Gemma 4 to compile
one instruction, SAM 3.1 plus RGB-D to ground it, and deterministic Panda
IK/position control to execute it.

## Implemented system

Five model-facing agents have distinct roles:

- **HRI:** sole user interface, clarification, proposal, confirmation, recovery.
- **Planner:** stateless previews and receding-horizon plans from fresh evidence.
- **Memory:** explicit retrieval/mutation of `preference.json` and
  EmbeddingGemma-backed `preference.npy`.
- **Monitor:** fenced evidence for the one active step.
- **Validator:** frozen checklist and independent whole-goal assessment.

Deterministic host code is equally important:

- `PrefMemRuntime` owns services, frames, publication, routing, and shutdown.
- `RecedingHorizonController` owns the frozen goal, trusted IDs, history,
  evidence, attention, and guards; it performs no model calls or I/O.
- `ExecutionService` fences the optional compiler/grounder/Panda controller.

Human mode uses the webcam/task page. MuJoCo atomically publishes a robot-hidden
`sam` RGB-D stream for grounding and a visible-robot `prefmem` stream for all
five upper-level agents. `TB6C-v2` uses distinct cameras; the legacy scene uses
one pose with different render visibility.

## Core workflow

1. HRI clarifies intent; an explicit current instruction overrides memory.
2. Planner returns a non-executable preview. Execution requires confirmation of
   its host-generated goal ID and revision.
3. Validator freezes its checklist and the confirmed `GoalContract`.
4. Every cycle uses a fresh frame. Planner may suggest three tasks, but the
   controller publishes only the first—execution horizon one.
5. Monitor accepts only fresh, matching, repeated evidence. Stable success or
   failure is recorded and triggers replanning toward the unchanged goal.
6. Only stable Validator `COMPLETE` finishes. `INCOMPLETE` replans;
   `NEEDS_EVIDENCE` requests another view without object movement.
7. Step timeouts default to bounded `AUTO_REPLAN`: retire the publication,
   record `INTERRUPTED/MONITOR_TIMEOUT`, capture a newer frame, and replan.
   Errors, blockers, final-validation timeouts, or exhausted guards enter
   `NEEDS_ATTENTION`.
8. Visual emergency output or the GUI stop latches `EMERGENCY_STOPPED`; the
   current hook is logging-only and not safety-rated.

## Non-negotiable policies

- Confirmation authorizes one run, never a preference write. Persistent writes
  require explicit future-facing consent or an update/forget request.
- Memory retrieves before mutation and makes at most one justified write. It
  defaults to top five per query, deterministic deduplication, a ten-item cap,
  and semantic filtering over capped records only.
- Failure or timeout never weakens the frozen goal or proves completion.
- MuJoCo's oracle grounding fallback is experiment-only and disabled by default.
  Assisted success never changes the strict result from `FAIL_GROUNDING`.
- Visual/software stops do not replace acknowledged robot and hardware safety.

## Evaluation and repository snapshot

- RoboPref/PrefMem is version `2.2.1` and requires Python 3.13+.
- Suite v2 fixes `TB6C-v2` and covers readiness through end-to-end evaluation.
- Status is `PROPOSED_NOT_FROZEN`; confirmatory execution is unauthorized.
  `CAL-X-03` and `CAL-X-04` remain unready; downstream implementation pins pass.
- Development evidence through 15 August 2026 records 28 `PASS` and 70
  `CAPABILITY_FAIL` gates. This is neither one model score nor confirmatory.
- `dataset/sim_datasets/` has 234 scripted two-frame fixtures, not rollouts.
  Manifests are scorer-only; prompts get the instruction, numbered images, and
  `BenchmarkEpisode.model_context()`.
- The package is correctly flattened at `experiments_suite_v2/`, but `.gitignore`
  still excludes it via `experiments*`; it is local-only despite tracked v2 tests.
- `README.md` lags camera, timeout, oracle, and suite behavior.

## Main source map

- Runtime/control/contracts: `src/prefmem/runtime.py`, `controller.py`,
  `contracts.py`
- Agents: `src/prefmem/agents/`
- Execution and simulation: `src/prefmem/execution/`, `src/simulation/`
- Human surfaces: `src/prefmem/stream_camera.py`, `task_publisher.py`, `src/ui/`
- Evaluation: local-only `experiments_suite_v2/`, tracked `dataset/`, and
  `tests/test_v2_*.py`

Do not assume the obsolete six-agent design, automatic summarisation, a frozen
action list, completion from subtask count, one shared MuJoCo image, a learned
VLA, oracle assistance as a strict pass, endpoint fixtures as rollouts, or a
frozen/confirmatory experiment programme.
