# PrefMem System Description V2

**Status:** implementation-grounded description of the current RoboPref workspace

**Scope:** the upper-level preference-aware orchestration system and its boundary with a lower-level VLA robot

## 1. Purpose and implementation scope

PrefMem is an upper-level Human–Robot Interaction (HRI) and task-assurance system for
preference-aware tabletop manipulation. It accepts a natural-language command and a
scene observation, resolves the user's intended outcome, retrieves relevant prior
context, plans robot-level subtasks, dispatches them through a VLA execution interface,
checks the resulting state, and may perform bounded re-observation or replanning for
supported recoverability outcomes.

The implementation separates two questions that the original concept treated too
similarly:

1. **What should the robot do for the current task?**
2. **What should the robot remember as a default for future tasks?**

A current-task choice does not by itself authorize a durable preference. Normal
terminal episodes are recorded as episodic history, while an explicitly future-scoped
memory request may be committed earlier in the same episode. Under the implemented HRI
policy, durable authority comes from direct future language such as “remember this” or
from a reply that HRI classifies as confirmation of a dedicated memory-consent
question. Direct requests receive an independent entailment check; replies to pending
memory questions receive structural prompt/action binding but no second semantic
affirmation check.

PrefMem does not train or fine-tune any model online. “Learning” in the current system
means updating structured, consent-approved preference records. The repository also
does not implement the lower-level VLA's VLM, cross-attention, diffusion action head, or
continuous control loop. Those remain below the `VLAExecutor` boundary.

## 2. Implemented architecture

The current implementation is best described as four model-backed agents, one VLA
execution boundary, and one deterministic assurance service. HRI, Planner, and
Validator can receive images; Memory uses the same JSON-model abstraction for
text-and-metadata semantic reasoning without image inputs.

```text
                                  +-----------------------------+
                                  | History and Preference JSON |
                                  +--------------+--------------+
                                                 ^
                                                 |
User + initial observation <-> HRI Orchestrator <-> Memory Agent
                                  |
                                  v
                            Planner Agent
                                  |
                           deterministic plan gate
                                  |
                                  v
                             VLA Executor
                                  |
                         deterministic execution gate
                                  |
                         final/re-observed image
                                  |
                                  v
                            Validator Agent
                                  |
                        deterministic validation gate
                                  |
               success | re-observe | replan | help | safety abort
                                  |
                                  v
                     terminal episodic-history update
```

The former standalone Summarising Agent is no longer a separate component.
Episode summarisation and rolling-history compaction are operations of the Memory
Agent. Task Assurance is an additional non-generative component that applies
deterministic confidence, structural, precondition, execution-status, goal-coverage,
recovery-budget, and safety gates. It prevents unchecked outputs from being dispatched
or treated as proof of success, but it does not independently prove that Planner's
subtask semantics are correct.

| Component | Implemented responsibility |
|---|---|
| **HRI Orchestrator** | The only user-facing component. It manages dialogue, typed pending questions, task contracts, component calls, recovery, terminal reporting, and the safety latch. |
| **Memory Agent** | Semantically retrieves history and approved preferences, summarises terminal episodes, proposes preference questions and transactions, and proposes lineage-preserving preference compaction. |
| **Planner Agent** | Converts a resolved task contract and the planning attempt's observation image into ordered VLA subtasks, preconditions, and one structured validation specification. |
| **VLA Executor** | Provides the boundary to recorded or live lower-level execution. It does not decide user intent, memory, or task completion. |
| **Validator Agent** | Independently checks the observed post-execution state against the Planner's frozen validation specification. |
| **Task Assurance** | Applies deterministic confidence, precondition, subtask-structure, execution-status, validation-coverage, recovery-budget, and safety gates. Detailed validation-schema parsing belongs to the Planner contract adapter. |

All orchestration is synchronous and owned by HRI. Planner calls, VLA subtasks,
validation, preference proposals, and history updates happen serially. There is no
concurrent “execute the first subtask while summarising” path in the current code.
Model and executor calls block `handle_user_message()`, so the interactive CLI cannot
accept a cancellation while a task call is running. Model calls have individual
timeouts, but there is no global command deadline.

## 3. Inputs, state, and inter-component contracts

### 3.1 Runtime inputs

The interactive prototype starts from:

- a user utterance;
- a participant namespace (`user_id`);
- a recorded episode directory containing at least two numerically named images;
- persistent history, preference, and history-outbox paths; and
- model, vision, confidence, and recovery configuration.

For a normal recorded episode, the numerically first image is the initial observation
and the numerically last image is the final observation. The HRI Agent receives the
initial image only on a new command. Planner receives the observation used for the
current planning attempt, while Validator receives the executor's final or refreshed
observation.

### 3.2 Active command state

For each command, HRI maintains:

- a session ID and episode ID;
- the initial user request and task-local visible dialogue;
- the memory context retrieved at command start;
- the current typed pending question, if any;
- memory events produced during the episode;
- the terminal task result while a post-task memory question is pending; and
- a process-local safety latch after an unsafe execution.

The raw dialogue is not used as unbounded cross-task memory. A new command starts with
a fresh task-local dialogue and a newly retrieved bounded memory context.

### 3.3 Typed HRI modes and questions

The HRI model must return one structured mode:

| Mode | Meaning |
|---|---|
| `ASK` | Ask an outcome-changing task clarification. |
| `CONFIRM` | Confirm a fully structured proposed task for the current episode. |
| `MEMORY_CONFIRM` | Ask a dedicated question about saving or deleting a future preference. |
| `EXECUTE` | Dispatch a complete task contract. |
| `REPORT` | Report a terminal non-execution outcome or an assured task result. |
| `RESTART` | Cancel the pending interaction and reinterpret the same utterance as a new command. |

The corresponding pending-question kinds are `TASK_CLARIFICATION`,
`TASK_CONFIRMATION`, and `MEMORY_CONSENT`. Their separation is a safety boundary: a
reply such as “yes” to a proposed action can authorize only the current task, never a
durable memory write.

HRI validates every model response and retries once with schema-repair instructions if
the response violates its contract. If the second response is also invalid, the turn
fails closed and no task is dispatched.

### 3.4 Task contract

Before execution, HRI normalises the resolved request into a task contract containing:

- a schema, task, episode, and user identity;
- a self-contained `confirmed_intent`;
- a list of relevant semantic object names and structured parameters;
- approved preference references, if used; and
- the turn that approved or directly specified the current task.

These are the fields enforced by host code. The HRI prompt also asks the model to
include a task type, but the current normaliser does not require it.

Only active, semantically retrieved `MATCH` preferences above the configured threshold
may be cited without another confirmation. One high-confidence `CONFLICT` disables
automatic citation of every retrieved preference for that task. HRI is instructed to
cite preferences it uses; the host validates any supplied IDs but does not require the
model to cite a preference whenever its chosen task semantics happen to agree with
memory.

The HRI policy instructs the model to let a current explicit instruction override
memory for the current task without altering the stored default. The host enforces
allowed preference IDs, but it does not independently prove this semantic precedence.

## 4. End-to-end command workflow

### 4.1 Start a command and retrieve memory

When a new command arrives, HRI:

1. retries any terminal-history checkpoints or failed history updates in the durable
   outbox;
2. sends the utterance, user ID, and opaque scene metadata to the Memory Agent;
3. receives a rolling history summary, semantically relevant prior episodes,
   semantically relevant approved preferences, repository versions, and warnings;
4. creates a new episode and fresh task-local dialogue; and
5. calls the HRI model with the memory context, scene image, and exact user message.

Memory is retrieved once at command start. Clarification turns continue with the same
retrieved context and the task-local dialogue.

### 4.2 Resolve ambiguity

The HRI policy instructs the model to ask only about ambiguity that can materially
change the requested final state. It may:

- ask an open clarification with `TASK_CLARIFICATION`;
- offer a concrete current-task interpretation with `TASK_CONFIRMATION`; or
- execute directly when the instruction, visible scene, and applicable approved
  preferences are sufficiently clear.

A task confirmation must carry the exact proposed task in structured form. The host
does not infer the proposal from assistant prose, and benchmark-oracle fields are
rejected. The current host structurally validates that proposal, but it does not yet
compare a later `EXECUTE` task contract with the task that was shown for confirmation.

If the HRI model classifies a reply as an unrelated new command and returns `RESTART`,
the host closes the old pending question, records the interrupted episode as cancelled,
retrieves fresh memory, and handles the same utterance as a new command. If the
physical task had already finished and only a memory-consent answer was pending, the
completed task is preserved before the new command starts.

### 4.3 Plan and gate the task

Planner receives the task contract, an observation image, and optional recovery
context. It returns a `PlanResult` with:

- status: `READY`, `ALREADY_SATISFIED`, `BLOCKED`, `UNSUPPORTED`, `UNSAFE`, or
  `UNKNOWN`;
- precondition records whose `satisfied` flags are checked by assurance;
- ordered, self-contained VLA subtask instructions;
- planner confidence and structured failure information; and
- for `READY` or `ALREADY_SATISFIED`, a `ValidationSpec`.

These are the statuses defined by the Planner prompt. The adapter uppercases, but does
not enum-reject, an unexpected status; Task Assurance handles an unrecognised value
conservatively.

Each validation goal has a unique ID, an explicit predicate, semantic arguments,
required/observable flags, and evidence modalities. Every required goal must be
observable. The specification's intent must exactly match the task contract.

The Planner adapter first parses and validates the structured validation specification.
Task Assurance permits VLA dispatch only when Planner confidence meets the threshold,
every precondition record has `satisfied = true`, at least one subtask exists, every
subtask is an object with a non-empty `task_instruction`, and a validation specification
is present. This production gate is structural: independent semantic grounding of
whether the proposed VLA instructions actually implement the task is performed only by
the out-of-band benchmark scorer.

The first validation specification returned with `READY` or `ALREADY_SATISFIED` is
frozen, even if the subsequent plan gate does not permit dispatch. Corrective plans
must preserve it exactly, including its ID, intent, goal IDs, predicates, arguments,
and ordering. Schema drift during recovery terminates the task workflow before further
dispatch rather than silently changing the definition of success.

If Planner returns `ALREADY_SATISFIED`, HRI does not call the VLA. It creates an
observation-only execution result and asks Validator to verify the claim.
The current branch terminates after that one validation; it does not re-observe or
replan if the already-satisfied claim is rejected. Runtime code also does not require
Planner's subtask list to be empty in this branch, although the benchmark scorer checks
for an explicitly empty plan and zero dispatches.

### 4.4 Dispatch the VLA

For a ready plan, HRI sends subtasks to the executor in order. When the executor exposes
the per-subtask API, HRI stops immediately if a subtask reports `FAILED`, `UNSAFE`,
`ABORTED`, or `CANCELLED`.

The default `RecordedEpisodeExecutor` issues no physical commands. It labels each
requested subtask `RECORDED_EXTERNAL_ATTEMPT` and returns the episode's recorded final
frame, but it does not establish that those instructions caused that frame and
explicitly sets `physical_execution_claimed = false`.

A live application may supply `CallableVLAExecutor` or another `VLAExecutor`. Any
high-frequency perception/action loop, trajectory control, and safety monitoring
remain application responsibilities outside this interface. When the executor exposes
`execute_subtask` and `observe`, PrefMem sequences task-level subtasks and can stop
between callbacks that report a terminal failure status. An implementation exposing
only whole-plan `execute()` owns its internal sequencing and must return the final
observation in its `ExecutionResult`. PrefMem then performs task-level validation and
recovery.

### 4.5 Validate the outcome

Only execution results with a validation-eligible reported status (`COMPLETED`,
`OBSERVED_RECORDED_ATTEMPT`, or `OBSERVATION_ONLY`) reach Validator. This status gate is
not independent proof of trajectory safety. Validator receives:

- the exact frozen `ValidationSpec`;
- path-redacted execution evidence; and
- the final observation image.

The dedicated `final_observation` path is replaced by an ID derived from the
observation bytes before model use. Executor-supplied `subtask_results` and `evidence`
are copied into model context, so live integrations must not place sensitive paths,
credentials, or benchmark-oracle data in those fields. Attempted actions and executor
claims are context, not proof that the requested final state was achieved.

Validator must return the same specification ID and exactly one check for every frozen
goal ID. It may report `SUCCESS`, `PARTIAL`, `FAILURE`, `UNKNOWN`, or `UNSAFE`.
Its prompt instructs it to use `UNKNOWN` for insufficient evidence such as occlusion,
blur, uncertain contact, or an inadequate viewpoint; host code validates the outcome
enum but cannot independently prove that the model chose the correct class.

Task Assurance declares success only when:

- Validator confidence meets the configured threshold;
- the complete frozen goal set was checked;
- Validator reports `SUCCESS` and `task_complete = true`; and
- every required goal is satisfied.

Only this path produces the host's `SUCCESS`/`SUCCESS_RECOVERED` task outcome and its
canonical `Task Complete.` message. A plausible plan, a locally completed VLA call, or
an HRI model claim cannot independently set the host outcome to success. The host does
not semantically inspect every arbitrary HRI `user_message`, so misleading success
wording paired with a non-success report is not separately blocked.

### 4.6 Bounded recovery and safety

The default configuration permits one corrective planning attempt after the initial
attempt and up to one re-observation within each execution attempt. Recovery is
therefore finite.

- A Planner or executor exception may consume a replanning attempt.
- A validation result or Validator failure requesting `REOBSERVE` can trigger a new
  observation and another validation without re-executing the task.
- `REPLAN` and `AUTO_LOCAL` both return to Planner in the current orchestrator, with
  instructions to plan only corrective actions for unmet frozen goals.
- Returned VLA failure states terminate before Validator; unlike a thrown executor
  exception, they are not automatically replanned.
- A non-proceeding plan gate is reported rather than automatically executing every
  suggested recovery action.
- When a recovery budget is exhausted, the system stops or requests user assistance
  instead of looping indefinitely.

The refreshed image used for validator re-observation is not currently assigned back
to Planner's `observation_path`; a later corrective plan therefore uses the preceding
execution observation rather than necessarily using that refreshed view.

A Planner-reported `UNSAFE` status is converted to `ABORTED_SAFETY` before dispatch. An
executor-reported `UNSAFE` result stops before Validator and also produces
`ABORTED_SAFETY`. Validator may likewise return `UNSAFE` for a validation-eligible
result. All three paths latch physical dispatch in the running process. A bare “retry”
or unrelated command does not clear the latch. The user must explicitly state that the
hazard has been checked and that it is safe to resume, and a separate HRI semantic
check must accept that clearance.

## 5. Persistent memory and preference learning

### 5.1 Two memory types

The current system uses two semantically different persistent memories.

**Episodic history** records what happened in individual task episodes. It can help HRI
interpret a repeated request or ask a better question, but it is never authority for a
future default.

**Preference memory** contains only user-authorized future rules. A preference contains
a self-contained semantic statement, scope, applicability, optional structured value,
user ownership, consent and evidence provenance, revision metadata, prior snapshots for
selected mutations, and merge lineage.

The storage format is structured JSON, not `memory.md`:

```text
memory/history.json          # schema-v2 episodic records and per-user summaries
memory/preferences.json      # schema-v2 preferences and committed transactions
memory/history_outbox.json   # schema-v1 terminal checkpoint and history retry queue
```

The repository attempts to append preference audit events beside the preference store
in a `*.audit.jsonl` file.

### 5.2 Semantic retrieval

At command start, the Memory Agent semantically ranks recent history and all active
preferences owned by the user. Its model reasons over paraphrases, word order, object
descriptions, scope, applicability, and conflicts; there is no exact-word or
hand-written alias gate.

Preference retrieval produces `MATCH`, `POSSIBLE`, or `CONFLICT` with confidence and a
reason. Unknown or inactive IDs returned by the model are rejected. By default, only a
`MATCH` at confidence `0.75` or above is usable without confirmation.

If semantic history retrieval fails, the service falls back to recent owned episodes
and attaches a warning. If preference retrieval fails, it returns no asserted
preference, preventing an unverified default from being applied.

### 5.3 Terminal history update

History is normally submitted after a terminal outcome, not during execution. If a
post-task preference question is proposed, HRI first checkpoints the unsummarised
terminal source in the outbox and postpones the history-repository append until the
user resolves the question or a later retry processes the checkpoint. HRI supplies the
Memory Agent with:

- the initial request and task-local dialogue;
- the resolved task contract;
- planned subtasks, executions, and validations across attempts;
- user-selected parameters;
- terminal assurance and failure information;
- the user-visible result; and
- any memory-consent events.

Before and after model summarisation, the system removes traces, hidden reasoning, raw
images, action tensors, confidence boilerplate, and other internal-only fields. If the
semantic summariser fails, deterministic code constructs a safe factual fallback
episode and still attempts persistence. If the repository write fails, HRI attempts to
queue the source for a later retry.

Each sanitized source has a fingerprint, making retries idempotent and rejecting the
same episode ID with changed content. After an append, Memory attempts to rebuild a
bounded rolling summary over the latest episodes. Summary failure does not roll back
the authoritative episode.

Normal successful, failed, blocked, cancelled, unknown, and safety-aborted outcomes can
therefore contribute to continuity, while none of them independently creates a
preference.

### 5.4 When the system asks to remember a preference

After a successful or recovered task, HRI may request a preference-question proposal.
It does not do so when the resolved task cites an existing saved preference, when a
preference was already committed in the episode, or when the task did not succeed.

The Memory model is instructed to propose a question only when the episodes are
semantically comparable, express the same meaningful choice, do not conflict, and are
not already covered by an active preference. Host code does not independently prove
those semantic judgments. It deterministically requires:

- at least two distinct supplied episode IDs, including the current episode;
- the configured minimum evidence count;
- no unknown episode IDs;
- a structured question and preference request; and
- model confidence of at least `0.75`.

Passing this gate does not save anything. It only allows HRI to ask a dedicated,
future-facing `MEMORY_CONSENT` question at the task boundary. Before waiting for the
answer, HRI checkpoints the completed task in the durable outbox.
If that checkpoint fails, HRI suppresses the preference question rather than waiting
for consent without durable retention of the completed task.

The user may then:

- confirm the proposed future preference;
- decline it;
- defer the decision;
- correct its value or scope and receive a revised memory question; or
- cancel the question.

For a pending memory question, the HRI model classifies the free-form reply into these
actions. The host binds the selected action to the displayed prompt and proposal and
preserves the exact quote, but it does not run the independent entailment check used
for direct, non-pending memory requests.

Declining, deferring, or cancelling creates no preference mutation. The physical task
is already terminal and is never executed again while resolving this question.

### 5.5 Direct future instructions

The user may also directly say, for example, “From now on, put printed items on the
left” or “Forget my saved block order.” Before accepting such a mutation, HRI performs
a second semantic entailment check against the exact utterance. The requested durable
operation must be entailed with confidence of at least `0.90`; current-task wording
alone is rejected.

### 5.6 Preference transactions and compaction

After valid consent, the Memory Agent proposes one or more semantic operations:
`ADD`, `UPDATE`, `MERGE`, `RETRACT`, `DELETE`, or `NOOP`. Deterministic repositories
then enforce:

- ownership and supplied-snapshot validity for preference target/source IDs;
- structural consent provenance and action binding, including the preserved quote,
  prompt, displayed proposal, and answer turn where applicable;
- optimistic repository versions;
- exact-request transaction idempotency and unique operation IDs within each
  transaction;
- all-or-nothing multi-operation commits; and
- retained consent, evidence, revisions, and lineage.

The repository validates these structural bindings; it does not independently prove
that the model's proposed operation is semantically identical to the user's words.

Deletion creates an auditable, non-physical tombstone and retains a pre-delete
snapshot. No undelete operation is currently implemented.

After a write, Memory may propose semantic compaction. An automatic merge requires at
least two active records, a lossless proposal at confidence `0.85` or above, and a
second independent semantic verification of equivalence, scope preservation, and
applicability preservation at the same threshold. Source records become
lineage-preserving inactive redirects. Conflicts, scope broadening, lossy abstraction,
and other non-merge changes are not auto-applied; if the model proposes them, they are
returned in `review_required`. Merge source records and lineage are retained, but no
explicit unmerge operation is currently implemented.

## 6. Persistence, failure isolation, and observability

The JSON repositories use in-process locks, cross-process lock files, version counters,
temporary files, atomic replacement, and last-known-good backups. Preference
transactions additionally enforce caller-supplied optimistic `expected_version`
checks. If a primary file is corrupt and a valid backup exists, the corrupt file is
quarantined and the backup is restored.

Persistence failures are isolated from task truth:

- a failed history write does not change a verified task outcome;
- HRI makes a best-effort outbox write of a partially cleaned terminal source; full
  history sanitisation occurs when Memory processes the retry;
- a failed preference transaction reports that nothing was committed;
- a failed post-write compaction does not undo the user's primary preference save; and
- a reasoning-free auxiliary audit is attempted for accepted and rejected preference
  mutations.

Outbox persistence can itself fail. Auxiliary audit writes are also best effort:
accepted transactions retain their authorization in the main preference store, while a
rejected mutation may have no JSONL audit entry if that append fails.

The `--display-all` mode exposes sanitized raw/parsed agent outputs and assurance gates.
Raw image bytes and hidden reasoning fields are omitted. Optional model telemetry can
record model configuration, timing, hashes, image counts, and response metadata when
an observer is injected, as in the evaluation runtime. Diagnostic output is not fed
back into agent context.

By default, all four model-backed components use the configured
`gemma4:31b-cloud` model through Ollama at temperature `0` with a 120-second per-call
timeout. The CLI can apply a common Ollama or OpenAI-compatible vLLM backend, model,
seed, and endpoint override. Image resizing is off by default, so validated source
bytes are preserved; optional resizing performs an aspect-preserving conversion within
the configured bounds.

The interactive CLI also enables `TerminalTranscript`, which mirrors typed input,
standard output, and standard error to `experiments/record.txt` by default. This is
useful for experiment reconstruction but should be treated as participant data when
planning storage, access control, and retention.

## 7. Updated example: “Tidy up the table”

Assume the scene contains printed items and electronic devices.

### First occurrence with no applicable saved preference

1. The user says, “Tidy up the table.”
2. HRI retrieves history and approved preferences, then examines the initial image.
3. Because the requested category-to-side mapping changes the final state, HRI must not
   invent a durable preference. It can ask an open clarification or make a structured
   current-task proposal, for example: “For this task, should I put printed items on the
   left and electronic devices on the right?”
4. If the user replies “Yes,” the reply approves only that current task.
5. If the user replies “No,” HRI asks how the table should be arranged.
6. If the user replies “No, do the opposite,” HRI resolves the current task as printed
   items on the right and electronic devices on the left.
7. HRI creates the task contract. Planner generates ordered VLA subtasks and the frozen
   validation goals. Task Assurance gates the plan.
8. HRI dispatches the subtasks sequentially, obtains the final observation, and calls
   Validator.
9. Only an assured validation success produces “Task Complete.”
10. A compact factual episode, including the user's chosen mapping, is submitted to
    history. No preference is created after this first task unless the user separately
    used explicit future-memory language.

### Repeated occurrence

On a later semantically similar request, the retrieved history may allow HRI to ask
whether the user wants the previous arrangement again. Agreeing still approves only
the current task. If a second comparable task succeeds with the same choice, Memory may
propose a separate post-task question:

> Would you like me to remember printed items on the right and electronic devices on
> the left as your default for future table-tidying tasks?

Under the HRI policy, an affirmative answer is classified as `COMMIT` and can create a
durable preference; “No” or “later” is classified as decline or defer and leaves
preference memory unchanged. The host structurally binds the classified action to the
pending proposal but does not independently reclassify the user's wording.

### Later use and override

For a future matching scene, a high-confidence active preference can resolve the
category-to-side ambiguity. HRI is instructed to attach its ID to the task contract for
provenance, and the host validates an ID when supplied, but citation is not currently
mandatory. If the user explicitly requests the opposite arrangement for one task, HRI
policy says the current instruction wins and the stored default remains unchanged.
Replacing the durable default requires a separate authorized memory operation.

## 8. Prototype and evaluation boundary

The plain interactive default is a recorded-episode adapter. Together with the
manifest-aware benchmark executors, the prototype can evaluate:

- task grounding and ambiguity handling;
- history and preference use;
- planning and structured VLA instructions;
- validation of recorded outcomes;
- bounded task-level recovery decisions; and
- response to an oracle-reported safety event in benchmark mode.

The simulation benchmark provides static endpoint packets for block stacking, category
sorting, and place setting, including success, wrong-complete, partial, uncertain,
unsafe-labelled, and already-satisfied cases. Unsafe cases use a private,
manifest-aware executor to inject an oracle `UNSAFE` status; their images neither
detect nor demonstrate the hazard. Manifest objects, target predicates, expected
outcomes, simulator state, and filesystem paths are not directly supplied to Planner
or Validator prompts. The injected `UNSAFE` result and a manifest-derived safety-event
label may subsequently appear as external execution evidence in history summarisation
or safety-clearance context.

Scripted counterfactual recovery reveals a paired success observation after a second
execution attempt or an explicit re-observation. It does not represent a physical
recovery trajectory.

These static image pairs do not demonstrate continuous VLA control, collision-free
trajectories, grasp stability, physical latency, hazard perception, or genuine
environment recovery. Those claims require a live executor and trajectory-level robot
evidence.

## 9. Main changes from the original concept

| Original concept | Current implementation |
|---|---|
| Six conceptual agent roles including a separate Summarising Agent | Four model-backed agents; summarisation is inside Memory; VLA is an executor boundary; Task Assurance is deterministic. |
| One free-form `memory.md` | Separate versioned JSON history, preference, and outbox stores, plus a best-effort JSONL mutation audit. |
| A task choice can be extracted as a user preference | The first and repeated choices remain history until explicit future-memory consent. |
| “Online learning” may imply model adaptation | No model weights are changed; learning is a transactional update to semantic preference records. |
| Summarisation starts alongside the first VLA subtask | Orchestration is synchronous and history summarisation occurs at the terminal task boundary. |
| Planner output proceeds directly to VLA | The Planner adapter validates goal structure, and Task Assurance gates confidence, preconditions, subtask structure, and schema presence. |
| Replanning may regenerate the validation schema | The first schema returned with `READY` or `ALREADY_SATISFIED` is frozen across recovery attempts. |
| VLA completion can lead to task completion | VLA completion is only local execution evidence; independent validation and assurance are mandatory. |
| Recovery repeats until validation succeeds | Recoverable failure or uncertainty may trigger bounded recovery; remaining blocked, failed, unknown, unsupported, or unsafe outcomes terminate conservatively. |
| The workspace contains a complete continuous-action VLA | The default adapter replays recorded observations; a live lower-level VLA must be integrated through the executor interface. |
| Safety is one validation outcome | Planner-, executor-, or Validator-reported `UNSAFE` produces a safety abort; executor `UNSAFE` stops before Validator, and every path latches dispatch until explicit clearance. |

## 10. Implementation map

- `main.py`: CLI construction and interactive entry point
- `agents/hri.py`: dialogue state, task orchestration, recovery, memory consent, and safety latch
- `agents/memory.py`: semantic retrieval, summarisation, preference proposals, updates, and compaction
- `agents/planner.py`: scene-grounded planning adapter
- `agents/vla.py`: recorded and callable VLA executor interfaces
- `agents/validator.py`: frozen-schema outcome validation
- `agents/contracts.py`: plan, execution, validation, and goal contracts
- `agents/model.py`, `agents/configs.py`, and `agents/vision.py`: model backends,
  defaults, image validation, and optional resizing
- `assurance/task_assurance.py`: deterministic gates and terminal decisions
- `memory/models.py`: memory queries, contexts, consent, and pending-question contracts
- `memory/repositories.py`: atomic history, preference, audit, backup, and outbox persistence
- `memory/migration.py`: conservative inspection and history-only import of legacy
  memory candidates; it never promotes them automatically to active preferences
- `simulation/benchmark/`: counterfactual static-observation evaluation infrastructure
