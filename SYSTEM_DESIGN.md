# PrefMem System Design

**Status:** implementation baseline  
**Scope:** the replacement for the implementation archived under `_old_1/`  
**Primary design decision:** the HRI Agent is the only user-facing orchestrator. The Memory Agent maintains two semantically different persistent memories: episodic history and explicitly approved preferences.

## 1. Purpose

PrefMem is an upper-level agentic control system for a Vision-Language-Action (VLA) robot. It resolves a user's task intent, recalls useful prior context, plans a sequence of robot skills, executes those skills through a VLA interface, validates the achieved state, and performs bounded recovery when necessary.

This design replaces the former candidate-preference mechanism. A choice made for one task is history, not a durable preference. It becomes a preference only when the user explicitly gives it future scope (for example, "remember this" or "always do this") or affirms a dedicated memory question. This distinction is independent of whether the robot succeeds or fails at the current task.

The document is both an architecture specification and an implementation contract. The words **must**, **must not**, **should**, and **may** are normative.

## 2. Goals and non-goals

### 2.1 Goals

The system must:

1. Preserve continuity between task conversations without retaining an unbounded raw transcript.
2. Keep episodic history separate from user-approved future preferences.
3. Use VLM reasoning for semantic memory retrieval, equivalence detection, and compaction, including complex preferences that cannot be handled reliably by token overlap or a fixed alias table.
4. Require explicit consent before adding or changing a durable preference.
5. Keep probabilistic semantic reasoning behind a deterministic, atomic, auditable transaction boundary.
6. Give the HRI Agent sole responsibility for user interaction and orchestration.
7. Pass one user-approved task contract from HRI to planning, execution, and validation without silently changing its meaning.
8. Report success only from outcome evidence, never from a plausible plan or an attempted action.
9. Bound automatic recovery and stop conservatively on safety events.
10. Make every memory use, memory mutation, task result, and recovery decision traceable to source evidence.

### 2.2 Non-goals

The first implementation is not required to:

- train or fine-tune a VLA model;
- infer a durable preference from implicit behaviour alone;
- implement a distributed MCP deployment;
- provide a high-scale vector database;
- preserve model chain-of-thought or hidden reasoning;
- prove physical contact or stability from a single monocular image when the available evidence cannot support that claim.

## 3. Architectural principles and invariants

These invariants take precedence over model output.

### 3.1 Interaction invariants

- The HRI Agent is the only component that reads free-form user input or emits user-facing dialogue.
- Other agents return typed results to HRI. They do not ask the user questions directly.
- A reply is interpreted relative to a typed pending question. An answer to "Use RGB for this task?" is current-task approval; it is not consent to remember RGB.
- A new command may reset the raw model message list, but HRI must rebuild its context from the newest `HistoryContext` and relevant approved preferences before reasoning about that command.

### 3.2 Memory invariants

- The first occurrence of a task choice is written to history only.
- Repeated choices are still history until explicit memory consent is obtained.
- Preference memory contains only user-authorised durable preferences and non-destructive provenance records such as merged redirects or retracted tombstones.
- Task success, task failure, planner output, validator output, and assistant suggestions never constitute preference consent.
- A current instruction always overrides a saved preference for the current task. It does not replace the saved preference unless the user separately authorises replacement.
- VLM reasoning decides semantic relevance and equivalence. The storage layer does not reduce semantic identity to exact words or hand-maintained aliases.
- Storage code validates authority, schema, referenced IDs, idempotency, atomicity, and provenance. It does not invent semantic meaning.
- Semantic compaction must be reversible. Source records and evidence remain addressable through merge lineage or an audit log.

### 3.3 Task-assurance invariants

- Planning readiness is not execution success.
- A VLA subtask result is not global task completion.
- The Validator evaluates the same goal conditions approved in the task contract and emitted in the plan's validation specification.
- `UNKNOWN` is neither `SUCCESS` nor `FAILURE`.
- No unparsed or schema-invalid agent output is dispatched to another operational component.
- Automatic retries and replans are bounded by configuration.
- Safety aborts are immediate and require explicit clearance before another physical attempt.

## 4. System context

```text
                         +-----------------------+
User <-----------------> | HRI Agent             |
                         | dialogue + orchestration
                         +-----------+-----------+
                                     |
              +----------------------+----------------------+
              |                      |                      |
              v                      v                      v
    +-------------------+   +-------------------+   +-------------------+
    | Memory Agent      |   | Planner Agent     |   | Task Assurance    |
    | history + prefs   |   | plan + goal spec  |   | policy + budgets  |
    +---------+---------+   +---------+---------+   +---------+---------+
              |                       |                       ^
              |                       v                       |
              |             +-------------------+             |
              |             | VLA Executor      |-------------+
              |             | robot adapter     |
              |             +---------+---------+
              |                       |
              |                       v
              |             +-------------------+
              +------------>| Validator Agent   |
                            | observed outcome  |
                            +-------------------+
```

Task Assurance is a deterministic policy service used by HRI; it is not a second orchestrator. The HRI Agent owns the control loop and invokes the next component selected by Task Assurance.

## 5. Component responsibilities

### 5.1 HRI Agent

The HRI Agent is the central core agent and only interaction port. It:

- accepts the user query and current observation;
- obtains a current memory context from the Memory Agent;
- grounds references and resolves task-changing ambiguity;
- distinguishes current-task questions from memory-consent questions;
- creates the user-approved `TaskContract`;
- invokes Planner, VLA Executor, Validator, Task Assurance, and Memory Agent;
- applies an approved preference or asks before using uncertain/conflicting context;
- maintains transient episode state and pending questions;
- reports verified progress, failure, recovery, and completion;
- sends the completed or aborted episode to history memory;
- decides when there is sufficient repeated historical evidence to offer a dedicated preference question;
- sends preference mutations to the Memory Agent only with consent evidence.

HRI must not directly edit memory files or reinterpret raw model output as a storage command.

### 5.2 Memory Agent

The Memory Agent is a semantic memory service with two persistent components.

**History memory** records compact episodes: what the user requested, what choices were resolved, what the agents did, what was observed, and how the episode ended. It contains no hidden reasoning and has no authority to set defaults.

**Preference memory** records explicitly approved future defaults. The Memory Agent uses a VLM to retrieve preferences by meaning, decide applicability in the current context, recognise paraphrases and synonyms, find duplicates or conflicts, and propose compacted records.

The Memory Agent exposes typed operations, validates VLM output, and delegates durable writes to a deterministic transaction layer. One update request may contain multiple operations, applied atomically.

The former standalone Summarising Agent is subsumed by history summarisation in the Memory Agent.

### 5.3 Planner Agent

The Planner Agent remains a scene-grounded long-horizon planner. It:

- inventories relevant visible objects and regions;
- verifies observable preconditions and declared robot capabilities;
- preserves the exact task contract;
- decomposes the task into short-horizon, self-contained subtasks;
- emits explicit observable goal conditions and validation checks;
- returns a non-ready status instead of inventing missing objects or relations.

The Planner does not interact with the user, read preferences independently, execute actions, or claim task completion.

### 5.4 VLA Executor

The VLA Executor is an adapter around the lower-level VLA robot or an episode-backed test double. It:

- accepts exactly one grounded subtask at a time;
- binds the subtask to the latest observation and robot state;
- streams or invokes continuous actions using the configured VLA policy;
- stops on completion, timeout, cancellation, controller fault, or safety signal;
- returns execution evidence and a new observation;
- never decides the user's global intent or writes memory.

The executor interface must allow a recorded-dataset implementation now and a physical robot implementation later.

### 5.5 Validator Agent

The Validator is independent of the Planner and VLA Executor. It receives:

- the approved `TaskContract`;
- the Planner's `ValidationSpec`;
- the latest observation and, where useful, additional viewpoints or intermediate evidence;
- execution metadata that does not itself count as proof of success.

It evaluates each required postcondition against visible evidence. It must return `UNKNOWN` when evidence is insufficient. It never derives preferences.

### 5.6 Task Assurance

Task Assurance applies deterministic gates to probabilistic component output. It:

- validates planner and validator schemas and confidence thresholds;
- enforces preconditions, recovery budgets, timeouts, and safety policy;
- selects `REOBSERVE`, `AUTO_LOCAL`, `REPLAN`, `USER_ASSIST`, `ABORT_UNSUPPORTED`, or `ABORT_SAFETY`;
- records attempt counts and prevents infinite loops;
- cannot execute recovery by itself or claim a recovery occurred.

## 6. Runtime state and persistence

The system has three state classes.

### 6.1 Transient working state

`EpisodeState` exists only for the current interaction and may be checkpointed for crash recovery. It is not preference memory.

```json
{
  "episode_id": "ep_...",
  "session_id": "session_...",
  "user_id": "default",
  "phase": "GROUNDING",
  "raw_user_query": "Stack the blocks",
  "task_contract": null,
  "plan": null,
  "current_subtask_index": 0,
  "attempt_counts": {
    "reobserve": 0,
    "local_retry": 0,
    "replan": 0
  },
  "pending_question": null,
  "memory_snapshot_version": 12,
  "cancel_requested": false
}
```

`phase` is one of:

```text
RECEIVED -> GROUNDING -> CLARIFYING -> READY_TO_PLAN -> PLANNING
         -> EXECUTING -> VALIDATING -> RECOVERING -> TERMINAL
```

Terminal outcomes are `SUCCESS`, `SUCCESS_RECOVERED`, `ALREADY_SATISFIED`, `PARTIAL`, `BLOCKED`, `FAILED`, `ABORTED_SAFETY`, `CANCELLED`, and `UNKNOWN`.

### 6.2 History memory

History is an append-oriented episode ledger plus a compact context snapshot. A representative `HistoryEpisode` is:

```json
{
  "schema_version": 1,
  "episode_id": "ep_...",
  "session_id": "session_...",
  "user_id": "default",
  "started_at": "RFC3339 timestamp",
  "ended_at": "RFC3339 timestamp",
  "initial_request": "Stack the blocks",
  "resolved_task": {
    "summary": "Stack the red, green, and blue blocks in RGB order from bottom to top.",
    "task_type_hint": "block stacking",
    "objects": ["red block", "green block", "blue block"],
    "constraints": ["red below green", "green below blue"]
  },
  "user_choices": [
    {
      "statement": "RGB order from bottom to top",
      "meaning": "red bottom, green middle, blue top",
      "source_turn_id": "turn_..."
    }
  ],
  "agent_actions": [
    {"agent": "planner", "action": "created 2 subtasks", "status": "READY"},
    {"agent": "vla", "action": "attempted 2 subtasks", "status": "SUCCEEDED"},
    {"agent": "validator", "action": "checked 2 goal relations", "status": "SUCCESS"}
  ],
  "outcome": {
    "status": "SUCCESS",
    "summary": "The final observation satisfied both required block relations.",
    "failure_code": null
  },
  "preference_events": [],
  "observation_refs": ["obs_initial_...", "obs_final_..."],
  "source_event_ids": ["event_..."],
  "summary_confidence": 0.93
}
```

History deliberately omits:

- chain-of-thought, hidden reasoning, HRI `trace`, and planner reasons intended only for model deliberation;
- large repeated fields such as full scene inventories and full validation payloads;
- raw image bytes and action tensors (store references instead);
- unsupported inferences presented as facts.

Uncertain statements must retain attribution, for example, "the Validator reported a possible gap" rather than "there was a gap."

`HistoryContext` supplied to HRI contains a bounded rolling summary, recent episodes, and semantically relevant older episodes. It is regenerated after every history update and replaces, rather than appends to, the prior HRI history context.

### 6.3 Preference memory

A preference is semantic and may be complex. Natural-language meaning is primary; structured hints help filtering and downstream grounding but must not be the only retrieval key.

```json
{
  "schema_version": 1,
  "id": "pref_...",
  "user_id": "default",
  "statement": "When stacking red, green, and blue blocks, put red at the bottom, green in the middle, and blue on top.",
  "applicability": "Applies to a tower containing the red, green, and blue blocks or equivalent cubes.",
  "scope": "block-stacking tasks with these three colours",
  "structured_hints": {
    "task_type": "stacking",
    "objects": ["red block", "green block", "blue block"],
    "relations": ["red below green", "green below blue"]
  },
  "status": "ACTIVE",
  "created_at": "RFC3339 timestamp",
  "updated_at": "RFC3339 timestamp",
  "revision": 1,
  "consent": {
    "kind": "DEDICATED_CONFIRMATION",
    "user_quote": "Yes, remember that order.",
    "episode_id": "ep_...",
    "turn_id": "turn_...",
    "question_id": "question_...",
    "recorded_at": "RFC3339 timestamp"
  },
  "evidence": [
    {"episode_id": "ep_...", "turn_id": "turn_...", "role": "consent"}
  ],
  "merged_from": [],
  "supersedes": [],
  "compaction_notes": []
}
```

Valid statuses are:

- `ACTIVE`: eligible for retrieval and use;
- `RETRACTED`: explicitly replaced or forgotten, retained as a tombstone;
- `MERGED`: a redirect to an active canonical record;
- `CONFLICT`: preserved but not silently applied until HRI resolves the conflict.

There is no persistent `candidate` status. A proposed preference and a deferred memory question live only in working state; the supporting choices remain in history.

## 7. Shared contracts

All inter-agent payloads must include `schema_version`, `request_id`, `episode_id`, and `user_id`. Unknown schema versions fail closed.

### 7.1 Task contract

```json
{
  "schema_version": 1,
  "task_id": "task_...",
  "episode_id": "ep_...",
  "user_request": "Stack the blocks",
  "resolved_intent": "Stack red, green, and blue from bottom to top.",
  "objects": [
    {"ref": "red_block", "description": "visible red block"}
  ],
  "constraints": [
    {"relation": "BELOW", "subject": "red_block", "object": "green_block"}
  ],
  "assumptions": [],
  "preference_refs": ["pref_..."],
  "history_refs": ["ep_..."],
  "user_approved": true,
  "approval_turn_id": "turn_..."
}
```

`user_approved` means the current task is sufficiently resolved to plan. It does not mean preference consent.

### 7.2 Plan and validation specification

```json
{
  "planning_status": "READY",
  "planner_confidence": 0.91,
  "preconditions": [
    {"id": "pre_1", "condition": "all three blocks are visible", "satisfied": true, "evidence": "..."}
  ],
  "subtasks": [
    {
      "id": "subtask_1",
      "instruction": "Place the green block securely on the red block.",
      "skill": "pick_and_place",
      "target": "green_block",
      "destination": "top of red_block",
      "completion_condition_ids": ["goal_1"]
    }
  ],
  "validation_spec": {
    "task_id": "task_...",
    "goal_conditions": [
      {
        "id": "goal_1",
        "description": "red block supports green block",
        "predicate": "ABOVE(green_block, red_block) AND CONTACT(green_block, red_block)",
        "observable_evidence": ["relative vertical position", "visible boundary/contact"],
        "required": true
      }
    ]
  },
  "failure": null
}
```

Planning statuses are `READY`, `ALREADY_SATISFIED`, `BLOCKED`, `UNSUPPORTED`, `UNSAFE`, and `UNKNOWN`. A non-ready result contains no executable subtasks.

### 7.3 Execution result

```json
{
  "subtask_id": "subtask_1",
  "attempt_id": "attempt_...",
  "status": "SUCCEEDED",
  "started_at": "RFC3339 timestamp",
  "ended_at": "RFC3339 timestamp",
  "before_observation_ref": "obs_...",
  "after_observation_ref": "obs_...",
  "action_log_ref": "actions_...",
  "controller_evidence": {
    "termination_reason": "policy reported subtask termination",
    "safety_stop": false
  },
  "failure": null
}
```

Execution statuses are `SUCCEEDED`, `FAILED`, `UNKNOWN`, `UNSAFE`, `CANCELLED`, and `TIMEOUT`. `SUCCEEDED` means the executor completed its local protocol; the Validator still decides whether the visual goal was achieved.

### 7.4 Validation result

```json
{
  "outcome": "SUCCESS",
  "task_complete": true,
  "validator_confidence": 0.91,
  "goal_checks": [
    {
      "condition_id": "goal_1",
      "satisfied": true,
      "confidence": 0.92,
      "evidence": "The green block is directly above and visually contacts the red block."
    }
  ],
  "discrepancies": [],
  "recoverability": "NONE",
  "recommended_action": "No further action.",
  "failure": null
}
```

Validation outcomes are `SUCCESS`, `PARTIAL`, `FAILURE`, `UNKNOWN`, and `UNSAFE`.

### 7.5 Failure contract

Every non-success operational result uses:

```json
{
  "stage": "GROUNDING",
  "code": "AMBIGUOUS_REFERENT",
  "expected": "one uniquely identified target block",
  "observed": "two indistinguishable red blocks",
  "confidence": 0.86,
  "severity": "MEDIUM",
  "recoverability": "USER_ASSIST",
  "safe_state": "no action was dispatched",
  "attempts": 0,
  "next_action": "ask which red block is intended",
  "user_message": "Which red block should I move?",
  "memory_effect": "NONE"
}
```

## 8. Memory Agent API

These are typed in-process service methods for the first implementation. They can later be exposed as MCP tools without changing their semantics. Calling internal methods "MCP servers" is unnecessary until a process or trust boundary exists.

### 8.1 Get complete HRI memory context

```python
def get_memory_context(request: MemoryContextRequest) -> MemoryContext:
    """Return the newest bounded history context and semantically relevant approved preferences."""
```

`MemoryContextRequest` contains the user ID, natural-language HRI request, scene description or observation reference, episode ID, desired limits, and request ID.

`MemoryContext` contains:

- `history_snapshot_version`;
- `history_summary`;
- recent and semantically relevant `HistoryEpisode` summaries;
- preference matches from `get_relative_preference_memory`;
- conflicts and uncertainty warnings;
- exact source IDs used for traceability.

### 8.2 Retrieve relative preference memory

The public name follows the proposed interface; `get_relevant_preferences` may be used as an internal alias.

```python
def get_relative_preference_memory(
    hri_request: PreferenceRetrievalRequest,
) -> PreferenceRetrievalResult:
    """Use VLM semantic reasoning to find and explain preferences applicable to the request."""
```

The VLM receives the query, scene context, and candidate or complete active preference set. It returns matches with `MATCH`, `POSSIBLE`, `CONFLICT`, or `NOT_APPLICABLE`, confidence, applied meaning, reasons, and preference IDs. A schema validator rejects references to IDs that were not supplied.

For the thesis-scale JSON store, the VLM may reason over all active preferences. When memory grows, a high-recall lexical/vector stage may shortlist candidates, but:

- it is an optimisation, not the semantic decision maker;
- exact token overlap must never be the only retrieval route;
- low shortlist confidence must fall back to a broader or full-store semantic pass;
- the final relevance decision belongs to the VLM Memory Agent.

HRI may silently apply only an `ACTIVE`, explicitly consented, non-conflicting `MATCH` above a configured threshold. It should tell the user which saved preference it is using and permit an override. `POSSIBLE` and `CONFLICT` require clarification.

### 8.3 Update preference memory

```python
def update_preference_memory(
    hri_request: PreferenceUpdateRequest,
) -> PreferenceUpdateResult:
    """Semantically interpret a requested mutation, compact affected preferences, and commit one atomic batch."""
```

The request wraps natural language in a typed envelope:

```json
{
  "request_id": "req_...",
  "episode_id": "ep_...",
  "user_id": "default",
  "instruction": "Remember red-bottom, green-middle, blue-top as my default for these block stacks.",
  "consent_evidence": {
    "kind": "DEDICATED_CONFIRMATION",
    "question_id": "question_...",
    "turn_id": "turn_...",
    "user_quote": "Yes, remember that order."
  },
  "history_evidence_ids": ["ep_..."],
  "expected_store_revision": 7
}
```

The VLM may propose multiple `ADD`, `UPDATE`, `MERGE`, `RETRACT`, `DELETE`, or `NOOP` operations. It is responsible for semantic interpretation, including recognising that "blue top, green middle, red bottom" and "RGB bottom-to-top" are equivalent. The result of a semantic proposal is typed and includes source IDs, target meaning, scope, confidence, and a concise rationale.

The deterministic transaction layer then checks:

1. every referenced record and evidence ID exists and belongs to the same user;
2. `ADD`, semantic `UPDATE`, scope broadening, replacement, and forgetting have appropriate explicit authority;
3. the pending question ID and answer turn match when consent is a confirmation;
4. the VLM did not reference records outside the supplied candidate set;
5. the expected store revision still matches;
6. the request ID has not already been committed;
7. all records satisfy the schema and preserve provenance;
8. the complete batch can be written atomically.

If any check fails, no operation is applied. The service returns a typed rejection to HRI.

### 8.4 Update history memory

```python
def update_history_memory(
    newest_conversation: HistoryUpdateRequest,
) -> HistoryUpdateResult:
    """Summarise the newest episode, append it, compact history context, and return the replacement snapshot."""
```

The request contains the user-visible turns and structured outputs from all agents. The VLM removes reasoning-only and redundant content, produces a `HistoryEpisode`, and updates the rolling summary. Deterministic code verifies identifiers, timestamps, enum values, source references, and terminal status before append.

The return value includes the new `HistoryContext`. HRI immediately replaces its cached history context with this snapshot. A history update failure does not change a task outcome; it is logged and retried idempotently.

### 8.5 Preference proposal support

Repeated history is evidence that a memory question may be useful, but not permission to write a preference.

```python
def propose_preference_question(
    request: PreferenceProposalRequest,
) -> PreferenceProposal | None:
    """Reason over history and active preferences and suggest a dedicated memory question."""
```

The Memory Agent may identify a repeated stable choice across semantically similar episodes. HRI decides whether and when to ask, considering interruption cost, previous deferrals/declines, conflict, and task state. A proposal lives in `EpisodeState`; it is not stored in preference memory.

## 9. Semantic preference retrieval and compaction

### 9.1 Division of responsibility

The semantic/deterministic boundary is:

```text
VLM Memory Agent
  - understands paraphrases, synonyms, relationships, and scene applicability
  - retrieves and reranks preferences by meaning
  - detects semantic duplicates and conflicts
  - proposes merged wording, structure, scope, and operations

Deterministic transaction layer
  - validates authority and referenced evidence
  - enforces consent and scope-change rules
  - applies idempotent atomic transactions
  - preserves revisions, source records, and audit lineage
  - never decides that two meanings are equivalent
```

This avoids the `stack`/`stacking` and `block`/`cube` mismatch bottleneck without allowing probabilistic output to make unauthorised or irreversible writes.

### 9.2 Compaction classes

The Memory Agent supports three semantic cases.

**Lossless equivalence merge.** Two explicitly authorised records express the same preference and scope using different language. The VLM may propose an automatic reversible merge. The canonical record retains all consent and evidence, and source records become `MERGED` redirects. No user prompt is required because the operation does not broaden or change the authorised meaning.

**Scope-preserving rewrite.** One record can be expressed more compactly without changing its applicability. The VLM may propose a new revision, preserving the prior revision and evidence.

**Lossy abstraction, scope broadening, or conflict resolution.** Combining records would create a more general rule, discard an exception, or choose between incompatible values. The VLM may propose the change, but HRI must obtain explicit user approval before commit.

Examples:

- "RGB bottom-to-top" and "blue on top, green in the middle, red on the bottom" can be a lossless merge if their contexts and scopes are semantically equivalent.
- "Use RGB for these three blocks" and "sort every object by spectrum order" must not be merged automatically; that broadens scope.
- "RGB for wooden blocks" and "BGR for foam blocks" are contextual preferences, not necessarily a conflict.
- "RGB" and "BGR" for the same context are a conflict and must not be silently compacted into one.

### 9.3 Compaction triggers

Compaction may run:

- in the affected neighbourhood after a committed add or update;
- when retrieval returns several highly similar records;
- when active-record count or prompt-token size crosses a configured threshold;
- as an explicit maintenance command.

Compaction runs under a store snapshot and expected revision. A concurrent change causes retry from a fresh snapshot. Physical deletion is reserved for a separate retention policy; normal compaction uses revisions, redirects, and tombstones.

## 10. Consent and pending-question model

### 10.1 Consent classes

Valid durable authority is:

- `EXPLICIT_FUTURE_STATEMENT`: the user directly says to remember, always use, make default, or an equivalent future-oriented instruction;
- `DEDICATED_CONFIRMATION`: the user affirmatively answers a clearly worded preference-memory question;
- `EXPLICIT_FORGET`: the user directly asks to forget or delete a saved preference;
- `EXPLICIT_REPLACEMENT`: the user confirms replacing an existing preference.

Current-task clarification, action approval, repetition, assistant inference, history similarity, task outcome, and validator confidence are not durable authority.

### 10.2 Typed pending question

```json
{
  "question_id": "question_...",
  "kind": "MEMORY_SAVE",
  "prompt": "Would you like me to remember RGB as your default for future block-stacking tasks?",
  "proposal": {
    "statement": "Use RGB bottom-to-top for these block stacks.",
    "scope": "block-stacking tasks with red, green, and blue blocks"
  },
  "allowed_responses": ["CONFIRM", "DECLINE", "DEFER", "CORRECT", "CANCEL"],
  "created_from_episode_ids": ["ep_1", "ep_2"]
}
```

HRI uses language understanding to classify free-form responses into this finite state. It must support phrasing such as:

- "Yes, remember it" -> `CONFIRM`;
- "No, only this time" -> `DECLINE`;
- "I want to decide later" -> `DEFER`;
- "Remember it, but only for the wooden blocks" -> `CORRECT`, followed by confirmation of the corrected scope if needed;
- a new unrelated command -> close or suspend the proposal and process the new command.

The final mutation is driven by the typed state and matching question ID, not by an exact hard-coded yes/no string set. `DEFER` closes the prompt without a preference mutation and records only a lightweight prompt-cooldown event in history or interaction metadata.

### 10.3 Question burden

HRI should ask a memory question only when the likely future value justifies interruption. Default proposal policy:

- do not ask after the first occurrence;
- after at least two semantically matching, independent user choices in relevant episodes, the Memory Agent may propose asking;
- do not treat the threshold as proof of preference;
- suppress repeated prompts after decline or defer according to configurable cooldowns;
- prioritise task/safety questions over memory questions;
- ask memory questions at a clear task boundary so they cannot be confused with action authorisation.

## 11. End-to-end workflows

### 11.1 First ambiguous task

```text
1. User: "Stack the blocks."
2. HRI obtains HistoryContext and relevant approved preferences; none apply.
3. HRI sees that order changes the outcome and asks for the order.
4. User: "RGB from bottom to top."
5. HRI creates the TaskContract and calls Planner.
6. HRI dispatches each ready subtask to the VLA Executor.
7. HRI calls Validator with the same TaskContract and ValidationSpec.
8. Task Assurance accepts success or chooses bounded recovery.
9. At terminal state, HRI calls update_history_memory.
10. History records the RGB choice and outcome. Preference memory remains unchanged.
```

No current-task wording in steps 1-9 authorises a preference write.

### 11.2 Repeated task before a preference exists

```text
1. User: "Stack the blocks."
2. HRI receives a history context containing the prior RGB episode.
3. HRI asks: "Last time you used red, green, blue from bottom to top. Use that order again?"
4. User agrees. This approves the current task only.
5. The task is planned, executed, assured, and appended to history.
6. The Memory Agent finds two semantically matching independent choices.
7. At the task boundary, HRI asks a dedicated future-memory question.
8. CONFIRM -> HRI calls update_preference_memory with matching consent evidence.
   DECLINE -> no preference write; apply decline cooldown.
   DEFER -> no preference write; apply defer cooldown.
```

The two history records may use different words such as `cube` and `block`; semantic history reasoning is responsible for recognising equivalence.

### 11.3 Task with an approved saved preference

```text
1. User: "Stack the blocks."
2. get_relative_preference_memory returns a high-confidence active preference match.
3. HRI grounds it against the current scene and says it will use the saved RGB order.
4. If no conflicting current instruction exists, HRI creates the contract and proceeds.
5. The TaskContract includes the preference ID for traceability.
6. The outcome is appended to history; merely using the preference creates no new preference revision.
```

If match confidence is only possible, the scene falls outside the stored scope, or several preferences conflict, HRI asks rather than silently choosing.

### 11.4 Direct future preference

```text
User: "From now on, stack these colours red, green, blue from bottom to top."
HRI: resolves both the current task and the explicit future-memory intent.
Memory Agent: proposes the semantic preference operation.
Transaction layer: verifies direct consent evidence and commits it.
HRI: reports what was saved and its scope, then continues the current task.
```

### 11.5 One-off override

```text
Saved preference: RGB.
User: "For this one, use blue, green, red."
HRI: current instruction wins; the TaskContract uses BGR.
Memory: no mutation.
History: records a one-off override.
HRI may later ask whether this should replace the default, but only a dedicated
EXPLICIT_REPLACEMENT answer changes preference memory.
```

### 11.6 Failure and recovery

```text
Planner/Executor/Validator returns structured non-success
        -> Task Assurance validates and selects bounded action
        -> HRI performs re-observation, local retry, or replan if budget remains
        -> otherwise HRI asks for assistance or reports terminal failure
        -> HRI appends the complete attempt and result to history
        -> preference memory is unchanged unless a separate explicit memory act occurred
```

## 12. Planning, execution, validation, and assurance loop

### 12.1 Dispatch gates

HRI dispatches a plan only if:

- the task contract is structurally valid;
- ambiguity that can materially change the outcome is resolved;
- planner status is `READY`;
- required preconditions are satisfied;
- planner confidence meets threshold;
- subtasks are non-empty and supported;
- no safety gate is active.

### 12.2 Sequential execution

For each subtask:

1. bind it to the latest observation;
2. invoke the VLA Executor;
3. store returned evidence references;
4. stop immediately on cancellation or unsafe result;
5. refresh the current observation before the next dependent subtask;
6. if local execution is unknown/failed, consult Task Assurance before continuing.

The plan may be invalidated by a changed scene. HRI then replans from the latest observation rather than blindly executing stale subtasks.

### 12.3 Validation

After all planned subtasks, HRI passes the exact `ValidationSpec` and current observation to the Validator. The Validator checks every required goal condition by ID. Conditions unsupported by the observation are `unknown`, not false. Overall rules are:

- all required checks true -> `SUCCESS`;
- some true and unmet work is preservable -> `PARTIAL`;
- adequate evidence shows any required condition false -> `FAILURE`;
- any required condition is not observable with adequate confidence -> `UNKNOWN`;
- an immediate hazard -> `UNSAFE`.

For contact-sensitive stacking, a single frame may be inadequate. The system should request another view or use simulation/robot state evidence when available. Disabling application-side resizing does not make a VLM geometrically reliable and must not be treated as a validation fix.

### 12.4 Default bounded recovery

- `REOBSERVE`: one additional frame/viewpoint;
- `AUTO_LOCAL`: one safe local retry;
- `REPLAN`: one materially different plan from the latest scene;
- `USER_ASSIST`: one concise request describing the missing condition;
- `ABORT_UNSUPPORTED`: no automatic retry;
- `ABORT_SAFETY`: immediate stop and explicit safety clearance before a new attempt.

Budgets are configurable. Exhaustion escalates local retry to replan and replan/reobserve to user assistance; it never loops indefinitely.

## 13. Context lifecycle

The old implementation reset HRI's model-visible conversation at each new command and retrieved preferences using exact token overlap. As a result, an identical second task could appear unrelated to the first. The new lifecycle is:

1. At process start, load the latest `HistoryContext` for the user.
2. At new command, create a fresh bounded HRI message list.
3. Attach the current system prompt, newest `HistoryContext`, semantically relevant active preferences, current observation, and user query.
4. During the task, append only current episode turns and typed tool results needed for the next decision.
5. At terminal state, call `update_history_memory`.
6. Replace the cached history context with the returned snapshot.
7. Begin the next command from this rebuilt context, not an empty conversational world and not the unlimited raw transcript.

This preserves continuity while controlling token growth and accumulated model confusion.

## 14. Persistence, transactions, and concurrency

The reference implementation may use JSON files, provided that the repository layer exposes a storage interface that can later be replaced by SQLite or another transactional store.

Recommended files:

```text
memory/
  history.json
  preferences.json
  audit.jsonl
  checkpoints/          # optional episode crash recovery
```

Required storage behaviour:

- write a complete temporary file and atomically replace the target;
- maintain monotonically increasing store revisions;
- use optimistic concurrency through `expected_store_revision`;
- use request IDs as idempotency keys;
- serialise concurrent writers per user/store;
- validate the post-write document before replacement;
- keep an append-only audit event for every accepted or rejected mutation;
- never log raw image bytes, action tensors, credentials, or hidden reasoning;
- support per-user isolation in retrieval and mutation;
- retain merge lineage and tombstones unless a separate explicit retention policy removes them.

A partially valid multi-operation request must not partially commit.

## 15. Failure handling by component

| Component | Failure | Required behaviour |
|---|---|---|
| HRI | malformed model output | Retry parse/response once; if still invalid, explain inability and do not dispatch. |
| Memory retrieval | VLM timeout or invalid IDs | Return no asserted match plus warning; HRI may clarify, but must not invent a preference. |
| History update | summariser/schema failure | Preserve source event bundle and queue idempotent retry; task outcome remains unchanged. |
| Preference update | consent or revision check fails | Reject entire transaction and report that nothing was saved. |
| Compaction | uncertain equivalence | Keep records separate and mark for review; do not discard information. |
| Planner | malformed/non-ready output | Do not execute; use Task Assurance recovery. |
| VLA | timeout/unknown robot state | Stop issuing actions, establish safe state, then reobserve or request help. |
| Validator | insufficient view | Return `UNKNOWN`; acquire more evidence within budget. |
| Safety monitor | hazard | Immediate abort; no automatic resume. |
| Persistence | corrupt file | Fail closed, preserve corrupt copy, load last valid snapshot if available, and surface an operator error. |

Model confidence is advisory evidence, not a guarantee. Schema validity and consent checks remain mandatory at any confidence.

## 16. Observability and audit

Every component call should emit a structured event containing:

- event, request, session, episode, task, and user IDs;
- component and operation name;
- input/output schema versions and referenced artifact IDs;
- latency, retry count, and model name/config version;
- parsed status, confidence, and failure code;
- preference IDs retrieved or mutated;
- history snapshot and store revisions;
- consent kind and source turn ID for mutations;
- no hidden reasoning.

User-visible messages should be traceable to these events. In particular, "Task complete" requires an accepted `SUCCESS`/`SUCCESS_RECOVERED` assurance result, and "I remembered this" requires a committed preference transaction ID.

## 17. Configuration

Agent and policy configuration belongs in `agents/configs.py` or a dedicated typed settings module. It should include:

- model host, base URL, model name, prompt path, temperature, and timeout per agent;
- memory file paths, history/context token limits, semantic match threshold, compaction threshold, and full-scan fallback limit;
- preference-prompt evidence threshold and decline/defer cooldowns;
- planner/validator confidence thresholds;
- maximum reobservations, local retries, and replans;
- VLA adapter type (`dataset`, `mock`, or `robot`) and capability manifest;
- image encoding settings. Preserve original images where validation quality matters and avoid unnecessary lossy JPEG conversion.

Tests must inject configuration and model clients; they must not depend on globally constructed live clients.

## 18. Recommended implementation layout

```text
agents/
  configs.py
  hri.py
  memory.py
  planner.py
  validator.py
  prompts/
memory/
  models.py             # history/preference/consent schemas
  repository.py         # atomic storage and revisions
  semantic.py           # VLM retrieval and compaction proposals
  service.py            # exposed Memory Agent API
  history.json
  preferences.json
orchestration/
  episode.py            # EpisodeState and HRI state transitions
  task_assurance.py
execution/
  base.py               # VLAExecutor protocol
  dataset.py
  mock.py
  robot.py              # future physical adapter
contracts/
  task.py
  planning.py
  execution.py
  validation.py
  failures.py
tests/
  unit/
  integration/
  scenarios/
main.py
SYSTEM_DESIGN.md
```

The exact module names may vary, but semantic reasoning, transaction safety, orchestration, and storage must remain separable and independently testable.

## 19. Testing strategy

### 19.1 Unit tests

**HRI state machine**

- first `Stack the blocks` choice writes history but no preference;
- new commands receive the latest history snapshot;
- current-action `yes` cannot satisfy a memory question;
- `CONFIRM`, `DECLINE`, `DEFER`, `CORRECT`, cancellation, and a new-command interruption transition correctly;
- an explicit current override wins without replacing memory;
- no non-HRI agent emits a user question.

**Semantic memory**

- `stack` retrieves `stacking` and `blocks` retrieves a preference worded with `cubes`;
- RGB bottom-to-top matches blue-top/green-middle/red-bottom;
- contextually distinct preferences are not incorrectly merged;
- complex conditional preferences are applied only in their semantic scope;
- unknown or low-confidence relevance is returned as `POSSIBLE`, not asserted;
- VLM hallucinated preference IDs are rejected.

**Transactions**

- add without durable consent is rejected;
- a repeated history choice alone is rejected as authority;
- matching question and affirmative turn commit exactly once;
- duplicate request IDs are idempotent;
- stale expected revisions fail without partial writes;
- multi-operation failure rolls back the full batch;
- lossless merge preserves both source records, evidence, and consent;
- scope broadening and conflict resolution require explicit approval;
- forgetting creates a retracted tombstone and stops retrieval.

**History**

- summaries omit trace/reasoning and large redundant fields;
- all agents' externally meaningful actions and terminal outcome are represented;
- uncertain validator statements retain attribution;
- a failed task still becomes a history episode;
- update returns a new replacement context version;
- retrying the same update does not duplicate an episode.

**Planner, Validator, and Assurance**

- goal IDs pass unchanged from contract/plan to validation;
- missing objects block dispatch;
- malformed outputs fail closed;
- executor local success cannot produce global success without validation;
- validator unknown triggers bounded reobservation;
- retry/replan budgets terminate;
- safety abort never automatically resumes.

### 19.2 Integration scenarios

At minimum, run these end-to-end scenarios with deterministic fake model responses and selected live-model evaluations:

1. First ambiguous RGB stack -> one history record, zero preferences.
2. Identical second stack -> prior history visible, current-order confirmation, dedicated memory proposal after task boundary.
3. Reply `I want to decide later` -> no preference, no repeated loop, deferred cooldown recorded.
4. Dedicated `yes` -> one active preference with exact consent provenance.
5. Third paraphrased request using `cubes` -> preference semantically retrieved and used.
6. Top-to-bottom wording -> same RGB meaning; lossless duplicate merge.
7. One-off BGR instruction -> current task BGR, saved RGB unchanged.
8. Explicitly replace RGB with BGR -> old record retracted, new record active.
9. Two contextual orders for different materials -> both preserved and correctly selected.
10. Validator cannot see contact -> `UNKNOWN`, reobserve, never false success.
11. Planner sees missing block -> no VLA call.
12. Executor timeout -> safe stop and bounded recovery.
13. Process restart -> history and approved preferences reconstruct context.
14. Concurrent preference writes -> one retries from new revision; no lost update.

### 19.3 Evaluation metrics

- false durable-write rate;
- missed explicit-preference rate;
- semantic retrieval precision/recall, including paraphrase and context changes;
- harmful semantic-merge rate and unresolved duplicate rate;
- memory question frequency, decline/defer rate, and repeated-prompt burden;
- continuity accuracy on repeated tasks;
- planner dispatch safety and malformed-output rejection rate;
- task success, partial success, false-success, false-failure, and unknown rates;
- recovery success and budget-exhaustion rate;
- end-to-end latency and VLM/token cost.

Live VLM tests are evaluation tests, not the sole regression suite. Core control and transaction behaviour must be reproducible with fakes.

## 20. Migration from `_old_1`

The archived code and memory files are evidence for testing, not state to copy blindly.

Migration rules:

1. Treat legacy `candidate` records without explicit durable evidence as historical task choices, not preferences.
2. Import a legacy durable record only when its evidence contains an explicit future statement or a dedicated memory confirmation that can be traced to the user.
3. Convert task conversations and assurance outcomes into `HistoryEpisode` records.
4. Use the VLM Memory Agent to propose semantic merges such as `cube`/`block` and reversed linguistic enumeration, then validate and preserve import provenance.
5. Leave ambiguous or conflicting legacy records inactive and surface them for user confirmation; do not guess.
6. Do not migrate chain-of-thought, HRI trace fields, raw repeated inventories, or model confidence as user preference evidence.
7. Preserve the original legacy files read-only for experimental reproducibility.

For `preferences_3.json`, choices made without explicit memory consent should become history evidence. The new preference store should remain empty until a future statement or dedicated confirmation authorises a preference.

## 21. Acceptance criteria

The replacement system is conformant when all of the following hold:

1. A first unmarked task choice cannot create any active preference record.
2. A second task receives model-visible history of the first even after raw dialogue reset.
3. Semantic retrieval succeeds for relevant paraphrases without exact token overlap.
4. Free-form defer responses close memory confirmation without mutation or an infinite yes/no loop.
5. Only the HRI Agent conducts dialogue and calls other agents/services.
6. Every active preference contains explicit consent provenance.
7. VLM-proposed duplicate merges are semantic, reversible, provenance-preserving, and atomically committed.
8. Scope broadening or conflict resolution cannot commit without user authority.
9. Planner goal conditions reach Validator unchanged by ID.
10. No task reports completion without an accepted validation/assurance result.
11. Recovery terminates within configured budgets and safety aborts require clearance.
12. History is updated for success, failure, cancellation, and abort, excludes hidden reasoning, and replaces HRI's prior history context.

## 22. Design summary

PrefMem separates **what happened** from **what should happen by default**:

- history records compact, provenance-bearing episodes and gives HRI conversational continuity;
- preferences record only explicitly approved future behaviour;
- the VLM Memory Agent performs the semantic work needed for complex retrieval, matching, and compaction;
- deterministic transactions enforce consent, atomicity, idempotency, and reversibility;
- HRI alone orchestrates the user, Planner, VLA Executor, Validator, Task Assurance, and Memory Agent;
- task assurance keeps planning, attempted execution, observed outcome, and preference learning evidentially distinct.

This architecture resolves the original exact-word retrieval failure without replacing it with a rigid alias system, while retaining an auditable safety boundary around long-term memory and robot action.

## 23. Reference implementation mapping

The repository-root implementation follows this specification with the following
concrete modules:

| Design responsibility | Implementation |
|---|---|
| HRI dialogue, typed pending questions, task-boundary memory proposals, orchestration, and safety latch | `agents/hri.py` |
| VLM semantic retrieval, episode summarisation, preference transactions, and semantic compaction | `agents/memory.py` |
| Consent, memory-query, context, and pending-question envelopes | `memory/models.py` |
| Atomic history/preference stores, optimistic revisions, process locks, audit events, tombstones, merge lineage, backups, and retry outbox | `memory/repositories.py` |
| Frozen planning/execution/validation contracts | `agents/contracts.py` |
| Scene-grounded planning and strict validator adapters | `agents/planner.py`, `agents/validator.py` |
| Per-subtask VLA boundary and recorded-dataset/live callback adapters | `agents/vla.py` |
| Deterministic dispatch, execution, validation, recovery, and safety gates | `assurance/task_assurance.py` |
| Central configuration | `agents/configs.py` |
| Conservative legacy inspection/import | `memory/migration.py` |

The default VLA implementation is deliberately a recorded-episode adapter: it uses the
numeric first and final dataset frames and never claims that Python physically executed
the robot. A live lower-level VLA is supplied through `CallableVLAExecutor`; HRI still
owns sequential dispatch, stopping, observation, validation, and bounded recovery.

Runtime stores are created lazily. Preference transactions are also recorded in a
reasoning-free `preferences.json.audit.jsonl`; last-valid JSON backups and corrupt-file
quarantine are managed beside each store. Failed history updates enter the configured
history outbox and are retried idempotently before a later command.

The legacy migrator is dry-run by default. For example:

```powershell
python -m memory.migration --legacy _old_1/memory/preferences_3.json --user-id default
```

Adding `--apply-history --history-store memory/history.json` imports unconsented legacy
choices as history only. Possible durable legacy evidence remains a review item; it is
never promoted to an active preference automatically.

The offline conformance suite uses injected scripted models and therefore requires no
Ollama service. Live-model visual accuracy and physical VLA behavior remain evaluation
concerns rather than deterministic unit-test claims.
