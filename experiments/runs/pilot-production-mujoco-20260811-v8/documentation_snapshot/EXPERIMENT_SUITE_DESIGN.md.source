# PrefMem Publication-Grade Experiment Suite Design

Status: proposed  
Project version at design time: 2.2.1  
Design date: 2026-08-11

## 1. Purpose

This document specifies a full, publication-grade experiment program for
PrefMem. It expands the existing 39-scenario campaign into a claim-driven,
statistically replicated suite covering:

- preference-aware human-robot interaction;
- task-level receding-horizon planning;
- step monitoring and failure recovery;
- independent holistic validation;
- confirmation, evidence, and emergency fencing;
- agent-topology ablations;
- mechanism ablations;
- component-level diagnosis;
- end-to-end MuJoCo evaluation;
- safety-screened real-system evaluation; and
- a human-participant study.

The suite is intended to support conference publications. It therefore
separates development data from locked evaluation data, defines primary
outcomes before experiments are run, uses matched repeated trials, and records
enough provenance to reproduce every reported figure and table.

This design follows the implemented workflow in
[PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) and the
[README System Workflow](../README.md#system-workflow).

## 2. Experimental principles

1. **Claims precede scenarios.** Every scenario must name the research claim
   and primary outcome it tests.
2. **Logical roles and agent count are distinct.** A system may retain five
   logical roles while implementing them with fewer independent model agents.
3. **Topology and mechanism ablations are separate.** Merging roles is not the
   same intervention as deleting Memory, Monitor, or Validator.
4. **The episode is the experimental unit.** Frames and steps within an episode
   are correlated observations, not independent samples.
5. **Hidden state is scoring-only.** MuJoCo poses and trigger internals must
   never be exposed to the evaluated agents.
6. **Model failures count.** Schema errors, timeouts, grounding failures, and
   motion failures are product outcomes unless a pre-registered infrastructure
   exclusion applies.
7. **Unsafe ablations remain simulated.** Systems without confirmation,
   monitoring, temporal stability, or trusted fencing must not control a real
   robot.
8. **No post-test tuning.** Prompt or product changes after inspecting locked
   test results require a new system version and a new campaign.
9. **No single opaque score.** Physical completion, preference satisfaction,
   false completion, safety, recovery, and cost are reported separately.
10. **All comparisons are matched.** Competing systems receive the same scene
    seed, language paraphrase, perturbation schedule, and execution conditions.

## 3. Research questions and hypotheses

| ID | Research question | Primary comparison | Pre-registered expectation |
|---|---|---|---|
| RQ1 | Does PrefMem improve safe holistic task completion? | Full vs open-loop and feedback-only baselines | Full improves contract success and safe completion |
| RQ2 | Does preference memory reduce interaction while preserving consent and overrides? | Full vs no-Memory | Full reduces clarification/correction burden without increasing unauthorized writes |
| RQ3 | Does receding-horizon planning improve recovery from physical changes? | Full vs open-loop | Full improves perturbation recovery |
| RQ4 | Does per-step visual monitoring prevent stale-success propagation and improve recovery? | Full vs no-Monitor | Full reduces false step success and downstream goal failure |
| RQ5 | Does independent final validation reduce false completion? | Full vs no-Validator | Full lowers false-completion rate |
| RQ6 | Do exact confirmation and host fencing prevent unauthorized execution? | Full vs no-confirmation plus deterministic fence tests | Full reduces pre-authorization publication to zero |
| RQ7 | Does repeated temporal evidence protect against transient observations? | Full vs single-evidence | Full reduces transient-frame terminal errors |
| RQ8 | What is the effect of model-agent decomposition? | 5-agent vs 4/3/2/1-agent topologies | Role isolation improves contract adherence, with a latency/cost tradeoff |
| RQ9 | How do success, safety, latency, and token usage trade off? | All systems | Exploratory Pareto analysis |
| RQ10 | Do the technical mechanisms improve interaction for people? | Human study: Full vs no-Memory | Full reduces user effort and improves preference-correct completion |

RQ1-RQ7 are confirmatory. RQ8-RQ10 may include confirmatory primary outcomes
and explicitly labelled exploratory secondary analyses.

## 4. Definition of an agent

For the topology experiment, an **agent** is an independent model instance with
its own role prompt and conversational state.

The following do not count as model agents:

- `PrefMemRuntime`;
- `RecedingHorizonController`;
- the preference database and embedding index;
- schema validators and deterministic oracles;
- the emergency coordinator;
- the task publisher;
- the fixed lower-level Gemma/SAM/RGB-D/Panda execution pipeline.

The lower-level execution pipeline remains identical across evaluated
topologies. Agent counts in the paper must state that they refer to upper-level
model agents.

## 5. Agent-topology systems

All topology variants retain the same five logical functions:

1. user interaction;
2. preference-memory access;
3. planning;
4. per-step assessment; and
5. holistic final assessment.

They differ only in which functions share a model instance and conversation
state.

### 5.1 T5-Full: five independent agents

Agents:

- HRI Agent;
- Memory Agent;
- Planner Agent;
- Monitor Agent; and
- Validator Agent.

Design:

- HRI is the only user-facing model agent.
- Memory owns semantic retrieval and consented mutation decisions.
- Planner is stateless and scene-grounded for previews and execution cycles.
- Monitor assesses only the active physical step.
- Validator compiles a frozen checklist and independently assesses the complete
  goal.
- The deterministic host retains all IDs, state transitions, publication
  fences, evidence thresholds, loop guards, and emergency control.

This is the reference system.

### 5.2 T4-InteractionMemory: four agents

Agents:

- InteractionMemory Agent;
- Planner Agent;
- Monitor Agent; and
- Validator Agent.

Design:

- HRI and Memory share one model instance and conversation.
- The merged agent receives direct retrieve/remember/update/forget tools.
- The persistent store and embedding implementation are unchanged.
- Planner, Monitor, and Validator remain independent.
- Consent rules remain identical to T5 and are scored explicitly.

This isolates the value of separating preference reasoning from user-facing
dialogue.

### 5.3 T3-ThreeRole: three agents

Agents:

- InteractionMemory Agent;
- Planner Agent; and
- Verifier Agent.

Design:

- Interaction and Memory are merged as in T4.
- Planner remains independent.
- Monitor and Validator are merged into one Verifier Agent with shared state.
- The host still calls the Verifier in two phase-specific modes:
  `STEP_ASSESS` and `FINAL_ASSESS`.
- A validation checklist is still frozen at confirmation, but the same model
  that assesses steps also performs holistic validation.

This tests whether independent step and goal verification reduces
self-consistency or confirmation-bias failures.

### 5.4 T2-ActorCritic: two agents

Agents:

- Actor Agent; and
- Critic Agent.

Design:

- Actor combines HRI, Memory, and Planner.
- Critic combines Monitor and Validator.
- The Actor sees user dialogue, preference tools, current scene evidence, goal
  history, and planning triggers.
- The Critic receives the phase-specific observation contract and visual
  evidence.
- The deterministic host preserves exact confirmation, goal freezing,
  horizon-one execution, trusted IDs, and emergency fencing.

This is a conventional actor/critic-style agent decomposition.

### 5.5 T1-Unified: one model agent

Agent:

- Unified Agent.

Design:

- One stateful model instance performs interaction, preference operations,
  preview, planning, step assessment, and final validation.
- The deterministic host invokes it in phase-specific modes:
  - `INTERACT`;
  - `PREVIEW`;
  - `PLAN`;
  - `STEP_ASSESS`;
  - `COMPILE_VALIDATION`; and
  - `FINAL_ASSESS`.
- Each mode retains the same strict output schema used by the corresponding
  independent agent in T5.
- All logical-role context is shared, including the agent's own prior plans and
  assessments.
- Host-owned contracts, IDs, safety gates, and physical execution remain
  unchanged.

This is the primary monolithic-agent baseline.

### 5.6 Topology comparison controls

Across T5-T1:

- use identical model weights and tokenizer;
- use identical image resolution and encoding;
- use identical decoding parameters;
- preserve the same phase-specific input evidence;
- preserve the same logical decision schedule;
- build merged prompts from the constituent role instructions;
- permit equal prompt-development effort on the development split;
- retain strict phase-specific schemas;
- retain all deterministic host safety mechanisms;
- report actual calls, input/output tokens, latency, peak context size, and
  failures by phase.

The topology experiment must not artificially handicap T1 with one unstructured
schema or give T5 privileged scene information.

## 6. Mechanism ablation systems

Mechanism ablations start from T5-Full and change one capability at a time.

| ID | Intervention | Agent count | Exact behavior | Allowed environment |
|---|---|---:|---|---|
| A-Memory | Remove Memory | 4 | No Memory tool or persistent store is available; underspecified preferences require clarification | Simulation, human study, safety-screened robot |
| A-HRI | Remove HRI | 4 | Raw user request goes to Planner; no model-led clarification or proposal dialogue | Simulation only |
| A-OpenLoop | Remove receding-horizon replanning | 5 | Planner emits one complete queue after confirmation; later observations never replace its remaining actions | Simulation and safety-screened robot |
| A-Monitor | Remove Monitor | 4 | MuJoCo executor `SETTLED` is converted to synthetic step success; no visual step verification occurs | Simulation only |
| A-Validator | Remove Validator | 4 | Planner sees a fresh final frame and emits `DECLARE_COMPLETE`; no frozen independent checklist is assessed | Simulation only |
| A-Confirmation | Remove exact confirmation | 5 | A valid preview auto-confirms and starts execution; clarification remains enabled | Simulation only |
| A-SingleEvidence | Remove temporal repetition | 5 | One accepted success/failure frame is terminal; stability interval is zero | Simulation only |
| A-DynamicGuard | Remove live scope protection | 5 | Open-category membership freezes at confirmation; new members do not cancel active motion | Simulation only |
| A-HostFence | Remove trusted evidence/publication identity | 5 | Model-supplied or stale identities are accepted | Synthetic event tests only |

### 6.1 A-Memory

- Remove the Memory Agent and memory tool from HRI.
- Use an empty isolated store for every episode.
- Do not inject a hidden default into the HRI prompt.
- Require clarification when task intent depends on a missing preference.
- Compare immediate, delayed, transferred, overridden, updated, and forgotten
  preferences.

### 6.2 A-HRI

- Send the raw user instruction directly to Planner preview.
- Keep a minimal host confirmation UI if required to prevent an uncontrolled
  safety comparison.
- Do not silently resolve ambiguous goals.
- Score unsupported actions, ambiguity, and constraint loss.

### 6.3 A-OpenLoop

- Planner returns the full action queue once after confirmation.
- The maximum queue length equals the full system's maximum allowed cycle
  budget.
- Monitor may assess each step, but:
  - success advances to the next frozen action;
  - failure terminates or requests human attention;
  - no new Planner call modifies the queue.
- Validator remains enabled so the ablation isolates replanning rather than
  final verification.

### 6.4 A-Monitor

- Retain Planner and Validator.
- Convert executor `SETTLED` to a synthetic success record.
- Provide no post-action VLM observation before the next cycle.
- Preserve execution faults as system errors.
- Restrict this variant to MuJoCo because human execution lacks an
  authoritative settled event.

### 6.5 A-Validator

- Retain per-step Monitor.
- Add `DECLARE_COMPLETE` to Planner's cycle decision schema.
- Give Planner a fresh final frame before this decision.
- Terminate when Planner declares completion.
- Do not infer completion from the number of successful steps.

This compares an independent holistic verifier with planner self-assessment.

### 6.6 A-Confirmation

- HRI still clarifies and requests a scene-grounded preview.
- The host auto-confirms a `READY` or
  `ALREADY_SATISFIED` proposal.
- The user never assents to the exact goal ID and revision.
- Run only in simulation.

### 6.7 A-SingleEvidence

- Set success confirmations to one.
- Set failure confirmations to one.
- Set stability duration to zero.
- Keep stale-publication and frame-order checks unchanged.

### 6.8 A-DynamicGuard

- Preserve open-category selection at confirmation.
- Freeze its member list at confirmation.
- Disable live new-member cancellation.
- Continue ordinary Monitor and Validator assessment.

### 6.9 A-HostFence

- Use a deterministic controller simulator with scripted publications and
  assessments.
- Compare trusted host IDs with model-provided IDs.
- Inject stale success, wrong-criterion evidence, delayed errors, and
  post-emergency reports.
- Never connect this variant to a physical or simulated execution service that
  can move objects.

## 7. Baselines

### 7.1 B1-OpenLoopLLM

A single LLM produces a complete task plan from the initial instruction and
scene. It receives no later scene feedback. The same lower-level executor runs
the plan.

This tests the value of closed-loop feedback against a common one-shot
language-planning design. It may be informed by, but must not be described as an
exact reproduction of, [Code as Policies](https://arxiv.org/abs/2209.07753)
unless that method's official implementation and evaluation protocol are used.

### 7.2 B2-FeedbackPlanner

A single planning agent receives:

- the high-level instruction;
- current scene evidence;
- the previous action;
- textual success/failure feedback; and
- task history.

It replans after each outcome but has:

- no persistent preference memory;
- no exact frozen goal contract;
- no separate HRI role; and
- no independent final Validator.

This is an Inner Monologue/RePLan-inspired baseline. Prior work specifically
studies closed-loop feedback and replanning:

- [Inner Monologue](https://arxiv.org/abs/2207.05608);
- [RePLan](https://arxiv.org/abs/2401.04157).

It must be called “inspired” unless the official methods are reproduced
faithfully.

### 7.3 B3-OracleCeiling

Use hidden state for:

- exact object identity;
- feasibility-aware action selection;
- step success;
- final goal completion.

The same deterministic controller executes actions. This is a non-deployable
ceiling that distinguishes high-level agent failures from perception, IK, and
motion-control limits.

### 7.4 B4-NoPersonalization

Use the complete closed-loop architecture with an empty memory store. This is
equivalent to A-Memory and serves as the main personalization baseline.

### 7.5 External-benchmark policy

The [Preference-based Planning benchmark](https://arxiv.org/abs/2502.00858)
contains hundreds of preference-planning cases and is relevant to external
validity. However, it commonly derives preferences from demonstrations, while
PrefMem requires explicitly consented textual memory. Any adapter must be
labelled clearly as an adapted benchmark rather than an official PbP result.

[BEHAVIOR-1K](https://behavior.stanford.edu/index.html) provides broad
human-centred household tasks. The current PrefMem Panda adapter is not a
generic robot interface, so BEHAVIOR evaluation is deferred until a genuine
compatible execution adapter exists.

## 8. Four experiment layers

### 8.1 Layer A: deterministic invariant suite

This layer evaluates the deterministic host with scripted model outputs and
randomized event schedules.

Required invariant families:

1. exact goal ID and revision confirmation;
2. stale/replaced goal rejection;
3. no publication before confirmation;
4. execution horizon of one;
5. candidate-tail discard;
6. single active publication;
7. retirement before replacement;
8. stale frame rejection;
9. wrong-publication rejection;
10. wrong-step and criterion-ID rejection;
11. success confirmation threshold;
12. success stability duration;
13. failure confirmation threshold;
14. transient `UNKNOWN` behavior;
15. visible contradiction behavior;
16. safe hold before replanning;
17. immutable terminal history;
18. frozen GoalContract;
19. frozen ValidationContract;
20. Monitor/Validator phase isolation;
21. multi-view validation accumulation;
22. no-progress guard;
23. repeated-failure guard;
24. cycle guard;
25. idempotent emergency latch;
26. no post-emergency publication;
27. memory retrieval before mutation;
28. no mutation without future-facing consent;
29. confirmation does not authorize memory mutation; and
30. one mutation maximum per request.

Run:

- direct unit tests for every transition;
- property-based state-machine tests;
- randomized interleavings of delayed Planner, Monitor, Validator, executor,
  and emergency events;
- at least hundreds of generated traces per invariant family.

Every temporal oracle must compare event sequence IDs or monotonic timestamps.
Checking that two events merely occurred is insufficient to establish order.

### 8.2 Layer B: component benchmarks

| Component | Initial target | Inputs | Primary metrics |
|---|---:|---|---|
| HRI/Planner | 500 instances | Images, dialogue, capability declarations, runtime state | clarification precision/recall, goal correctness, schema validity, unsupported-action rejection |
| Memory | 300 interaction sequences | Preferences, paraphrases, updates, overrides, distractors | retrieval precision/recall, mutation accuracy, consent violations, interference |
| Monitor/Validator | 1,000 labelled frames or clips | Clean, contradictory, and occluded evidence | confusion matrices, false terminal rate, appropriate abstention |
| Grounding/execution | 500 configurations | Object selectors, RGB-D, target relations, poses | detection success, ambiguity rate, depth validity, reachability, controller completion |

### HRI/Planner strata

- clear versus genuinely ambiguous goals;
- object-reference ambiguity;
- constraint conflict;
- supported versus unsupported primitives;
- already-satisfied goals;
- missing explicit objects;
- exact revision and confirmation language;
- paraphrases, typos, and concise user messages;
- closed versus open categories;
- scenes with distractors.

### Memory strata

- remember;
- retrieve;
- update;
- forget;
- explicit one-run override;
- conflicting old and new preferences;
- irrelevant semantic neighbours;
- same preference expressed with paraphrases;
- delayed retrieval after unrelated interactions;
- no-consent statements;
- confirmation-only statements;
- long-store interference;
- preference transfer to new scenes.

### Monitor/Validator labels

- `MET`;
- `NOT_MET`;
- `UNKNOWN`;
- partial occlusion;
- complete occlusion;
- blur and lighting degradation;
- single-frame contradiction;
- persistent contradiction;
- moved source/target;
- fallen stack;
- protected-object displacement;
- visible emergency proxy.

Report per-class precision, recall, F1, confusion matrices, false-success rate,
false-failure rate, and appropriate-abstention rate.

### Grounding/execution strata

- colour and spatial references;
- duplicate objects;
- flat pads versus three-dimensional targets;
- visually similar target and source;
- edge-of-workspace poses;
- reachable/unreachable pairs;
- corrupted and sparse depth;
- source/target motion after grounding;
- post-cancellation restart;
- stack-height stress.

### 8.3 Layer C: end-to-end simulation benchmark

The proposed locked catalogue contains 62 scenario templates.

| Family | Count | Purpose |
|---|---:|---|
| Intent and confirmation | 8 | Goal clarity, capability honesty, revision, and authorization |
| Preference memory | 12 | Consent, retrieval, transfer, override, update, forgetting, interference |
| Nominal manipulation | 10 | Task horizons, partial goals, scope, distractors, multiple targets |
| Perturbation and recovery | 14 | Scene changes at different execution boundaries and recoverability levels |
| Evidence and validation | 8 | Temporal evidence, occlusion, stale reports, holistic incompleteness |
| Safety, faults, and capability | 10 | Service faults, emergency, workspace, unsupported actions, loop guards |
| **Total** | **62** | |

### Intent and confirmation templates

1. clear fully specified request;
2. vague physical outcome;
3. duplicate-object reference ambiguity;
4. conflicting constraints;
5. rejected proposal and revised goal;
6. stale goal/revision confirmation attempt;
7. unsupported manipulation primitive;
8. user instruction change during execution.

### Preference-memory templates

1. remember a stacking order;
2. immediate retrieval;
3. delayed retrieval;
4. transfer to a new layout;
5. paraphrased request;
6. explicit one-run override;
7. persistent update;
8. explicit forgetting;
9. conflicting preferences;
10. irrelevant retrieval candidate;
11. statement without mutation consent;
12. long-store interference and capacity.

Every mutation template must separately score:

- whether a write occurred;
- whether it was authorized;
- whether the correct record changed;
- whether later planning applied it;
- whether an override incorrectly persisted.

### Nominal-manipulation templates

1. already-satisfied goal, horizon zero;
2. single pick/place;
3. ordered three-object task;
4. correct partial state;
5. six-step long-horizon task;
6. ten-step stress task;
7. closed category with distractors;
8. open category;
9. multiple targets/boards;
10. held-out colours, layouts, or detector-supported objects.

### Perturbation and recovery templates

1. source moved before grounding;
2. source moved after grounding;
3. target moved after grounding;
4. new dynamic member during motion;
5. new dynamic member after settling;
6. irrelevant object entering a closed goal;
7. required object removed after confirmation;
8. top object removed before validation;
9. partial stack collapse;
10. protected object displaced;
11. recoverable controller miss;
12. transient detector miss;
13. repeated failure of the same action;
14. persistent infeasible blocker.

Each trigger must have:

- an intended boundary;
- a required firing count;
- an event-order oracle;
- a hidden-state postcondition;
- a recovery oracle;
- a maximum allowed response time.

### Evidence and validation templates

1. one-frame Monitor occlusion;
2. persistent Monitor occlusion;
3. partial final-validation occlusion;
4. persistent final-validation occlusion;
5. wrong-publication success;
6. one valid success frame;
7. alternating contradictory frames;
8. all local steps successful but holistic goal incomplete.

### Safety, fault, and capability templates

1. invalid Planner schema;
2. invalid Monitor enum/schema;
3. invalid Validator schema;
4. invalid execution-compiler output;
5. corrupted RGB-D frame;
6. camera timeout;
7. out-of-workspace target;
8. missing explicit object;
9. visible emergency proxy with external trigger;
10. repeated-action/cycle guard.

### 8.4 Layer D: real-system and human evaluation

### Real-system subset

Use twelve safety-screened templates:

- four nominal tasks;
- four interaction/preference tasks;
- two recoverable perturbations; and
- two validation/occlusion tasks.

Evaluate:

- T5-Full;
- B2-FeedbackPlanner; and
- the most informative safe ablation selected before real-system testing.

Use at least five repetitions per task/system after a hardware pilot.

Do not run:

- A-Confirmation;
- A-Monitor;
- A-SingleEvidence;
- A-DynamicGuard;
- A-HostFence; or
- any variant that failed the simulation safety gate.

## 9. Dataset splits

### 9.1 Development split

Includes:

- all scenarios from `full_20260809`;
- known failure reproductions;
- prompt-development examples;
- debugging scenes;
- component examples inspected by developers.

Development results must never be presented as held-out performance.

### 9.2 Pilot split

Contains unseen:

- layouts;
- language paraphrases;
- trigger timings;
- memory histories; and
- visual conditions.

Use it to:

- verify instrumentation;
- estimate outcome variance;
- select the repetition count through power analysis;
- preflight task feasibility;
- freeze prompts and thresholds.

Pilot trials are excluded from confirmatory results.

### 9.3 Locked test split

Lock and hash:

- scenario templates;
- scene seeds;
- language variants;
- preference histories;
- perturbation schedules;
- primary outcomes;
- exclusion rules; and
- analysis scripts.

No product, prompt, oracle, or threshold changes are allowed after unlocking
test results. A necessary change creates a new campaign and version.

### 9.4 Template-level separation

Do not split only by random pose seed. Hold out combinations of:

- task semantics;
- layout families;
- prompt formulations;
- preference types;
- distractor types; and
- perturbation boundaries.

This better measures generalization than evaluating new coordinates for already
seen scenarios.

## 10. Scenario-validity policy

Maintain two distinct catalogues.

### 10.1 Mechanism-isolation catalogue

Before the locked run:

- verify target detection;
- verify source/target identity;
- verify reachability and collision-free feasibility;
- verify trigger hooks;
- verify that the requested manipulation lies within executor capabilities.

The preflight uses separate calibration trials, not the evaluated agent.

### 10.2 End-to-end capability catalogue

Do not pre-exclude:

- grounding ambiguity;
- detector failures;
- reachability failures;
- Planner capability errors.

These failures count against the full system because users experience them.

Results from the two catalogues must be reported separately:

- mechanism success conditional on reaching the intended boundary;
- unconditional end-to-end system success.

## 11. Parameterization and randomization

Each scenario template should generate episode instances over:

- scene/layout seed;
- object positions;
- prompt paraphrase;
- object identities and distractors;
- task horizon;
- camera pose where allowed;
- lighting and texture;
- perturbation occurrence and timing;
- dynamic-scope condition;
- preference-history length.

Use common random numbers:

- system variants receive the same generated instance;
- perturbations fire from the same event predicate;
- the same paraphrase and memory state are used;
- execution/controller seeds match.

Randomize run order within blocks to reduce service drift and thermal/time
effects. Record the order.

## 12. Trial matrix and sample size

Let `r` be the matched repetitions per scenario template. Determine
`r` using a separate pilot and simulation-based power analysis.

Expected range:

- minimum: 10 repetitions per template/system;
- likely main-study range: 15-25;
- higher for rare safety events or unstable model outputs.

### 12.1 Main system comparison

Run:

```text
62 templates × r repetitions ×
{T5-Full, B1-OpenLoopLLM, B2-FeedbackPlanner, B4-NoPersonalization}
```

### 12.2 Agent-topology comparison

Select 24 templates stratified across all six families:

```text
24 templates × r repetitions × {T5, T4, T3, T2, T1}
```

The selected set must include:

- ambiguity and confirmation;
- consented and non-consented memory;
- nominal short and long horizons;
- recoverable perturbations;
- occlusion and holistic validation;
- schema and safety faults.

### 12.3 Mechanism-ablation comparison

Run each ablation only on claim-relevant scenarios:

| Ablation | Required template families |
|---|---|
| A-Memory | Preference memory, ambiguity, override, update, forget |
| A-HRI | Ambiguity, reference resolution, capability limits, revisions |
| A-OpenLoop | Long horizon, all perturbations, collapses, target/source movement |
| A-Monitor | Misses, moved objects, partial regressions, protected-object changes |
| A-Validator | Already satisfied, holistic incompleteness, regression, occlusion |
| A-Confirmation | Revisions, overrides, constraint conflict, unsupported tasks |
| A-SingleEvidence | One-frame success, transient occlusion, alternating evidence |
| A-DynamicGuard | Open-scope entry during motion and after settling |
| A-HostFence | Synthetic stale/wrong-ID event traces |

### 12.4 Model sensitivity

Freeze three open-weight VLM backbones before testing:

- the primary project model;
- a comparable alternative;
- a smaller model.

Run:

```text
20 representative templates × 10 repetitions ×
{T5-Full, B2-FeedbackPlanner} × 3 backbones
```

Do not change models after inspecting results. Record exact hashes and decoding
settings.

## 13. Outcomes and metrics

### 13.1 Primary outcomes

### Physical goal completion

All hidden physical goal predicates pass at episode end:

- required spatial relations;
- object membership;
- preservation constraints;
- stability and support;
- target placement.

This is independent of the controller's declared state.

### Contract success

The entire scenario contract passes:

- physical goal;
- explicit constraints;
- stored and current-run preferences;
- confirmation requirements;
- publication and evidence fences;
- trigger response;
- terminal-state expectation;
- safety requirements.

### False completion

The controller enters `COMPLETE` while any required physical,
preference, or authorization condition is unmet.

Report:

- episode-level false-completion rate;
- false-completion causes;
- time and state at false completion.

### Safe completion

Physical completion with:

- no unauthorized publication;
- no protected-object violation;
- no unhandled collision;
- no workspace violation;
- no emergency-policy violation;
- no false completion.

### Perturbation recovery

For episodes with a fired perturbation:

- intended trigger fired;
- system detected or safely handled the change;
- no critical safety violation occurred;
- original frozen goal was eventually completed within the limit.

Report both:

- conditional recovery given trigger firing;
- unconditional end-to-end success.

### Preference satisfaction

Proportion of applicable:

- explicit current instructions;
- stored defaults;
- preservation preferences;
- ordering preferences;
- negative constraints;
- one-run overrides

that are correctly reflected in the final outcome.

### 13.2 Secondary outcomes

Interaction:

- clarification turns;
- unnecessary clarification;
- user correction turns;
- time to executable confirmation;
- unsupported-action explanation accuracy.

Memory:

- retrieval precision/recall;
- unauthorized mutation rate;
- duplicate-write rate;
- update/forget accuracy;
- override persistence errors;
- cross-task interference.

Planning/execution:

- physical publication count;
- planning cycles;
- repeated actions;
- action efficiency relative to an oracle plan;
- controller path length;
- task time;
- recovery time.

Observation:

- Monitor/Validator confusion matrices;
- appropriate `UNKNOWN` rate;
- false success/failure;
- schema correction count;
- no-progress timeout rate.

Safety/liveness:

- protected-object displacement;
- collision/contact violations;
- emergency-stop latency;
- post-emergency publications;
- attention-state rate;
- loop-guard activation;
- unresolved timeout rate.

Cost:

- model calls by role;
- input/output and reasoning tokens;
- per-call and episode latency;
- GPU time;
- execution time;
- storage footprint;
- optional energy estimate.

## 14. Statistical analysis plan

### 14.1 Experimental unit

The independent experimental unit is an episode.

Do not treat:

- video frames;
- Monitor polls;
- Planner cycles; or
- individual actions

as independent samples for system-level hypothesis tests.

### 14.2 Models

Binary outcomes:

- mixed-effects logistic regression.

Count outcomes:

- negative-binomial mixed-effects models;
- Poisson only if dispersion diagnostics support it.

Positive continuous outcomes:

- log-normal or Gamma mixed models.

Completion/recovery time with timeouts:

- survival analysis or censored accelerated-failure-time models.

Participant ordinal responses:

- cumulative-link mixed models.

### 14.3 Fixed and random effects

Candidate fixed effects:

- system condition;
- scenario family;
- task horizon;
- perturbation type;
- ambiguity level;
- memory-history length;
- occlusion condition;
- model backbone;
- pre-registered interactions.

Candidate random effects:

- scenario template;
- generated instance/seed;
- prompt paraphrase;
- participant;
- task set for the human study.

### 14.4 Reporting

Report:

- estimated marginal means;
- absolute and relative differences;
- odds ratios or rate ratios where appropriate;
- 95% confidence intervals;
- per-template raw values;
- paired differences;
- failure taxonomy;
- model diagnostics.

Use Holm correction within each pre-registered family of pairwise comparisons.
Do not correct exploratory analyses together with confirmatory outcomes; label
them separately.

### 14.5 Power and repetition count

Run a pilot with approximately five matched repetitions per selected cell.

Use simulation-based power analysis for the planned mixed model. Define a
minimum effect worth detecting before test execution, such as:

- a 10-percentage-point difference in contract success;
- a 5-percentage-point difference in false completion;
- a practically meaningful reduction in user corrections.

Target at least 90% power for primary comparisons when feasible.

### 14.6 Exclusions

Permitted exclusions:

- corrupted artifact caused by the harness;
- external service outage documented independently of system behavior;
- human-study withdrawal according to the consent protocol;
- pre-registered hardware safety cancellation.

Not permitted as exclusions:

- malformed model output;
- model timeout;
- SAM ambiguity;
- execution fault;
- Planner or Monitor error;
- task timeout;
- unexpected attention state.

All exclusions must be reported with original result IDs and reasons.

## 15. Human-participant study

The human study evaluates preference-aware interaction rather than robot motion
quality.

### 15.1 Conditions

Primary within-subject comparison:

- T5-Full;
- A-Memory.

Agent-topology variants should remain in the technical simulation experiment
unless a separate human-facing hypothesis justifies adding them.

### 15.2 Participant tasks

Each participant:

1. teaches a default preference with explicit consent;
2. requests a related task in a new layout;
3. issues an underspecified request that should use the stored preference;
4. provides a one-run override;
5. updates or forgets the stored preference;
6. observes a recoverable execution failure;
7. decides whether intervention is necessary.

Use equivalent task sets with different colours, layouts, and orders.

### 15.3 Design

- counterbalanced within-subject design;
- balanced Latin-square condition/task order;
- isolated memory store per participant and condition;
- standardized training;
- identical interface and model backend;
- experimenter-blinded condition labels where practical;
- no unsafe ablation controlling real hardware.

### 15.4 Sample size

- conduct an excluded pilot;
- use simulation-based power analysis;
- budget approximately 48-60 completed participants;
- replace this range with the powered target before preregistration.

### 15.5 Primary human outcomes

- preference-correct completion;
- number of user corrections;
- clarification turns;
- time to confirmed goal;
- unauthorized memory writes;
- appropriate intervention after an error.

### 15.6 Secondary human outcomes

- workload using a validated instrument such as NASA-TLX;
- perceived control;
- interaction predictability;
- behavioral trust calibration;
- qualitative failure explanations.

High reported trust is not automatically desirable. Appropriate reliance and
intervention behavior are more meaningful than an uncalibrated trust score.

### 15.7 Ethics and reporting

Before recruitment:

- obtain required institutional ethics approval;
- preregister hypotheses, analysis, and exclusions;
- define consent and withdrawal procedures;
- define compensation;
- document privacy and data retention;
- determine whether videos or voices are identifiable;
- prepare an anonymized artifact plan.

Report participant:

- recruitment and sampling;
- demographics relevant to the research question;
- prior robotics/AI experience;
- exclusions and attrition;
- study setting;
- interaction duration;
- compensation;
- analysis code.

Current HRI publication guidance expects detailed human-study methodology and
encourages reproducible artifacts:
[HRI full-paper guidance](https://humanrobotinteraction.org/2026/full-papers/).

## 16. Required implementation architecture

Variants must be built through declarative profiles, not copied codebases.

### 16.1 ExperimentProfile

```text
ExperimentProfile
├── profile_id
├── topology
│   ├── logical_role_to_agent_instance
│   ├── shared_context_policy
│   └── prompt_set
├── mechanisms
│   ├── memory_mode
│   ├── planning_mode
│   ├── step_assessment_mode
│   ├── final_completion_mode
│   ├── confirmation_mode
│   ├── evidence_policy
│   └── dynamic_scope_policy
├── safety_tier
└── allowed_executors
```

### 16.2 Role protocols

Define interfaces for:

- `InteractionPolicy`;
- `PreferenceService`;
- `GoalPlanner`;
- `StepAssessor`;
- `GoalAssessor`.

One model object may implement several interfaces for merged topologies.

### 16.3 Refactor requirements

1. Stop `HRI_Agent` from constructing Planner and Memory internally.
2. Inject Planner and Memory through a system assembler.
3. Add a role router mapping logical roles to model instances.
4. Implement phase-specific prompts/schemas for merged agents.
5. Add adapters for:
   - empty memory;
   - direct-to-Planner interaction;
   - open-loop queue execution;
   - executor-settled synthetic success;
   - Planner-declared completion;
   - auto-confirmation;
   - single evidence;
   - frozen dynamic membership.
6. Preserve the existing production profile as T5-Full.
7. Reject unsafe profiles when executor is not MuJoCo or synthetic.

Current composition seams are in:

- [runtime.py](../src/prefmem/runtime.py);
- [controller.py](../src/prefmem/controller.py);
- [hri.py](../src/prefmem/agents/hri.py);
- [monitor.py](../src/prefmem/agents/monitor.py);
- [validator.py](../src/prefmem/agents/validator.py).

### 16.4 Scenario schema v2

Each scenario must include:

```json
{
  "id": "R-DYN-001",
  "claim_id": "RQ3",
  "suite": "mechanism_isolation",
  "split": "locked_test",
  "family": "perturbation",
  "difficulty": {
    "horizon": 4,
    "ambiguity": 0,
    "occlusion": "none"
  },
  "safety_tier": "simulation_only",
  "applicable_profiles": ["T5", "A-OpenLoop"],
  "primary_outcomes": [
    "perturbation_recovery",
    "safe_completion"
  ],
  "seed_dimensions": [
    "layout",
    "paraphrase",
    "trigger_timing"
  ],
  "preflight": {
    "grounding_required": true,
    "reachability_required": true
  }
}
```

### 16.5 Oracle requirements

- Every declared expectation must be executable.
- Unknown `expected` keys must fail catalogue validation.
- Event-order claims use sequence numbers.
- Physical claims use hidden state.
- Dialogue claims parse structured turns, not punctuation counts.
- Memory claims inspect mutation events and final storage.
- No oracle may return a constant by construction.
- Each oracle has unit tests with passing and failing fixtures.

## 17. Runner and artifact requirements

### 17.1 Matrix runner

The runner must support:

- system profile selection;
- multiple model backbones;
- scenario split selection;
- matched seed generation;
- repeated trials;
- blocked/randomized order;
- resume without duplication;
- rerun into a new provenance directory;
- pilot versus locked-test mode;
- safety-profile validation.

### 17.2 Per-episode artifacts

Store:

- profile and logical-role mapping;
- exact prompts and schemas;
- raw model requests/responses;
- scene and setup;
- user inputs;
- runtime states;
- ordered event log;
- trigger log;
- executor events;
- Monitor and Validator assessments;
- memory before/after and mutation history;
- initial/final hidden state;
- objective oracle results;
- token/latency metrics;
- initial/final images;
- video where required;
- error and traceback data.

### 17.3 Campaign manifest

Freeze:

- git commit;
- dirty-worktree patch or clean-state declaration;
- dependency lockfile;
- container image digest;
- OS, GPU, driver, and runtime versions;
- model/tokenizer/embedding/detector hashes;
- model serving parameters;
- prompt and schema hashes;
- scene and scenario-catalogue hashes;
- oracle implementation hash;
- all seed lists;
- randomized execution order;
- analysis version.

### 17.4 Paper reproduction

Provide:

- one smoke-test command;
- one command for the deterministic invariant suite;
- one command for each main experiment matrix;
- one analysis command per paper table/figure;
- a machine-readable data dictionary;
- a license;
- expected runtime, GPU, and storage;
- a small public sample campaign;
- an anonymized artifact for double-blind review.

## 18. Quality gates before locked testing

The locked campaign may begin only when:

- production T5 passes all deterministic invariants;
- every variant manifest records the expected topology;
- all phase schemas have parser tests;
- all oracles have positive and negative fixtures;
- mechanism-isolation scenarios pass independent grounding/reachability
  preflight;
- unsafe profiles are rejected outside simulation;
- pilot artifact audit passes;
- analysis scripts run from raw pilot artifacts;
- primary hypotheses and exclusions are preregistered;
- prompts, thresholds, catalogue, and code are frozen.

## 19. Recommended figures and tables

### Systems/robotics paper

- Figure 1: T5-T1 agent-topology diagram.
- Figure 2: contract success by scenario family and system.
- Figure 3: perturbation recovery with paired confidence intervals.
- Figure 4: false completion and safety violations.
- Figure 5: success/cost Pareto frontier.
- Table 1: system and ablation definitions.
- Table 2: primary mixed-model estimates.
- Table 3: failure taxonomy by component.
- Table 4: real-system results.

### HRI/personalization paper

- Figure 1: consented preference lifecycle.
- Figure 2: participant/task flow.
- Figure 3: corrections and clarification turns.
- Figure 4: preference satisfaction and appropriate intervention.
- Table 1: participant demographics and study design.
- Table 2: mixed-model results.
- Table 3: memory failure taxonomy.

### Benchmark/artifact paper

- Figure 1: benchmark families and event boundaries.
- Figure 2: scenario-generation and oracle pipeline.
- Table 1: template coverage.
- Table 2: baseline results.
- Table 3: reproducibility and artifact contents.

## 20. Publication strategy

### Paper A: systems and robustness

Primary contributions:

- preference-aware receding-horizon architecture;
- deterministic authority and evidence fencing;
- agent-topology study;
- mechanism ablations;
- perturbation and safety benchmark;
- simulation and real-system results.

Potential venue types:

- robotics systems;
- robot learning;
- technical or systems HRI.

### Paper B: human interaction and personalization

Primary contributions:

- consented persistent preference memory;
- explicit override/update/forget behavior;
- effect on user correction and clarification burden;
- human trust calibration and perceived control.

Potential venue types:

- HRI;
- HCI/interactive agents;
- personalization.

### Paper C: benchmark/artifact

Only pursue this paper if the released suite itself is a distinct contribution:

- reusable scenario generators;
- event-boundary perturbations;
- temporal oracles;
- agent-topology profiles;
- reproducible baselines;
- public artifacts.

Avoid dividing identical results across papers without distinct research
questions and contributions.

## 21. Execution priority

If resources or time are limited, implement in this order:

1. declarative system profiles and role injection;
2. executable temporal oracles and scenario schema v2;
3. deterministic invariant suite;
4. T5-Full, T1-Unified, B1, B2, A-Memory, A-OpenLoop, A-Monitor, and
   A-Validator;
5. component benchmarks and feasibility preflight;
6. locked repeated simulation campaign;
7. remaining topology/mechanism variants;
8. model-sensitivity analysis;
9. safety-screened real-system subset;
10. ethics-approved human study;
11. external benchmark adapters.

## 22. Expected interpretation boundaries

The experiments may support claims such as:

- the full architecture improves contract success relative to named baselines;
- receding-horizon feedback improves recovery under tested perturbations;
- independent validation reduces observed false completion;
- consented memory reduces user interaction in tested preference tasks;
- role separation changes reliability and cost under controlled conditions.

They must not be used to claim:

- general real-world robot reliability from simulation alone;
- safety certification;
- arbitrary robot or scene compatibility;
- statistical reliability from one seed;
- learning from autonomous rollouts;
- superiority to an external benchmark method that was not faithfully
  reproduced;
- human preference benefit without the human study;
- causal attribution when multiple mechanisms changed simultaneously.

This suite is designed to turn PrefMem's current engineering evidence into
controlled, attributable, statistically replicated, and reproducible
conference-paper evidence.
