# Analyse project experiments

> Must read:
> - Project context: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
> - Project workflow: `System Workflow` section in [README.md](README.md)
> Read when needed:
> - Project experiments: [experiments](experiments/)
>
> TODO: Analyse the experiments I did.

<details><summary>3 previous messages</summary>

> I’ll first absorb the project context and the README’s system workflow, then inventory the experiment artifacts and trace results back to the intended hypotheses and metrics. I’ll keep this read-only and finish with an evidence-backed analysis, including anomalies and concrete next experiments.
>
> The experiment suite is a single canonical 39-scenario MuJoCo run, so I’ll treat “24/39 passed” as contract coverage from one campaign—not as a statistical success rate. I’m now cross-checking the authored report against the raw per-scenario oracle outputs and the scenario definitions, especially the 15 failures and any rerun provenance.
>
> The raw evidence confirms a useful pattern: failures were generally contained rather than falsely declared successful. Twelve failed cases ended in `NEEDS_ATTENTION`, two unsupported requests stopped at confirmation, and S12 ended `COMPLETE` only after the physically correct scene became visible again. The weak area is recovery under perturbation; the strongest area is HRI/memory and deterministic fencing. I’m doing a final methodological audit now, including whether each declared expectation is actually enforced by an oracle.
</details>

## Bottom line

The experiments validate PrefMem’s orchestration and safety fencing, but not yet robust autonomous recovery. The defensible result is **24/39 scenario contracts passed (61.5%) in one simulated campaign**, not a general success rate.

The strongest evidence is for nominal stacking, confirmation/memory behavior, and fail-safe handling. The weakest area is perturbation recovery.

## Scorecard

| Category | Raw result | Excluding non-isolating scenarios* | Interpretation |
|---|---:|---:|---|
| Nominal / scene | 6/10 | 6/7 | Standard stacking and scope handling work; longer or novel tasks remain fragile |
| Perturbation | 3/9 | 3/8 | Principal weakness: only R04, R08, and R09 passed |
| Boundary / fault | 7/12 | 7/10 | Host fences generally work; capability and occlusion handling do not |
| HRI / memory | 8/8 | 8/8 | Strong proof-of-path, but only scripted single trials |
| **Total** | **24/39** | **24/33** | Diagnostic coverage, not statistical reliability |

\*The adjusted column excludes the six cases classified as scenario-design infeasibilities. It should not replace the raw end-to-end result: failures to ground pads or detect reachability are still limitations users would experience.

The campaign and its classifications are documented in the [canonical report](./experiments/runs/full_20260809/EXPERIMENT_REPORT.md:5) and [failure analysis](./experiments/runs/full_20260809/failure_analysis.md:23).

## What the evidence supports

- The receding-horizon workflow works on the clean core: S01–S06 passed single-cube, ordered-stack, partial-stack, already-satisfied, closed-scope, and open-scope tasks.
- HRI behavior is promising: clarification, memory consent, one-run override, goal revision fencing, duplicate-object ambiguity, invalid Planner output, and the empty-memory ablation all passed.
- Deterministic guards behaved well: stale/single-frame evidence fencing, depth/compiler/frame failures, reachability rejection, and emergency latching.
- Failures were generally contained. Of the 15 failed contracts, 12 ended in `NEEDS_ATTENTION`, two unsupported requests stopped at confirmation, and S12 reached `COMPLETE` only after the physically correct scene was visible again. No unmet physical goal was falsely declared complete.
- Artifact quality is unusually strong: **39/39 scenarios and 585/585 files passed audit**, with full video, state, trigger, dialogue, and physical-pose evidence ([artifact audit](./experiments/runs/full_20260809/artifact_audit.md:1)).

## Main failure pattern

The nine cleanly isolated product failures cluster into three areas:

1. **Monitor/Validator semantics — 4 cases**

   S07 emitted an invalid `MET` task-status enum; R01 and R05 kept visible contradictions `ONGOING` until timeout; S12 interpreted occlusion as `NOT_MET` instead of `UNKNOWN`.

2. **Planning and capability awareness — 4 cases**

   R03 exceeded Planner schema bounds; R07 published a task for an absent object; B03 and B04 offered confirmation for unsupported rotation and pushing. R10’s destructive action on an already-correct tower is an additional serious Planner issue, despite that scenario being classified as non-isolating.

3. **Dynamic recovery — 1 case**

   R02 correctly cancelled twice on scene changes, but later failed to settle after recovery. This suggests the fencing logic is stronger than the resumed motion path.

The six non-isolating cases also reveal useful system boundaries: SAM could not reliably distinguish grey/red pads in S08, S09, and R06; S10 exposed reachability preflight weakness; R10 failed to reach validation; F04 successfully tested stale-publication rejection but lacked a final-evidence recovery turn.

## Methodological cautions

- Every scenario was run once with seed 7. Model variability, confidence intervals, and long-horizon reliability are therefore unknown.
- Tasks use one MuJoCo cube-stacking domain. The evidence does not cover a real robot, human/webcam execution, varied lighting, unfamiliar objects, or broad language variation.
- There is no full-system baseline. A01 is a useful memory ablation, but the deferred open-loop and validator-disabled ablations mean the experiments cannot yet quantify the benefit of receding-horizon control or independent validation.
- Most evidence is objective, but some automated oracles are weaker than their names imply. For example, `safe_hold_before_replan` checks that cancellation and a scene change both occurred, not their order; `replan_after_trigger` only checks that a trigger occurred and the final cycle exceeded one ([oracle implementation](./experiments/harness/results.py:274)).
- Only `expected.terminal_states` is automatically evaluated. Other `expected` fields matter only when duplicated by explicit oracles ([evaluator](./experiments/harness/results.py:60)). Consequently, declarations such as S04’s `preview_status`, H03’s confirmation-time memory-write count, and B03/B04’s unsupported-operation explanation are not directly part of the pass calculation.
- H01, H04, and R03 contain re-evaluation histories, but the report only explains the H01/R03 oracle corrections. H04’s re-evaluation should also be documented.
- Trace reproducibility is good, but rerun reproducibility is incomplete: the [run manifest](./experiments/runs/full_20260809/run_manifest.json:45) lacks the code commit, dependency/environment snapshot, model hashes, decoding settings, and service/hardware versions.

## Recommended next iteration

1. Add capability- and feasibility-aware preview gates, absent-object checks, and an already-satisfied/no-destructive-action guard.
2. Normalize bounded model enums and schema violations, then make persistent visible `NOT_MET` evidence terminal while reserving `UNKNOWN` for occlusion.
3. Harden cancellation recovery and debounce duplicate dynamic-scene changes.
4. Redesign S08, S09, S10, R06, R10, and F04 before rerunning.
5. Replace descriptive `expected` fields with executable oracles and make temporal oracles check event sequence numbers.
6. Run repeated trials across seeds and prompt paraphrases, with matched full-system ablations and a fully pinned provenance manifest.

Overall: this is a strong engineering validation suite and a credible demonstration of safety-oriented orchestration. It currently supports “the architecture and several invariants work in simulation,” but not yet “the autonomous system is reliable under perturbations.” No files were changed.

> I want to design a full experiment suite. Make sure also include ablation experiments.
>
> Note:
> 1. I suppose ablation experiments will need to modify the agentic system code (several system with different amount of agents). This is fine, just make sure you include the system designs for each ablation experiments.
> 2. This experiment suite results will be use for publish conference papers.

<details><summary>5 previous messages</summary>

> I’ll design this as a publication-grade evaluation program, not merely expand the current 39 scenarios. I’ll anchor each experiment to an explicit research claim, define implementable system variants for every ablation, separate deterministic invariant tests from stochastic end-to-end trials and human-subject evaluation, and specify the statistical analysis and reproducibility package. I’ll also check current robotics/HRI evaluation practice and inspect PrefMem’s actual composition points so the ablations are technically honest.
>
> <details><summary>Explored 4 files, 2 searches, a list, searched the web</summary>
>
> <details><summary>Explored 8 files, 2 searches, a list</summary>
>
> - Listed files in `prefmem`
> - Searched for `^class |^def |^async def |Agent|Service|RecedingHorizon|ValidationContract|GoalContract|Memory` in `runtime.py`
> - Read `./src/prefmem/runtime.py`
> - Read `./src/prefmem/runtime.py`
> - Read `./src/prefmem/controller.py`
> - Read `./src/prefmem/runtime.py`
> - Read `./src/prefmem/agents/hri.py`
> - Read `./src/prefmem/agents/config.py`
> - Searched for `def _run_planning_cycle|def _on_monitor|def _on_validator|def _publish_task|def _on_execution|def context_dict|def confirm_goal|def submit_planner|def submit` in `runtime.py`
> - Read `./src/prefmem/runtime.py`
> - Read `./src/prefmem/controller.py`
>
> </details>
>
> Searched the web for `site:humanrobotinteraction.org 2026 papers evaluation reproducibility human subjects HRI conference guidelines ...`
>
> Searched the web for `site:openreview.net/forum robot manipulation LLM planner replanning preference memory benchmark ICLR CoRL 2025 ...`
>
> Searched the web for `"MEMORYBENCH" robotic manipulation official project benchmark ...`
>
> Searched the web for `site:ntrs.nasa.gov NASA Task Load Index Hart Staveland original pdf ...`
>
> </details>
>
> The clean implementation strategy is to add a profile-driven system assembler rather than maintain separate forks. PrefMem already dependency-injects Planner, Monitor, Validator, controller, and executor; the main refactor is to stop `HRI_Agent` from constructing Planner and Memory internally. That will let every ablation share identical host logic, physics, perception, seeds, and scoring while changing only the declared agent topology or mechanism under test. Unsafe variants such as “no confirmation” or “no Monitor” should be simulation-only.
>
> The suite will use two distinct ablation families: an agent-topology ladder (5→4→3→2→1 model agents) and mechanism ablations (memory, receding horizon, Monitor, Validator, confirmation, temporal evidence, and dynamic-scope guard). Keeping those separate is important: otherwise a “fewer agents” result cannot tell whether role merging or removal of a specific capability caused the effect. The main benchmark will use matched scene seeds, language paraphrases, and perturbation schedules across variants, with hidden state used only for scoring.
>
> <details><summary>Plan</summary>
>
> The complete suite is specified: research claims, topology and mechanism variants, scenario strata, matched trial protocol, metrics, statistical models, human study, implementation seams, and artifact requirements.
>
> - [x] Inspect PrefMem composition and identify clean ablation seams
> - [x] Review current primary-source robotics/HRI evaluation practice and baselines
> - [x] Define claims, system variants, scenarios, metrics, and statistical protocol
> - [x] Deliver a complete conference-ready experiment-suite specification
>
> </details>
</details>

# Recommended experiment-suite design

Build the next suite as a claim-driven, versioned benchmark with two separate ablation families:

1. **Agent-topology ablations:** the same five logical roles implemented using 5, 4, 3, 2, or 1 model agents.
2. **Mechanism ablations:** remove Memory, receding-horizon replanning, Monitor, Validator, confirmation, temporal evidence, or dynamic-scope guarding one at a time.

Keeping these families separate is essential. Otherwise, a result from “fewer agents” cannot distinguish role merging from loss of a particular capability.

The existing 39 scenarios should become the **development and regression set**, not the final paper test set, because they have already influenced debugging and analysis.

## 1. Research questions

| ID | Research question | Primary comparison |
|---|---|---|
| RQ1 | Does PrefMem improve safe holistic task completion? | Full vs open-loop and feedback-only baselines |
| RQ2 | Does preference memory reduce interaction while preserving consent and overrides? | Full vs no-Memory |
| RQ3 | Does receding-horizon replanning improve recovery from physical changes? | Full vs open-loop |
| RQ4 | Does per-step visual monitoring improve recovery and prevent stale-success propagation? | Full vs no-Monitor |
| RQ5 | Does independent final validation reduce false completion? | Full vs no-Validator |
| RQ6 | Do exact confirmation and host fencing prevent unauthorized execution? | Full vs no-confirmation; deterministic fence tests |
| RQ7 | Does repeated temporal evidence reduce decisions based on transient frames? | Full vs single-evidence |
| RQ8 | What is the effect of agent decomposition? | 5-agent vs 4/3/2/1-agent topologies |
| RQ9 | How do quality, latency, token usage, and failure containment trade off? | All systems |
| RQ10 | Do technical improvements translate into better human interaction? | Human study: Full vs no-Memory |

Pre-register directional hypotheses for RQ1–RQ7. Treat RQ8 as partly exploratory: more agents may improve role isolation but also increase calls, latency, and failure surfaces.

## 2. What counts as an agent

For the topology experiment, define an agent as:

> An independent model instance with its own role prompt and conversational state.

Do not count:

- `PrefMemRuntime`;
- `RecedingHorizonController`;
- the memory database;
- schema validation or oracle code;
- the fixed lower-level Gemma/SAM/Panda execution pipeline.

Those components remain constant unless a specific mechanism ablation explicitly changes them.

## 3. Agent-topology systems

All five systems receive the same evidence, tools, physical executor, phase-specific output schemas, and logical decision opportunities. The treatment is which roles share model state.

| ID | Agents | System design |
|---|---:|---|
| **T5-Full** | 5 | Independent HRI, Memory, Planner, Monitor, and Validator agents. This is the reference system. |
| **T4-InteractionMemory** | 4 | Merge HRI and Memory into one `InteractionMemoryAgent`. It has direct retrieval/mutation tools and handles consent. Planner, Monitor, and Validator remain independent. |
| **T3-ThreeRole** | 3 | `InteractionMemoryAgent`; independent Planner; combined `VerifierAgent` for both per-step monitoring and holistic validation. |
| **T2-ActorCritic** | 2 | `ActorAgent` combines HRI, Memory, and Planner. `CriticAgent` combines Monitor and Validator. The host still freezes goals and routes phase-specific calls. |
| **T1-Unified** | 1 | One stateful `UnifiedAgent` performs interaction, memory operations, preview/planning, step assessment, and final assessment. The host invokes it in phase-specific modes with strict schemas. |

Important controls:

- Use the same model weights, decoding settings, image resolution, and tool implementations.
- Construct merged prompts from the constituent role instructions rather than writing deliberately weaker prompts.
- Preserve phase-specific schemas. Do not make T1 solve an artificially harder unstructured-output problem.
- Keep the same host-owned IDs, goal contracts, publication fences, and emergency coordinator.
- Report actual model calls, tokens, wall time, and peak context size rather than forcing identical costs.

This comparison measures the effect of **role isolation and state sharing**, not the value of individual mechanisms.

## 4. Mechanism ablation systems

| ID | Change from T5-Full | Exact system behavior | Safety scope |
|---|---|---|---|
| **A-Memory** | Remove Memory Agent | Memory tool always returns an empty result; no persistent writes. HRI must clarify underspecified preferences. | Simulation and human study |
| **A-HRI** | Remove HRI Agent | Raw user instruction goes directly to Planner. No dialogue-level clarification; confirmation may remain as a host UI control. | Simulation |
| **A-OpenLoop** | Remove receding-horizon replanning | Planner produces one complete queue after confirmation. Monitor may assess steps, but the remaining queue is never regenerated from new observations. | Simulation and safety-screened robot tasks |
| **A-Monitor** | Remove Monitor Agent | In MuJoCo, executor `SETTLED` becomes a synthetic step success. Planner replans without visual verification. | Simulation only |
| **A-Validator** | Remove Validator Agent | Planner receives a fresh final frame and may emit `DECLARE_COMPLETE`; the host accepts this without a frozen independent checklist. | Simulation only |
| **A-Confirmation** | Disable explicit confirmation | A valid preview is automatically confirmed and executed. Clarification remains enabled so this isolates confirmation rather than dialogue. | Simulation only |
| **A-SingleEvidence** | Disable temporal stability | One accepted success/failure frame is terminal; stability interval is zero. | Simulation only |
| **A-DynamicGuard** | Disable live open-category scope | Membership freezes at confirmation; new objects do not trigger cancellation or replanning. | Simulation only |
| **A-HostFence** | Disable trusted evidence/publication IDs | Accept model-supplied or stale identities. Use scripted assessments only; never route actions to a robot. | Synthetic event tests only |

Avoid a vague “no Validator” implementation that declares success after a fixed number of actions. Let the Planner explicitly decide completion from a fresh frame; this is a meaningful comparison with independent verification.

## 5. External and architecture baselines

Use at least these:

### B1: Open-loop LLM planner

One LLM creates the full task sequence from the initial observation. It receives no subsequent scene feedback. This resembles the common one-shot language-plan pattern and tests the value of closed-loop control.

### B2: Feedback-only single-agent planner

One planning agent receives scene descriptions and step-success feedback, then replans. It has no preference memory, exact goal contract, or independent final validator.

This should be described as **Inner Monologue/RePLan-inspired**, not an exact reproduction unless their official code and protocol are used. Both prior systems specifically study the value of environmental feedback and replanning: [Inner Monologue](https://arxiv.org/abs/2207.05608) and [RePLan](https://arxiv.org/abs/2401.04157).

### B3: Oracle ceiling

Use hidden scene state to select feasible actions and verify outcomes. It does not represent a deployable system; it separates high-level agent failures from perception, IK, and controller limits.

### B4: No-personalization closed-loop system

Equivalent to T5-Full without persistent memory. This is both a baseline and the primary Memory ablation.

For external validity, consider an adapted preference-planning evaluation based on the tasks in the [PbP benchmark](https://arxiv.org/abs/2502.00858). Clearly label it adapted because PrefMem learns only explicitly consented textual preferences, whereas PbP commonly derives preferences from demonstrations.

BEHAVIOR-1K offers broader household tasks and scenes, but PrefMem’s current Panda adapter is not compatible with it. Do not claim BEHAVIOR results without building a genuine executor adapter that respects its observation and evaluation rules. [BEHAVIOR-1K](https://behavior.stanford.edu/index.html) is a useful longer-term external benchmark.

## 6. Experiment layers

A paper-quality suite should have four layers.

### Layer A: Deterministic invariant tests

Use scripted model outputs and property-based event generation. These should test the host, not model intelligence.

Required families:

- exact goal/revision confirmation;
- stale confirmation rejection;
- horizon-one publication;
- stale/wrong-publication evidence;
- evidence from an older frame;
- success and failure confirmation thresholds;
- task retirement before replacement publication;
- safe hold before replan;
- final-validation routing isolation;
- immutable goal/checklist;
- loop and repeated-failure guards;
- emergency latching and idempotence;
- post-emergency publication rejection;
- consented versus unauthorized memory writes.

Generate hundreds of randomized event sequences. Every temporal oracle must compare ordered event sequence numbers, not merely check that two events both occurred.

### Layer B: Component benchmarks

These prevent an end-to-end failure from obscuring which component broke.

| Component | Initial target | Examples |
|---|---:|---|
| HRI/Planner | 500 instances | ambiguity, conflicts, unsupported actions, exact revisions, schema validity |
| Memory | 300 interaction sequences | remember, retrieve, update, forget, override, interference, no-consent cases |
| Monitor/Validator | 1,000 labeled frames or short clips | `MET`, `NOT_MET`, `UNKNOWN`, partial/persistent occlusion, contradiction |
| Grounding/execution | 500 configurations | selector accuracy, ambiguous matches, depth validity, IK/reachability, target movement |

Report confusion matrices and error types, not only scalar accuracy.

### Layer C: End-to-end simulation benchmark

Use 62 scenario templates:

| Family | Templates | Coverage |
|---|---:|---|
| Intent and confirmation | 8 | clear request, vague outcome, referential ambiguity, conflicting constraints, revision, stale confirmation, unsupported primitive, mid-task instruction |
| Preference memory | 12 | remember, immediate recall, delayed transfer, paraphrase, override, update, forget, conflict, irrelevant retrieval, no consent, confirmation-not-write, interference |
| Nominal manipulation | 10 | horizons 0/1/3/6/10, partial goal, already satisfied, closed/open scope, distractors, multiple targets, multiple boards |
| Perturbation and recovery | 14 | source/target movement, dynamic entry at several phases, removed objects, collapse, miss, recoverable fault, protected-object displacement, repeated failure |
| Evidence and validation | 8 | transient/persistent/partial occlusion, stale assessment, single success frame, alternating evidence, incomplete final goal, satisfied-goal no-op |
| Safety, faults, capability | 10 | invalid schemas/enums, corrupt depth, camera timeout, emergency, out-of-workspace, missing object, unsupported action, loop guard |

Each template should generate instances across:

- scene/layout seeds;
- prompt paraphrases;
- object identities and distractors;
- task horizons;
- perturbation timing;
- camera and lighting conditions;
- open versus closed object scope.

### Layer D: Real-system and human evaluation

Use a safety-screened subset only:

- four nominal tasks;
- four preference/interaction tasks;
- two recoverable perturbations;
- two validation/occlusion tasks.

Run T5-Full, the feedback-only baseline, and the most informative ablation. Do not run no-confirmation, no-Monitor, or disabled-fence variants on hardware.

## 7. Data splits and scenario validity

Use three disjoint splits:

- **Development:** current 39 scenarios plus newly authored debugging cases.
- **Pilot:** unseen layouts and paraphrases for estimating variance and checking instrumentation.
- **Locked test:** held-out templates, layouts, language, and perturbation schedules.

Do not tune prompts or product code on locked-test results. If a defect requires a code change, close the campaign, increment the system version, and begin a new campaign.

Maintain two separate suites:

1. **Mechanism-isolation suite:** grounding and reachability are preflighted so the intended boundary is guaranteed to execute.
2. **End-to-end capability suite:** no preflight exclusion; perception and execution failures count against the system.

This prevents another situation where a pad-detection error invalidates a recovery experiment while still preserving that detection failure as an end-to-end result.

## 8. Trial matrix

Let \(r\) be matched repetitions per template. Determine it using a pilot and simulation-based power analysis; expect approximately 15–25 rather than one run.

Recommended matrix:

- Main comparison: 62 templates × \(r\) × {T5, B1, B2, B4}.
- Agent topology: 24 stratified templates × \(r\) × {T5, T4, T3, T2, T1}.
- Mechanism ablations: each ablation runs only on its 8–16 relevant templates.
- Model sensitivity: 20 representative templates × 10 repetitions × three frozen VLM backbones.
- Real-system subset: 12 tasks × at least five repeats × three safe systems.

Use common random numbers: the same scene seed, prompt paraphrase, and perturbation schedule should be paired across systems.

## 9. Metrics

Do not collapse everything into one score.

### Primary outcomes

1. **Physical goal completion:** hidden-state goal predicates all pass.
2. **Contract success:** goal, preferences, preservation constraints, confirmation, and safety oracles all pass.
3. **False-completion rate:** controller says `COMPLETE` while a physical or preference condition is unmet.
4. **Safe completion:** physical success with no critical authorization, collision, protected-object, or emergency violation.
5. **Perturbation recovery rate:** completed after a fired perturbation without operator rescue.
6. **Preference satisfaction:** proportion of applicable explicit and stored preferences obeyed.

### Secondary outcomes

- user clarification turns;
- unnecessary clarification rate;
- unauthorized memory mutation rate;
- override/update/forget correctness;
- physical publication count;
- cycles and repeated actions;
- recovery latency;
- timeout/attention rate;
- Planner, Monitor, and Validator schema-error rates;
- `MET`/`NOT_MET`/`UNKNOWN` confusion matrices;
- protected-object displacement;
- collision and workspace violations;
- emergency-stop latency;
- model calls, tokens, inference latency, execution time, and compute cost.

Report both conditional recovery—given the intended trigger fired—and unconditional end-to-end success.

## 10. Statistical analysis

Use the episode as the experimental unit. Frames and steps within an episode are not independent samples.

- Binary outcomes: mixed-effects logistic regression.
- Counts: negative-binomial mixed models.
- Completion/recovery time with timeouts: survival analysis or censored accelerated-failure-time models.
- Continuous positive latency/cost: log-normal mixed models.
- Participant Likert responses: cumulative-link mixed models rather than treating individual items as continuous.
- Random effects: scenario template, generated instance/seed, and participant where applicable.
- Fixed effects: system condition, difficulty, perturbation family, task horizon, and selected interactions.
- Report effect sizes, estimated marginal means, and 95% confidence intervals.
- Apply Holm correction within each pre-registered family of pairwise comparisons.
- Include all model and execution failures. Exclude only verified harness/infrastructure failures using criteria fixed before testing.
- Publish per-seed values and bootstrap intervals, not only pooled averages.

The pilot should estimate variance and power for a pre-registered minimum effect, for example a 10-percentage-point change in contract success or false completion.

## 11. Human-subject experiment

For the personalization claim, use a counterbalanced within-subject study:

- **Conditions:** T5-Full and A-Memory.
- **Participants:** determine through simulation-based power after an excluded pilot; budget approximately 48–60 completed participants.
- **Task blocks:** teach a default, retrieve it later in a new scene, apply a one-run override, update or forget it, and handle a recoverable failure.
- **Counterbalancing:** balanced Latin square with different but equivalent colors/layouts per condition.
- **Primary endpoints:** preference satisfaction, number of corrections, clarification turns, and task completion time.
- **Secondary endpoints:** workload, perceived control, and behavioral trust calibration.
- **Behavioral trust:** whether participants appropriately intervene after visible errors, not merely whether they report high trust.
- **Workload:** a validated measure such as NASA-TLX may be used; avoid inventing an unvalidated composite score.

Obtain ethics approval before recruitment, preregister hypotheses and exclusions, record demographics and sampling, and document compensation and consent. Current HRI guidance explicitly expects detailed human-study methodology and encourages reproducible artifacts ([HRI full-paper guidance](https://humanrobotinteraction.org/2026/full-papers/)).

## 12. Required code architecture

Implement variants through configuration, not branches or copied repositories.

```text
ExperimentProfile
├── topology
│   ├── role_to_agent_instance
│   └── shared_context_policy
├── mechanisms
│   ├── memory_mode
│   ├── planning_mode
│   ├── step_assessment_mode
│   ├── completion_mode
│   ├── confirmation_mode
│   ├── evidence_policy
│   └── dynamic_scope_policy
└── safety_tier
```

Recommended changes:

1. Refactor `HRI_Agent` so Planner and Memory are injected instead of constructed internally.
2. Define protocols for interaction, preference, planning, step assessment, and final assessment.
3. Add a role router mapping logical roles onto shared or independent agent instances.
4. Implement open-loop, executor-settled, Planner-completion, and single-evidence adapters.
5. Add `profile_id` and complete role mapping to every result manifest.
6. Extend scenarios with `claim_id`, split, safety tier, difficulty, applicable profiles, and primary outcome.
7. Replace descriptive `expected` fields with executable oracles.
8. Add monotonic event IDs so every temporal claim is directly scored.
9. Hash every prompt, model, scene, configuration, and oracle implementation.

The current injection points in [runtime.py](./src/prefmem/runtime.py:63) and the role construction in [hri.py](./src/prefmem/agents/hri.py:149) make this refactor practical.

## 13. Publication artifact requirements

Every campaign should freeze:

- git commit and dirty-worktree status;
- dependency lockfile and container image digest;
- model, tokenizer, embedding, and detector hashes;
- prompts and schema hashes;
- decoding parameters;
- GPU, driver, OS, and service versions;
- profile and role mapping;
- scenario catalogue and split hashes;
- all seeds;
- raw model responses;
- state/event traces;
- initial/final hidden states;
- videos and endpoint images;
- analysis scripts that regenerate every table and figure.

Provide one command for a smoke test and one command per paper table/figure. HRI currently encourages artifacts that enable reproduction and recreation, while ACM artifact practice expects documented, complete, exercisable packages.

## 14. Paper strategy

A coherent publication program would be:

1. **Robotics/systems paper:** architecture, perturbation benchmark, topology ladder, mechanism ablations, safety containment, and real-system subset.
2. **HRI/personalization paper:** consented memory, overrides, correction burden, and the human study.
3. **Benchmark/artifact paper:** only if the released scenarios, generators, temporal oracles, and baselines constitute a genuinely reusable benchmark.

Avoid splitting the same experiment table across papers without distinct research contributions.

## Execution priority

If resources are limited, implement in this order:

1. profile-driven architecture and rigorous temporal oracles;
2. T5, T1, open-loop, feedback-only, no-Memory, no-Monitor, and no-Validator;
3. locked simulation benchmark with repeated matched trials;
4. remaining topology and safety ablations;
5. real-system subset;
6. ethics-approved human study;
7. external benchmark adapters.

This design would support substantially stronger claims than the present single-run catalogue while keeping every comparison attributable, reproducible, and safe. No files were changed.

> Write this experiment suite design to a markdown file.

<details><summary>2 previous messages</summary>

> I’ll add the complete design as `experiments/EXPERIMENT_SUITE_DESIGN.md`, preserving the research questions, exact ablation topologies, protocols, statistics, human study, implementation architecture, and publication requirements from the proposal.
>
> <details><summary>Created a file, edited a file, ran 3 commands</summary>
>
> - Ran `test -e experiments/EXPERIMENT_SUITE_DESIGN.md; printf '%s\n' $?`
> - Wrote `./experiments/EXPERIMENT_SUITE_DESIGN.md` (+1513 -0)
> - Ran ````wc -l experiments/EXPERIMENT_SUITE_DESIGN.md && rg -n '@@BT@@|^#|^```' experiments/EXPERIMENT_SUITE_DESIGN.md | sed -n '1,220p' && git diff -- experiments/EXPERIMENT_SUITE_DESIGN.md | sed -n '1,120p'````
> - Wrote `./experiments/EXPERIMENT_SUITE_DESIGN.md` (+6 -6)
> - Ran ````printf 'code_fences='; rg -c '^```' experiments/EXPERIMENT_SUITE_DESIGN.md; printf 'placeholder_hits='; rg -c '@@BT@@' experiments/EXPERIMENT_SUITE_DESIGN.md || true; git status --short -- experiments/EXPERIMENT_SUITE_DESIGN.md; sed -n '1,45p' experiments/EXPERIMENT_SUITE_DESIGN.md; tail -n 28 experiments/EXPERIMENT_SUITE_DESIGN.md````
>
> </details>
</details>

Created [EXPERIMENT_SUITE_DESIGN.md](./experiments/EXPERIMENT_SUITE_DESIGN.md).

It contains the full publication-grade suite, including agent-topology and mechanism ablations, system designs, baselines, scenario catalogue, metrics, statistics, human study, implementation architecture, and reproducibility requirements.

> I will start a new CodeX session with effort level "Ultra", and let CodeX carry out the experiment for me. How should I insctuct CodeX to carry out the experiment professionally? Also, I want full video recordings for each experiment:
> - A SETUP.md including all setups for this experiment.
> - a experiment_record.md file to record thie experiment (such as user input, system output, camera frame at each request to the system)
> - a diagnosis.md for diagnosis or comment on each system behaviour. (This should have links or reference to the experiment_record.md)
> - Any more ideas from you?
>
> The servers of Gemma, SAM, and EmbeddingGemma are all running, and forwarding to localhost port with the following commands. The connection sometime can be disconnected, halt, or holding, you can use these commands to establish connection again.
> Gemma 4 26B A4B it:
> ssh -N -L 8000:127.0.0.1:8000 \
>   -p 21314 -i ~/.ssh/id_ed25519 \
>   root@185.216.21.214
>
> SAM 3.1:
> ssh -N -L 9000:127.0.0.1:9000 \
>   -p 11204 -i ~/.ssh/id_ed25519 \
>   root@213.173.103.97
>
> EmbeddingGemma:
> ssh -N -L 8080:127.0.0.1:8080 \
>   -p 11204 -i ~/.ssh/id_ed25519 \
>   root@213.173.103.97

<details><summary>3 previous messages</summary>

> I’m using the OpenAI/Codex documentation skill because this is specifically about orchestrating a long-running Codex session. I’ll turn your requirements into a paste-ready master instruction, add professional run-control and provenance safeguards, and refine the artifact set so recordings, diagnoses, interruptions, reruns, and tunnel recovery remain auditable.
>
> <details><summary>Searched the web</summary>
>
> <details><summary>Explored a file</summary>
>
> - Read `~/.codex/skills/.system/openai-docs/SKILL.md`
>
> </details>
>
> Searched the web for `site:developers.openai.com/codex long running tasks AGENTS.md prompting Codex ...`
>
> Searched the web for `site:developers.openai.com/codex/use-cases "Follow a goal"`
>
> </details>
>
> Context compaction
>
> Source: automatic
> Status: completed
</details>

The professional approach is a two-stage campaign:

1. Codex implements the harness, validates recording, and runs pilots.
2. You review and approve a frozen protocol before Codex starts the publication runs.

This prevents implementation bugs or prompt tuning from contaminating the locked test set. Because this is long-running work, use a durable Codex goal if the interface offers it; OpenAI specifically describes goals as a way to give Codex a durable objective for long-running work. [OpenAI Codex use cases](https://developers.openai.com/codex/use-cases)

Paste the following into the new Ultra-effort session.

## Master instruction for Codex

```text
You are the principal experiment engineer and reproducibility lead for the
RoboPref research project.

Your objective is to implement and execute the publication-grade experiment
suite described in this repository. Do not merely advise me or produce a plan:
inspect the system, implement the necessary experiment infrastructure, test it,
run pilots, freeze the protocol, and—after the mandatory approval gate—execute
the locked experiment campaign and produce auditable results.

If this Codex interface supports durable goals, create a goal for this objective
and maintain it until the campaign is genuinely complete.

======================================================================
1. MANDATORY READING
======================================================================

Before changing code or starting experiments, read completely:

1. PROJECT_CONTEXT.md
2. The "System Workflow" section of README.md
3. experiments/EXPERIMENT_SUITE_DESIGN.md
4. experiments/README.md
5. Existing reports, audits, failure analyses, run artifacts, and experiment
   scripts under experiments/
6. The relevant agent, runtime, Webots, recording, monitoring, validation, and
   model-client source code

Treat EXPERIMENT_SUITE_DESIGN.md as the target scientific protocol. Treat the
actual source code as authoritative about current implementation behavior.

Create a requirements crosswalk mapping every experiment-suite requirement to:

- implementation status;
- source file or experiment profile;
- test or preflight check;
- generated artifact;
- unresolved gap.

Do not silently omit or weaken any experiment, ablation, metric, scenario,
recording requirement, or statistical control.

======================================================================
2. AUTHORITY AND SAFETY
======================================================================

You are authorized to:

- inspect and modify code inside this repository;
- implement experiment profiles, ablation systems, instrumentation, recording,
  auditing, and analysis tools;
- execute simulation experiments;
- reconnect the three explicitly listed SSH tunnels when necessary;
- resume interrupted experiment batches according to the retry protocol below.

You are not authorized to:

- expose, read, copy, or print private SSH-key contents;
- publish or commit credentials or private connection details;
- kill unknown processes or use broad commands such as killall;
- push commits or open pull requests unless I separately ask;
- run human-participant studies without confirmed ethics/consent procedures;
- run unsafe real-robot ablations without separate authorization;
- alter unrelated user work in a dirty working tree.

Preserve existing changes. Record the repository commit and a content/diff hash
for every campaign. A clean, frozen commit is strongly preferred for locked
runs.

======================================================================
3. NON-NEGOTIABLE SCIENTIFIC RULES
======================================================================

1. Separate development, calibration, pilot, and locked-test data.
2. Never tune prompts, thresholds, scenarios, stopping rules, or code using
   locked-test outcomes.
3. Freeze before the locked campaign:
   - hypotheses and estimands;
   - scenario catalog and split;
   - agent-system profiles and model mappings;
   - prompts and thresholds;
   - seeds and randomized execution schedule;
   - oracle definitions;
   - retry/exclusion rules;
   - metrics and statistical analysis scripts.
4. Hash the frozen protocol and all relevant configuration files.
5. Do not edit the frozen configuration after locked execution begins.
6. If a substantive fix becomes necessary, terminate that campaign version,
   increment the campaign version, document the change, and restart every
   affected condition. Do not mix incompatible campaign versions.
7. Preserve every attempt, including failures, timeouts, interruptions, aborted
   runs, and invalid harness attempts. Never overwrite or cherry-pick runs.
8. A valid system failure is experimental data and must not be retried merely
   because it failed.
9. Infrastructure interruptions may be retried only under the predeclared
   infrastructure retry policy.
10. Run variants sequentially by default to avoid resource contention unless
    deterministic resource isolation has been demonstrated.

Use explicit run statuses:

- VALID_PASS
- VALID_SYSTEM_FAILURE
- INFRA_INTERRUPTED
- INVALID_HARNESS
- ABORTED_SAFETY
- NOT_RUN

Keep the oracle verdict separate from the later root-cause diagnosis.

======================================================================
4. EXECUTION PHASES AND APPROVAL GATES
======================================================================

Phase A — Audit and scaffold

- Inspect the repository, existing experiments, services, and working tree.
- Produce the requirement crosswalk and implementation plan.
- Estimate total run count, inference calls, wall-clock time, and video storage.
- Check available disk space with a safety margin.
- Create a deterministic campaign directory and run registry.
- Do not stop after planning; proceed directly to implementation.

Phase B — Implement the suite

- Implement the full-system profiles, topology ablations, mechanism ablations,
  baselines, scenarios, metrics, oracles, and configuration validation specified
  by EXPERIMENT_SUITE_DESIGN.md.
- Keep variants configuration-driven where scientifically valid.
- Where topology ablations require different code paths, implement explicit
  profile factories and verify the actual instantiated agent graph.
- Record the logical agent topology, model mapping, and enabled mechanisms in
  every setup artifact.

Phase C — Implement provenance and recording

- Implement all artifact requirements in Section 5 below.
- Make the machine-readable event log the source of truth.
- Generate the Markdown records from the machine-readable data.
- Add artifact integrity checks and checksums.

Phase D — Test and pilot

- Run deterministic unit and integration tests.
- Run preflight checks for every system profile.
- Verify that a requested ablation is genuinely active rather than only renamed.
- Run at least one successful pilot and one controlled failure pilot for each
  materially distinct recording/execution path.
- Confirm that videos, request frames, timestamps, events, outputs, oracle
  results, and diagnoses are mutually aligned.
- Test tunnel disconnection handling without relabeling a system failure as
  infrastructure failure.
- Audit pilot artifacts independently.

MANDATORY GATE 1 — Protocol freeze

After the pilot, stop and present:

- implementation summary;
- changed-file summary;
- test and pilot results;
- unresolved limitations;
- total locked-run count and resource estimate;
- storage estimate and available capacity;
- frozen protocol path and SHA-256 hash;
- scenario/config/prompt/source hashes;
- proposed randomized schedule;
- exact command that will start or resume the locked campaign;
- artifact examples with links.

Do not begin locked runs until I send:

APPROVE LOCKED CAMPAIGN <campaign_id> <protocol_sha256>

Phase E — Locked campaign

After approval:

- verify that the supplied protocol hash matches;
- re-run preflight without changing the protocol;
- execute the frozen schedule;
- checkpoint the run registry after every attempt;
- preserve partial results and remain safely resumable;
- provide concise progress updates;
- never silently reduce the number of seeds, repetitions, scenarios, variants,
  recordings, metrics, or artifact checks.

Phase F — Analysis and final audit

- Run only the predeclared statistical analysis on locked data.
- Generate all tables and plots directly from machine-readable artifacts.
- Produce robustness, ablation, failure-mode, latency, and cost analyses defined
  by the suite.
- Clearly separate confirmatory, secondary, exploratory, and post-hoc results.
- Run an independent completeness and checksum audit.
- Report exclusions and missing data without hiding them.
- Do not claim completion while scheduled entries remain unresolved.

======================================================================
5. CAMPAIGN AND EPISODE ARTIFACT CONTRACT
======================================================================

Use existing repository conventions if compatible. Otherwise use:

experiments/runs/<campaign_id>/
  CAMPAIGN_MANIFEST.json
  PROTOCOL.md
  PROTOCOL.sha256
  REQUIREMENTS_CROSSWALK.md
  RUN_STATE.json
  run_schedule.csv
  run_index.csv
  EXCLUSIONS.md
  CHANGELOG.md
  environment/
  analysis/
  artifact_audit/
  episodes/
    <schedule_id>/
      attempt_<number>/

Every attempt directory, including interrupted attempts, must contain:

- SETUP.md
- setup.json
- experiment_record.md
- diagnosis.md
- events.jsonl
- frame_requests.jsonl
- model_calls.jsonl
- runtime_states.jsonl
- inputs.json
- results.json
- results.md
- terminal.log
- initial_state.json
- final_state.json when available
- initial.png
- final.png when available
- frames/
- model_calls/
- video.mp4
- robot_camera.mp4
- video_metadata.json
- artifact_checks.json
- checksums.sha256

If a process dies before an artifact can be finalized, retain the partial file,
mark it clearly, and report its status. Never pretend it is complete.

======================================================================
6. SETUP.md REQUIREMENTS
======================================================================

SETUP.md must be generated from setup.json and include:

- campaign ID, schedule ID, attempt ID, and run status;
- UTC start/end time and monotonic clock origin;
- repository commit and dirty-tree/source-manifest hash;
- operating system, dependency versions, simulator version, and relevant
  hardware/GPU information;
- experiment family, variant/profile, repetition, scenario, and seed;
- instantiated logical agent graph;
- model/service mapping for every logical agent;
- model IDs reported by the servers;
- prompt, configuration, profile, scenario, and oracle hashes;
- all relevant timeouts, thresholds, budgets, and stopping rules;
- scene/world/controller and initial-state information;
- user task and expected oracle conditions;
- randomized schedule position;
- video and camera configuration;
- local service endpoints and health-check results;
- tunnel epoch identifiers without exposing secrets or private key contents;
- exact reproducibility command;
- known deviations, if any.

Do not include private SSH-key material or secrets. Public artifacts should use
service labels and localhost ports rather than publishing private infrastructure
details unnecessarily.

======================================================================
7. experiment_record.md REQUIREMENTS
======================================================================

events.jsonl is the append-only source of truth. experiment_record.md is its
human-readable chronological rendering.

Every event must have a stable event ID such as E000123 and include, where
applicable:

- UTC and monotonic timestamps;
- elapsed video time;
- robot/simulator state before and after;
- user input;
- receiving logical agent;
- exact model request ID;
- camera/frame reference;
- system or model output;
- parsed output;
- tool/action request and result;
- command publication and execution events;
- Monitor and Validator decisions;
- trigger/interrupt events;
- oracle observations;
- latency, timeout, retry, and error information.

Use stable anchors so diagnosis.md can link directly to evidence, for example:

[Event E000123](./experiment_record.md#event-e000123)

Link both the relevant request frame and video timestamp:

[request frame](./frames/<sha256>.png)
[video 00:01:23.400](./video.mp4#t=83.4)

For every request made to HRI, Planner, Executor, Monitor, Validator, or another
agent, save the exact camera bytes actually supplied to that request.

frame_requests.jsonl must map:

- request ID;
- logical agent;
- capture timestamp;
- camera/view ID;
- frame sequence number;
- image path;
- SHA-256 of the raw image;
- SHA-256 of the encoded request payload;
- prompt/request-body hash;
- corresponding response and event IDs.

Identical images may be content-deduplicated, but every request must retain its
own logical reference. Do not substitute a later screenshot for the image that
was actually sent to the model.

Store each raw model request and response under model_calls/ with secrets
redacted. Preserve malformed or unsuccessful responses as evidence.

======================================================================
8. diagnosis.md REQUIREMENTS
======================================================================

Create diagnosis.md only after the run outcome and raw record are finalized.
Diagnosis must never feed back into the running episode.

Include:

- run status and oracle verdict;
- concise behavioral summary;
- expected versus observed behavior;
- safety and liveness consequences;
- first divergence point;
- causal timeline;
- likely failure layer:
  - perception/SAM;
  - embedding/retrieval;
  - HRI;
  - planning;
  - execution;
  - monitoring;
  - validation;
  - coordination/state management;
  - simulator/robot;
  - model service/infrastructure;
  - recorder/oracle/harness;
- primary diagnosis and confidence;
- competing explanations and disconfirming evidence;
- whether behavior reflects the studied system or invalid infrastructure;
- recommended follow-up, without modifying locked results;
- reviewer notes and unresolved ambiguity.

Every factual diagnosis claim must cite one or more experiment_record.md event
anchors, request frames, raw model calls, or video timestamps. Example:

The monitor failed to react for 4.2 seconds after obstacle appearance
([E000143](./experiment_record.md#event-e000143),
[video 00:02:14.100](./video.mp4#t=134.1)).

Do not present speculative root causes as established facts.

======================================================================
9. FULL VIDEO RECORDING REQUIREMENTS
======================================================================

Record every attempt, including pilots, valid failures, infrastructure
interruptions, and safety aborts.

Follow the experiment-suite recording specification. Unless the existing design
specifies otherwise, use a synchronized 1920×1080, 10 FPS composite video with:

- third-person simulator view;
- exact robot-camera view;
- current episode/variant/seed;
- UTC and monotonic elapsed time;
- current system state and active logical agent;
- latest user request and high-level system output;
- safety/monitor/validator events;
- a terminal/event-log pane with secrets redacted.

Also save robot_camera.mp4 as the clean first-person stream.

Start recording before scene reset and before the first user input. Stop only
after the final state, oracle result, and a short final-state hold have been
captured.

Use a single monotonic time origin to align video, events, camera frames, model
calls, and runtime state. Store the mapping in video_metadata.json.

Validate automatically:

- file exists and is non-empty;
- expected codec, resolution, and frame rate;
- decodable first, middle, and last frames;
- sensible frame count and duration;
- no blank/frozen pane beyond a declared threshold;
- coverage from pre-task setup through final hold;
- event/video duration agreement;
- required overlay visibility;
- request-frame timestamps fall within the recording interval.

A run cannot pass its artifact audit if its required recording is missing or
corrupt.

======================================================================
10. MODEL SERVICES AND TUNNEL RECOVERY
======================================================================

Services are forwarded locally as follows:

- Gemma:          127.0.0.1:8000
- SAM 3.1:        127.0.0.1:9000
- EmbeddingGemma: 127.0.0.1:8080

The following exact tunnel destinations are authorized for reconnection. Use
persistent terminal sessions and retain local operational logs. Keep public
artifacts redacted.

Gemma:

ssh -N -T \
  -L 8000:127.0.0.1:8000 \
  -p 21314 \
  -i ~/.ssh/id_ed25519 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  root@185.216.21.214

SAM:

ssh -N -T \
  -L 9000:127.0.0.1:9000 \
  -p 11204 \
  -i ~/.ssh/id_ed25519 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  root@213.173.103.97

EmbeddingGemma:

ssh -N -T \
  -L 8080:127.0.0.1:8080 \
  -p 11204 \
  -i ~/.ssh/id_ed25519 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  root@213.173.103.97

Before creating a tunnel:

1. Check whether the local port is already listening.
2. Test whether the existing service is healthy.
3. Reuse a healthy tunnel.
4. If unhealthy, identify the exact owning process.
5. Never terminate an unknown process. Ask me if ownership is ambiguous.
6. Do not weaken SSH host-key checking.

Discover actual service protocols and model IDs from the clients and health
endpoints. For OpenAI-compatible services, /v1/models may be used. Inspect the
SAM client to determine its correct health check instead of guessing.

Use finite connection and response timeouts for all service calls. Before every
batch, require two consecutive successful health checks and one small
non-mutating inference/embedding/segmentation smoke test.

If a tunnel or service fails during an episode:

1. stop the robot safely;
2. stop and finalize the current recording where possible;
3. mark the attempt INFRA_INTERRUPTED;
4. preserve all partial artifacts;
5. record the precise failure and health-check evidence;
6. restore and verify the service;
7. restart the scheduled episode from a fresh scene and a new attempt directory,
   using the same scheduled seed;
8. never splice two attempts into one canonical recording.

Allow at most three infrastructure attempts per scheduled episode unless the
frozen protocol states otherwise. After that, quarantine the entry and report
the blocker. Do not retry a valid semantic, planning, monitoring, execution, or
safety failure.

======================================================================
11. COMPLETION CRITERIA
======================================================================

The campaign is complete only when:

- every scheduled entry has a valid final outcome or a documented unresolved
  infrastructure disposition;
- every topology and mechanism ablation was verified as genuinely instantiated;
- every required video and request frame passed integrity checks;
- every diagnosis links to its supporting record evidence;
- no attempt has been overwritten or silently excluded;
- campaign manifests and checksums validate;
- tables and plots regenerate from immutable raw artifacts;
- exclusions and deviations are fully documented;
- the requirements crosswalk has no silent omissions;
- a final report identifies confirmatory, secondary, exploratory, and post-hoc
  findings separately;
- exact reproduction and resume commands are documented.

At final handoff, report:

- completed and unresolved run counts by status;
- suite coverage;
- artifact-audit results;
- protocol and source hashes;
- primary results with uncertainty estimates;
- ablation results;
- safety and failure-mode findings;
- compute, latency, token, and storage costs;
- limitations and threats to validity;
- paths to the campaign manifest, run index, analyses, videos, representative
  episode records, and diagnoses.

Begin now by reading the mandatory files and auditing the current repository.
Do not stop after writing a plan; continue through implementation, tests, and
pilots until Mandatory Gate 1.
```

## Approval message after the pilot

Once Codex reports the frozen hash, reply with exactly:

```text
APPROVE LOCKED CAMPAIGN <campaign_id> <protocol_sha256>

Verify the hash, rerun preflight without changing the frozen protocol, and
execute the complete locked schedule. Preserve every attempt and follow the
predeclared retry and exclusion rules. Stop only for an unresolved safety,
integrity, storage, or infrastructure blocker.
```

## Additional artifacts I strongly recommend

Beyond your three Markdown files, the most important additions are:

- `events.jsonl`: append-only, machine-readable evidence. The Markdown record should be a generated view, not the only record.
- `frame_requests.jsonl`: proves exactly which camera image each agent received.
- `model_calls/`: exact prompts, raw responses, timestamps, and parse outcomes.
- `run_schedule.csv`: prevents selective execution or cherry-picking.
- `PROTOCOL.sha256`: establishes that the protocol was frozen before test runs.
- `checksums.sha256`: detects missing or modified evidence.
- `artifact_checks.json`: automated video/log/frame completeness checks.
- `EXCLUSIONS.md`: every exclusion with its predeclared rule and supporting evidence.
- `RUN_STATE.json`: allows a new Codex session to resume safely after interruption.
- `robot_camera.mp4`: clean first-person footage in addition to the composite presentation video.
- `REQUIREMENTS_CROSSWALK.md`: demonstrates that every suite requirement was implemented and evaluated.

For conference use, I would also have the final diagnoses reviewed blindly where possible: reviewers should see anonymized system IDs rather than knowing which result came from the full system or an ablation. This reduces interpretive bias. If human participants ever appear in recordings, ethics approval, informed consent, privacy redaction, and a publication-release policy must be established before recording begins.

> What are these?

:codex-annotation{index="1"} This is a **manual approval gate**, not a shell command.

After Codex completes pilot experiments, it should pause and report:

- `campaign_id`: a unique name for the experiment campaign, such as `robopref_suite_v1_20260811`.
- `protocol_sha256`: a cryptographic fingerprint of the frozen experiment protocol, such as `a8149c...`. It proves the experiment design was not changed after approval.

You would replace the placeholders and send something like:

```text
APPROVE LOCKED CAMPAIGN robopref_suite_v1_20260811 a8149c73b0e...

Verify the hash, rerun preflight without changing the frozen protocol, and
execute the complete locked schedule.
```

This tells Codex: “I reviewed the pilot and authorize the real, publication-quality runs using exactly this protocol.”

The gate prevents Codex from immediately launching hundreds of expensive runs before you have confirmed that the system designs, recordings, metrics, scenarios, and storage estimates are correct. It also establishes a defensible protocol-freeze point for a paper.

You do not generate these values yourself—Codex should generate and show them at the end of the pilot stage.
