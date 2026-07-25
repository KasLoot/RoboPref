# PrefMem evaluation workflow

This workflow evaluates PrefMem against the generated packets in
`dataset/sim_datasets` while keeping simulator truth outside the agent context. It
has two complementary studies:

1. A cold-memory endpoint study runs one canonical, fully specified task per fresh
   PrefMem instance.
2. A stateful memory study runs the generated multi-conversation protocols with
   history and preference state preserved only inside a protocol.

The two studies should be reported separately. A cold trial measures task
interpretation, planning, dispatch control, validation, recovery, and terminal
memory side effects. The protocol study measures cross-conversation history,
preference consent, semantic retrieval, one-off overrides, duplicate avoidance,
and user isolation.

## What this benchmark does and does not validate

The runner sends only the human instruction, the initial image, and the recorded
terminal observation through the normal PrefMem interfaces. `manifest.json` is
opened by the experiment runner for selection and is passed to the strict scorer
only after the HRI turn has returned. Scenario IDs are not used as participant IDs
or inserted into model context.

The recorded `BenchmarkEpisodeExecutor` tests the upper-level agent system. It does
not measure continuous VLA action quality, real trajectory dynamics, contact
forces, or recovery policies on a physical robot. Unsafe trajectory evidence is a
declared simulator fixture used to test PrefMem's safety gate.

## Preflight

Run both dry runs before calling Ollama:

```bash
python -m simulation.benchmark validate dataset/sim_datasets

python -m simulation.benchmark evaluate dataset/sim_datasets \
  --output experiments/prefmem-evaluation/cold-pilot \
  --seeds 1 \
  --repetitions 1 \
  --model-seed 1000 \
  --dry-run

python -m simulation.benchmark evaluate-memory dataset/sim_datasets \
  --output experiments/prefmem-evaluation/memory-pilot \
  --repetitions 1 \
  --model-seed 2000 \
  --dry-run
```

A dry run validates packet integrity and selectors, prints the planned matrix, makes
zero model calls, and does not create the output directory.

## Pilot

Start with one seed and a small stratified sample:

```bash
python -m simulation.benchmark evaluate dataset/sim_datasets \
  --output experiments/prefmem-evaluation/cold-pilot \
  --seeds 1 \
  --max-scenarios 24 \
  --repetitions 1 \
  --condition full-pilot \
  --model-seed 1000
```

Use `--display-all` only for a small debugging run. Structured agent events and
hashed model-call telemetry are persisted per trial even when terminal display is
off. A run whose production telemetry or agent-event stream cannot be persisted is
marked failed with `ARTIFACT_INCOMPLETE`; logging failures cannot silently produce
an apparently valid experiment.

Inspect:

- `run_config.json` for the immutable experiment configuration, prompt hashes,
  selected manifest-and-frame digest, models, sampling parameters, callable
  fingerprints, and runtime provenance;
- `episode_results.jsonl` for one durable record per trial;
- `checks.csv` for check-level error analysis;
- `summary.json` for aggregate and stratified metrics;
- `trials/.../result.json` for a readable copy of an individual result;
- `trials/.../agent_events.jsonl` for sanitized structured background-agent output;
- `trials/.../model_calls.jsonl` for call timing, token metadata when available, and
  hashes rather than raw prompts or images; and
- each trial's `memory/` directory for its isolated history, preference, and outbox
  state.

Runtime errors are persisted as completed attempts. Re-running an identical command
resumes by `(condition, repetition, scenario_id)` and does not silently retry them.
Resume validation re-scores completed evidence against the packet oracle and rejects
missing, duplicate, malformed, or self-inconsistent checks. Use a new output
directory when changing any semantic experiment setting.

## Full cold-memory study

Use multiple model repetitions and keep the randomization seed fixed:

```bash
python -m simulation.benchmark evaluate dataset/sim_datasets \
  --output experiments/prefmem-evaluation/cold-full \
  --repetitions 3 \
  --condition full \
  --shuffle-seed 7301 \
  --model-seed 3000
```

Useful filters are `--families`, `--scene-variants`, `--target-ids`, `--outcomes`,
`--seeds`, `--scenario-ids`, `--exclude-controls`, and `--max-scenarios`. Filtering
is applied before repetitions and randomization.

Strict scoring includes:

- HRI resolution of the requested semantic target;
- exact Planner preservation of the confirmed intent;
- complete structured Planner goal predicates;
- expected HRI and Planner terminal states;
- whether Validator was called;
- Validator outcome and `task_complete`;
- execution status and VLA dispatch count;
- recovery decision;
- terminal history delta; and
- absence of an unconsented preference write.

Missing required evidence is a failure, not a skipped success. The summary reports
skipped checks explicitly.

## Stateful memory protocols

Run all generated protocols with fresh stores per protocol repetition:

```bash
python -m simulation.benchmark evaluate-memory dataset/sim_datasets \
  --output experiments/prefmem-evaluation/memory-full \
  --repetitions 3 \
  --model-seed 4000
```

Limit a run with `--protocol-ids ID [ID ...]`, or pass a different validated bundle
with `--protocols-path`. A named initial preference fixture is applied through
`MemoryAgent.update_preference_memory` with validated consent evidence; the runner
never writes a fixture directly into repository JSON.

Protocol results resume by `(protocol_id, repetition)`. Each attempt has isolated
stores and rotates deterministically across compatible scene variants and dataset
seeds. Every turn that produces task-execution evidence is also passed through the
same strict target, Planner, execution, and Validator scorer used by the cold study.
This prevents a protocol from passing merely because a wrong plan is internally
self-consistent.

Before any output directory, fixture, agent, or model call is created, the evaluator
validates every expectation and resolves the complete repetition schedule. Selectors
within one run must remain on the same physical scene; the schedule and its digest are
stored in the run configuration. Resumed task-producing steps are re-scored from
their persisted result and selected manifest.

Retrieval checks use the `MemoryContext` records actually delivered to HRI, rather
than IDs written by HRI into its own trace. The evaluator verifies every delivered
history and preference ID against the active user's repositories. Unverifiable
ownership or a foreign-user record is a mandatory protocol failure. The delivered
IDs and ownership decision are saved with each step for audit.

## Memory ablations

The runner supports:

- `full`
- `no-memory`
- `history-only`
- `preference-only`

Use a distinct condition label and output directory for every cold study:

```bash
python -m simulation.benchmark evaluate dataset/sim_datasets \
  --output experiments/prefmem-evaluation/cold-no-memory \
  --condition no-memory \
  --memory-mode no-memory \
  --repetitions 3 \
  --shuffle-seed 7301 \
  --model-seed 3000
```

Keep packet selection, prompt versions, model, seeds, temperature, and recovery
budgets identical across paired conditions. The cold evaluator retains the raw
full-contract result and identifies checks disabled by an intervention so task
behaviour is not confused with a deliberately suppressed memory write. In the
stateful protocol study, failure of a retrieval or persistence expectation under a
memory ablation is an intended experimental outcome and remains visible.

## Reporting

Report at least:

- strict endpoint pass rate with a 95% interval;
- HRI target-resolution accuracy;
- exact Planner goal-schema accuracy;
- Validator outcome accuracy and false-completion rate;
- unsafe-case gate compliance;
- zero-dispatch accuracy for already-satisfied controls;
- recovery-decision accuracy;
- history and unconsented-preference correctness;
- protocol pass rate, scene-cluster bootstrap interval, and every protocol check
  rate;
- expected-check coverage and unconditional accuracy, so runtime failures cannot
  disappear from component denominators, plus excess-evidence counts so unexpected
  or duplicate checks cannot inflate a rate above 100%;
- infrastructure/output-contract failure-event counts and the distinct number of
  affected trials; and
- median and 95th-percentile trial duration.

Break results down by family, target, scene variant, outcome, seed, and control/core
packet. Core counterfactual packets sharing `(family, scene_variant, seed)` are
correlated; already-satisfied controls are target-specific physical scenes and use
`target_id` as an additional cluster field. Do not describe all 234 packets as
independent observations. Use model repetitions and physical-scene clusters when
estimating uncertainty, and retain per-check records for qualitative failure
analysis.

Do not tune prompts on the final split. A practical workflow is seed 1 for
development, seed 2 for a frozen-prompt validation pass, and seed 3 for the final
held-out report. Record any departure from that split.

## Current protocol scope and trust boundary

The generated stateful protocols currently use block stacking because it provides
two clean, counterfactual order preferences (RGB and BGR). Category sorting and
place setting are covered by the cold endpoint matrix, but their long-term
preference retrieval, paraphrase, override, and isolation behaviour is not yet
tested. Do not generalize the stateful-memory result to those families until
equivalent protocols are added.

The protocols assert that repeated equivalent choices produce one active
preference, but they do not force two active records through the Memory Agent's
compaction threshold. Merge/conflict review and delete/deactivate behaviour require
a separate compaction protocol before they can be claimed as validated.

Default runners preserve the oracle boundary. An injected Python orchestrator
factory is trusted experiment code and can access richer runner objects, so custom
factories must not add manifest truth or protocol expectations to model prompts.
Run configuration records source and callable hashes, but a mutable Ollama model
tag, server build, and hardware stack are not yet independently attested; record
those externally for thesis-grade reproducibility.
