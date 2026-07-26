# RoboPref counterfactual simulation benchmark

The simulation benchmark generates deterministic, paired initial/final image packets
for evaluating PrefMem's task reasoning and assurance chain. It is intentionally
separate from success-only robot demonstrations: failure, ambiguity, uncertainty,
safety interruption, and already-satisfied controls are first-class test cases.

For the complete conversation methodology and thesis reporting guidance, see
[`docs/PREFMEM_EVALUATION.md`](../docs/PREFMEM_EVALUATION.md).

## Dataset contents

Each packet contains two numeric frames and a private manifest:

```text
dataset/sim_datasets/
  index.json
  episodes/
    ep-<opaque-id>/
      1.png
      2.png
      manifest.json
```

The manifest declares the scene family, target, expected endpoint label, object
state, ground-truth predicates, hashes, and expected agent behavior. It is an
out-of-band oracle. Models receive only the current image and an observation ID
derived from the initial pixels; they never receive scenario IDs, target IDs,
outcome labels, simulator state, or benchmark expectations.

Families:

- `block_stack`: RGB or BGR from bottom to top;
- `category_sort`: printed items left/electronics right, or the opposite;
- `place_setting`: right-handed or mirrored left-handed layouts.

Core endpoint labels:

- `success`;
- `wrong_complete`;
- `partial`;
- `near_miss`;
- `unknown` (occluded/insufficient evidence);
- `unsafe`.

Optional `already_satisfied` controls use `expected_outcome=success` and require
observation-only validation with zero VLA dispatches.

## Generate and validate

Synthetic example:

```bash
python -m simulation.benchmark generate \
  --output dataset/sim_benchmark \
  --families block_stack category_sort place_setting \
  --seeds 1 2 3 \
  --backend synthetic \
  --include-controls

python -m simulation.benchmark validate dataset/sim_benchmark
```

MuJoCo rendering is optional:

```bash
python -m simulation.benchmark generate \
  --output dataset/sim_benchmark_mujoco \
  --families block_stack category_sort place_setting \
  --seeds 1 \
  --backend mujoco \
  --include-controls
```

Use `--overwrite` only when intentionally regenerating an existing benchmark root.
Validation checks the declared family/seed matrix, opaque IDs, manifest schemas,
counterfactual initial-frame identity, final-frame hashes, target/outcome coverage,
and material visual distinction between endpoints.

Print the deterministic core catalogue without rendering:

```bash
python -m simulation.benchmark list \
  --families block_stack category_sort place_setting \
  --seeds 1
```

## Interactive packet execution

Use one recorded packet with the ordinary PrefMem CLI:

```bash
python main.py \
  --dataset dataset/sim_datasets/episodes/ep-<opaque-id> \
  --benchmark \
  --display-all
```

`BenchmarkEpisodeExecutor` returns the recorded final observation and explicitly
does not claim physical execution. For an unsafe packet, the harness uses the
private manifest to inject an `UNSAFE` execution gate so Validator must not be
called. This tests PrefMem's abort/latch response after a perfect hazard signal,
not hazard detection.

## Full conversation evaluation

Inspect a model-free plan:

```bash
python -m simulation.benchmark evaluate-conversations dataset/sim_datasets \
  --output experiments/prefmem-evaluation/full-v1 \
  --dry-run
```

The checked-in dataset expands to 594 isolated conversations and 810 commands,
including 54 balanced memory cases across all three task families.

Run all five suites:

```bash
python -m simulation.benchmark evaluate-conversations dataset/sim_datasets \
  --output experiments/prefmem-evaluation/full-v1 \
  --suites endpoint dialogue memory recovery safety \
  --repetitions 3 \
  --shuffle-seed 7301 \
  --model-seed 3000 \
  --model-provider vllm \
  --model /workspace/models/gemma-4-26B-A4B-it \
  --model-base-url http://localhost:8000/v1
```

Suites:

| Suite | What it evaluates |
|---|---|
| `endpoint` | Explicit task contract, exact predicates, independently grounded planner-to-VLA action semantics, VLA evidence, Validator class, recovery action, terminal assurance, and controls. |
| `dialogue` | Ambiguity handling, structured confirmations, history use, and correction of an opposite proposal. |
| `memory` | Dedicated consent, cross-scene reuse, one-off override, and user isolation in block stacking, category sorting, and place setting. |
| `recovery` | First-attempt failure/uncertainty detection plus semantically compatible scripted paired-endpoint replan/re-observation. |
| `safety` | Abort, persistent latch, refusal of a bare retry, explicit clearance, and zero post-abort dispatch after an oracle-signaled `UNSAFE` result. |

The evaluator defaults to local vLLM and the configured Gemma model. Override with
`--model-provider`, `--model`, and the provider-specific connection option. Common
selection fields are `--families`, `--scene-variants`, `--target-ids`, `--outcomes`,
`--seeds`, `--scenario-ids`, `--exclude-controls`, `--max-cases`, and
`--shuffle-seed`. Limited pilots use deterministic round-robin stratification over
suite/profile/family/outcome rather than an unstratified shuffled prefix.
Interactive runs display a resume-aware progress bar; use `--no-progress` to
disable it for redirected logs or batch schedulers.

Recovery defaults to one replan and one re-observation. Near misses default to the
`perceptual` policy; select `--near-miss-policy strict` for geometric failure
expectations. Memory modes are `full`, `no-memory`, `history-only`, and
`preference-only`.

Every case/repetition receives fresh history, preference, and outbox files. Results
resume only when semantic configuration and source/data hashes match, and every
existing record passes strict case/seed/status/command-ledger validation. Interrupted
unrecorded attempts are preserved while retries use a fresh suffixed directory. The
output includes `experiment.json`, `cases.json`, `results.jsonl`, `summary.json`,
run-local memory, model telemetry, and agent events. Summary v2 separates functional
from audited success, retains planned validator denominators, and bootstraps
connected physical-scene dependency components only for complete valid batches.

## Python API

```python
from simulation.benchmark import (
    ConversationEvaluationConfig,
    evaluate_conversations,
    generate_benchmark,
    validate_benchmark,
)

validation = validate_benchmark("dataset/sim_datasets")
assert validation.valid, validation.errors

report = evaluate_conversations(
    ConversationEvaluationConfig(
        benchmark_root="dataset/sim_datasets",
        output_dir="experiments/prefmem-evaluation/endpoint-pilot",
        suites=("endpoint",),
        max_cases=12,
    )
)
print(report.to_dict())
```

The strict scorer is out-of-band. Do not pass an `EpisodeDescriptor.manifest`, a
case oracle, or any score back into HRI, Memory, Planner, Validator, or VLA prompts.

## Scope

Static image pairs can test semantic reasoning, dialogue, memory boundaries,
validation labels, safety gating, and scripted counterfactual recovery. They cannot
measure continuous control quality, collision-free trajectories, grasp stability,
latency on a physical robot, or genuine environment recovery. Those claims require
a live VLA/environment adapter and trajectory-level instrumentation.
