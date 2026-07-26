# RoboPref counterfactual simulation benchmark

This package generates paired endpoint episodes for evaluating PrefMem's HRI,
planning, validation, recovery, history, and preference behaviour. It is separate
from the runtime agents on purpose: simulator truth is an offline scoring oracle and
must never be added to a model prompt.

The archived collector in `_old_1/simulation/` remains unchanged. The new generator
reuses its useful design choices—deterministic scene seeds, stable object IDs,
ARX L5 asset, camera framing, explicit predicates, and atomic output—but does not
reuse its success-only HDF5 recorder. A counterfactual benchmark needs to retain
failures and uncertainty as first-class examples.

## Scenario matrix

For every seed, the catalog contains:

| Family | Scene variants | Targets |
|---|---|---|
| `block_stack` | wide and compact scatter | RGB or BGR, bottom to top |
| `category_sort` | front-row and interleaved scatter | printed items left or electronics left |
| `place_setting` | front and radial scatter | right-handed or mirrored left-handed setting |

Every `(scene, target)` is paired with six endpoints:

| Outcome | Meaning | Expected agent behaviour |
|---|---|---|
| `success` | All goal and safety predicates hold | Validator `SUCCESS` |
| `wrong_complete` | The alternate target is completed | Validator `FAILURE`, then replan |
| `partial` | A strict, non-empty goal subset holds | Validator `PARTIAL`, then replan |
| `near_miss` | Visually close, but outside a declared tolerance | Validator `FAILURE`, then replan |
| `unknown` | Physical goal holds but the final view is occluded | Validator `UNKNOWN`, then re-observe |
| `unsafe` | A safety predicate/evidence limit fails | Abort; do not report task completion |

There are 72 scenarios per seed:

`3 families × 2 scenes × 2 targets × 6 outcomes`.

`--include-controls` adds six non-Cartesian `already_satisfied` packets per seed
(one for each family/target). Their initial and terminal observations are identical;
the expected Planner status is `ALREADY_SATISFIED`, execution is
`OBSERVATION_ONLY`, and expected VLA dispatch count is zero.

Initial object state and pixels depend only on `(family, scene variant, seed)`.
Changing the target or endpoint outcome cannot alter the initial observation.
`wrong_complete` uses the exact successful final state of the alternate target.

## Generate

The dependency-light renderer uses Pillow and is suitable for fast experiment
development and CI:

```bash
python -m simulation.benchmark generate \
  --output dataset/sim_benchmark \
  --families block_stack category_sort place_setting \
  --seeds 1 2 3 \
  --backend synthetic \
  --include-controls \
  --write-memory-protocols
```

This creates 234 packets (216 matrix scenarios plus 18 controls). Re-running the
same command is idempotent and preserves
existing file bytes and modification times. Use `--overwrite` only when deliberately
regenerating packets after a renderer/schema change.

The optional MuJoCo snapshot renderer loads the archived ARX L5 XML and meshes:

```bash
python -m simulation.benchmark generate \
  --output dataset/sim_benchmark_mujoco \
  --families block_stack category_sort place_setting \
  --seeds 1 \
  --backend mujoco
```

The MuJoCo backend renders privileged initial/final object states in the robot scene.
It does **not** claim that the arm executed the transition. The archived
DiffIK/scripted expert is a reference for a future trajectory-rollout backend.
Objects are static snapshot geometry; contact-force, collision, and safety values are
scripted counterfactual fixtures, not measured MuJoCo telemetry. Spatial predicates
are recomputed deterministically from terminal object state, while every manifest
marks the provenance of non-visual evidence explicitly.

Both renderers use a warm procedural oak surface and high-contrast task regions:
category sorting uses mustard-yellow and teal mats, while place setting uses a
mustard woven placemat. The MuJoCo renderer additionally uses a three-point studio
light rig, cast/contact shadows, classic-renderer reflective materials, a head-on
camera mounted opposite the robot, and multi-part geometry for ceramics, utensils,
electronics, books, and blocks. The Pillow backend mirrors the colour separation,
wood grain, highlights, and contact shadows as a deterministic CI fallback; it is
intentionally illustrative rather than a physics renderer.

Renderer changes alter PNG bytes and therefore their manifest SHA-256 values. After
updating visual materials, lighting, geometry, or camera settings, regenerate a
benchmark output with `--overwrite` rather than mixing old and new packets.
Pillow output is byte-deterministic for a fixed environment. MuJoCo rasterisation can
vary by a few least-significant channel values across OpenGL drivers or fresh render
contexts, so SHA-256 remains an integrity check rather than a perceptual metric.
Success/near-miss validation ignores those tiny quantisation differences and requires
a material number of changed pixels.

## Packet and oracle boundary

Each generated episode is directly compatible with `DatasetEpisode`:

```text
dataset/sim_benchmark/
  index.json
  protocols.json                 # optional
  episodes/
    ep-<opaque-id>/
      1.png                      # initial observation
      2.png                      # terminal observation
      manifest.json              # private benchmark oracle
```

Directory names are opaque, so paths shown to HRI do not reveal the family, target,
or outcome. PNGs have no target/outcome metadata. `manifest.json` contains the human
instruction, object states, canonical predicates, margins, failure mode, safety
evidence, per-frame SHA-256 digests, and expected agent behaviour. It is for
experiment setup and scoring only.

Use the packet with PrefMem by passing the opaque episode directory:

```bash
python main.py \
  --dataset dataset/sim_benchmark/episodes/ep-<opaque-id> \
  --benchmark \
  --display-all
```

Enter the instruction from the private manifest as the user's query. Do not attach
the manifest or its `ground_truth`/`benchmark_expectations` fields to HRI, Planner,
VLA, or Validator context. `dataset.BenchmarkEpisode.model_context()` provides the
safe manifest-derived boundary: only an opaque observation ID and available
modalities. The observation ID is derived only from the initial-frame bytes, so
target/outcome counterfactual siblings receive the same model-visible ID.

## Validate

```bash
python -m simulation.benchmark validate \
  dataset/sim_benchmark
```

Validation checks:

- opaque IDs and manifest-directory identity;
- exact `1.png`, `2.png`, `manifest.json` packets;
- valid PNG files and frame references;
- numbered-frame digest integrity;
- declared families, scenes, targets, and outcomes;
- success/failure/unknown predicate consistency;
- byte-identical initial frames across counterfactual siblings;
- materially distinct success and near-miss terminal evidence (not just a different
  image hash);
- canonical optional memory protocols bound to the declared matrix;
- index-to-filesystem agreement; and
- completeness of the declared family/seed Cartesian matrix.

## Evaluate PrefMem

The batch evaluator validates the dataset first, creates fresh history,
preference, and outbox stores for every trial, runs the canonical instruction
through the real HRI orchestrator, and scores the returned structured result only
afterward:

Both batch commands default to vLLM model
`/workspace/models/gemma-4-26B-A4B-it` through `http://localhost:8000/v1`; start
that server before a non-dry run or supply explicit model options.

They show a resume-aware progress bar on standard error. Pass `--no-progress` to
disable it without changing result artifacts or the JSON summary on standard output.

```bash
python -m simulation.benchmark evaluate dataset/sim_datasets \
  --output experiments/prefmem-evaluation/cold-full \
  --repetitions 3 \
  --shuffle-seed 7301 \
  --model-seed 3000
```

Use `--dry-run` first to verify the selected matrix without creating artifacts or
calling a model. Runs are durable and resumable. Each output includes a frozen run
configuration, JSONL trial results, check-level CSV, aggregate summary, isolated
memory stores, sanitized structured agent events, and hashed model-call telemetry.
The frozen digest covers the selected manifests and numbered frame bytes, and
completed resume rows are oracle-re-scored before they are trusted.

Stateful memory behaviour is evaluated separately:

```bash
python -m simulation.benchmark evaluate-memory dataset/sim_datasets \
  --output experiments/prefmem-evaluation/memory-full \
  --repetitions 3 \
  --model-seed 4000
```

Task-producing protocol turns receive the same strict oracle-side semantic score
as cold trials. Memory retrieval evidence comes from the context actually returned
to HRI and is ownership-checked against the active user. Production runs also fail
closed if either structured agent events or hashed model-call telemetry cannot be
persisted. Protocol preflight resolves the full repetition schedule before any
fixture or model call and rejects selectors that drift across physical scenes.

Both commands support `--memory-mode full|no-memory|history-only|preference-only`.
Use a separate output directory and stable `--condition` label for each cold
ablation. The full methodology, filters, artifacts, metrics, held-out split, and
interpretation limits are documented in
[`docs/PREFMEM_EVALUATION.md`](../docs/PREFMEM_EVALUATION.md).

## Memory protocols

`--write-memory-protocols` adds semantic, multi-conversation fixtures without exact
wording assertions. They cover:

- first one-off choice becoming history but not preference;
- repeat followed by “decide later” (no consent);
- later explicit consent producing one approved preference without duplicates;
- paraphrase-based retrieval;
- a one-off BGR override that preserves the RGB default; and
- cross-user isolation.

These stateful cases currently cover block stacking only; compaction threshold,
conflict/delete behaviour, category-sort preferences, and place-setting preferences
remain explicit follow-up coverage.

Protocol generation and benchmark validation are offline and never call a model service.
Executing a protocol calls the injected `HRIOrchestrator`, so it uses whichever
models that orchestrator is configured with (Ollama, vLLM, or scripted test models).
The checks inspect structured outputs and repository state, never hidden
model reasoning. Their exact RGB/BGR comparison is an out-of-band test oracle only;
runtime preference retrieval, paraphrase matching, and compaction remain VLM-reasoned.

`run_memory_protocol(...)` executes a protocol through an injected
`HRIOrchestrator`, switches opaque scenario packets through a resolver callback, and
measures history/preference repository deltas around every turn. Named initial
fixtures are semantic, consent-bearing requests rather than repository JSON; the
caller must provide `apply_fixture` so setup goes through the experiment's actual
memory API. Run every protocol with fresh per-user history, preference, and outbox
namespaces; the runner enforces this for both the protocol user and any different
fixture owner so results do not depend on protocol order.

## Programmatic API

```python
from simulation.benchmark import (
    EvaluationConfig,
    FAMILY_DEFINITIONS,
    build_catalog,
    evaluate_memory_protocols,
    generate_benchmark,
    run_cold_memory_evaluation,
    score_agent_result,
    validate_benchmark,
)

scenarios = build_catalog(
    families=["block_stack"],
    seeds=[1, 2],
)
report = generate_benchmark(
    "dataset/sim_benchmark",
    families=["block_stack"],
    seeds=[1, 2],
)
validation = validate_benchmark(report.output_root)
```

`score_agent_result(manifest, structured_result)` is an out-of-band comparison helper.
Oracle truth must flow into the scorer after an agent run, never back into agent
context. Strict scoring is the default: a missing expected observation makes the
overall result fail. `allow_partial=True` scores the available HRI result fields while
reporting skipped checks; use it for diagnostics, not final benchmark metrics. A full
result envelope should add `history_delta` and `preference_delta` measured from store
snapshots (the protocol runner does this automatically).

`--benchmark` selects `BenchmarkEpisodeExecutor`. For unsafe cases it reports the
simulated interruption at the execution boundary, so PrefMem exercises its “do not
call Validator after unsafe execution” contract without exposing the expected label
to a model. Programmatic experiments can construct this executor from
`BenchmarkEpisode.from_path(...)` and inject it into `HRIOrchestrator`.

## CLI command forms

Use either `python -m simulation.benchmark` or the installed
`robopref-benchmark` console alias. All subcommands accept `-h`/`--help`; place it
after the subcommand to see that command's fields.

**Top-level syntax**

```text
python -m simulation.benchmark {generate|validate|list|evaluate|evaluate-memory} [FIELDS]
```

| Command | Purpose |
|---|---|
| `generate` | Create deterministic endpoint packets and optional memory protocols. |
| `validate` | Check packet structure, hashes, scenario semantics, and index consistency. |
| `list` | Print the declared scenario matrix without rendering packets. |
| `evaluate` | Run isolated cold-memory task trials and strict oracle-side scoring. |
| `evaluate-memory` | Run stateful multi-conversation memory protocols. |

### Generate packets: `generate`

**Syntax**

```text
python -m simulation.benchmark generate --output PATH [FIELDS]
```

| Field | Required | Value / default | Meaning and usage |
|---|---:|---|---|
| `--output PATH` | Yes | No default | Destination benchmark root. Generation creates `index.json` and `episodes/`; choose a new path unless intentionally regenerating. |
| `--families FAMILY ...` | No | All three families | Space-separated subset of `block_stack`, `category_sort`, and `place_setting`. At least one value must follow the flag. |
| `--seeds N ...` | No | `1` | One or more integer scene seeds. Use multiple seeds to create independent physical-scene variants. |
| `--backend BACKEND` | No | `synthetic` | `synthetic` uses deterministic Pillow rendering; `mujoco` uses the archived ARX L5 scene for static snapshots. |
| `--overwrite` | No | Off | Regenerates existing packets after renderer or schema changes. Without it, matching files are preserved idempotently. |
| `--include-controls` | No | Off | Adds already-satisfied, observation-only control packets with zero expected VLA dispatches. |
| `--write-memory-protocols` | No | Off | Writes `protocols.json` with semantic multi-conversation memory fixtures matched to the generated catalog. |
| `-h`, `--help` | No | Off | Prints the `generate` parser help and exits. |

Values following `--families` and `--seeds` are space-separated, not comma-separated.

### Validate packets: `validate`

**Syntax**

```text
python -m simulation.benchmark validate ROOT
```

| Field | Required | Value / default | Meaning and usage |
|---|---:|---|---|
| `ROOT` | Yes | No default | Generated benchmark directory containing `index.json` and `episodes/`. Validation is read-only and exits nonzero when errors are found. |
| `-h`, `--help` | No | Off | Prints the `validate` parser help and exits. |

### List the scenario matrix: `list`

**Syntax**

```text
python -m simulation.benchmark list [FIELDS]
```

| Field | Required | Value / default | Meaning and usage |
|---|---:|---|---|
| `--families FAMILY ...` | No | All three families | Space-separated subset of `block_stack`, `category_sort`, and `place_setting`. |
| `--seeds N ...` | No | `1` | Integer seeds for which catalog entries should be declared. |
| `-h`, `--help` | No | Off | Prints the `list` parser help and exits. |

The command prints JSON metadata only; it does not render images or write a benchmark.

### Run one generated packet interactively: `main.py`

**Syntax**

```text
python main.py --dataset PACKET --benchmark [OPTIONS]
```

| Field | Required | Value / default | Meaning and usage |
|---|---:|---|---|
| `--dataset PACKET` | Yes in benchmark mode | No benchmark default | Opaque episode directory under `ROOT/episodes/`; do not pass `manifest.json` itself. |
| `--benchmark` | Yes for this use | Off | Selects `BenchmarkEpisodeExecutor`, including manifest-aware safety interruption without exposing oracle labels to models. |
| `--display_all`, `--display-all` | No | Off | Shows structured diagnostics from all agents and assurance gates while debugging the packet. |
| `-h`, `--help` | No | Off | Prints every `main.py` field and exits. |

The complete model, memory-store, recovery, transcript, and user fields for this same
parser are documented in the root README's [interactive PrefMem form](../README.md#interactive-prefmem-mainpy).

### Common batch-evaluation fields

The following fields are parsed by both `evaluate` and `evaluate-memory`.

| Field | Required | Value / default | Meaning and usage |
|---|---:|---|---|
| `ROOT` | Yes | No default | Valid generated benchmark root. The evaluator keeps it immutable and requires the output directory to be disjoint. |
| `--output PATH` | Yes | No default | Directory for frozen run configuration, durable JSONL results, summaries, and isolated run artifacts. |
| `--repetitions N` | No | `1` | Positive number of model repetitions per selected scenario or protocol. |
| `--dry-run` | No | Off | Validates the dataset and selection plan, prints planned counts, makes zero model calls, and creates no output directory. |
| `--no-resume` | No | Resume enabled | Refuses an output directory that already contains durable results instead of skipping compatible completed keys. |
| `--model MODEL_ID` | No | `/workspace/models/gemma-4-26B-A4B-it` with default vLLM | Model identifier applied to HRI, Memory, Planner, and Validator. It must match the server-advertised vLLM ID. |
| `--model-provider PROVIDER` | No | `vllm` | Backend for all four VLM agents: `vllm` or `ollama`. Pass `ollama` explicitly to use the interactive default backend. |
| `--model-base-url URL` | No | `http://localhost:8000/v1` for vLLM | OpenAI-compatible API root ending in `/v1`; credentials belong in `VLLM_API_KEY`, not in this URL. |
| `--ollama-host URL` | No | Ollama client default | Ollama endpoint override. Use only with `--model-provider ollama`. |
| `--temperature FLOAT` | No | `0.0` | Non-negative sampling temperature applied to all four VLM agents. |
| `--model-seed N` | No | Unset | Non-negative base sampling seed. Repetition `r` uses `N + r - 1`. |
| `--timeout-seconds FLOAT` | No | `120.0` | Positive per-model-call timeout for every VLM agent. |
| `--resize-images` | No | Off | Resizes model-bound images using PrefMem's configured `640×480` vision bounds. |
| `--max-replans N` | No | `1` | Non-negative maximum number of additional planning/recovery attempts. Use `0` to disable replanning. |
| `--max-reobservations N` | No | `1` | Non-negative maximum number of re-observations after `UNKNOWN`. Use `0` to disable them. |
| `--memory-mode MODE` | No | `full` | Evaluation-only ablation: `full`, `no-memory`, `history-only`, or `preference-only`; disabled channels are removed for both reads and writes. |
| `--display_all`, `--display-all` | No | Off | Prints structured background-agent diagnostics in addition to persisting sanitized events. |
| `--no-progress` | No | Progress shown | Disables the resume-aware tqdm bar on standard error; JSON output on standard output is unchanged. |
| `-h`, `--help` | No | Off | Prints the selected evaluation subcommand's parser help and exits. |

Changing any frozen semantic setting—including provider, model, endpoint, filters, seeds,
memory mode, prompts, or recovery budgets—requires a new output directory. vLLM URLs require
`--model-provider vllm`; `--ollama-host` is incompatible with that provider.

### Cold-memory evaluation: `evaluate`

**Syntax**

```text
python -m simulation.benchmark evaluate ROOT --output PATH [COMMON FIELDS] [FIELDS]
```

Use all fields in the common batch-evaluation form above, plus these cold-study fields:

| Field | Required | Value / default | Meaning and usage |
|---|---:|---|---|
| `--condition LABEL` | No | `full` | Stable experiment label written to every record and used in the durable resume key. Use a distinct label for each ablation or experimental condition. |
| `--families FAMILY ...` | No | All families | Filters to `block_stack`, `category_sort`, and/or `place_setting`. |
| `--scene-variants VALUE ...` | No | All variants | Filters scene layouts. Core values are `wide_scatter`, `compact_scatter`, `front_row`, `interleaved_scatter`, `front_scatter`, and `radial_scatter`; controls use `control_already_satisfied`. |
| `--target-ids VALUE ...` | No | All targets | Filters semantic targets: `rgb_bottom_to_top`, `bgr_bottom_to_top`, `printed_left`, `electronics_left`, `right_handed`, or `left_handed`. |
| `--outcomes VALUE ...` | No | All outcomes | Filters to `success`, `wrong_complete`, `partial`, `near_miss`, `unknown`, and/or `unsafe`. |
| `--seeds N ...` | No | All generated seeds | Filters existing packets by integer scene seed; it does not generate new packets. |
| `--scenario-ids ID ...` | No | All scenario IDs | Selects exact opaque IDs from `ROOT/index.json`, useful for reproducing individual failures. |
| `--exclude-controls` | No | Controls included | Removes already-satisfied observation-only controls from the selected set. |
| `--max-scenarios N` | No | Unlimited | Positive cap applied with deterministic balanced sampling before repetitions are expanded. Useful for pilots. |
| `--shuffle-seed N` | No | `0` | Integer seed controlling randomized trial order; it does not change packet contents or model sampling. |
| `--fail-fast` | No | Off | After durably recording an infrastructure/runtime `ERROR`, stops instead of continuing to later trials. Ordinary strict-check failures do not trigger it. |

Selection filters are combined: a packet must satisfy every supplied filter. A request
that selects zero packets is rejected. Keep selection fields and `--shuffle-seed` fixed
when comparing model or memory conditions.

### Stateful memory evaluation: `evaluate-memory`

**Syntax**

```text
python -m simulation.benchmark evaluate-memory ROOT --output PATH [COMMON FIELDS] [FIELDS]
```

Use all fields in the common batch-evaluation form above, plus these protocol fields:

| Field | Required | Value / default | Meaning and usage |
|---|---:|---|---|
| `--protocol-ids ID ...` | No | Every protocol in the bundle | Space-separated exact protocol IDs, such as `memory-repeat-defer-then-consent`; unknown or duplicate selections are rejected. |
| `--protocols-path PATH` | No | `ROOT/protocols.json` | Validated semantic protocol bundle override. Use it to evaluate a separately versioned bundle against the same packet catalog. |

Protocol preflight resolves every selector for every repetition before creating output
or calling a model. Resume keys are `(protocol_id, repetition)`, and each run gets fresh
