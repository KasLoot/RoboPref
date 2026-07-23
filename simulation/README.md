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

```powershell
.\.venv\Scripts\python.exe -m simulation.benchmark generate `
  --output dataset/sim_benchmark `
  --families block_stack category_sort place_setting `
  --seeds 1 2 3 `
  --backend synthetic `
  --include-controls `
  --write-memory-protocols
```

This creates 234 packets (216 matrix scenarios plus 18 controls). Re-running the
same command is idempotent and preserves
existing file bytes and modification times. Use `--overwrite` only when deliberately
regenerating packets after a renderer/schema change.

The optional MuJoCo snapshot renderer loads the archived ARX L5 XML and meshes:

```powershell
uv sync --extra simulation

.\.venv\Scripts\python.exe -m simulation.benchmark generate `
  --output dataset/sim_benchmark_mujoco `
  --families block_stack category_sort place_setting `
  --seeds 1 `
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

```powershell
.\.venv\Scripts\python.exe .\main.py `
  --dataset dataset/sim_benchmark/episodes/ep-<opaque-id> `
  --benchmark `
  --display-all
```

Enter the instruction from the private manifest as the user's query. Do not attach
the manifest or its `ground_truth`/`benchmark_expectations` fields to HRI, Planner,
VLA, or Validator context. `dataset.BenchmarkEpisode.model_context()` provides the
safe manifest-derived boundary: only an opaque observation ID and available
modalities. The observation ID is derived only from the initial-frame bytes, so
target/outcome counterfactual siblings receive the same model-visible ID.

## Validate

```powershell
.\.venv\Scripts\python.exe -m simulation.benchmark validate `
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

## Memory protocols

`--write-memory-protocols` adds semantic, multi-conversation fixtures without exact
wording assertions. They cover:

- first one-off choice becoming history but not preference;
- repeat followed by “decide later” (no consent);
- later explicit consent producing one compact preference;
- paraphrase-based retrieval;
- a one-off BGR override that preserves the RGB default; and
- cross-user isolation.

Protocol generation and benchmark validation are offline and never call Ollama.
Executing a protocol calls the injected `HRIOrchestrator`, so it uses whichever
models that orchestrator is configured with (real Ollama models or scripted test
models). The checks inspect structured outputs and repository state, never hidden
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
    FAMILY_DEFINITIONS,
    build_catalog,
    generate_benchmark,
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
