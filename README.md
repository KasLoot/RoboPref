# RoboPref / PrefMem

PrefMem is an HRI-centred agentic layer for preference-aware robot manipulation. The
current implementation separates two persistent memory types:

- **Episodic history** records compact facts about completed, failed, blocked, and
  cancelled tasks. It preserves continuity but is never treated as preference authority.
- **Preference memory** stores only user-authorized semantic defaults. A VLM performs
  semantic retrieval, equivalence reasoning, and reversible compaction; deterministic
  repositories enforce consent, ownership, atomicity, idempotency, and provenance.

The HRI Agent is the sole user-facing orchestrator. It retrieves memory context, resolves
the task, calls Planner, dispatches a VLA executor, validates against Planner's frozen
goal schema, runs deterministic task assurance, and finally updates history.

The implementation deliberately separates semantic and deterministic responsibilities:

- The VLM Memory Agent reasons over paraphrases, complex applicability, conflicts, and
  duplicate meaning. There is no exact-word or hand-built alias retrieval gate.
- The repositories enforce user ownership, typed consent, proposal/action binding,
  optimistic revisions, atomic multi-operation commits, idempotency, tombstones, and
  reversible merge lineage.
- A first task choice is history only. After a second semantically matching completed
  task, Memory may propose a dedicated future-preference question at the task boundary.
  `DECLINE`, `DEFER`, `CORRECT`, and cancellation do not create a preference.
- Planner creates one frozen validation specification. Recovery cannot replace it, and
  failed or unsafe execution never reaches Validator as a successful attempt.

See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for the complete design and contracts.

For the deterministic counterfactual task/outcome generator (block stacking,
semantic category sorting, place settings, failures, unsafe endpoints, and occluded
observations), see [simulation/README.md](simulation/README.md).

## Run the recorded-episode prototype

```bash
./.venv/bin/python ./main.py \
  --dataset dataset/v3 \
  --history-store memory/history.json \
  --preference-store memory/preferences.json \
  --user-id participant-01 \
  --max-replans 1 \
  --max-reobservations 1
```

### Use the localhost vLLM model

Install the locked dependencies once, including the official OpenAI Python SDK:

```bash
uv sync
```

Then select the `vllm` backend and the exact model ID advertised by the server:

```bash
uv run python ./main.py \
  --model-provider vllm \
  --model /workspace/models/gemma-4-26B-A4B-it \
  --model-base-url http://localhost:8000/v1 \
  --dataset dataset/v3 \
  --history-store memory/history.json \
  --preference-store memory/preferences.json \
  --user-id participant-01
```

`--model-base-url` defaults to `http://localhost:8000/v1`; it must point to the
API root ending in `/v1`, not to `/v1/chat/completions`. Unauthenticated vLLM
servers use the SDK's non-secret `EMPTY` placeholder. If the server was started
with `--api-key`, set that value in `VLLM_API_KEY`; it is never written to model
telemetry or benchmark run configuration. Text and prepared scene images are sent
through OpenAI-compatible Chat Completions, with JSON-object output requested.

The default executor is a recorded-dataset adapter: it dispatches no physical commands
and uses the episode's final frame as external execution evidence. Integrate a real VLA
through `agents.vla.CallableVLAExecutor` or the `VLAExecutor` protocol.

Interactive model, prompt, memory, vision, semantic threshold, and recovery defaults
are in `agents/configs.py`; interactive PrefMem remains Ollama-backed by default.
The `evaluate` and `evaluate-memory` batch commands instead default to vLLM model
`/workspace/models/gemma-4-26B-A4B-it` at `http://localhost:8000/v1`. Use
`--model-provider`, `--model`, and `--model-base-url` to override those evaluation
settings. To evaluate with Ollama, pass `--model-provider ollama` and optionally
`--ollama-host`.
Add `--display_all` to print the complete structured outputs from HRI, Memory, Planner,
VLA, Validator, and Task Assurance. This diagnostic output is also captured by the
configured terminal transcript; raw image bytes and hidden reasoning fields are omitted.

## Safe legacy-memory inspection

Legacy schema-v1 candidates are not valid durable preferences. Inspect a legacy file
without writing anything:

```bash
./.venv/bin/python -m memory.migration \
  --legacy _old_1/memory/preferences_3.json \
  --user-id default
```

Add `--apply-history --history-store memory/history.json` to import those choices as
history-only episodes. Records with possible durable evidence are reported for semantic
and user review; the migrator never silently activates them. Files under `_old_1/` are
read-only inputs and remain unchanged.

## Test

```bash
./.venv/bin/python -m unittest discover -s tests -q
```

The offline suite uses injected scripted models and never calls an external model. It covers
semantic retrieval/compaction, consent and defer behavior,
post-task memory proposals, persistence rollback/idempotency/revisions, frozen schemas,
unsafe execution, history sanitization, dataset ordering, and lossless image handling.

## Command forms

The tables below are command forms: choose a launcher, then fill in the fields needed
for the run. Use either the existing virtual environment (`.venv/bin/python`) or
`uv run python`; examples using plain `python` assume the environment is already active.

READMEs under `_old_1/` describe archived, read-only code, and the ARX L5 asset README
is vendored upstream documentation. Their historical commands are intentionally unchanged.

### Interactive PrefMem: `main.py`

**Syntax**

```text
{.venv/bin/python | uv run python | python} main.py [OPTIONS]
```

| Field | Required | Value / default | Meaning and usage |
|---|---:|---|---|
| `--dataset PATH` | No | `dataset/v3` | Episode directory containing numbered frames such as `1.png`, `2.png`, and so on. Set it to one generated packet directory when using `--benchmark`. |
| `--benchmark` | No | Off | Enables the manifest-aware recorded benchmark executor. It requires an explicit `--dataset`; oracle labels remain outside model context. |
| `--history-store PATH` | No | `memory/history.json` | JSON store for participant episodic history. Use a participant- or experiment-specific path to isolate runs. |
| `--history-outbox PATH` | No | Beside the history store as `history_outbox.json` | Durable retry queue for history writes that could not be persisted immediately. |
| `--preference-store PATH` | No | `memory/preferences.json` | JSON store for consent-approved semantic preferences. |
| `--memory-store PATH` | No | None | Deprecated alias for `--preference-store`; do not pass both. |
| `--user-id ID` | No | `default` | Non-identifying participant namespace used for history and preference ownership. |
| `--transcript PATH` | No | `experiments/record.txt` | Captures the terminal session. Parent directories are created by the transcript writer. |
| `--display_all`, `--display-all` | No | Off | Prints structured HRI, Memory, Planner, VLA, Validator, and assurance diagnostics. Both spellings are equivalent. |
| `--resize-images` | No | Off | Resizes model-bound images to the configured `640×480` bounds before inference. |
| `--max-replans N` | No | `1` | Maximum additional planning/recovery attempts after the initial plan. Use `0` to disable replanning. |
| `--max-reobservations N` | No | `1` | Maximum new observations requested after an `UNKNOWN` validation result. Use `0` to disable re-observation. |
| `--model MODEL_ID` | No | `gemma4:31b-cloud` | Overrides the model identifier for all four VLM agents. For vLLM, use the exact ID returned by the server. |
| `--model-provider PROVIDER` | No | `ollama` | Backend for all four VLM agents: `ollama` or `vllm`. |
| `--model-base-url URL` | No | `http://localhost:8000/v1` for vLLM | OpenAI-compatible vLLM API root. It must be an absolute HTTP(S) URL ending in `/v1`, without credentials, query parameters, or fragments. |
| `--model-seed N` | No | Provider default | Sets the same integer sampling seed on HRI, Memory, Planner, and Validator. Use it for repeatable debugging. |
| `--ollama-host URL` | No | Ollama client default | Overrides the Ollama service endpoint. It is incompatible with `--model-provider vllm`. |
| `-h`, `--help` | No | Off | Prints the authoritative parser help and exits. |

Constraint notes:

- `--benchmark` requires `--dataset` to be present on the command line.
- `--model-base-url` requires `--model-provider vllm`.
- `--ollama-host` cannot be combined with `--model-provider vllm`.
- Negative interactive recovery budgets are clamped to `0`.

### Install or synchronize dependencies: `uv sync`

**Syntax**

```text
uv sync
```

| Field | Required | Value / default | Meaning and usage |
|---|---:|---|---|
| `uv` | Yes | Executable on `PATH` | Runs the uv package/environment manager. |
| `sync` | Yes | Subcommand | Makes the project environment match `pyproject.toml` and `uv.lock`, including OpenAI, Ollama, Pillow, and tqdm. |
| `-h`, `--help` | No | Off | Shows uv's `sync` options; use this before adding resolver, extra, or environment flags not shown here. |

Run this after cloning, after dependency metadata changes, or when recreating `.venv`.

### Inspect or migrate legacy memory: `memory.migration`

**Syntax**

```text
python -m memory.migration --legacy PATH [OPTIONS]
```

| Field | Required | Value / default | Meaning and usage |
|---|---:|---|---|
| `--legacy PATH` | Yes | No default | Read-only schema-v1 preferences JSON to inspect. The source file is never rewritten. |
| `--user-id ID` | No | `default` | Selects records owned by this participant namespace; cross-user records are skipped with warnings. |
| `--history-store PATH` | Conditional | No default | Destination history-v2 JSON. It is required only when `--apply-history` is used. |
| `--apply-history` | No | Off (inspection only) | Appends safe history-only candidates to `--history-store`; it never activates a durable preference. |
| `-h`, `--help` | No | Off | Prints parser help and exits. |

`--apply-history` requires `--history-store`. Omit it for a non-mutating inspection report.

### Run the offline tests: `unittest discover`

**Syntax**

```text
.venv/bin/python -m unittest discover -s tests -q
```

| Field | Required | Value / default | Meaning and usage |
|---|---:|---|---|
| `.venv/bin/python` | Yes | Project interpreter | Uses the locked project environment; `uv run python` is an equivalent launcher. |
| `-m unittest` | Yes | Python module mode | Runs Python's standard-library unit-test command. |
| `discover` | Yes | Subcommand | Recursively discovers test modules instead of naming them individually. |
| `-s tests` | No | Discovery default is `.` | Sets `tests/` as the discovery start directory. |
| `-q` | No | Normal verbosity | Quiet mode: suppresses per-test names while retaining failures and the final summary. |
| `-h`, `--help` | No | Off | Shows the standard `unittest` or discovery options. |

The repository suite uses scripted models and does not call Ollama or vLLM.
