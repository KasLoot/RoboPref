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
