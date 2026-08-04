# RoboPref
A robotic agent system, enhancing human-robot interaction capabilities by leveraging interactive history and experiences.

Version: `2.1.0`

## System workflow

```mermaid
flowchart TD
    USER([User]) -->|request and clarification| HRI["HRI Agent<br/>only user-facing agent"]
    CAMERA["Webcam server<br/>stream, snapshots, and task API"] -->|current frame| HRI

    HRI -.->|retrieve or consented mutation| MEMORY["Memory Agent"]
    MEMORY <--> STORE[("Preference JSON<br/>and embeddings")]

    HRI -->|clarified goal| PREVIEW["Runtime captures a fresh frame<br/>and requests a preview"]
    CAMERA -->|fresh snapshot| PREVIEW
    PREVIEW --> PLANNER["Planner<br/>stateless and scene-grounded"]
    PLANNER -->|nominal strategy| PROPOSAL["HRI presents goal, constraints,<br/>expected outcome, and outline"]
    PROPOSAL --> CONFIRM{"Exact goal ID and revision<br/>confirmed?"}
    CONFIRM -->|No: reject or revise| HRI
    CONFIRM -->|Yes| CHECKLIST["Validator compiles and freezes<br/>the final evidence checklist"]
    CAMERA -->|confirmation frame| CHECKLIST

    CHECKLIST --> PLAN["Controller starts a planning cycle<br/>with frozen goal and execution history"]
    CAMERA -->|fresh cycle frame| PLAN
    PLAN -->|goal, history, trigger, and frame| PLANNER
    PLANNER --> DECISION{"Planner decision"}

    DECISION -->|ACT| FIRST["Select candidate task 1 only<br/>execution horizon = 1"]
    DECISION -->|REQUEST_FINAL_VALIDATION| FINAL_TASK["Build final-validation publication"]
    FIRST --> PUBLISH["Runtime publishes one current task"]
    FINAL_TASK --> PUBLISH
    PUBLISH -->|PUT /api/task| CAMERA
    PUBLISH --> PHASE{"Publication phase"}

    PHASE -->|STEP| MONITOR["Monitor service"]
    CAMERA -->|post-publication frames| MONITOR
    CAMERA -->|browser shows STEP instruction| EXECUTOR["Physical task executor<br/>human in the current implementation"]
    EXECUTOR -->|changes physical scene| CAMERA
    MONITOR --> MONITOR_RESULT{"Monitor assessment"}
    MONITOR_RESULT -->|ONGOING| MONITOR
    MONITOR_RESULT -->|stable SUCCESS or FAIL| HISTORY["Append terminal observation<br/>and failure evidence to history"]
    HISTORY --> PLAN

    PHASE -->|FINAL_VALIDATION| VALIDATOR["Validator service<br/>accumulates evidence across views"]
    CAMERA -->|fresh validation views| VALIDATOR
    CAMERA -->|browser requests camera-only views| CAMERA_VIEW["Move only the camera;<br/>keep scene objects unchanged"]
    CAMERA_VIEW --> CAMERA
    VALIDATOR --> VALIDATION_RESULT{"Host-derived result"}
    VALIDATION_RESULT -->|NEEDS_EVIDENCE| CAMERA_VIEW
    VALIDATION_RESULT -->|stable INCOMPLETE| HISTORY
    VALIDATION_RESULT -->|stable COMPLETE| COMPLETE([COMPLETE])
    COMPLETE -->|broad checklist and status| HRI

    DECISION -->|BLOCKED or NEEDS_USER_INPUT| ATTENTION["NEEDS_ATTENTION"]
    MONITOR -->|service error or no-progress timeout| ATTENTION
    VALIDATOR -->|service error or no-progress timeout| ATTENTION
    ATTENTION -->|reason and runtime state| HRI
    ATTENTION -->|resume current task| PUBLISH
    ATTENTION -->|replan with guidance| PLAN

    MONITOR -->|emergency_stop = true| ESTOP["Shared emergency coordinator<br/>latch and emergency_stop hook"]
    VALIDATOR -->|emergency_stop = true| ESTOP
    ESTOP --> STOPPED([EMERGENCY_STOPPED])
```

The confirmed goal remains fixed while the controller repeatedly plans from a
fresh frame, publishes one task, and uses visual evidence to decide what comes
next. The current implementation delegates physical actions to the human
watching the camera page; `emergency_stop()` is a placeholder integration hook,
not a safety-rated stop.

## Quickstart

### Clone the repository

```bash
git clone https://github.com/KasLoot/RoboPref.git
cd RoboPref
uv sync
```

### Serve a Reasoning VLM using vLLM

#### Install vLLM

```bash
workspace=<path-to-workspace>
mkdir -p $workspace/vllm
cd $workspace/vllm
uv venv -p 3.12
source .venv/bin/activate

# CUDA 13.0
uv pip install vllm --extra-index-url https://wheels.vllm.ai/0.25.1/cu130 --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

```

#### Download VLM model: Gemma-4-26B-A4B-it
```bash
mkdir -p $workspace/models/gemma-4-26B-A4B-it

hf download google/gemma-4-26B-A4B-it --local-dir $workspace/models/gemma-4-26B-A4B-it
```

#### Run vLLM server
- You will need a GPU with at least 80GB of VRAM (e.g., A100 80GB, H100 80GB, or RTX Pro 6000 Blackwell).
- You may use a quantized model to reduce VRAM requirements, but this may affect reasoning performance.

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve /workspace/models/gemma-4-26B-A4B-it \
  --max-model-len 32768 \
  --enable-per-request-metrics \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --mm-processor-kwargs '{"max_soft_tokens": 560}' \
  --moe-backend triton

# Supported values: 70, 140, 280 (default), 560, 1120 tokens per image.
```

#### Forward the vLLM server port to your local machine

```bash
ssh -N -L 8000:127.0.0.1:8000 \
  -p 27217 -i ~/.ssh/id_ed25519 \
  root@82.221.170.234
```

### Serve Embedding model using vLLM

```bash
mkdir -p $workspace/models/embeddinggemma-300m
hf download google/embeddinggemma-300m --local-dir $workspace/models/embeddinggemma-300m

# RTX 4070 Ti with 12GB VRAM, using bfloat16 precision and 70% GPU memory utilization.
vllm serve $workspace/models/embeddinggemma-300m --dtype bfloat16 \
  --gpu-memory-utilization 0.70 \
  --hf-overrides '{"matryoshka_dimensions":[768]}' \
  --port 8080
```

### Run the RoboPref agent

#### Webcam streaming

Stream camera 0 to a browser on this machine:

```bash
uv sync
uv run stream_camera
```

Keep this process running and open `http://127.0.0.1:1234`. The page shows the
live feed and, below it, the controller's single current task, expected visual
observations, monitor observation, and execution state. Restart `stream_camera`
after updating PrefMem so the page and `/api/task` use the matching contracts.

The operating system is detected automatically; use `--system` to select one explicitly.

Useful options:

```bash
# Find an available camera index
uv run stream_camera --list-cameras

# Select camera 1 and use Windows DirectShow
uv run stream_camera --system windows --camera 1 --backend dshow

# Use Linux V4L2 with MJPEG transport
uv run stream_camera --system linux

# Make the stream available on the local network (no authentication)
uv run stream_camera --host 0.0.0.0
```

#### Open another terminal to run the RoboPref agent:
```bash
uv run prefmem \
  --model-config vllm \
  --camera-base-url http://127.0.0.1:1234 \
  --memory-store-path ./memory_store_test
```

#### Operator GUI

With the camera server, Gemma 4 tunnel, and EmbeddingGemma server running, start
the local operator console in place of the terminal `prefmem` process:

```bash
uv run ui --memory-store-path ./memory_store_test
```

Open `http://127.0.0.1:8090`. The console combines the live camera feed, HRI
conversation, exact goal confirmation, current execution state, validation
results, history, and service health. It listens on loopback only and uses port
8090 so it does not conflict with EmbeddingGemma on port 8080. The console has
no login and exposes a shared transcript and robot controls, so do not publish,
reverse-proxy, or tunnel port 8090 without adding authentication and operator
arbitration.

On desktop, the workspace is split evenly: the live camera, Monitor details,
and compact agent-status cards are on the left, while the full-height chat is
on the right. User and HRI messages use opposing bubbles; streamed reasoning,
tool calls, tool results, and other internal output stay inside collapsed
**Internal activity** sections. **New chat** clears the shared transcript and
starts a fresh HRI checkpoint thread without resetting the physical goal,
current task, Monitor, validation state, emergency latch, or saved preferences.

Do not run `uv run prefmem` and `uv run ui` at the same time. They are two front
ends for the same runtime, and each process would compete for the camera page's
single current-task slot. The GUI starts the agent lazily when you press **Start
PrefMem** or send the first message, so a missing camera/model service is shown
as a recoverable startup or health error instead of preventing the page from
opening.

Useful GUI options:

```bash
# Use the default ./memory_store and open the browser automatically
uv run ui --open-browser

# Point the console at another local camera-stream origin
uv run ui --camera-base-url http://127.0.0.1:1234
```

The GUI's red software-stop control latches PrefMem and stops further task
publication, but the current `emergency_stop()` integration is logging-only.
It is not a safety-rated or hardware emergency stop.

This configuration expects the Gemma chat endpoint at
`http://localhost:8000/v1` and EmbeddingGemma at
`http://localhost:8080/v1`.

### Receding-horizon execution

PrefMem treats the confirmed high-level goal and constraints as a frozen goal
contract, not as a frozen stack of actions. Confirmation starts a new Planner
call with a fresh camera frame. On every cycle the Planner predicts a short
one-to-three-task horizon, while the controller publishes only its first task
to the camera page and Monitor (execution horizon 1).

After stable `SUCCESS`, the result is appended to execution history and a fresh
Planner cycle chooses what to do next. A stable task-level `FAIL` also replans
with the visible observation and failure reason, allowing recovery from changed
scene state. Camera/model/JSON errors and no-progress timeouts instead pause in
`NEEDS_ATTENTION`; they are never interpreted as task failure.

At confirmation time, the independent Validator expands the frozen broad goal
outcomes into a detailed visual checklist and freezes it for the full run. When
Planner later requests final validation, only Validator receives that
publication; Monitor remains responsible for ordinary sub-tasks. Validator can
accumulate evidence while the operator moves only the camera across several
views, but the scene objects must remain unchanged. The host derives
`COMPLETE`, `INCOMPLETE`, or `NEEDS_EVIDENCE` from the checklist. An incomplete
result is appended to execution history and replanned; missing evidence asks
for another camera view. HRI reports one overall sentence followed by the
broad checklist and each item's `MET`, `NOT_MET`, or `UNKNOWN` status, while
the detailed checklist remains internal.

Monitor and Validator output include an immediate `emergency_stop` key. If it
is true, the runtime latches `EMERGENCY_STOPPED`, invokes the placeholder
`emergency_stop()` hook once, stops publishing work, and exits the interactive
session. Replace that placeholder with an acknowledged robot-specific stop API
before connecting physical hardware; the visual model is not a safety-rated
E-stop.

`--memory-store-path`<br>
&emsp;The path to store the memory. Default: `./memory_store`

`--think`<br>
&emsp;Per-agent CoT reasoning. Default: `["HRI"]` Supported: `all`, `HRI`, `Memory`, `Planner`. Monitor and Validator are always disabled.
