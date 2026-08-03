# RoboPref
A robotic agent system, enhancing human-robot interaction capabilities by leveraging interactive history and experiences.

Version: `2.0.0.dev1`

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
  -p 11774 -i ~/.ssh/id_ed25519 \
  root@69.8.146.87
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
`NEEDS_ATTENTION`; they are never interpreted as task failure. Once the Planner
believes the goal is satisfied, Monitor performs a final holistic validation.

Monitor output includes an immediate `emergency_stop` key. If it is true, the
runtime latches `EMERGENCY_STOPPED`, invokes the placeholder
`emergency_stop()` hook once, stops publishing work, and exits the interactive
session. Replace that placeholder with an acknowledged robot-specific stop API
before connecting physical hardware; the visual model is not a safety-rated
E-stop.

`--memory-store-path`<br>
&emsp;The path to store the memory. Default: `./memory_store`

`--think`<br>
&emsp;Per-agent CoT reasoning. Default: `["HRI"]` Supported: `all`, `HRI`, `Memory`, `Planner`, `Validator`.
