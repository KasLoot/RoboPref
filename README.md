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
  -p 13671 -i ~/.ssh/id_ed25519 \
  root@157.157.221.177
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
uv run prefmem --memory-store-path ./memory_store
```

`--memory-store-path`<br>
&emsp;The path to store the memory. Default: `./memory_store`

`--think`<br>
&emsp;Per-agent CoT reasoning. Default: `["HRI"]` Supported: `all`, `HRI`, `Memory`, `Planner`, `Validator`.
