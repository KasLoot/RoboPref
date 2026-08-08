#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${SAM3_REPO_DIR:-${WORKSPACE}/sam3}"
MODEL_DIR="${SAM3_MODEL_DIR:-${WORKSPACE}/models/sam3.1}"
CHECKPOINT="${SAM3_CHECKPOINT:-${MODEL_DIR}/sam3.1_multiplex.pt}"
VENV="${SAM3_VENV:-${WORKSPACE}/.venvs/sam31}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9000}"

# Conservative defaults. Enable these after the basic server works.
SAM3_USE_FA3="${SAM3_USE_FA3:-0}"
SAM3_COMPILE="${SAM3_COMPILE:-0}"
SAM3_MAX_OBJECTS="${SAM3_MAX_OBJECTS:-16}"
SAM3_MULTIPLEX_COUNT="${SAM3_MULTIPLEX_COUNT:-16}"

export SAM3_CHECKPOINT
export SAM3_USE_FA3
export SAM3_COMPILE
export SAM3_MAX_OBJECTS
export SAM3_MULTIPLEX_COUNT

# -----------------------------------------------------------------------------
# Preconditions
# -----------------------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is not installed."
    echo "Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: SAM 3.1 checkpoint not found:"
    echo "  ${CHECKPOINT}"
    echo
    echo "Expected download command:"
    echo "  hf download facebook/sam3.1 --local-dir ${MODEL_DIR}"
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU / nvidia-smi not found."
    echo "SAM 3.1's official implementation requires CUDA."
    exit 1
fi

# -----------------------------------------------------------------------------
# Clone/update official SAM 3 repo
# -----------------------------------------------------------------------------

mkdir -p "${WORKSPACE}"

if [[ -d "${REPO_DIR}/.git" ]]; then
    echo "Updating SAM 3 repo..."
    git -C "${REPO_DIR}" pull --ff-only
else
    echo "Cloning SAM 3 repo..."
    git clone https://github.com/facebookresearch/sam3.git "${REPO_DIR}"
fi


# -----------------------------------------------------------------------------
# SAM 3.1 upstream compatibility patch
#
# Current SAM 3 main passes offload_state_to_cpu to the SAM 3.1 multiplex
# init_state(), which does not accept that argument.
# See: https://github.com/facebookresearch/sam3/issues/544
# -----------------------------------------------------------------------------

SAM3_BASE_PREDICTOR="${REPO_DIR}/sam3/model/sam3_base_predictor.py"

"${VENV}/bin/python" - "${SAM3_BASE_PREDICTOR}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

old = """        inference_state = self.model.init_state(**init_kwargs)"""

new = """        # SAM 3 / SAM 3.1 have different init_state signatures.
        # Filter arguments to those accepted by the concrete model.
        import inspect

        sig = inspect.signature(self.model.init_state)
        valid_params = set(sig.parameters.keys())
        filtered_init_kwargs = {
            k: v for k, v in init_kwargs.items() if k in valid_params
        }

        inference_state = self.model.init_state(**filtered_init_kwargs)"""

if "filtered_init_kwargs" in text:
    print("SAM 3.1 start_session compatibility patch already applied.")
elif old in text:
    text = text.replace(old, new, 1)
    path.write_text(text)
    print("Applied SAM 3.1 start_session compatibility patch.")
else:
    raise RuntimeError(
        "Could not locate expected start_session implementation. "
        "The upstream SAM 3 source may have changed."
    )
PY

# -----------------------------------------------------------------------------
# Python environment
# -----------------------------------------------------------------------------

echo "Creating uv environment..."
mkdir -p "$(dirname "${VENV}")"

if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python 3.12 "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "Installing PyTorch..."
uv pip install \
    torch==2.10.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128

echo "Installing SAM 3..."
uv pip install -e "${REPO_DIR}"

echo "Installing API dependencies..."
uv pip install \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pillow \
    einops \
    pycocotools \
    psutil

# Optional optimized inference dependencies.
if [[ "${SAM3_USE_FA3}" == "1" ]]; then
    echo "Installing FlashAttention 3..."
    uv pip install einops ninja

    uv pip install \
        flash-attn-3 \
        --no-deps \
        --index-url https://download.pytorch.org/whl/cu128

    uv pip install \
        "git+https://github.com/ronghanghu/cc_torch.git"
fi

# -----------------------------------------------------------------------------
# REST server
# -----------------------------------------------------------------------------

SERVER_PY="${REPO_DIR}/sam31_server.py"

cat > "${SERVER_PY}" <<'PY'
import asyncio
import base64
import io
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from sam3.model_builder import build_sam3_predictor


CHECKPOINT = os.environ.get(
    "SAM3_CHECKPOINT",
    "/workspace/models/sam3.1/sam3.1_multiplex.pt",
)

USE_FA3 = os.environ.get("SAM3_USE_FA3", "0") == "1"
COMPILE = os.environ.get("SAM3_COMPILE", "0") == "1"
MAX_OBJECTS = int(os.environ.get("SAM3_MAX_OBJECTS", "16"))
MULTIPLEX_COUNT = int(os.environ.get("SAM3_MULTIPLEX_COUNT", "16"))


predictor = None

# One GPU model instance; serialize inference requests.
gpu_lock = asyncio.Lock()


def masks_to_detections(outputs):
    """
    Convert SAM 3.1 output masks into pixel-space XYXY bounding boxes.

    Bounding boxes are derived from the final masks so their coordinate
    convention is unambiguous.
    """
    obj_ids = np.asarray(outputs.get("out_obj_ids", []))
    masks = np.asarray(outputs.get("out_binary_masks", []))
    raw_scores = outputs.get("out_probs", outputs.get("out_scores", []))
    scores = np.asarray([] if raw_scores is None else raw_scores).reshape(-1)

    if masks.size == 0:
        return []

    # Common shape:
    #   [N, 1, H, W]
    # Convert to:
    #   [N, H, W]
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if masks.ndim != 3:
        raise RuntimeError(
            f"Unexpected out_binary_masks shape: {masks.shape}"
        )

    detections = []

    for i, mask in enumerate(masks):
        mask = mask.astype(bool)

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        x1 = int(xs.min())
        y1 = int(ys.min())

        # Exclusive upper edge, conventional XYXY representation.
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1

        object_id = int(obj_ids[i]) if i < len(obj_ids) else i
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG", optimize=True)

        detection = {
            "object_id": object_id,
            "box_xyxy": [x1, y1, x2, y2],
            "mask_area": int(mask.sum()),
            "mask_png_base64": base64.b64encode(
                mask_buffer.getvalue()
            ).decode("ascii"),
        }
        if i < len(scores) and np.isfinite(scores[i]):
            score = float(scores[i])
            if 0.0 <= score <= 1.0:
                detection["score"] = score
        detections.append(detection)

    return detections


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for SAM 3.1")

    if not Path(CHECKPOINT).is_file():
        raise RuntimeError(f"Checkpoint not found: {CHECKPOINT}")

    print(f"Loading SAM 3.1 checkpoint: {CHECKPOINT}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FlashAttention 3: {USE_FA3}")
    print(f"torch.compile: {COMPILE}")

    predictor = build_sam3_predictor(
        version="sam3.1",
        checkpoint_path=CHECKPOINT,
        max_num_objects=MAX_OBJECTS,
        multiplex_count=MULTIPLEX_COUNT,
        use_fa3=USE_FA3,
        use_rope_real=USE_FA3,
        compile=COMPILE,
        warm_up=False,
        async_loading_frames=False,
    )

    print("SAM 3.1 loaded.")

    yield

    if predictor is not None:
        predictor.shutdown()


app = FastAPI(
    title="SAM 3.1 Object Detection API",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "sam3.1",
        "checkpoint": CHECKPOINT,
        "cuda": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }


@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    threshold: float = Form(0.5),
):
    """
    Detect all objects matching a natural-language prompt.

    Input:
      image      image file
      prompt     e.g. "person wearing a red shirt"
      threshold  SAM output probability threshold

    Output:
      pixel-space bounding boxes [x1, y1, x2, y2], lossless PNG masks,
      and model scores when SAM exposes calibrated probabilities
    """
    if not prompt.strip():
        raise HTTPException(400, "prompt must not be empty")

    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(
            400,
            "threshold must be between 0 and 1",
        )

    try:
        pil_image = Image.open(image.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            400,
            f"Invalid image: {exc}",
        ) from exc

    width, height = pil_image.size

    # SAM 3.1 is exposed by Meta through the multiplex video predictor.
    # A directory containing a single numerically-named JPEG is therefore
    # used for image-only detection.
    with tempfile.TemporaryDirectory(prefix="sam31_") as tmp:
        frame_path = Path(tmp) / "00000.jpg"
        pil_image.save(frame_path, format="JPEG", quality=100)

        session_id = None

        async with gpu_lock:
            try:
                start = predictor.handle_request(
                    {
                        "type": "start_session",
                        "resource_path": tmp,
                    }
                )

                session_id = start["session_id"]

                response = predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": prompt,
                        "output_prob_thresh": threshold,
                    }
                )

                outputs = response["outputs"]

                detections = masks_to_detections(outputs)

            except Exception as exc:
                raise HTTPException(
                    500,
                    f"SAM 3.1 inference failed: {exc}",
                ) from exc

            finally:
                if session_id is not None:
                    try:
                        predictor.handle_request(
                            {
                                "type": "close_session",
                                "session_id": session_id,
                            }
                        )
                    except Exception:
                        pass

    return {
        "prompt": prompt,
        "threshold": threshold,
        "image": {
            "width": width,
            "height": height,
        },
        "count": len(detections),
        "detections": detections,
    }
PY

# -----------------------------------------------------------------------------
# Start server
# -----------------------------------------------------------------------------

cd "${REPO_DIR}"

echo
echo "Starting SAM 3.1 server"
echo "  checkpoint: ${CHECKPOINT}"
echo "  API:        http://${HOST}:${PORT}"
echo "  docs:       http://${HOST}:${PORT}/docs"
echo
echo "Example:"
echo "  curl -X POST http://127.0.0.1:${PORT}/detect \\"
echo "    -F 'prompt=person wearing a red shirt' \\"
echo "    -F 'threshold=0.5' \\"
echo "    -F 'image=@/path/to/image.jpg'"
echo

exec "${VENV}/bin/uvicorn" \
    sam31_server:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers 1
