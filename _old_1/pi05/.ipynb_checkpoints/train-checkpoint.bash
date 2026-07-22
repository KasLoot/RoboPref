uv tool install hf
export PATH="/root/.local/bin:$PATH"
uv tool update-shell
hf auth login
cd /workspace
mkdir datasets
mkdir models

apt update
apt-get install git-lfs
git lfs install

cd /workspace/datasets
git clone https://huggingface.co/datasets/KasLoot/RoboPref_dataset

cd /workspace/models
git clone https://huggingface.co/KasLoot/pi05_base_pytorch


cd /workspace/RoboPref
uv sync
source .venv/bin/activate
uv remove torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

mkdir -p ~/.cache/openpi/big_vision
curl -o ~/.cache/openpi/big_vision/paligemma_tokenizer.model \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

DS=/workspace/datasets/RoboPref_dataset_v3/stacking_blocks_ambiguous
uv run python -m pi05.norm_stats --dataset $DS
uv run python -m pi05.train --dataset $DS \
    --checkpoint-in /workspace/models/pi05_base_pytorch \
    --checkpoint-out /workspace/models/pi05_arx_ambiguous_v3d \
    --embodiment arx_l5_3cam


DS=/workspace/datasets/RoboPref_dataset_v3/stacking_blocks_decomposed
uv run python -m pi05.norm_stats --dataset $DS
uv run python -m pi05.train --dataset $DS \
    --checkpoint-in /workspace/models/pi05_base_pytorch \
    --checkpoint-out /workspace/models/pi05_arx_decomposed_v3d \
    --embodiment arx_l5_3cam


DS=/workspace/datasets/RoboPref_dataset_v3/stacking_blocks_ordered
uv run python -m pi05.norm_stats --dataset $DS
uv run python -m pi05.train --dataset $DS \
    --checkpoint-in /workspace/models/pi05_base_pytorch \
    --checkpoint-out /workspace/models/pi05_arx_ordered_v3d \
    --embodiment arx_l5_3cam