uv tool install hf
export PATH="/root/.local/bin:$PATH"
uv tool update-shell
cd /workspace
mkdir datasets
mkdir models

cd datasets
hf download KasLoot/RoboPref_dataset --repo-type=dataset

cd ../models
hf download KasLoot/pi05_base_pytorch

cd ..
git clone https://github.com/KasLoot/RoboPref.git

cd RoboPref
uv sync
source .venv/bin/activate

DS=/workspace/datasets/RoboPref_dataset/stacking_blocks_ambiguous
uv run python -m pi05.norm_stats --dataset $DS
uv run python -m pi05.train --dataset $DS \
    --checkpoint-in /workspace/models/pi05_base_pytorch \
    --checkpoint-out /workspace/models/pi05_arx_ambiguous \
    --micro-batch 256


DS=/workspace/datasets/RoboPref_dataset/stacking_blocks_decomposed
uv run python -m pi05.norm_stats --dataset $DS
uv run python -m pi05.train --dataset $DS \
    --checkpoint-in /workspace/models/pi05_base_pytorch \
    --checkpoint-out /workspace/models/pi05_arx_decomposed \
    --micro-batch 256


DS=/workspace/datasets/RoboPref_dataset/stacking_blocks_ordered
uv run python -m pi05.norm_stats --dataset $DS
uv run python -m pi05.train --dataset $DS \
    --checkpoint-in /workspace/models/pi05_base_pytorch \
    --checkpoint-out /workspace/models/pi05_arx_ordered \
    --micro-batch 256