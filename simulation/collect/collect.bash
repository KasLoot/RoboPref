uv run python simulation/collect/collect.py --category all --root /data/datasets/RoboPref_dataset_temp  --overwrite --preview

uv run python simulation/collect/run_policy.py \
    --dataset ordered \
    --task 1 \
    --episodes 1 \
    --viewer \
    --checkpoint /data/models/pi05_arx_ordered/best/