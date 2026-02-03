# Molmo Heatmap Fine-Tuning Pipeline

This directory contains a standalone pipeline to fine-tune Molmo (VLM) with a
heatmap prediction head. The ground-truth heatmap is generated from contact
points using the same distance-transform approach as
`/home/aiza/heatmaps/preprocessing/make_heatmap.py`, which produces lower
values near the polygon borders and higher values near the center.

## What it does
- Loads Molmo (`allenai/Molmo-7B-D-0924`) with `trust_remote_code=True`.
- Extracts image-token embeddings from a selectable hidden layer.
- Predicts a heatmap with a small CNN head.
- Computes BCE + soft IoU loss against distance-transform heatmaps.
- Supports LoRA fine-tuning of Molmo layers.

## Quick start

Install dependencies (adjust to your env as needed):

```bash
pip install -r /home/aiza/molmo_heatmap_pipeline/requirements.txt
```

Run fine-tuning:

```bash
python /home/aiza/molmo_heatmap_pipeline/train_heatmap.py \
  --csv /home/aiza/surg_found_copy/dataset_pairs_with_frames.csv \
  --train-split train \
  --val-split val \
  --batch-size 1 \
  --epochs 3 \
  --heatmap-size 224 224 \
  --hidden-layer -1 \
  --lora \
  --amp
```

## Notes
- To use earlier Molmo layers, set `--hidden-layer` (e.g., `-6`).
- LoRA is enabled by default; pass `--no-lora` to disable it.
- The heatmap gradients shrink near polygon borders because the target is a
  normalized distance transform (same as `make_heatmap.py`).
