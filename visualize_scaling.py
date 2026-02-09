from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data import generate_heatmap_from_polygon
from dataset_vlm import SurgActMolmoDataset




def _heatmap_overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    img = np.array(image.convert("RGB"))
    heatmap = np.asarray(heatmap)
    if heatmap.ndim > 2:
        heatmap = np.squeeze(heatmap)
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}")

    target_size = (img.shape[1], img.shape[0])
    heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.clip(heatmap_resized * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    return Image.fromarray(overlay)

def _create_visual_strip(image: Image.Image, target: np.ndarray, pred: np.ndarray) -> Image.Image:
    """Creates a horizontal comparison strip: Image | Ground Truth Overlay | Prediction Overlay"""
    gt_overlay = _heatmap_overlay(image, target)
    pred_overlay = _heatmap_overlay(image, pred)
    
    total_width = image.width + gt_overlay.width + pred_overlay.width
    height = image.height
    
    strip = Image.new("RGB", (total_width, height))
    strip.paste(image, (0, 0))
    strip.paste(gt_overlay, (image.width, 0))
    strip.paste(pred_overlay, (image.width + gt_overlay.width, 0))
    return strip

def _concat_horiz(left: Image.Image, right: Image.Image) -> Image.Image:
    width = left.width + right.width
    height = max(left.height, right.height)
    row = Image.new("RGB", (width, height), color=(0, 0, 0))
    row.paste(left, (0, 0))
    row.paste(right, (left.width, 0))
    return row


def _pad_to_width(img: Image.Image, target_width: int) -> Image.Image:
    if img.width >= target_width:
        return img
    padded = Image.new("RGB", (target_width, img.height), color=(0, 0, 0))
    padded.paste(img, (0, 0))
    return padded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize original vs resized frame/heatmap scaling."
    )
    parser.add_argument("--csv", required=True, help="Path to dataset CSV.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--N_scale_division", type=int, default=4)
    parser.add_argument("--decord-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--output-path",
        default="/home/aiza/molmo_heatmap_pipeline/infer_outputs/scaling_preview.png",
    )
    args = parser.parse_args()

    dataset = SurgActMolmoDataset(
        csv_path=args.csv,
        split_name=args.split,
        device=args.decord_device,
    )
    sample = dataset[args.index]
    image: Image.Image = sample["image"]
    contacts = sample.get("points", {}).get("contacts", [])
    orig_w, orig_h = image.size

    orig_heatmap = generate_heatmap_from_polygon((orig_h, orig_w), contacts)

    target_w = int(orig_w / args.N_scale_division)
    target_h = int(orig_h / args.N_scale_division)
    resized_image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    scaled_contacts = [
        {"x": p.get("x", 0.0) * scale_x, "y": p.get("y", 0.0) * scale_y}
        for p in contacts
    ]
    scaled_heatmap = generate_heatmap_from_polygon(
        (target_h, target_w), scaled_contacts
    )

    row1 = _concat_horiz(image, _heatmap_overlay(image, orig_heatmap))
    row2 = _concat_horiz(resized_image, _heatmap_overlay(resized_image, scaled_heatmap))
    max_width = max(row1.width, row2.width)
    row1 = _pad_to_width(row1, max_width)
    row2 = _pad_to_width(row2, max_width)

    preview = Image.new("RGB", (max_width, row1.height + row2.height), color=(0, 0, 0))
    preview.paste(row1, (0, 0))
    preview.paste(row2, (0, row1.height))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path)
    print(f"Saved preview to: {output_path}")


if __name__ == "__main__":
    main()
