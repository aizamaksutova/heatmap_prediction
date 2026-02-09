from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataset_vlm import SurgActMolmoDataset


def generate_heatmap_from_polygon(
    image_shape: tuple[int, int],
    contact_points: list[dict[str, Any]],
) -> np.ndarray:
    """
    Generates a binary heatmap (0.0 to 1.0) where all pixels inside the
    polygon are 1.0 and outside are 0.0.
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if not contact_points:
        return np.zeros((height, width), dtype=np.float32)

    pts = []
    for p in contact_points:
        x, y = p.get("x", 0), p.get("y", 0)
        pts.append([int(x), int(y)])

    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)

    heatmap = (mask > 0).astype(np.float32)
    return heatmap


def generate_heatmap_from_polygon_distance(
    image_shape: tuple[int, int],
    contact_points: list[dict[str, Any]],
) -> np.ndarray:
    """
    Generates a grayscale heatmap (0.0 to 1.0) where the center of the
    polygon is the highest intensity (1.0) and edges are lower.
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if not contact_points:
        return np.zeros((height, width), dtype=np.float32)

    pts = []
    for p in contact_points:
        x, y = p.get("x", 0), p.get("y", 0)
        pts.append([int(x), int(y)])

    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)

    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_val = dist_map.max()
    if max_val > 0:
        heatmap = dist_map / max_val
        heatmap = heatmap.astype(np.float32)
        return heatmap
    else:
        return np.zeros((height, width), dtype=np.float32)


@dataclass
class HeatmapSample:
    image: Image.Image
    orig_image: Image.Image
    image_size: tuple[int, int]
    prompt: str
    heatmap: np.ndarray
    orig_heatmap: np.ndarray


class HeatmapMolmoDataset:
    def __init__(
        self,
        csv_path: str,
        split_name: str | None = None,
        N: int = 4,
        heatmap_size: tuple[int, int] | None = (224, 224),
    ) -> None:
        self.dataset = SurgActMolmoDataset(csv_path=csv_path, split_name=split_name)
        self.N = N
        self.heatmap_size = heatmap_size
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> HeatmapSample:
        sample = self.dataset[idx]
        image: Image.Image = sample["image"]
        orig_w, orig_h = image.size

        target_w = int(orig_w / self.N)
        target_h = int(orig_h / self.N)

        resized_image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        # 3. Adjust the contact point coordinates
        contacts = sample.get("points", {}).get("contacts", [])
        orig_heatmap = generate_heatmap_from_polygon((orig_h, orig_w), contacts)
        scaled_contacts = []
        for p in contacts:
            scaled_contacts.append({
                "x": p.get("x", 0) * scale_x,
                "y": p.get("y", 0) * scale_y
            })
            
        heatmap = generate_heatmap_from_polygon((target_h, target_w), scaled_contacts)
        if self.heatmap_size is not None:
            heatmap = cv2.resize(
                heatmap,
                (int(self.heatmap_size[1]), int(self.heatmap_size[0])),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)

        return HeatmapSample(
            image=resized_image,  # Now returning the 224x224 image
            orig_image=image,
            image_size=(target_w, target_h),
            prompt=sample.get("prompt", ""),
            heatmap=heatmap,
            orig_heatmap=orig_heatmap,
        )


def collate_heatmap_batch(
    samples: list[HeatmapSample],
    processor,
    prompt_type: str = "contact",
) -> dict[str, Any]:
    images = [s.image for s in samples]
    prompts = []
    for s in samples:
        if prompt_type == "empty":
            prompts.append("")
        else:
            prompts.append(s.prompt)

    processed = [
        processor.process(images=[img], text=prompt)
        for img, prompt in zip(images, prompts)
    ]

    input_ids_list = []
    max_len = 0
    for item in processed:
        input_ids = item["input_ids"]
        if input_ids.dim() == 2 and input_ids.size(0) == 1:
            input_ids = input_ids.squeeze(0)
        max_len = max(max_len, int(input_ids.shape[0]))
        input_ids_list.append(input_ids)

    padded_ids = []
    for input_ids in input_ids_list:
        pad_len = max_len - int(input_ids.shape[0])
        if pad_len > 0:
            pad = input_ids.new_full((pad_len,), -1)
            input_ids = torch.cat([input_ids, pad], dim=0)
        padded_ids.append(input_ids)
    input_ids = torch.stack(padded_ids, dim=0)
    attention_mask = input_ids.ne(-1)

    inputs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}

    for key in ("images", "image_masks", "image_input_idx"):
        tensors = []
        for item in processed:
            val = item.get(key)
            if val is None:
                continue
            if val.dim() > 0 and val.size(0) == 1:
                val = val.squeeze(0)
            tensors.append(val)
        if tensors:
            max_n = max(int(t.size(0)) for t in tensors)
            padded = []
            for t in tensors:
                pad_n = max_n - int(t.size(0))
                if pad_n > 0:
                    pad_shape = (pad_n, *t.shape[1:])
                    pad = t.new_zeros(pad_shape)
                    t = torch.cat([t, pad], dim=0)
                padded.append(t)
            inputs[key] = torch.stack(padded, dim=0)

    heatmaps = torch.from_numpy(
        np.stack([s.heatmap for s in samples], axis=0)
    ).unsqueeze(1)

    return {
        "model_inputs": inputs,
        "heatmaps": heatmaps,
        "image_sizes": [s.image_size for s in samples],
        "prompts": prompts,
        "images": images,
        "orig_images": [s.orig_image for s in samples],
        "orig_heatmaps": [s.orig_heatmap for s in samples],
    }
