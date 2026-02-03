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

from molmoact_exps.preprocessing.dataset_vlm import SurgActMolmoDataset


def generate_heatmap_from_polygon(
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
    else:
        heatmap = dist_map
    return heatmap.astype(np.float32)


@dataclass
class HeatmapSample:
    image: Image.Image
    image_path: str
    image_size: tuple[int, int]
    prompt: str
    heatmap: np.ndarray


class HeatmapMolmoDataset:
    def __init__(
        self,
        csv_path: str,
        split_name: str | None = None,
        heatmap_size: tuple[int, int] | None = (224, 224),
    ) -> None:
        self.dataset = SurgActMolmoDataset(csv_path=csv_path, split_name=split_name)
        self.heatmap_size = heatmap_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> HeatmapSample:
        sample = self.dataset[idx]
        image: Image.Image = sample["image"]
        width, height = image.size
        contacts = sample.get("points", {}).get("contacts", [])
        heatmap = generate_heatmap_from_polygon((height, width), contacts)

        if self.heatmap_size is not None:
            heatmap = cv2.resize(
                heatmap,
                (int(self.heatmap_size[1]), int(self.heatmap_size[0])),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)

        return HeatmapSample(
            image=image,
            image_path=sample["image_path"],
            image_size=(width, height),
            prompt=sample.get("prompt", ""),
            heatmap=heatmap,
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
        "image_paths": [s.image_path for s in samples],
        "prompts": prompts,
    }
