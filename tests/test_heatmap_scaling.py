from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import data


class _StubDataset:
    def __init__(self, sample: dict) -> None:
        self._sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        return self._sample


def test_heatmap_contact_scaling_matches_resized_image(monkeypatch) -> None:
    orig_w, orig_h = 200, 100
    target_h, target_w = 50, 80
    image = Image.new("RGB", (orig_w, orig_h), color=(0, 0, 0))

    contacts = [
        {"x": 50, "y": 20},
        {"x": 150, "y": 20},
        {"x": 150, "y": 80},
        {"x": 50, "y": 80},
    ]
    sample = {"image": image, "points": {"contacts": contacts}, "prompt": "p"}

    monkeypatch.setattr(data, "SurgActMolmoDataset", lambda **_: _StubDataset(sample))

    dataset = data.HeatmapMolmoDataset(
        csv_path="unused.csv",
        heatmap_size=(target_h, target_w),
    )
    result = dataset[0]

    assert result.image.size == (target_w, target_h)
    assert result.heatmap.shape == (target_h, target_w)

    mask = result.heatmap > 0
    ys, xs = np.where(mask)
    assert ys.size > 0

    centroid_y = float(ys.mean())
    centroid_x = float(xs.mean())

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    expected_center_x = (50 + 150) / 2 * scale_x
    expected_center_y = (20 + 80) / 2 * scale_y

    assert abs(centroid_x - expected_center_x) <= 1.0
    assert abs(centroid_y - expected_center_y) <= 1.0
