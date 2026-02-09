from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# --- Decord Imports ---
try:
    from decord import VideoReader, cpu, gpu
except ImportError:
    raise ImportError("Please install decord: pip install decord")

@dataclass
class SampleRecord:
    case_dir: str
    clip_stem: str
    json_path: Path
    mp4_path: Path
    frames_dir: Path
    start_frame: int
    end_frame: int
    split_name: str


def _load_csv(csv_path: str | Path) -> list[SampleRecord]:
    csv_path = Path(csv_path)
    records: list[SampleRecord] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                SampleRecord(
                    case_dir=row["case_dir"],
                    clip_stem=row["clip_stem"],
                    json_path=Path(row["json_path"]),
                    mp4_path=Path(row["mp4_path"]),
                    frames_dir=Path(row["frames_dir"]),
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    split_name=row.get("split_name", ""),
                )
            )
    return records


# --- Helper Functions for Decord ---
def decord_frame_to_numpy(frame) -> np.ndarray:
    """Decord frame -> numpy (H, W, 3) uint8, RGB."""
    arr = frame.asnumpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """numpy -> PIL (RGB)."""
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"Unexpected array shape for image: {arr.shape}")

def load_frame_from_video(
    video_path: str | Path,
    frame_index: int,
    device: str = "cpu",
) -> Image.Image:
    """
    Extracts a single frame from an MP4 file using decord.
    Returns a PIL Image.
    """
    video_path = str(video_path)
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Initialize Decord VideoReader
    # Note: For heavy dataloading, you might want to keep the VideoReader persistent
    # or manage context carefully, but for single frame access this works.
    ctx = gpu(0) if device == "gpu" else cpu(0)
    vr = VideoReader(video_path, ctx=ctx)

    if frame_index < 0 or frame_index >= len(vr):
        raise IndexError(
            f"Frame index {frame_index} out of bounds for video {video_path} (len={len(vr)})"
        )

    # Extract frame
    dec = vr[frame_index]
    arr = decord_frame_to_numpy(dec)
    return numpy_to_pil(arr)

def _load_json(json_path: Path) -> dict[str, Any]:
    with json_path.open("r") as f:
        return json.load(f)


def _select_frame_index(
    payload: dict[str, Any],
    default_frame: int,
) -> int:
    affordance_range = payload.get("affordance_range") or {}
    action_range = payload.get("action_range") or {}
    import random
    # Priority 2: Random frame within affordance range (if end exists)
    if "start" in affordance_range:
        start = int(affordance_range["start"])
        # Check if 'end' exists to create a range, otherwise stick to start
        if "end" in affordance_range:
            end = int(affordance_range["end"])
            if "start" in action_range:
                action_start = int(action_range["start"])
                end = min(end, action_start - 1)
            if start <= end:
                return random.randint(start, end)
            return int(default_frame)

    return int(default_frame)


def _frame_path(frames_dir: Path, start_frame: int, frame_index: int) -> Path:
    relative_index = frame_index - start_frame + 1
    if relative_index < 1:
        raise ValueError(
            f"Frame index {frame_index} precedes clip start {start_frame}"
        )
    return frames_dir / f"frame_{relative_index:05d}.png"


def construct_prompt_heatmap(payload: dict[str, Any], frame_index: int) -> str:
    return (
        """You are given a surgical video frame from a cholecystectomy surgery.
        Predict the tissue affordance region for the given action and tool.
        Surgery: cholecystectomy
        Action: {action}
        Tool: {tool}
        """
    )

def _reorder_centroid(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(points) <= 1:
        return points
    pts = [(p["x"], p["y"]) for p in points]
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)

    def angle(p):
        return math.atan2(p[1] - cy, p[0] - cx)

    sort_index = sorted(range(len(pts)), key=lambda i: angle(pts[i]), reverse=True)
    sorted_points = [points[i] for i in sort_index]

    def start_key(p):
        return (p["y"], p["x"])

    start_i = min(range(len(sorted_points)), key=lambda i: start_key(sorted_points[i]))
    return sorted_points[start_i:] + sorted_points[:start_i]


def extract_points_for_frame(
    payload: dict[str, Any],
    frame_index: int,
) -> dict[str, list[dict[str, Any]]]:
    points = payload.get("points", [])
    frame_points = [p for p in points if p.get("frameIndex") == frame_index]

    anchors = [p for p in frame_points if p.get("type") == "tool_anchor"]
    tips = [p for p in frame_points if p.get("type") == "tool_tip"]
    contacts = [p for p in frame_points if p.get("type") == "contact"]

    tips = sorted(tips, key=lambda p: p["x"])
    contacts = _reorder_centroid(contacts)

    return {
        "anchors": anchors[:1],
        "tips": tips[:2],
        "contacts": contacts[:4],
    }


def count_tool_tips_for_frame(payload: dict[str, Any], frame_index: int) -> int:
    points = payload.get("points", [])
    return sum(
        1
        for p in points
        if p.get("frameIndex") == frame_index and p.get("type") == "tool_tip"
    )


def count_tool_anchors_for_frame(payload: dict[str, Any], frame_index: int) -> int:
    points = payload.get("points", [])
    return sum(
        1
        for p in points
        if p.get("frameIndex") == frame_index and p.get("type") == "tool_anchor"
    )


def compute_contact_bbox(contacts: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    if not contacts:
        return None
    xs = [float(c.get("x", 0.0)) for c in contacts]
    ys = [float(c.get("y", 0.0)) for c in contacts]
    return (min(xs), min(ys), max(xs), max(ys))


class SurgActMolmoDataset:
    """
    Loads frames and JSON annotations referenced by dataset_pairs_with_frames.csv.
    """

    def __init__(
        self,
        csv_path: str | Path,
        split_name: str | None = None,
        device: str = "cpu",
    ):
        self.records = _load_csv(csv_path)
        if split_name:
            self.records = [
                r for r in self.records if r.split_name == split_name
            ]
        
        self.device = device

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        payload = _load_json(rec.json_path)
        global_frame_index = _select_frame_index(payload, rec.start_frame)
        video_relative_index = global_frame_index - rec.start_frame
        
        if video_relative_index < 0:
            print("LOADED FRAME INDEX < 0")

            video_relative_index = 0 

        # 3. Load Frame using Decord
        image = load_frame_from_video(
            video_path=rec.mp4_path,
            frame_index=video_relative_index,
            device=self.device
        )
        prompt = construct_prompt_heatmap(payload, global_frame_index)
        points = extract_points_for_frame(payload, global_frame_index)
        # contact_bbox = compute_contact_bbox(points.get("contacts", []))
        # num_tool_tips = count_tool_tips_for_frame(payload, frame_index)
        # num_tool_anchors = count_tool_anchors_for_frame(payload, frame_index)
        # tips_prompt = construct_prompt_tool_tips(payload, num_tool_tips)
        # anchors_prompt = construct_prompt_tool_anchors(payload, num_tool_anchors)

        return {
            "image": image,
            # "image_path": str(frame_path),
            "frame_index": global_frame_index,
            "json_path": str(rec.json_path),
            "action": payload.get("action"),
            "tool": payload.get("tool"),
            "prompt": prompt,
            # "tips_prompt": tips_prompt,
            # "anchors_prompt": anchors_prompt,
            "points": points,
            # "contact_bbox": contact_bbox,
            # "num_tool_tips": num_tool_tips,
            # "num_tool_anchors": num_tool_anchors,
            "meta": {
                "case_dir": rec.case_dir,
                "clip_stem": rec.clip_stem,
                "start_frame": rec.start_frame,
                "end_frame": rec.end_frame,
                "split_name": rec.split_name,
            },
        }
