from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor
from peft import PeftModel

from dataset_vlm import SurgActMolmoDataset
from losses import bce_iou_loss
from models import MolmoHeatmapModel


def generate_heatmap_from_polygon(
    image_shape: tuple[int, int],
    contact_points: list[dict[str, Any]],
) -> np.ndarray:
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


class HeatmapInferenceDataset:
    def __init__(
        self,
        csv_path: str,
        split_name: str | None = None,
        heatmap_size: tuple[int, int] | None = (224, 224),
        device: str = "cpu",
    ) -> None:
        self.dataset = SurgActMolmoDataset(
            csv_path=csv_path,
            split_name=split_name,
            device=device,
        )
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
        meta = sample.get("meta", {})
        image_path = (
            f"{meta.get('clip_stem','')}_frame_{sample.get('frame_index','')}.png"
        )
        return HeatmapSample(
            image=image,
            image_path=image_path,
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
        "images": images,
    }


def _move_inputs_to_device(model_inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for k, v in model_inputs.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _download_checkpoint(checkpoint_url: str, checkpoint_dir: Path) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filename = checkpoint_url.split("?")[0].split("/")[-1]
    if not filename:
        filename = "checkpoint.bin"
    download_path = checkpoint_dir / filename
    if not download_path.exists():
        urlretrieve(checkpoint_url, download_path)
    if filename.endswith(".zip"):
        with zipfile.ZipFile(download_path, "r") as zf:
            zf.extractall(checkpoint_dir)
    elif filename.endswith(".tar") or filename.endswith(".tar.gz") or filename.endswith(".tgz"):
        with tarfile.open(download_path, "r:*") as tf:
            tf.extractall(checkpoint_dir)
    return checkpoint_dir


def _load_model_from_checkpoint(
    model_name: str,
    checkpoint_dir: Path,
    hidden_layer: int,
    use_lora: bool,
    head_channels: int,
    load_in_4bit: bool,
) -> MolmoHeatmapModel:
    model = MolmoHeatmapModel(
        model_name=model_name,
        hidden_layer=hidden_layer,
        lora=use_lora,
        head_channels=head_channels,
        load_in_4bit=load_in_4bit,
    )
    head_path = checkpoint_dir / "heatmap_head.pt"
    if not head_path.exists():
        raise FileNotFoundError(f"Missing heatmap head: {head_path}")
    model.heatmap_head.load_state_dict(torch.load(head_path, map_location="cpu"))

    full_dir = checkpoint_dir / "molmo_full"
    lora_dir = checkpoint_dir / "molmo_lora"
    if full_dir.exists():
        model.model = model.model.from_pretrained(
            str(full_dir),
            trust_remote_code=True,
            torch_dtype="auto",
        )
    elif use_lora and lora_dir.exists():
        model.model = PeftModel.from_pretrained(model.model, str(lora_dir))
    elif use_lora and not lora_dir.exists():
        raise FileNotFoundError(
            f"Missing LoRA adapter directory: {lora_dir}. "
            "Run without --no-lora only if a LoRA adapter is available."
        )
    return model


def _to_overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
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
    if colored.shape[:2] != img.shape[:2]:
        colored = cv2.resize(colored, target_size, interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    return Image.fromarray(overlay)


def _per_sample_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )           
    bce = bce.mean(dim=(1, 2, 3))
    probs = torch.sigmoid(logits)
    iou = soft_iou_loss(probs, targets)
    total = bce + iou
    return total, bce, iou

def main() -> None:
    parser = argparse.ArgumentParser(description="Infer Molmo heatmap predictions.")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--model-name", default="allenai/Molmo-7B-D-0924")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-url", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heatmap-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--hidden-layer", type=int, default=-1)
    parser.add_argument("--prompt-type", choices=["contact", "empty"], default="contact")
    parser.add_argument("--head-channels", type=int, default=256)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--decord-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--output-dir", default="/home/aiza/molmo_heatmap_pipeline/infer_outputs")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if args.checkpoint_url:
        checkpoint_dir = _download_checkpoint(args.checkpoint_url, checkpoint_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    dataset = HeatmapInferenceDataset(
        csv_path=args.csv,
        split_name=args.split,
        heatmap_size=(int(args.heatmap_size[0]), int(args.heatmap_size[1])),
        device=args.decord_device,
    )

    def _collate(samples):
        return collate_heatmap_batch(samples, processor, prompt_type=args.prompt_type)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_checkpoint(
        model_name=args.model_name,
        checkpoint_dir=checkpoint_dir,
        hidden_layer=args.hidden_layer,
        use_lora=not args.no_lora,
        head_channels=args.head_channels,
        load_in_4bit=args.load_in_4bit,
    ).to(device)
    model.eval()

    results_path = output_dir / "predictions.jsonl"
    processed = 0
    with torch.no_grad(), results_path.open("w") as handle:
        for batch in tqdm(loader, desc="infer", unit="batch"):
            model_inputs = _move_inputs_to_device(batch["model_inputs"], device)
            heatmaps = batch["heatmaps"].to(device)
            image_sizes = batch["image_sizes"]
            target_size = (heatmaps.size(-2), heatmaps.size(-1))
            logits = model(model_inputs, image_sizes=image_sizes, target_size=target_size)
            loss, _ = bce_iou_loss(logits, heatmaps, bce_weight=1.0, iou_weight=1.0)
            per_total, per_bce, per_iou = _per_sample_losses(logits, heatmaps)
            preds = torch.sigmoid(logits).cpu().numpy()
            targets = heatmaps.cpu().numpy()

            for i in range(len(batch["image_paths"])):
                pred_map = preds[i, 0]
                tgt_map = targets[i, 0]
                print('pred_map shape:', pred_map.shape)
                print('tgt_map shape:', tgt_map.shape) 
                overlay_pred = _to_overlay(batch["images"][i], pred_map)
                overlay_tgt = _to_overlay(batch["images"][i], tgt_map)
                combined = Image.new(
                    "RGB",
                    (overlay_pred.width * 2, overlay_pred.height),
                )
                combined.paste(overlay_tgt, (0, 0))
                combined.paste(overlay_pred, (overlay_pred.width, 0))
                out_name = f"{Path(batch['image_paths'][i]).stem}_tgt_pred.png"
                combined.save(output_dir / out_name)

                payload = {
                    "image_path": batch["image_paths"][i],
                    "loss_total": float(per_total[i].detach().cpu().item()),
                    "loss_bce": float(per_bce[i].detach().cpu().item()),
                    "loss_iou": float(per_iou[i].detach().cpu().item()),
                }
                handle.write(json.dumps(payload) + "\n")
            processed += len(batch["image_paths"])
            if args.max_samples and processed >= args.max_samples:
                break


if __name__ == "__main__":
    main()
