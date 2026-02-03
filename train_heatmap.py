from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from transformers import AutoProcessor

from data import HeatmapMolmoDataset, collate_heatmap_batch
from losses import bce_iou_loss
from models import MolmoHeatmapModel


def _save_checkpoint(
    out_dir: Path,
    model: MolmoHeatmapModel,
    processor,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    save_full_model: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"step": step, "epoch": epoch}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    torch.save(optimizer.state_dict(), out_dir / "optimizer.pt")
    torch.save(model.heatmap_head.state_dict(), out_dir / "heatmap_head.pt")
    processor.save_pretrained(out_dir / "processor")
    if save_full_model:
        model.model.save_pretrained(out_dir / "molmo_full")
    else:
        model.model.save_pretrained(out_dir / "molmo_lora")


def _load_checkpoint(
    checkpoint_dir: Path,
    model: MolmoHeatmapModel,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    meta_path = checkpoint_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Checkpoint meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    model.heatmap_head.load_state_dict(
        torch.load(checkpoint_dir / "heatmap_head.pt", map_location="cpu")
    )
    opt_path = checkpoint_dir / "optimizer.pt"
    if optimizer is not None and opt_path.exists():
        optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
    return meta


def _move_inputs_to_device(model_inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for k, v in model_inputs.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Molmo for heatmap prediction.")
    parser.add_argument("--csv", required=True, help="Path to dataset CSV.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--model-name", default="allenai/Molmo-7B-D-0924")
    parser.add_argument("--output-dir", default="/home/aiza/molmo_heatmap_pipeline/outputs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--heatmap-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--hidden-layer", type=int, default=-1)
    parser.add_argument("--prompt-type", choices=["contact", "empty"], default="contact")
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=1.0)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="*",
        default=None,
        help="Overrides default target modules for LoRA.",
    )
    parser.add_argument("--head-channels", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-full-model", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="molmo-heatmap")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    use_lora = args.lora or not args.no_lora
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    heatmap_size = (int(args.heatmap_size[0]), int(args.heatmap_size[1]))
    train_dataset = HeatmapMolmoDataset(
        csv_path=args.csv,
        split_name=args.train_split,
        heatmap_size=heatmap_size,
    )
    val_dataset = HeatmapMolmoDataset(
        csv_path=args.csv,
        split_name=args.val_split,
        heatmap_size=heatmap_size,
    )

    def _collate(samples):
        return collate_heatmap_batch(samples, processor, prompt_type=args.prompt_type)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )

    model = MolmoHeatmapModel(
        model_name=args.model_name,
        hidden_layer=args.hidden_layer,
        lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        head_channels=args.head_channels,
    ).to(device)

    if hasattr(model.model, "print_trainable_parameters"):
        model.model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        meta = _load_checkpoint(Path(args.resume), model, optimizer)
        start_epoch = int(meta.get("epoch", 0))
        global_step = int(meta.get("step", 0))

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_losses = []
        progress = tqdm(train_loader, desc=f"train epoch {epoch+1}", unit="batch")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            model_inputs = _move_inputs_to_device(batch["model_inputs"], device)
            heatmaps = batch["heatmaps"].to(device)
            image_sizes = batch["image_sizes"]
            target_size = (heatmaps.size(-2), heatmaps.size(-1))

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(model_inputs, image_sizes=image_sizes, target_size=target_size)
                loss, metrics = bce_iou_loss(
                    logits,
                    heatmaps,
                    bce_weight=args.bce_weight,
                    iou_weight=args.iou_weight,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(metrics["total"])
            progress.set_postfix({"loss": f"{metrics['total']:.4f}"})
            global_step += 1
            if args.wandb:
                wandb.log(
                    {
                        "train/loss_total": metrics["total"],
                        "train/loss_bce": metrics["bce"],
                        "train/loss_iou": metrics["iou"],
                        "train/step": global_step,
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
                )

        model.eval()
        val_losses = []
        with torch.no_grad():
            progress = tqdm(val_loader, desc=f"val epoch {epoch+1}", unit="batch")
            for batch in progress:
                model_inputs = _move_inputs_to_device(batch["model_inputs"], device)
                heatmaps = batch["heatmaps"].to(device)
                image_sizes = batch["image_sizes"]
                target_size = (heatmaps.size(-2), heatmaps.size(-1))
                logits = model(model_inputs, image_sizes=image_sizes, target_size=target_size)
                loss, metrics = bce_iou_loss(
                    logits,
                    heatmaps,
                    bce_weight=args.bce_weight,
                    iou_weight=args.iou_weight,
                )
                val_losses.append(metrics["total"])
                progress.set_postfix({"loss": f"{metrics['total']:.4f}"})
                if args.wandb:
                    wandb.log(
                        {
                            "val/loss_total": metrics["total"],
                            "val/loss_bce": metrics["bce"],
                            "val/loss_iou": metrics["iou"],
                            "val/step": global_step,
                            "val/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

        epoch_dir = output_dir / f"epoch_{epoch+1}"
        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(
                epoch_dir,
                model,
                processor,
                optimizer,
                step=global_step,
                epoch=epoch + 1,
                save_full_model=args.save_full_model,
            )

        summary = {
            "epoch": epoch + 1,
            "train_loss": float(sum(train_losses) / max(1, len(train_losses))),
            "val_loss": float(sum(val_losses) / max(1, len(val_losses))),
        }
        (epoch_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        if args.wandb:
            wandb.log(
                {
                    "epoch/train_loss": summary["train_loss"],
                    "epoch/val_loss": summary["val_loss"],
                    "epoch": epoch + 1,
                },
                step=global_step,
            )


if __name__ == "__main__":
    main()
