from __future__ import annotations

import math
from typing import Iterable

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def _best_grid_shape(num_tokens: int, width: int, height: int) -> tuple[int, int]:
    if num_tokens <= 0:
        return 1, 1
    ratio = float(width) / max(float(height), 1.0)
    best = None
    limit = int(math.sqrt(num_tokens)) + 1
    for h in range(1, limit):
        if num_tokens % h != 0:
            continue
        w = num_tokens // h
        aspect = float(w) / float(h)
        score = abs(math.log(aspect / max(ratio, 1e-6)))
        if best is None or score < best[0]:
            best = (score, h, w)
    if best is not None:
        return best[1], best[2]
    return 1, num_tokens


def _reshape_tokens_to_grid(
    tokens: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    num_tokens = tokens.size(0)
    height, width = _best_grid_shape(num_tokens, image_size[0], image_size[1])
    total = height * width
    if total == 0:
        raise ValueError("Cannot reshape tokens: empty grid.")
    if total < num_tokens:
        tokens = tokens[:total]
    elif total > num_tokens:
        pad = tokens.new_zeros((total - num_tokens, tokens.size(-1)))
        tokens = torch.cat([tokens, pad], dim=0)
    return tokens.view(height, width, tokens.size(-1))


class HeatmapHead(nn.Module):
    def __init__(self, hidden_size: int, head_channels: int = 256) -> None:
        super().__init__()
        mid = max(1, head_channels // 2)
        self.proj = nn.Conv2d(hidden_size, head_channels, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(head_channels, mid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid, mid // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(mid // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.proj(x))

class MolmoHeatmapModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        hidden_layer: int = -1,
        lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Iterable[str] | None = None,
        head_channels: int = 256,
        load_in_4bit: bool = True,
    ) -> None:
        super().__init__()

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 # or torch.bfloat16 if your GPU supports it
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            quantization_config=quantization_config,
        )
        self.hidden_layer = hidden_layer
        self.hidden_size = int(self.model.config.hidden_size)
        self.heatmap_head = HeatmapHead(self.hidden_size, head_channels=head_channels)

        if lora:
            self.model.gradient_checkpointing_enable()
            if load_in_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            target_modules = list(lora_target_modules or [])
            if not target_modules:
                target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                ]
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_config)
        else:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.heatmap_head = HeatmapHead(self.hidden_size, head_channels=head_channels)
        self.heatmap_head.to(dtype=torch.float32)
        self.heatmap_head.train()


    def forward(
        self,
        model_inputs: dict[str, torch.Tensor],
        image_sizes: list[tuple[int, int]],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        outputs = self.model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[self.hidden_layer]

        image_input_idx = model_inputs.get("image_input_idx")
        if image_input_idx is None:
            raise ValueError("model_inputs missing image_input_idx from processor.")

        logits = []
        for i in range(hidden_states.size(0)):
            idx = image_input_idx[i]
            if idx.dim() > 1:
                idx = idx.reshape(-1)
            idx = idx[idx >= 0]
            if idx.numel() == 0:
                raise ValueError("No image tokens found for sample.")
            tokens = hidden_states[i].index_select(0, idx)
            grid = _reshape_tokens_to_grid(tokens, image_sizes[i])
            grid = grid.permute(2, 0, 1).unsqueeze(0)
            pred = self.heatmap_head(grid)
            pred = nn.functional.interpolate(
                pred,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            logits.append(pred)
        return torch.cat(logits, dim=0)
