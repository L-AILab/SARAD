from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import random
import numpy as np
import torch
import yaml


class EarlyStopping:
    def __init__(self, min_epochs: int, patience: int, min_delta: float):
        self.min_epochs = int(min_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.best_epoch = 0
        self.bad_epochs = 0

    def update(self, epoch: int, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = float(value)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return epoch >= self.min_epochs and self.bad_epochs >= self.patience

    def state_dict(self) -> dict:
        return {
            "min_epochs": self.min_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "EarlyStopping":
        output = cls(state["min_epochs"], state["patience"], state["min_delta"])
        output.best = float(state["best"])
        output.best_epoch = int(state["best_epoch"])
        output.bad_epochs = int(state["bad_epochs"])
        return output


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _atomic_checkpoint(checkpoint: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def resolve_amp(device: torch.device, cfg: dict):
    if device.type != "cuda":
        return (torch.float32, False)
    name = str(cfg.get("amp_dtype", "bfloat16")).lower()
    if name in {"bfloat16", "bf16"}:
        return (torch.bfloat16, False)
    if name in {"float16", "fp16"}:
        return (torch.float16, True)
    raise ValueError(f"Unsupported AMP dtype: {name}")
