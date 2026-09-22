from __future__ import annotations
from models.sarad import SARAD
from models.losses import SARADLoss
from datasets.speech_lvis import SpeechLVISDataset, speech_lvis_collate
from utils.training import (
    EarlyStopping,
    _atomic_checkpoint,
    load_config,
    seed_everything,
    resolve_amp,
)
import argparse
import json
import time
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import DataLoader, Sampler, Subset


class FeatureLocalSampler(Sampler[int]):
    def __init__(self, dataset: SpeechLVISDataset):
        groups: dict[tuple[str, str], list[int]] = {}
        for (index, row) in enumerate(dataset.items):
            image = dataset.image_index[int(row["image_id"])]
            key = (image["feature_shard"], row["target_shard"])
            groups.setdefault(key, []).append(index)
        self.groups = list(groups.values())

    def __iter__(self):
        for group_index in torch.randperm(len(self.groups)).tolist():
            group = self.groups[group_index]
            for offset in torch.randperm(len(group)).tolist():
                yield group[offset]

    def __len__(self) -> int:
        return sum(map(len, self.groups))


def _dataset(cfg: dict, split: str) -> SpeechLVISDataset:
    return SpeechLVISDataset(
        cfg[f"{split}_manifest"],
        target_root=cfg[f"{split}_target_root"],
        image_indices=cfg[f"{split}_image_indices"],
        image_feature_root=cfg[f"{split}_image_feature_root"],
        speech_root=cfg["speech_root"],
        text_features=cfg["text_features"],
        sample_rate=int(cfg.get("sample_rate", 16000)),
        max_samples=int(cfg.get("sample_rate", 16000) * float(cfg.get("max_seconds", 6))),
        max_prototypes=int(cfg.get("max_prototypes", 5)),
    )


def _loader(dataset, cfg: dict, training: bool, smoke: bool, device: torch.device):
    workers = 0 if smoke else int(cfg.get("workers", 8))
    sampler = (
        FeatureLocalSampler(dataset)
        if training and isinstance(dataset, SpeechLVISDataset)
        else None
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=speech_lvis_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=training,
    )


def _build_loss(cfg: dict) -> SARADLoss:
    names = {
        "speech_text_weight",
        "speech_visual_weight",
        "mask_weight",
        "structure_weight",
        "intra_weight",
        "background_weight",
        "temperature",
        "distill_temperature",
        "background_margin",
    }
    return SARADLoss(**{name: cfg[name] for name in names if name in cfg})


def _run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    totals: Dict[str, float] = {}
    examples = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for (step, batch) in enumerate(loader, start=1):
        moved = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for (key, value) in batch.items()
        }
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                outputs = model(
                    waveform=moved["waveform"],
                    attention_mask=moved["attention_mask"],
                    target_start_samples=moved["target_start_samples"],
                    target_end_samples=moved["target_end_samples"],
                    patch_features=moved["patch_features"],
                    teacher_text_features=moved["teacher_text_features"],
                    structure_features=moved["structure_features"],
                    structure_valid_mask=moved["structure_valid_mask"],
                )
                losses = criterion(
                    outputs,
                    masks=moved["mask"],
                    foreground_patch_masks=moved["foreground_patch_masks"],
                    structure_valid_mask=moved["structure_valid_mask"],
                    grid_size=batch["grid_size"],
                    category_ids=batch["category_ids"],
                )
                scaled = losses["total"] / grad_accum_steps
        if training:
            scaler.scale(scaled).backward()
            if step % grad_accum_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        count = moved["waveform"].shape[0]
        examples += count
        for (name, value) in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
    return {name: value / max(examples, 1) for (name, value) in totals.items()}


def train(config_path: str, smoke: bool = False, resume: bool = False) -> Path:
    cfg = load_config(config_path)
    train_cfg = cfg["train"]
    seed_everything(int(train_cfg.get("seed", 0)))
    device = torch.device(train_cfg.get("device", "cuda:0"))
    model_cfg = cfg["model"]
    model = SARAD.from_pretrained(
        speech_backbone_path=model_cfg["speech_backbone"],
        ifp_checkpoint=model_cfg.get("ifp_checkpoint"),
        image_dim=model_cfg.get("image_dim"),
        text_dim=model_cfg.get("text_dim"),
        projection_dim=model_cfg.get("projection_dim"),
        head_hidden_dim=model_cfg.get("head_hidden_dim"),
        freeze_speech_backbone=bool(model_cfg.get("freeze_speech_backbone", False)),
        adapter_hidden_dim=int(model_cfg.get("adapter_hidden_dim", 512)),
        visual_adapter_hidden_dim=int(model_cfg.get("visual_adapter_hidden_dim", 256)),
        visual_adapter_scale=float(model_cfg.get("visual_adapter_scale", 0.1)),
        visual_pool_temperature=float(model_cfg.get("visual_pool_temperature", 0.1)),
        structure_topk=int(model_cfg.get("structure_topk", 5)),
    ).to(device)
    train_dataset = _dataset(cfg["data"], "train")
    val_dataset = _dataset(cfg["data"], "val")
    if smoke:
        limit = min(int(train_cfg.get("smoke_samples", 16)), len(train_dataset))
        train_dataset = Subset(train_dataset, range(limit))
        val_dataset = Subset(val_dataset, range(min(limit, len(val_dataset))))
    train_loader = _loader(train_dataset, train_cfg, True, smoke, device)
    val_loader = _loader(val_dataset, train_cfg, False, smoke, device)
    criterion = _build_loss(cfg.get("loss", {}))
    (amp_dtype, use_scaler) = resolve_amp(device, train_cfg)
    (speech, visual, backbone) = ([], [], [])
    for (name, parameter) in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("visual_adapter"):
            visual.append(parameter)
        elif name.startswith("speech_backbone"):
            backbone.append(parameter)
        else:
            speech.append(parameter)
    groups = [
        {"params": speech, "lr": float(train_cfg.get("lr", 0.0002))},
        {"params": visual, "lr": float(train_cfg.get("visual_lr", 5e-05))},
    ]
    if backbone:
        groups.append({"params": backbone, "lr": float(train_cfg.get("backbone_lr", 1e-05))})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(train_cfg.get("weight_decay", 0.0001)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    stopping = EarlyStopping(
        min_epochs=1 if smoke else int(train_cfg.get("min_epochs", 11)),
        patience=1 if smoke else int(train_cfg.get("patience", 10)),
        min_delta=float(train_cfg.get("min_delta", 0.0001)),
    )
    output_dir = Path(cfg["output"]["dir"] + ("_smoke" if smoke else ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    history = []
    start_epoch = 1
    if resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_speech_state_dict(saved["speech_ifp_v3"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history = saved.get("history", [])
        start_epoch = int(saved["epoch"]) + 1
        stopping = EarlyStopping.from_state_dict(saved["early_stopping"])
    max_epochs = 1 if smoke else int(train_cfg.get("max_epochs", 100))
    started = time.time()
    stop_reason = "max_epochs"
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_started = time.time()
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            grad_accum_steps=int(train_cfg.get("grad_accum_steps", 1)),
            amp_dtype=amp_dtype,
        )
        val_metrics = _run_epoch(model, val_loader, criterion, device, amp_dtype=amp_dtype)
        previous_best = stopping.best
        should_stop = stopping.update(epoch, val_metrics["total"])
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "bad_epochs": stopping.bad_epochs,
            "elapsed_sec": time.time() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint = {
            "version": 3,
            "epoch": epoch,
            "speech_ifp_v3": model.speech_state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "early_stopping": stopping.state_dict(),
            "config": cfg,
        }
        _atomic_checkpoint(checkpoint, last_path)
        if stopping.best < previous_best:
            _atomic_checkpoint(checkpoint, output_dir / "best.pt")
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if should_stop:
            stop_reason = "early_stopping"
            break
    summary = {
        "status": "complete",
        "version": 3,
        "smoke": smoke,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "epochs_completed": history[-1]["epoch"],
        "stop_reason": stop_reason,
        "best_epoch": stopping.best_epoch,
        "best_val_loss": stopping.best,
        "elapsed_sec": time.time() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return output_dir / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.config, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    main()
