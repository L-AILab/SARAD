from __future__ import annotations
import numpy as np
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable
import h5py
import torch
from torch.utils.data import Dataset


class H5LRU:
    def __init__(self, root: Path | str, max_open: int = 8):
        self.root = Path(root)
        self.max_open = int(max_open)
        self.handles: OrderedDict[str, h5py.File] = OrderedDict()

    def get(self, name: str) -> h5py.File:
        handle = self.handles.get(name)
        if handle is None:
            handle = h5py.File(self.root / name, "r", swmr=True)
            self.handles[name] = handle
            while len(self.handles) > self.max_open:
                (_, old) = self.handles.popitem(last=False)
                old.close()
        else:
            self.handles.move_to_end(name)
        return handle

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handles"] = OrderedDict()
        return state

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __del__(self):
        self.close()


def _read_jsonl(path: Path | str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class SpeechLVISDataset(Dataset):
    def __init__(
        self,
        manifest: Path | str,
        target_root: Path | str,
        image_indices: Iterable[Path | str],
        image_feature_root: Path | str,
        speech_root: Path | str,
        text_features: Path | str,
        sample_rate: int = 16000,
        max_samples: int | None = None,
        max_prototypes: int = 5,
    ):
        self.items = _read_jsonl(manifest)
        self.targets = H5LRU(target_root)
        self.images = H5LRU(image_feature_root)
        self.sample_rate = int(sample_rate)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.max_prototypes = int(max_prototypes)
        self.speech_root = Path(speech_root)
        self.image_index = {}
        for path in image_indices:
            for row in _read_jsonl(path):
                image_id = int(row["image_id"])
                if image_id in self.image_index:
                    raise ValueError(f"Duplicate image index: {image_id}")
                self.image_index[image_id] = row
        speech = defaultdict(list)
        for path in sorted((self.speech_root / "metadata").glob("*/*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            speech[int(value["lvis"]["class_id"])].append(value)
        self.speech = dict(speech)
        text = torch.load(text_features, map_location="cpu")
        self.text = {
            int(category_id): feature.float()
            for (category_id, feature) in zip(text["category_ids"], text["features"])
        }
        missing_images = {int(row["image_id"]) for row in self.items} - self.image_index.keys()
        missing_speech = {int(row["category_id"]) for row in self.items} - self.speech.keys()
        missing_text = {int(row["category_id"]) for row in self.items} - self.text.keys()
        if missing_images or missing_speech or missing_text:
            raise ValueError(
                f"Incomplete joins: images={len(missing_images)}, speech={len(missing_speech)}, text={len(missing_text)}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.items[index]
        category_id = int(row["category_id"])
        image_row = self.image_index[int(row["image_id"])]
        image_handle = self.images.get(image_row["feature_shard"])
        patches = torch.from_numpy(
            image_handle["patch_features"][int(image_row["feature_row"])]
        ).float()
        target_handle = self.targets.get(row["target_shard"])
        coverage = (
            torch.from_numpy(target_handle["mask_coverage"][int(row["target_row"])]).float() / 255.0
        )
        grid_size = tuple(row["grid_size"])
        mask = coverage.reshape(1, *grid_size)
        foreground = coverage > 0
        structures = select_structural_prototypes(patches, foreground, self.max_prototypes)
        valid_count = min(self.max_prototypes, max(1, int(foreground.sum())))
        structure_valid = torch.arange(self.max_prototypes) < valid_count
        utterances = self.speech[category_id]
        utterance = utterances[int(torch.randint(len(utterances), (1,)).item())]
        waveform = _load_audio(self.speech_root / utterance["audio"]["path"], self.sample_rate)
        if self.max_samples is not None:
            waveform = waveform[: self.max_samples]
        target_start = min(int(utterance["timing"]["target_start_sample"]), waveform.numel() - 1)
        target_end = min(int(utterance["timing"]["target_end_sample"]), waveform.numel())
        target_end = max(target_end, target_start + 1)
        return {
            "pair_id": row["pair_id"],
            "image_id": int(row["image_id"]),
            "category_id": category_id,
            "class_name": row["category_name"],
            "waveform": waveform,
            "target_start_sample": target_start,
            "target_end_sample": target_end,
            "patch_features": patches,
            "teacher_text_features": self.text[category_id],
            "mask": mask,
            "foreground_patch_mask": foreground,
            "structure_features": structures,
            "structure_valid_mask": structure_valid,
            "grid_size": grid_size,
            "utterance_id": utterance["utterance_id"],
        }


def speech_lvis_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_length = max((item["waveform"].numel() for item in batch))
    waveform = torch.zeros(len(batch), max_length)
    attention = torch.zeros(len(batch), max_length, dtype=torch.long)
    for (index, item) in enumerate(batch):
        length = item["waveform"].numel()
        waveform[index, :length] = item["waveform"]
        attention[index, :length] = 1
    grids = {item["grid_size"] for item in batch}
    if len(grids) != 1:
        raise ValueError("All Speech-LVIS samples in a batch must share grid_size")
    return {
        "waveform": waveform,
        "attention_mask": attention,
        "target_start_samples": torch.tensor(
            [item["target_start_sample"] for item in batch], dtype=torch.long
        ),
        "target_end_samples": torch.tensor(
            [item["target_end_sample"] for item in batch], dtype=torch.long
        ),
        "patch_features": torch.stack([item["patch_features"] for item in batch]),
        "teacher_text_features": torch.stack([item["teacher_text_features"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "foreground_patch_masks": torch.stack([item["foreground_patch_mask"] for item in batch]),
        "structure_features": torch.stack([item["structure_features"] for item in batch]),
        "structure_valid_mask": torch.stack([item["structure_valid_mask"] for item in batch]),
        "category_ids": [item["category_id"] for item in batch],
        "class_name": [item["class_name"] for item in batch],
        "pair_id": [item["pair_id"] for item in batch],
        "utterance_id": [item["utterance_id"] for item in batch],
        "grid_size": grids.pop(),
    }


def _load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    try:
        import torchaudio

        (waveform, observed_rate) = torchaudio.load(str(path))
        if observed_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, observed_rate, sample_rate)
    except Exception:
        try:
            import soundfile as sf

            (values, observed_rate) = sf.read(str(path), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(values.T)
        except ImportError:
            import wave

            with wave.open(str(path), "rb") as handle:
                observed_rate = handle.getframerate()
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                raw = handle.readframes(handle.getnframes())
            if width != 2:
                raise ValueError(f"Wave fallback only supports 16-bit PCM: {path}")
            values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            waveform = torch.from_numpy(values.reshape(-1, channels).T.copy())
        if observed_rate != sample_rate:
            import librosa

            waveform = torch.from_numpy(
                librosa.resample(waveform.numpy(), orig_sr=observed_rate, target_sr=sample_rate)
            )
    return waveform.mean(0).float()


def select_structural_prototypes(
    patch_features: torch.Tensor, foreground_mask: torch.Tensor, max_prototypes: int = 8
) -> torch.Tensor:
    foreground = patch_features[foreground_mask]
    if foreground.shape[0] == 0:
        foreground = patch_features
    count = min(max_prototypes, foreground.shape[0])
    center = foreground.mean(0, keepdim=True)
    first = ((foreground - center) ** 2).sum(-1).argmax()
    selected = [int(first)]
    distances = ((foreground - foreground[first]) ** 2).sum(-1)
    for _ in range(1, count):
        index = int(distances.argmax())
        selected.append(index)
        distances = torch.minimum(distances, ((foreground - foreground[index]) ** 2).sum(-1))
    prototypes = foreground[selected]
    if prototypes.shape[0] < max_prototypes:
        repeat_indices = torch.arange(max_prototypes) % prototypes.shape[0]
        prototypes = prototypes[repeat_indices]
    return prototypes
