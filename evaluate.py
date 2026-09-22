import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.functional.classification import binary_average_precision

from models.encoders import ImageEncoder
from utils.inference import add_model_arguments, load_model, predict
from utils.metrics import denseav_multi_iou_flat, resize_center_crop


def evaluate(args):
    rows = [
        json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()
    ]
    root = Path(args.data_root)
    model = load_model(args.checkpoint, args.hubert, args.device)
    encoder = ImageEncoder(args.dino_repo, args.dino_weights).to(args.device)
    groups = defaultdict(list)
    for row in rows:
        if args.protocol == "cityscapes" and "valid_labels" not in row:
            raise ValueError("Cityscapes queries must specify valid_labels to exclude void pixels.")
        groups[(row["class_id"], row.get("voice", ""))].append(row)
    results = []
    for (class_id, voice), items in sorted(groups.items()):
        predictions, targets = [], []
        for row in items:
            with Image.open(root / row["image"]) as file:
                image = file.convert("RGB")
            with Image.open(root / row["mask"]) as file:
                mask = file.copy()
            if args.protocol == "ade20k":
                image = resize_center_crop(image, 224)
                mask = resize_center_crop(mask, 224, nearest=True)
            elif args.protocol == "cityscapes":
                mask = mask.resize((448, 224), Image.Resampling.NEAREST)
            labels = np.asarray(mask)
            if labels.ndim != 2:
                raise ValueError("Evaluation expects a single-channel semantic or binary mask.")
            target = torch.from_numpy(labels == row.get("mask_label", 1)).flatten()
            valid = torch.ones_like(target)
            if "valid_labels" in row:
                valid = torch.from_numpy(np.isin(labels, row["valid_labels"])).flatten()
            elif "ignore_label" in row:
                valid = torch.from_numpy(labels != row["ignore_label"]).flatten()
            score = predict(model, encoder, image, root / row["audio"], row["start"], row["end"])
            score = F.interpolate(
                score[None, None], labels.shape, mode="bilinear", align_corners=False
            ).flatten()
            predictions.append(score[valid])
            targets.append(target[valid])
        scores, target = torch.cat(predictions), torch.cat(targets)
        scores = (scores - scores.min()) / (scores.max() - scores.min()).clamp_min(1e-12)
        result = dict(
            class_id=class_id,
            voice=voice,
            samples=len(items),
            mAP=100 * float(binary_average_precision(scores, target.long())),
            mIoU=100 * float(denseav_multi_iou_flat(scores, target)),
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = dict(
        protocol=args.protocol,
        samples=len(rows),
        groups=len(results),
        mAP=float(np.mean([r["mAP"] for r in results])),
        mIoU=float(np.mean([r["mIoU"] for r in results])),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_class.json").write_text(json.dumps(results, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="Data")
    parser.add_argument(
        "--protocol", choices=["ade20k", "cityscapes", "pascal", "isic", "isaid"], required=True
    )
    parser.add_argument("--output", default="outputs/evaluation")
    evaluate(parser.parse_args())
