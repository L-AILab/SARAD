import argparse
from pathlib import Path

import numpy as np
import torch.nn.functional as F
from PIL import Image

from models.encoders import ImageEncoder
from utils.inference import add_model_arguments, load_model, predict


def main():
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--image", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--start", type=float, required=True, help="Target word start, seconds")
    parser.add_argument("--end", type=float, required=True, help="Target word end, seconds")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/prediction")
    args = parser.parse_args()
    model = load_model(args.checkpoint, args.hubert, args.device)
    encoder = ImageEncoder(args.dino_repo, args.dino_weights).to(args.device)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    score = predict(model, encoder, image, args.audio, args.start, args.end)
    score = F.interpolate(
        score[None, None], (image.height, image.width), mode="bilinear", align_corners=False
    )[0, 0]
    response = (score - score.min()) / (score.max() - score.min()).clamp_min(1e-12)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "response.npy", score.numpy())
    Image.fromarray((response.numpy() > args.threshold).astype("uint8") * 255).save(
        out / "mask.png"
    )


if __name__ == "__main__":
    main()
