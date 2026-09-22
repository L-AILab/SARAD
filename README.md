# SARAD

Speech-guided image segmentation with semantic focus alignment and regional affinity distillation.

## Model

SARAD takes an image, a spoken category prompt and the target word's time interval.
HuBERT encodes the speech. Attention-pooled and frame-level speech features are
matched to DINOv3 image patches, and the two responses are averaged. During
training, category text and foreground regions supervise the speech focus;
foreground prototypes provide the regional distillation targets.

This repository contains the full HuBERT Large + DINOv3 ViT-H/16+ model.
Inference uses the speech and image branches, without text or SAM.

```text
SARAD/
  models/          # Network, frozen image encoder and losses
  datasets/        # Speech-LVIS training data loader
  configs/         # Full-model training configuration
  utils/           # Checkpoint loading, metrics and training utilities
  train.py
  inference.py
  evaluate.py
  requirements.txt
```

## Installation

Use Python 3.10 or newer. Install PyTorch and torchvision for your CUDA version,
then install the remaining dependencies:

```bash
pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov3.git external/dinov3
```

Prepare the following local files:

```text
checkpoints/
  sarad.pt
  hubert-large-ls960-ft/   # Hugging Face configuration and component weights
  dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
```

HuBERT uses `facebook/hubert-large-ls960-ft`. Obtain the DINOv3 checkpoint from
the official repository under its license. Model weights and datasets are not
included in this source archive. Load only trusted PyTorch checkpoints.

The released model loader accepts the existing `speech_ifp_v3` checkpoint key.
The key is retained so saved training weights do not need conversion.

## Training

Training consumes the prepared Speech-LVIS image features, mask coverage shards,
category text features and speech metadata. Set their locations in
`configs/sarad.yaml`; paths are relative to the repository's working directory.
The exact shard layout is documented in [datasets/README.md](datasets/README.md).

```bash
python train.py --config configs/sarad.yaml
python train.py --config configs/sarad.yaml --resume
```

The configuration uses a batch size of 8, four accumulation steps and AdamW.
The checkpoint with the lowest validation total loss is saved as `best.pt`.
The supplied configuration uses BF16 and requires a GPU that supports it.
`--smoke` runs a small one-epoch check of a prepared dataset.

## Inference

Provide the target word's start and end times in seconds within the audio file.

```bash
python inference.py \
  --checkpoint checkpoints/sarad.pt \
  --dino-weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
  --image examples/image.jpg --audio examples/prompt.wav \
  --start 0.8 --end 1.4 --output outputs/example
```

Replace the example paths and times with your input. The output contains the raw
response (`response.npy`) and a binary mask (`mask.png`). The default 0.5 threshold
applies to the min-max-normalized image response; it is a visualization setting,
not the threshold-search benchmark metric.

Images are resized to 512 x 512, producing a 32 x 32 patch grid. The image
preprocessing deliberately follows the trained version: RGB values in [0, 1]
without ImageNet normalization.

## Evaluation

Create a JSONL manifest with one image-category-speech query per line:

```json
{"image":"images/example.jpg","mask":"masks/example.png","audio":"speech/prompt.wav","class_id":1,"mask_label":1,"ignore_label":255,"start":0.8,"end":1.4,"voice":"speaker1"}
```

Image, mask and audio paths are relative to `--data-root`. Masks must be
single-channel labels. Set `mask_label` to the foreground label actually stored
in the mask (for example, 255 for a 0/255 binary mask); do not mark that same value
as `ignore_label`. Cityscapes rows must provide `valid_labels` to exclude void.
Decode ADE20K's color-encoded annotations into semantic labels before indexing.

```bash
python evaluate.py \
  --checkpoint checkpoints/sarad.pt \
  --dino-weights checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
  --manifest Data/PASCAL/evaluation.jsonl --data-root Data/PASCAL \
  --protocol pascal --output outputs/pascal
```

Supported geometries are `ade20k` (224 center crop), `cityscapes` (448 x 224
nearest-neighbor GT) and native-resolution `pascal`, `isic`, `isaid`.
Evaluation pools valid pixels within each class and speaker group. It reports
exact average precision and the maximum IoU over 20 thresholds, then averages
groups. The original speech prompts, masks, class grouping and splits are needed
to reproduce a published table. A new manifest is not automatically the official
benchmark split.

## Acknowledgements

The projection and focus/structure formulation follows IFP. The model uses
[HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert),
[DINOv3](https://github.com/facebookresearch/dinov3) and training text features from
[CLIP](https://github.com/openai/CLIP). The threshold-sweep evaluation follows
[DenseAV](https://github.com/mhamilton723/DenseAV). Their code and checkpoints
retain their respective licenses. The repository layout was informed by
[PromptPilot](https://github.com/L-AILab/PromptPilot); its model code is not bundled.
