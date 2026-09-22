# Speech-LVIS

The training loader joins each image-category pair to an image-feature shard, a
mask-coverage shard, category text features and a bank of spoken prompts.

## Pair manifest

Each row in `train.jsonl` or `val.jsonl` contains:

```json
{"pair_id":"image_category","image_id":1,"category_id":1,"category_name":"object","target_shard":"part_000.h5","target_row":0,"grid_size":[32,32]}
```

The target HDF5 file contains `mask_coverage`, an array of flattened 32 x 32
coverage maps stored as uint8. Values are divided by 255 during loading. LVIS
instances belonging to the same category are united before generating coverage.

## Image features

Image index rows contain `image_id`, `feature_shard` and `feature_row`.
Each image-feature HDF5 shard contains `patch_features` with shape
`[images, 1024, 1280]`. Extract the last DINOv3 H/16+ patch features at 512 x 512,
using RGB [0, 1] inputs and the same transform as `models/encoders.py`.
The original prepared shards store FP16 features; the loader converts them to FP32.
Join images by their IDs rather than assuming that the LVIS validation split
uses only the COCO validation image directory.

## Text features

The text feature file is a PyTorch dictionary with `category_ids` and `features`.
Features are the 512-dimensional CLIP ViT-B/16 text embeddings from the original
category prompts. CLIP is used in preprocessing, not in the training loop.

## Speech bank

Place per-utterance metadata under `speech_root/metadata/<speaker>/*.json`:

```json
{"utterance_id":"object_speaker1","lvis":{"class_id":1},"audio":{"path":"audio/object.wav"},"timing":{"target_start_sample":12800,"target_end_sample":22400}}
```

Audio paths are relative to `speech_root`. Timing is in samples at 16 kHz.
One utterance is sampled per pair when loading. The full spoken prompt and target
word interval are retained. Training truncates audio at six seconds as specified
in the configuration. The original speech bank was generated with Microsoft
Neural TTS through edge-tts; there is no local TTS checkpoint in this repository.

## External data

This source package does not redistribute COCO/LVIS images, annotations, the
prepared Speech-LVIS feature shards or generated speech. Use the original
prepared split for reproduction. The loader is not a downloader or a replacement
for the data preparation pipeline.
