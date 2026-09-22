import torch
from PIL import Image


def resize_center_crop(image: Image.Image, size: int, nearest: bool = False) -> Image.Image:
    (width, height) = image.size
    scale = size / min(width, height)
    resized = image.resize(
        (round(width * scale), round(height * scale)),
        Image.Resampling.NEAREST if nearest else Image.Resampling.BICUBIC,
    )
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def denseav_multi_iou_flat(
    prediction: torch.Tensor, target: torch.Tensor, k: int = 20
) -> torch.Tensor:
    """Compute DenseAV's strict threshold sweep without materializing k full masks."""
    prediction = torch.as_tensor(prediction).detach().float().flatten()
    target = torch.as_tensor(target).detach().bool().flatten()
    if prediction.numel() != target.numel() or prediction.numel() == 0:
        raise ValueError("prediction and target must be nonempty and have equal size")
    thresholds = torch.linspace(
        prediction.min(), prediction.max(), k, device=prediction.device, dtype=prediction.dtype
    )
    buckets = torch.bucketize(prediction, thresholds, right=False)
    all_counts = torch.bincount(buckets, minlength=k + 1)
    positive_counts = torch.bincount(buckets[target], minlength=k + 1)

    def suffix_sum(values: torch.Tensor) -> torch.Tensor:
        return values.flip(0).cumsum(0).flip(0)

    predicted_positive = suffix_sum(all_counts)[1:]
    intersection = suffix_sum(positive_counts)[1:]
    union = predicted_positive + target.sum() - intersection
    return (intersection.float() / union.clamp_min(1).float()).max()
