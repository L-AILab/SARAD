from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F


def multi_positive_contrastive(
    student: torch.Tensor, teacher: torch.Tensor, labels, temperature: float
) -> torch.Tensor:
    logits = student @ teacher.T / temperature
    positive = torch.tensor(
        [[left == right for right in labels] for left in labels],
        dtype=torch.bool,
        device=logits.device,
    )

    def direction(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positive_values = values.masked_fill(~mask, torch.finfo(values.dtype).min)
        return (torch.logsumexp(values, dim=-1) - torch.logsumexp(positive_values, dim=-1)).mean()

    return 0.5 * (direction(logits, positive) + direction(logits.T, positive.T))


def distribution_kl(
    student: torch.Tensor, teacher: torch.Tensor, temperature: float
) -> torch.Tensor:
    return (
        F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            F.softmax(teacher.detach() / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )


def soft_mask_loss(
    logits: torch.Tensor, masks: torch.Tensor, grid_size: Tuple[int, int]
) -> torch.Tensor:
    (gh, gw) = grid_size
    maps = logits.reshape(-1, 1, gh, gw)
    maps = F.interpolate(maps, masks.shape[-2:], mode="bilinear", align_corners=False)
    target = masks.float().reshape_as(maps)
    bce = F.binary_cross_entropy_with_logits(maps, target)
    probs = maps.sigmoid()
    intersection = (probs * target).sum(dim=(1, 2, 3))
    dice = (
        1.0 - ((2 * intersection + 1) / (probs.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()
    )
    return bce + dice


class SARADLoss(nn.Module):
    """GT-supervised category-speech loss preserving IFP focus and structure."""

    def __init__(
        self,
        speech_text_weight: float = 1.0,
        speech_visual_weight: float = 1.0,
        mask_weight: float = 1.0,
        structure_weight: float = 0.5,
        intra_weight: float = 0.5,
        background_weight: float = 0.5,
        temperature: float = 0.07,
        distill_temperature: float = 0.2,
        background_margin: float = 0.2,
    ):
        super().__init__()
        self.weights = {
            "speech_text": float(speech_text_weight),
            "speech_visual": float(speech_visual_weight),
            "mask": float(mask_weight),
            "structure": float(structure_weight),
            "intra": float(intra_weight),
            "background": float(background_weight),
        }
        self.temperature = float(temperature)
        self.distill_temperature = float(distill_temperature)
        self.background_margin = float(background_margin)

    @staticmethod
    def _region_focuses(
        image_tokens: torch.Tensor,
        masks: torch.Tensor,
        grid_size: Tuple[int, int],
        foreground_patch_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if foreground_patch_masks is None:
            foreground_patch_masks = F.interpolate(
                masks.float(), grid_size, mode="nearest"
            ).flatten(1)
        foreground_patch_masks = foreground_patch_masks.to(image_tokens.dtype)
        background_patch_masks = 1.0 - foreground_patch_masks
        foreground_weights = foreground_patch_masks / foreground_patch_masks.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        background_weights = background_patch_masks / background_patch_masks.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        foreground = F.normalize(
            torch.einsum("bn,bnd->bd", foreground_weights, image_tokens), dim=-1
        )
        background = F.normalize(
            torch.einsum("bn,bnd->bd", background_weights, image_tokens), dim=-1
        )
        return (foreground, background, foreground_patch_masks.bool())

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        masks: torch.Tensor,
        grid_size: Tuple[int, int],
        category_ids,
        foreground_patch_masks: Optional[torch.Tensor] = None,
        structure_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        required = {"speech_focus", "text_focus", "speech_map", "image_tokens"}
        missing = required - outputs.keys()
        if missing:
            raise KeyError(f"Missing Speech-IFP V3 outputs: {sorted(missing)}")
        speech = outputs["speech_focus"]
        text = outputs["text_focus"].detach()
        (foreground, background, _) = self._region_focuses(
            outputs["image_tokens"], masks, grid_size, foreground_patch_masks
        )
        losses: Dict[str, torch.Tensor] = {
            "speech_text": multi_positive_contrastive(speech, text, category_ids, self.temperature),
            "speech_visual": multi_positive_contrastive(
                speech, foreground, category_ids, self.temperature
            ),
            "mask": soft_mask_loss(outputs["speech_map"], masks, grid_size),
        }
        if "speech_structure_logits" in outputs and "teacher_structure_logits" in outputs:
            global_structure = distribution_kl(
                outputs["speech_structure_logits"],
                outputs["teacher_structure_logits"],
                self.distill_temperature,
            )
            temporal_structure = distribution_kl(
                outputs["speech_temporal_structure_logits"],
                outputs["teacher_temporal_structure_logits"],
                self.distill_temperature,
            )
            losses["structure"] = 0.5 * (global_structure + temporal_structure)
            similarities = F.cosine_similarity(
                outputs["structure_tokens"], foreground[:, None, :], dim=-1
            )
            if structure_valid_mask is not None:
                valid = structure_valid_mask.to(similarities.dtype)
                losses["intra"] = ((1.0 - similarities) * valid).sum() / valid.sum().clamp_min(1.0)
            else:
                losses["intra"] = (1.0 - similarities).mean()
        else:
            losses["structure"] = speech.new_zeros(())
            losses["intra"] = speech.new_zeros(())
        foreground_similarity = F.cosine_similarity(speech, foreground, dim=-1)
        background_similarity = F.cosine_similarity(speech, background, dim=-1)
        losses["background"] = F.relu(
            self.background_margin - foreground_similarity + background_similarity
        ).mean()
        losses["total"] = sum((self.weights[name] * losses[name] for name in self.weights))
        return losses
