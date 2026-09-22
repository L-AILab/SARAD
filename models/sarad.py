from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Architecture-compatible copy of the official IFP projection head."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def _head_from_state(state: Dict[str, torch.Tensor]) -> ProjectionHead:
    (hidden_dim, in_dim) = state["net.0.weight"].shape
    (out_dim, hidden_dim_2) = state["net.2.weight"].shape
    if hidden_dim != hidden_dim_2:
        raise ValueError("Invalid IFP projection-head dimensions")
    head = ProjectionHead(in_dim, out_dim, hidden_dim)
    head.load_state_dict(state)
    return head


class MaskedAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self, sequence: torch.Tensor, frame_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(sequence).squeeze(-1)
        if frame_mask is not None:
            logits = logits.masked_fill(~frame_mask.bool(), torch.finfo(logits.dtype).min)
        weights = logits.softmax(dim=-1)
        return (torch.einsum("bt,btd->bd", weights, sequence), weights)


def interval_to_frame_mask(
    starts: torch.Tensor,
    ends: torch.Tensor,
    waveform_lengths: torch.Tensor,
    frame_count: int,
    valid_frame_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Map half-open sample intervals to valid HuBERT frame intervals."""
    device = starts.device
    if valid_frame_mask is None:
        valid_frame_mask = torch.ones(starts.shape[0], frame_count, dtype=torch.bool, device=device)
    else:
        valid_frame_mask = valid_frame_mask.to(device=device, dtype=torch.bool)
    valid_counts = valid_frame_mask.sum(-1).clamp_min(1)
    lengths = waveform_lengths.to(device).clamp_min(1)
    starts = starts.to(device).clamp_min(0)
    ends = torch.maximum(ends.to(device), starts + 1)
    starts = torch.minimum(starts, lengths - 1)
    ends = torch.minimum(ends, lengths)
    frame_starts = torch.floor(starts.float() / lengths * valid_counts).long()
    frame_ends = torch.ceil(ends.float() / lengths * valid_counts).long()
    frame_starts = torch.minimum(frame_starts, valid_counts - 1)
    frame_ends = torch.maximum(frame_ends, frame_starts + 1)
    steps = torch.arange(frame_count, device=device)[None]
    selected = (steps >= frame_starts[:, None]) & (steps < frame_ends[:, None]) & valid_frame_mask
    empty = ~selected.any(-1)
    if empty.any():
        fallback = (valid_counts[empty] - 1).clamp_min(0)
        selected[empty] = False
        selected[empty, fallback] = True
    return selected


class SARAD(nn.Module):
    def __init__(
        self,
        speech_backbone: nn.Module,
        image_head: nn.Module,
        text_head: nn.Module,
        freeze_speech_backbone: bool = True,
        freeze_modal_heads: bool = True,
        adapter_hidden_dim: int = 1024,
        visual_adapter_hidden_dim: int = 256,
        visual_adapter_scale: float = 0.1,
        visual_pool_temperature: float = 0.1,
        structure_topk: int = 8,
    ):
        super().__init__()
        self.speech_backbone = speech_backbone
        self.image_head = image_head
        self.text_head = text_head
        self.freeze_speech_backbone = freeze_speech_backbone
        self.freeze_modal_heads = freeze_modal_heads
        speech_dim = int(speech_backbone.config.hidden_size)
        projection_dim = image_head.net[2].out_features
        self.speech_pool = MaskedAttentionPool(speech_dim)
        self.speech_projector = nn.Sequential(
            nn.LayerNorm(speech_dim),
            nn.Linear(speech_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_dim, projection_dim),
        )
        if freeze_modal_heads:
            for module in (self.image_head, self.text_head):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if freeze_speech_backbone:
            self.speech_backbone.eval()
            for parameter in self.speech_backbone.parameters():
                parameter.requires_grad_(False)
        speech_dim = int(speech_backbone.config.hidden_size)
        dim = self.projection_dim
        self.temporal_projector = nn.Sequential(
            nn.LayerNorm(speech_dim),
            nn.Linear(speech_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_hidden_dim, dim),
        )
        self.visual_adapter = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, visual_adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(visual_adapter_hidden_dim, dim),
        )
        self.visual_adapter_scale = float(visual_adapter_scale)
        self.visual_pool_temperature = float(visual_pool_temperature)
        self.structure_topk = int(structure_topk)

    @property
    def image_dim(self) -> int:
        return self.image_head.net[0].in_features

    @property
    def text_dim(self) -> int:
        return self.text_head.net[0].in_features

    @property
    def projection_dim(self) -> int:
        return self.image_head.net[2].out_features

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_modal_heads:
            self.image_head.eval()
            self.text_head.eval()
        if self.freeze_speech_backbone:
            self.speech_backbone.eval()
        return self

    def project_image(self, patch_features: torch.Tensor) -> torch.Tensor:
        if self.freeze_modal_heads:
            with torch.no_grad():
                return self.image_head(patch_features)
        return self.image_head(patch_features)

    def project_teacher(self, text_features: torch.Tensor) -> torch.Tensor:
        if self.freeze_modal_heads:
            with torch.no_grad():
                return self.text_head(text_features)
        return self.text_head(text_features)

    @classmethod
    def from_pretrained(
        cls,
        speech_backbone_path: str,
        ifp_checkpoint: Optional[str],
        image_dim: Optional[int] = None,
        text_dim: Optional[int] = None,
        projection_dim: Optional[int] = None,
        head_hidden_dim: Optional[int] = None,
        freeze_speech_backbone: bool = True,
        adapter_hidden_dim: int = 1024,
        visual_adapter_hidden_dim: int = 256,
        visual_adapter_scale: float = 0.1,
        visual_pool_temperature: float = 0.1,
        structure_topk: int = 8,
        local_files_only: bool = True,
    ) -> "SARAD":
        from transformers import AutoModel

        if ifp_checkpoint is None:
            dimensions = (image_dim, text_dim, projection_dim, head_hidden_dim)
            if any((value is None for value in dimensions)):
                raise ValueError(
                    "Component-only initialization requires image_dim, text_dim, projection_dim, and head_hidden_dim"
                )
            image_head = ProjectionHead(image_dim, projection_dim, head_hidden_dim)
            text_head = ProjectionHead(text_dim, projection_dim, head_hidden_dim)
            freeze_modal_heads = False
        else:
            checkpoint = torch.load(Path(ifp_checkpoint), map_location="cpu", weights_only=False)
            image_state = checkpoint.get("img_head", checkpoint.get("img"))
            text_state = checkpoint.get("txt_head", checkpoint.get("txt"))
            if image_state is None or text_state is None:
                raise KeyError("IFP checkpoint must contain img_head/txt_head or img/txt")
            image_head = _head_from_state(image_state)
            text_head = _head_from_state(text_state)
            freeze_modal_heads = True
        backbone = AutoModel.from_pretrained(
            speech_backbone_path, local_files_only=local_files_only
        )
        return cls(
            backbone,
            image_head,
            text_head,
            freeze_speech_backbone=freeze_speech_backbone,
            freeze_modal_heads=freeze_modal_heads,
            adapter_hidden_dim=adapter_hidden_dim,
            visual_adapter_hidden_dim=visual_adapter_hidden_dim,
            visual_adapter_scale=visual_adapter_scale,
            visual_pool_temperature=visual_pool_temperature,
            structure_topk=structure_topk,
        )

    def _speech_hidden(
        self, waveform: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        context = torch.no_grad() if self.freeze_speech_backbone else nullcontext()
        with context:
            hidden = self.speech_backbone(
                input_values=waveform, attention_mask=attention_mask
            ).last_hidden_state
        frame_mask = None
        if attention_mask is not None:
            if hasattr(self.speech_backbone, "_get_feature_vector_attention_mask"):
                frame_mask = self.speech_backbone._get_feature_vector_attention_mask(
                    hidden.shape[1], attention_mask
                )
            else:
                lengths = attention_mask.sum(-1)
                output_lengths = (
                    (lengths.float() / waveform.shape[1] * hidden.shape[1]).ceil().long()
                )
                steps = torch.arange(hidden.shape[1], device=hidden.device)[None]
                frame_mask = steps < output_lengths[:, None]
        return (hidden, frame_mask)

    def encode_image(self, features: torch.Tensor) -> torch.Tensor:
        base = self.project_image(features)
        adapted = base + self.visual_adapter_scale * self.visual_adapter(base)
        return F.normalize(adapted, dim=-1)

    @staticmethod
    def dense_audio_visual_logits(
        speech_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        similarities = torch.einsum("atd,bnd->abtn", speech_tokens, image_tokens)
        best_spatial = similarities.max(dim=-1).values
        if frame_mask is None:
            return best_spatial.mean(dim=-1)
        valid = frame_mask[:, None, :].to(best_spatial.dtype)
        return (best_spatial * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1.0)

    def encode_category_speech(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        target_start_samples: torch.Tensor,
        target_end_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (hidden, valid_frame_mask) = self._speech_hidden(waveform, attention_mask)
        if attention_mask is None:
            waveform_lengths = torch.full(
                (waveform.shape[0],), waveform.shape[1], device=waveform.device
            )
        else:
            waveform_lengths = attention_mask.sum(-1)
        target_frame_mask = interval_to_frame_mask(
            target_start_samples,
            target_end_samples,
            waveform_lengths,
            hidden.shape[1],
            valid_frame_mask,
        )
        (pooled, pool_weights) = self.speech_pool(hidden, target_frame_mask)
        focus = F.normalize(self.speech_projector(pooled), dim=-1)
        tokens = F.normalize(self.temporal_projector(hidden), dim=-1)
        return (focus, tokens, pool_weights, target_frame_mask)

    @staticmethod
    def temporal_response(
        speech_tokens: torch.Tensor, image_tokens: torch.Tensor, target_frame_mask: torch.Tensor
    ) -> torch.Tensor:
        similarities = torch.einsum("btd,bnd->btn", speech_tokens, image_tokens)
        weights = target_frame_mask[:, :, None].to(similarities.dtype)
        return (similarities * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def forward(
        self,
        waveform: torch.Tensor,
        patch_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_start_samples: Optional[torch.Tensor] = None,
        target_end_samples: Optional[torch.Tensor] = None,
        teacher_text_features: Optional[torch.Tensor] = None,
        structure_features: Optional[torch.Tensor] = None,
        structure_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if target_start_samples is None or target_end_samples is None:
            lengths = (
                attention_mask.sum(-1)
                if attention_mask is not None
                else torch.full((waveform.shape[0],), waveform.shape[1], device=waveform.device)
            )
            target_start_samples = torch.zeros_like(lengths)
            target_end_samples = lengths
        (
            speech_focus,
            speech_tokens,
            pool_weights,
            target_frame_mask,
        ) = self.encode_category_speech(
            waveform, attention_mask, target_start_samples, target_end_samples
        )
        image_tokens = self.encode_image(patch_features)
        global_map = torch.einsum("bnd,bd->bn", image_tokens, speech_focus)
        temporal_map = self.temporal_response(speech_tokens, image_tokens, target_frame_mask)
        speech_map = 0.5 * (global_map + temporal_map)
        output: Dict[str, torch.Tensor] = {
            "speech_focus": speech_focus,
            "speech_tokens": speech_tokens,
            "global_speech_map": global_map,
            "temporal_speech_map": temporal_map,
            "speech_map": speech_map,
            "image_tokens": image_tokens,
            "pool_weights": pool_weights,
            "target_frame_mask": target_frame_mask,
            "dense_av_logits": self.dense_audio_visual_logits(
                speech_tokens, image_tokens, target_frame_mask
            ),
        }
        if teacher_text_features is not None:
            text_focus = self.project_teacher(teacher_text_features)
            teacher_map = torch.einsum("bnd,bd->bn", image_tokens, text_focus)
            output.update({"text_focus": text_focus, "teacher_map": teacher_map})
        structures = None
        if structure_features is not None:
            structures = self.encode_image(structure_features)
        elif "teacher_map" in output:
            count = min(self.structure_topk, image_tokens.shape[1])
            indices = output["teacher_map"].detach().topk(count, dim=-1).indices
            structures = image_tokens.gather(
                1, indices[:, :, None].expand(-1, -1, image_tokens.shape[-1])
            )
        if structures is not None:
            global_logits = torch.einsum("bkd,bd->bk", structures, speech_focus)
            temporal_logits = torch.einsum("btd,bkd->btk", speech_tokens, structures)
            temporal_logits = temporal_logits.masked_fill(
                ~target_frame_mask[:, :, None], torch.finfo(temporal_logits.dtype).min
            )
            counts = target_frame_mask.sum(1, keepdim=True).clamp_min(1)
            temporal_logits = torch.logsumexp(temporal_logits, dim=1) - counts.log()
            if structure_valid_mask is not None:
                invalid = ~structure_valid_mask.bool()
                global_logits = global_logits.masked_fill(invalid, -10000.0)
                temporal_logits = temporal_logits.masked_fill(invalid, -10000.0)
            output.update(
                {
                    "structure_tokens": structures,
                    "speech_structure_logits": global_logits,
                    "speech_temporal_structure_logits": temporal_logits,
                }
            )
            if "text_focus" in output:
                teacher_logits = torch.einsum("bkd,bd->bk", structures, output["text_focus"])
                if structure_valid_mask is not None:
                    teacher_logits = teacher_logits.masked_fill(
                        ~structure_valid_mask.bool(), -10000.0
                    )
                output["teacher_structure_logits"] = teacher_logits
                output["teacher_temporal_structure_logits"] = teacher_logits
        return output

    def speech_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        state = {
            "speech_pool": self.speech_pool.state_dict(),
            "speech_projector": self.speech_projector.state_dict(),
        }
        if not self.freeze_speech_backbone:
            state["speech_backbone"] = self.speech_backbone.state_dict()
        if not self.freeze_modal_heads:
            state["image_head"] = self.image_head.state_dict()
            state["text_head"] = self.text_head.state_dict()
        state["temporal_projector"] = self.temporal_projector.state_dict()
        state["visual_adapter"] = self.visual_adapter.state_dict()
        return state

    def load_speech_state_dict(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        self.speech_pool.load_state_dict(state["speech_pool"])
        self.speech_projector.load_state_dict(state["speech_projector"])
        if "speech_backbone" in state:
            self.speech_backbone.load_state_dict(state["speech_backbone"])
        if "image_head" in state:
            self.image_head.load_state_dict(state["image_head"])
        if "text_head" in state:
            self.text_head.load_state_dict(state["text_head"])
        self.temporal_projector.load_state_dict(state["temporal_projector"])
        self.visual_adapter.load_state_dict(state["visual_adapter"])
