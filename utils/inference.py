import torch
from transformers import AutoConfig, AutoModel

from models.sarad import SARAD, _head_from_state
from datasets.speech_lvis import _load_audio


def load_model(checkpoint, hubert, device):
    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = saved["speech_ifp_v3"]
    config = AutoConfig.from_pretrained(str(hubert), local_files_only=True)
    if config.hidden_size != 1024:
        raise ValueError("This release uses HuBERT Large (hidden size 1024).")
    backbone = AutoModel.from_config(config)
    model_config = saved.get("config", {}).get("model", {})
    model = SARAD(
        backbone,
        _head_from_state(state["image_head"]),
        _head_from_state(state["text_head"]),
        freeze_speech_backbone=False,
        freeze_modal_heads=False,
        adapter_hidden_dim=state["speech_projector"]["1.weight"].shape[0],
        visual_adapter_hidden_dim=state["visual_adapter"]["1.weight"].shape[0],
        visual_adapter_scale=model_config.get("visual_adapter_scale", 0.1),
        visual_pool_temperature=model_config.get("visual_pool_temperature", 0.1),
        structure_topk=model_config.get("structure_topk", 5),
    )
    expected = set(model.speech_state_dict())
    if set(state) != expected:
        raise ValueError(f"Checkpoint modules differ: {set(state) ^ expected}")
    model.load_speech_state_dict(state)
    return model.to(device).eval()


@torch.inference_mode()
def predict(model, encoder, image, audio, start, end):
    device = next(model.parameters()).device
    waveform = _load_audio(audio, 16000).to(device)
    if not 0 <= start < end <= len(waveform) / 16000 + 1 / 16000:
        raise ValueError("Target-word interval is outside the audio duration.")
    features = encoder(image)
    output = model(
        waveform[None],
        features,
        attention_mask=torch.ones_like(waveform[None], dtype=torch.long),
        target_start_samples=torch.tensor([int(start * 16000)], device=device),
        target_end_samples=torch.tensor([int(end * 16000)], device=device),
    )
    return output["speech_map"].reshape(32, 32).cpu()


def add_model_arguments(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hubert", default="checkpoints/hubert-large-ls960-ft")
    parser.add_argument("--dino-repo", default="external/dinov3")
    parser.add_argument("--dino-weights", required=True)
    parser.add_argument("--device", default="cuda:0")
