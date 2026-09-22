from pathlib import Path

import torch
from torch import nn
from torchvision import transforms


class ImageEncoder(nn.Module):
    def __init__(self, repository, checkpoint):
        super().__init__()
        self.backbone = torch.hub.load(
            str(repository), "dinov3_vith16plus", source="local", pretrained=False
        )
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        # Match the released checkpoint's preprocessing: RGB [0, 1], no normalization.
        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

    @torch.inference_mode()
    def forward(self, image):
        device = next(self.backbone.parameters()).device
        image = self.transform(image.convert("RGB")).unsqueeze(0).to(device)
        tokens = self.backbone.get_intermediate_layers(image, n=1)[0]
        if tokens.shape[1] == 1025:
            tokens = tokens[:, 1:]
        if tokens.shape[1:] != (1024, 1280):
            raise ValueError(f"Unexpected DINOv3 patch shape: {tuple(tokens.shape)}")
        return tokens.float()
