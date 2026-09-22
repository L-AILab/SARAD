from types import SimpleNamespace
import unittest

import torch
from torch import nn

from models.sarad import ProjectionHead, SARAD
from utils.metrics import denseav_multi_iou_flat


class TinySpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.projection = nn.Linear(1, 16)

    def forward(self, input_values, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.projection(input_values[..., None]))


class ModelTest(unittest.TestCase):
    def test_checkpoint_roundtrip(self):
        def build():
            return SARAD(
                TinySpeechEncoder(),
                ProjectionHead(12, 8, 16),
                ProjectionHead(10, 8, 16),
                freeze_speech_backbone=False,
                freeze_modal_heads=False,
                adapter_hidden_dim=16,
                visual_adapter_hidden_dim=12,
            ).eval()

        first, second = build(), build()
        second.load_speech_state_dict(first.speech_state_dict())
        waveform = torch.randn(2, 20)
        patches = torch.randn(2, 16, 12)
        left, right = first(waveform, patches), second(waveform, patches)
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        self.assertEqual(left["speech_map"].shape, (2, 16))

    def test_iou_thresholds(self):
        scores = torch.rand(200)
        target = torch.rand(200) > 0.5
        thresholds = torch.linspace(scores.min(), scores.max(), 20)
        expected = max(
            ((scores > t) & target).sum() / ((scores > t) | target).sum().clamp_min(1)
            for t in thresholds
        )
        torch.testing.assert_close(denseav_multi_iou_flat(scores, target), expected)


if __name__ == "__main__":
    unittest.main()
