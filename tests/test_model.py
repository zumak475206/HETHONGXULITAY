"""
tests/test_model.py
===================
Kiểm thử tự động Mạng Nơ-ron MultiScaleBiLSTMAttention trong PyTorch.
"""

import unittest
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from src.models.bilstm_attention import MultiScaleBiLSTMAttention, MultiScaleTemporalCNN, TemporalSelfAttention


class TestModelArchitecture(unittest.TestCase):

    def test_multiscale_cnn_dimensions(self):
        """Kiểm tra tầng Conv1D đa thang đo k=3, 7, 15 ghép nối chính xác 96 kênh."""
        cnn = MultiScaleTemporalCNN(in_channels=config.FEATURE_DIM, out_channels_per_branch=32)
        x = torch.randn(4, config.SEQUENCE_LENGTH, config.FEATURE_DIM)
        out = cnn(x)
        self.assertEqual(out.shape, (4, config.SEQUENCE_LENGTH, 96))

    def test_attention_weights_sum_to_one(self):
        """Kiểm tra tổng trọng số chú ý qua SEQUENCE_LENGTH frame bằng đúng 1.0 (Softmax)."""
        attn = TemporalSelfAttention(hidden_dim=256)
        h = torch.randn(2, config.SEQUENCE_LENGTH, 256)
        context, weights = attn(h)

        self.assertEqual(context.shape, (2, 256))
        self.assertEqual(weights.shape, (2, config.SEQUENCE_LENGTH))

        sums = torch.sum(weights, dim=1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_full_model_forward_and_backward(self):
        """Kiểm tra forward pass và lan truyền ngược đạo hàm toàn mạng với 158D và 45 frames."""
        model = MultiScaleBiLSTMAttention(input_dim=config.FEATURE_DIM, num_classes=10)
        x = torch.randn(2, config.SEQUENCE_LENGTH, config.FEATURE_DIM, requires_grad=True)

        logits, attn = model(x)
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(attn.shape, (2, config.SEQUENCE_LENGTH))

        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_flexible_temporal_lengths(self):
        """Kiểm tra mô hình hỗ trợ linh hoạt các độ dài chuỗi thời gian khác nhau (30, 45, 60 frames)."""
        model = MultiScaleBiLSTMAttention(input_dim=config.FEATURE_DIM, num_classes=10)
        model.eval()
        with torch.no_grad():
            for t_len in [30, 45, 60]:
                x_test = torch.randn(1, t_len, config.FEATURE_DIM)
                l, a = model(x_test)
                self.assertEqual(l.shape, (1, 10))
                self.assertEqual(a.shape, (1, t_len))


if __name__ == "__main__":
    unittest.main()
