"""
src/models/bilstm_attention.py
==============================
Kiến trúc Mạng Nơ-ron Chuỗi Thời gian Đa Thang đo (Multi-Scale Temporal CNN + Bi-LSTM + Self-Attention).

Bản chất kiến trúc Deep Learning:
1. Tầng Multi-Scale Temporal CNN (Conv1D đa thang đo):
   Thay vì đưa trực tiếp 30 frame đặc trưng vào Bi-LSTM, mạng sử dụng 3 nhánh Conv1D song song
   với các kích thước Kernel khác nhau để trích xuất đặc trưng biến thiên theo thời gian:
   - Branch 1 (Kernel 3):  Bắt các vi chuyển động cục bộ tức thời (Local patterns: Click, Tap, Chụm ngón).
   - Branch 2 (Kernel 7):  Bắt các chuyển động tốc độ trung bình (Mid patterns: Vuốt, Hất tay).
   - Branch 3 (Kernel 15): Bắt các quỹ đạo không gian diện rộng (Global patterns: Vẽ vòng tròn, Thu phóng hai tay).
   Đầu ra của 3 nhánh được ghép nối (Concatenate) thành vector 96 chiều tại mỗi bước thời gian.

2. Tầng Bi-LSTM (Bidirectional Long Short-Term Memory):
   - Xử lý chuỗi thời gian theo 2 chiều (Quá khứ -> Tương lai và Tương lai -> Quá khứ).
   - hidden_size = 128, 2 layers, bidirectional=True -> Trả về vector 256 chiều tại mỗi frame.

3. Tầng Temporal Self-Attention:
   - Tự động học trọng số chú ý alpha_t trong [0, 1] cho từng frame trong chuỗi 30 frame.
   - alpha_t = softmax(v^T * tanh(W * h_t + b))
   - Context vector c = sum(alpha_t * h_t) tập trung vào "khoảnh khắc vàng" của cử chỉ.
   - Trọng số alpha_t có thể trích xuất trực tiếp để vẽ biểu đồ nhiệt Attention Heatmap lên màn hình.

4. Tầng Phân loại (Dense Head):
   - Linear(256, 64) -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear(64, num_classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class MultiScaleTemporalCNN(nn.Module):
    """3 nhánh Conv1D song song bắt các mẫu thời gian ngắn, vừa và dài."""

    def __init__(self, in_channels: int = 158, out_channels_per_branch: int = 32):
        super().__init__()
        self.conv_local = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels_per_branch,
            kernel_size=3,
            padding=1  # padding 'same' cho k=3
        )
        self.bn_local = nn.BatchNorm1d(out_channels_per_branch)

        self.conv_mid = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels_per_branch,
            kernel_size=7,
            padding=3  # padding 'same' cho k=7
        )
        self.bn_mid = nn.BatchNorm1d(out_channels_per_branch)

        self.conv_global = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels_per_branch,
            kernel_size=15,
            padding=7  # padding 'same' cho k=15
        )
        self.bn_global = nn.BatchNorm1d(out_channels_per_branch)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Time, Dim=158)
        Trả về: (Batch, Time, OutDim=96)
        """
        # Đổi chiều sang (Batch, Channels=158, Time) cho Conv1D
        x_trans = x.transpose(1, 2)

        feat_local  = self.act(self.bn_local(self.conv_local(x_trans)))
        feat_mid    = self.act(self.bn_mid(self.conv_mid(x_trans)))
        feat_global = self.act(self.bn_global(self.conv_global(x_trans)))

        # Ghép nối 3 nhánh: (Batch, 32 + 32 + 32 = 96, Time=30)
        concat_feat = torch.cat([feat_local, feat_mid, feat_global], dim=1)

        # Đổi lại chiều về (Batch, Time=30, Channels=96) cho LSTM
        return concat_feat.transpose(1, 2)


class TemporalSelfAttention(nn.Module):
    """Cơ chế tự chú ý gán trọng số cho từng frame trong chuỗi thời gian."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 64)
        self.v = nn.Linear(64, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: (Batch, Time=30, HiddenDim=256)
        Trả về:
          context: (Batch, HiddenDim=256)
          weights: (Batch, Time=30)
        """
        # u_t = tanh(W * h_t + b) -> (Batch, Time, 64)
        u = torch.tanh(self.proj(h))
        # score = v^T * u_t -> (Batch, Time, 1)
        scores = self.v(u)
        # alpha_t = softmax(score) -> (Batch, Time, 1)
        weights = F.softmax(scores, dim=1)

        # Context vector c = sum(alpha_t * h_t) -> (Batch, HiddenDim)
        context = torch.sum(weights * h, dim=1)

        return context, weights.squeeze(-1)


class MultiScaleBiLSTMAttention(nn.Module):
    """
    Kiến trúc mạng hoàn chỉnh tích hợp:
    Multi-Scale Conv1D -> Bi-LSTM -> Temporal Self-Attention -> Classification Head.
    """

    def __init__(
        self,
        input_dim: int = 158,
        cnn_branch_channels: int = 32,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        num_classes: int = 10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # 1. Multi-Scale Temporal CNN
        self.temporal_cnn = MultiScaleTemporalCNN(
            in_channels=input_dim,
            out_channels_per_branch=cnn_branch_channels
        )
        cnn_out_dim = cnn_branch_channels * 3  # 96

        # 2. Bi-LSTM
        self.bilstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0
        )
        bilstm_out_dim = lstm_hidden_size * 2  # 256 (2 chiều)

        # 3. Temporal Self-Attention
        self.attention = TemporalSelfAttention(hidden_dim=bilstm_out_dim)

        # 4. Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(bilstm_out_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: Tensor đầu vào shape (Batch, Time, Dim=158)
        Trả về:
          logits: (Batch, num_classes)
          attention_weights: (Batch, Time)
        """
        # 1. Multi-scale CNN: (B, T, 158) -> (B, T, 96)
        x_cnn = self.temporal_cnn(x)

        # 2. Bi-LSTM: (B, T, 96) -> (B, T, 256)
        lstm_out, _ = self.bilstm(x_cnn)

        # 3. Temporal Self-Attention: (B, T, 256) -> Context (B, 256), Weights (B, T)
        context, attn_weights = self.attention(lstm_out)

        # 4. Classification Head: (B, 256) -> (B, num_classes)
        logits = self.classifier(context)

        return logits, attn_weights
