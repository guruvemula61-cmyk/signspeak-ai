"""
model.py
Lightweight CNN-LSTM gesture classifier, sized to run efficiently on the
Snapdragon Hexagon NPU after INT8/FP16 quantization via Qualcomm AI Hub.

Input:  (batch, sequence_length, feature_dim) landmark sequences
Output: (batch, num_classes) gesture logits
"""

import torch
import torch.nn as nn

from dataset import SEQUENCE_LENGTH, FEATURE_DIM, DATASET_LABELS


class SignGestureClassifier(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, num_classes: int = len(DATASET_LABELS),
                 hidden_size: int = 64, cnn_channels: int = 32):
        super().__init__()

        # 1D conv over the temporal axis to learn short-range motion patterns
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
        )

        # LSTM to capture the longer-range gesture dynamics
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature_dim) -> conv expects (batch, feature_dim, seq_len)
        x = x.permute(0, 2, 1)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1)  # back to (batch, seq_len, cnn_channels)

        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)

        return self.classifier(last_hidden)


if __name__ == "__main__":
    model = SignGestureClassifier()
    dummy_input = torch.randn(4, SEQUENCE_LENGTH, FEATURE_DIM)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # (4, num_classes)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
