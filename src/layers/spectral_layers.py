
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SpectralGating(nn.Module):

    def __init__(self, channels: int, height: int, width: int,
                 threshold: float = 0.1, complex_init: str = "kaiming"):

        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.threshold = threshold

        self.register_buffer(
            "freq_shape",
            torch.tensor([channels, height, width // 2 + 1], dtype=torch.long)
        )

        self.weight_real = nn.Parameter(
            torch.zeros(channels, height, width // 2 + 1)
        )
        self.weight_imag = nn.Parameter(
            torch.zeros(channels, height, width // 2 + 1)
        )

        self._init_weights(complex_init)

    def _init_weights(self, strategy: str = "kaiming"):

        if strategy == "identity":

            nn.init.ones_(self.weight_real)
            nn.init.zeros_(self.weight_imag)
        elif strategy == "kaiming":

            fan_in = self.height * (self.width // 2 + 1)
            std = (2.0 / fan_in) ** 0.5
            nn.init.normal_(self.weight_real, 0, std)
            nn.init.normal_(self.weight_imag, 0, std)
        else:
            raise ValueError(f"Unknown init strategy: {strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        x_freq = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

        complex_weight = (
            self.weight_real.unsqueeze(0) +
            1j * self.weight_imag.unsqueeze(0)
        )

        x_filtered = x_freq * complex_weight

        if self.threshold > 0:
            magnitude = torch.abs(x_filtered)
            mask = magnitude > self.threshold
            x_filtered = x_filtered * mask.float()

        output = torch.fft.irfft2(x_filtered, s=(H, W), dim=(-2, -1), norm="ortho")

        return output

class FrequencyLoss(nn.Module):

    def __init__(self, weight: float = 0.1):

        super().__init__()
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred_freq = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
        target_freq = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")

        loss_real = F.mse_loss(pred_freq.real, target_freq.real)
        loss_imag = F.mse_loss(pred_freq.imag, target_freq.imag)

        return loss_real + loss_imag

if __name__ == "__main__":

    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    spec_gate = SpectralGating(channels, height, width)
    output = spec_gate(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Module parameters: {sum(p.numel() for p in spec_gate.parameters())}")
