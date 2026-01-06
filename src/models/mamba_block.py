
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class DepthwiseSeparableConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DirectionalScanner(nn.Module):

    def __init__(self, channels: int, scan_dim: int = 64):

        super().__init__()
        self.channels = channels
        self.scan_dim = scan_dim

        self.proj_in = nn.Linear(channels, scan_dim)

        self.gru_cell = nn.GRUCell(scan_dim, scan_dim)

        self.proj_out = nn.Linear(scan_dim, channels)

    def _scan_direction(self, x: torch.Tensor, direction: str) -> torch.Tensor:

        B, C, H, W = x.shape

        if direction == "right":

            x = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        elif direction == "down":

            x = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        elif direction == "left":

            x = x.permute(0, 2, 3, 1).flip(1).reshape(B * H, W, C)
        elif direction == "up":

            x = x.permute(0, 3, 2, 1).flip(1).reshape(B * W, H, C)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        x = self.proj_in(x)

        outputs = []
        h = torch.zeros(x.shape[0], self.scan_dim, device=x.device, dtype=x.dtype)

        for t in range(x.shape[1]):
            h = self.gru_cell(x[:, t], h)
            outputs.append(h)

        x = torch.stack(outputs, dim=1)

        x = self.proj_out(x)

        if direction == "right":
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        elif direction == "down":
            x = x.reshape(B, W, H, C).permute(0, 3, 2, 1)
        elif direction == "left":
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).flip(-1)
        elif direction == "up":
            x = x.reshape(B, W, H, C).permute(0, 3, 2, 1).flip(-2)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        scan_right = self._scan_direction(x, "right")
        scan_down = self._scan_direction(x, "down")
        scan_left = self._scan_direction(x, "left")
        scan_up = self._scan_direction(x, "up")

        output = (scan_right + scan_down + scan_left + scan_up) / 4.0

        return output

class VSSBlock(nn.Module):

    def __init__(self, channels: int, hidden_dim: Optional[int] = None,
                 scan_dim: int = 64, expansion_ratio: float = 2.0):

        super().__init__()
        self.channels = channels
        hidden_dim = hidden_dim or int(channels * expansion_ratio)

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.conv_expand = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=True)

        self.scanner = DirectionalScanner(hidden_dim, scan_dim=scan_dim)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=hidden_dim, eps=1e-6)
        self.conv_contract = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x

        x = self.norm1(x)
        x = self.conv_expand(x)
        x = F.gelu(x)

        x = self.scanner(x)

        x = self.norm2(x)
        x = self.conv_contract(x)

        output = x + residual

        return output

class MambaBlockStack(nn.Module):

    def __init__(self, channels: int, depth: int = 2, **kwargs):

        super().__init__()
        self.blocks = nn.ModuleList([
            VSSBlock(channels, **kwargs) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

if __name__ == "__main__":

    batch_size, channels, height, width = 2, 64, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    vss_block = VSSBlock(channels, scan_dim=32)
    output = vss_block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Module parameters: {sum(p.numel() for p in vss_block.parameters())}")
