
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import math

try:
    from .mamba_block import VSSBlock, MambaBlockStack
    from ..layers.spectral_layers import SpectralGating
except (ImportError, ValueError):
    try:
        from models.mamba_block import VSSBlock, MambaBlockStack
        from layers.spectral_layers import SpectralGating
    except ImportError:
        try:
            import sys
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.dirname(current_dir)
            if src_dir not in sys.path:
                sys.path.append(src_dir)

            from models.mamba_block import VSSBlock, MambaBlockStack
            from layers.spectral_layers import SpectralGating
        except ImportError as e:
            raise ImportError(f"Could not import required modules (mamba_block, spectral_layers): {e}")

class SpectralVSSBlock(nn.Module):

    def __init__(self, channels: int, height: int, width: int,
                 mamba_depth: int = 2, expansion_ratio: float = 2.0,
                 spectral_threshold: float = 0.1,
                 use_mamba: bool = True, use_spectral: bool = True):
        super().__init__()

        self.channels = channels
        self.height = height
        self.width = width
        self.use_mamba = use_mamba
        self.use_spectral = use_spectral

        if use_mamba and mamba_depth > 0:
            self.vss_blocks = MambaBlockStack(
                channels,
                depth=mamba_depth,
                expansion_ratio=expansion_ratio,
                scan_dim=min(64, channels)
            )
        else:

            self.vss_blocks = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(min(32, channels), channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(min(32, channels), channels),
                nn.GELU()
            )

        if use_spectral:
            self.spectral_gate = SpectralGating(
                channels, height, width,
                threshold=spectral_threshold,
                complex_init="kaiming"
            )
        else:
            self.spectral_gate = None

        if use_mamba and use_spectral:
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.fusion_weight = None

        self.norm = nn.GroupNorm(min(32, channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        spatial_out = self.vss_blocks(x)

        if self.use_spectral and self.spectral_gate is not None:
            spectral_out = self.spectral_gate(x)

            if self.fusion_weight is not None:
                weight = torch.sigmoid(self.fusion_weight)
                output = weight * spatial_out + (1 - weight) * spectral_out
            else:
                output = (spatial_out + spectral_out) / 2.0
        else:

            output = spatial_out

        output = self.norm(output)

        return output

class MultiScaleFusion(nn.Module):

    def __init__(self, high_channels: int, low_channels: int, scale_factor: int = 2):
        super().__init__()

        self.scale_factor = scale_factor

        self.high_to_low = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, kernel_size=3,
                     stride=scale_factor, padding=1, bias=False),
            nn.GroupNorm(min(32, low_channels), low_channels),
            nn.GELU()
        )

        self.low_to_high = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(low_channels, high_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, high_channels), high_channels),
            nn.GELU()
        )

        self.high_gate = nn.Parameter(torch.ones(1) * 0.5)
        self.low_gate = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, high_feat: torch.Tensor,
                low_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        high_to_low = self.high_to_low(high_feat)

        low_to_high = self.low_to_high(low_feat)

        h_gate = torch.sigmoid(self.high_gate)
        l_gate = torch.sigmoid(self.low_gate)

        new_high = high_feat + h_gate * low_to_high
        new_low = low_feat + l_gate * high_to_low

        return new_high, new_low

class HRNetStage(nn.Module):

    def __init__(self, channels: int, height: int, width: int,
                 num_blocks: int = 2, mamba_depth: int = 2,
                 use_mamba: bool = True, use_spectral: bool = True):
        super().__init__()

        self.blocks = nn.ModuleList([
            SpectralVSSBlock(
                channels, height, width,
                mamba_depth=mamba_depth,
                use_mamba=use_mamba,
                use_spectral=use_spectral
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for block in self.blocks:
            x = block(x)
        return x

class HRNetStem(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 64, stride: int = 4):
        super().__init__()

        mid_channels = out_channels // 2

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                              stride=2, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(32, mid_channels), mid_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3,
                              stride=2, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x

class AggregationLayer(nn.Module):

    def __init__(self, high_channels: int, low_channels: int,
                 out_channels: int, scale_factor: int = 2):
        super().__init__()

        self.scale_factor = scale_factor

        self.upsample_low = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(low_channels, low_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, low_channels), low_channels)
        )

        total_channels = high_channels + low_channels
        self.projection = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.GELU()
        )

    def forward(self, high_feat: torch.Tensor,
                low_feat: torch.Tensor) -> torch.Tensor:

        low_up = self.upsample_low(low_feat)

        concat = torch.cat([high_feat, low_up], dim=1)

        output = self.projection(concat)

        return output

class HRNetV2MambaBackbone(nn.Module):

    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_stages: int = 4, blocks_per_stage: int = 2,
                 mamba_depth: int = 2, img_size: int = 256,
                 use_mamba: bool = True, use_spectral: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.img_size = img_size
        self.use_mamba = use_mamba
        self.use_spectral = use_spectral

        self.stem = HRNetStem(in_channels, base_channels, stride=4)

        high_res_size = img_size // 4
        low_res_size = img_size // 8

        high_channels = base_channels
        low_channels = base_channels * 2

        self.create_low_stream = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, low_channels), low_channels),
            nn.GELU()
        )

        self.high_res_stages = nn.ModuleList()

        self.low_res_stages = nn.ModuleList()

        self.fusion_modules = nn.ModuleList()

        for stage_idx in range(num_stages):

            self.high_res_stages.append(
                HRNetStage(
                    channels=high_channels,
                    height=high_res_size,
                    width=high_res_size,
                    num_blocks=blocks_per_stage,
                    mamba_depth=mamba_depth,
                    use_mamba=use_mamba,
                    use_spectral=use_spectral
                )
            )

            self.low_res_stages.append(
                HRNetStage(
                    channels=low_channels,
                    height=low_res_size,
                    width=low_res_size,
                    num_blocks=blocks_per_stage,
                    mamba_depth=mamba_depth,
                    use_mamba=use_mamba,
                    use_spectral=use_spectral
                )
            )

            self.fusion_modules.append(
                MultiScaleFusion(
                    high_channels=high_channels,
                    low_channels=low_channels,
                    scale_factor=2

                )
            )

        self.aggregation = AggregationLayer(
            high_channels=high_channels,
            low_channels=low_channels,
            out_channels=high_channels + low_channels,
            scale_factor=2
        )

        self.out_channels = high_channels + low_channels
        self.feature_size = high_res_size

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        high = self.stem(x)

        low = self.create_low_stream(high)

        for stage_idx in range(self.num_stages):

            high = self.high_res_stages[stage_idx](high)
            low = self.low_res_stages[stage_idx](low)

            high, low = self.fusion_modules[stage_idx](high, low)

        features = self.aggregation(high, low)

        return {
            'features': features,
            'high_res': high,
            'low_res': low
        }

if __name__ == "__main__":
    print("=" * 60)
    print("Testing HRNetV2-Mamba Backbone")
    print("=" * 60)

    print("\n[1] Testing SpectralVSSBlock...")
    block = SpectralVSSBlock(channels=64, height=64, width=64, mamba_depth=2)
    x = torch.randn(2, 64, 64, 64)
    out = block(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    print("\n[2] Testing MultiScaleFusion...")
    fusion = MultiScaleFusion(high_channels=64, low_channels=128, scale_factor=2)
    high = torch.randn(2, 64, 64, 64)
    low = torch.randn(2, 128, 32, 32)
    new_high, new_low = fusion(high, low)
    print(f"High: {high.shape} → {new_high.shape}")
    print(f"Low: {low.shape} → {new_low.shape}")

    print("\n[3] Testing HRNetV2MambaBackbone...")
    backbone = HRNetV2MambaBackbone(
        in_channels=3,
        base_channels=64,
        num_stages=4,
        blocks_per_stage=2,
        mamba_depth=2,
        img_size=256
    )

    x = torch.randn(2, 3, 256, 256)
    outputs = backbone(x)

    print(f"Input: {x.shape}")
    print(f"Features: {outputs['features'].shape}")
    print(f"High-res: {outputs['high_res'].shape}")
    print(f"Low-res: {outputs['low_res'].shape}")
    print(f"Output channels: {backbone.out_channels}")

    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
