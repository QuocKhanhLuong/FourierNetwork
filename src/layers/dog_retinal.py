import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class DoGFilter(nn.Module):
    def __init__(self, in_channels=1, sigma_center=1.0, sigma_surround=2.0, kernel_size=15):
        super().__init__()
        self.in_channels = in_channels
        self.sigma_center = sigma_center
        self.sigma_surround = sigma_surround
        self.kernel_size = kernel_size

        kernel = self._create_dog_kernel(sigma_center, sigma_surround, kernel_size)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(in_channels, 1, 1, 1)
        self.register_buffer('kernel', kernel)

    def _create_dog_kernel(self, sigma_c, sigma_s, size):
        x = torch.arange(size, dtype=torch.float32) - size // 2
        y = torch.arange(size, dtype=torch.float32) - size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        r2 = xx**2 + yy**2

        g_center = torch.exp(-r2 / (2 * sigma_c**2))
        g_center = g_center / g_center.sum()

        g_surround = torch.exp(-r2 / (2 * sigma_s**2))
        g_surround = g_surround / g_surround.sum()

        dog = g_center - g_surround
        return dog

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.kernel_size // 2, groups=self.in_channels)


class MultiScaleDoG(nn.Module):
    def __init__(self, in_channels=1, scales=[(1.0, 2.0), (2.0, 4.0), (4.0, 8.0)], kernel_size=15):
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = len(scales)
        
        self.dog_filters = nn.ModuleList([
            DoGFilter(in_channels, sigma_c, sigma_s, kernel_size)
            for sigma_c, sigma_s in scales
        ])

    def forward(self, x):
        responses = [dog(x) for dog in self.dog_filters]
        return torch.cat(responses, dim=1)

    @property
    def out_channels(self):
        return self.in_channels * self.num_scales


class RetinalLayer(nn.Module):
    def __init__(self, in_channels=1, scales=[(1.0, 2.0), (2.0, 4.0)], 
                 kernel_size=15, concat_original=True):
        super().__init__()
        self.concat_original = concat_original
        self.in_channels = in_channels
        
        self.dog = MultiScaleDoG(in_channels, scales, kernel_size)
        
        dog_ch = self.dog.out_channels
        total_ch = in_channels + dog_ch if concat_original else dog_ch
        
        self.proj = nn.Sequential(
            nn.Conv2d(total_ch, total_ch, 1, bias=False),
            nn.BatchNorm2d(total_ch),
            nn.ReLU(inplace=True)
        )
        
        self._out_channels = total_ch

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x):
        dog_response = self.dog(x)
        
        if self.concat_original:
            out = torch.cat([x, dog_response], dim=1)
        else:
            out = dog_response
            
        return self.proj(out)


class OnOffPathway(nn.Module):
    def __init__(self, in_channels=1, sigma_center=1.0, sigma_surround=2.5, kernel_size=15):
        super().__init__()
        self.dog = DoGFilter(in_channels, sigma_center, sigma_surround, kernel_size)

    def forward(self, x):
        response = self.dog(x)
        on_path = F.relu(response)
        off_path = F.relu(-response)
        return torch.cat([on_path, off_path], dim=1)


if __name__ == "__main__":
    x = torch.randn(2, 1, 256, 256)
    
    layer = RetinalLayer(in_channels=1, scales=[(1.0, 2.0), (2.0, 4.0)], concat_original=True)
    out = layer(x)
    print(f"RetinalLayer: {x.shape} -> {out.shape}")
    print(f"Out channels: {layer.out_channels}")
