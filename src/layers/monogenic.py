
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class RieszTransform(nn.Module):

    def __init__(self, epsilon: float = 1e-8):

        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = x.shape

        freq_y = torch.fft.fftfreq(H, device=x.device, dtype=x.dtype)
        freq_x = torch.fft.fftfreq(W, device=x.device, dtype=x.dtype)
        freq_y, freq_x = torch.meshgrid(freq_y, freq_x, indexing='ij')

        radius = torch.sqrt(freq_x**2 + freq_y**2 + self.epsilon)

        kernel_x = freq_x / radius
        kernel_y = freq_y / radius

        kernel_x[0, 0] = 0
        kernel_y[0, 0] = 0

        x_fft = torch.fft.fft2(x)

        riesz_x_fft = -1j * x_fft * kernel_x.unsqueeze(0).unsqueeze(0)
        riesz_y_fft = -1j * x_fft * kernel_y.unsqueeze(0).unsqueeze(0)

        riesz_x = torch.fft.ifft2(riesz_x_fft).real
        riesz_y = torch.fft.ifft2(riesz_y_fft).real

        return riesz_x, riesz_y

class LogGaborFilter(nn.Module):

    def __init__(self, num_scales: int = 4, num_orientations: int = 6,
                 min_wavelength: float = 3.0, mult: float = 2.1,
                 sigma_on_f: float = 0.55):

        super().__init__()
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.min_wavelength = min_wavelength
        self.mult = mult
        self.sigma_on_f = sigma_on_f

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        freq_y = torch.fft.fftfreq(H, device=device, dtype=dtype)
        freq_x = torch.fft.fftfreq(W, device=device, dtype=dtype)
        freq_y, freq_x = torch.meshgrid(freq_y, freq_x, indexing='ij')

        radius = torch.sqrt(freq_x**2 + freq_y**2)
        radius[0, 0] = 1
        theta = torch.atan2(freq_y, freq_x)

        x_fft = torch.fft.fft2(x)

        outputs = []

        for scale in range(self.num_scales):
            wavelength = self.min_wavelength * (self.mult ** scale)
            fo = 1.0 / wavelength

            log_gabor_radial = torch.exp(
                -(torch.log(radius / fo) ** 2) / (2 * math.log(self.sigma_on_f) ** 2)
            )
            log_gabor_radial[0, 0] = 0

            for orient in range(self.num_orientations):
                angle = orient * math.pi / self.num_orientations

                ds = torch.sin(theta - angle)
                dc = torch.cos(theta - angle)
                dtheta = torch.abs(torch.atan2(ds, dc))

                angular_spread = torch.exp(
                    -(dtheta ** 2) / (2 * (math.pi / self.num_orientations) ** 2)
                )

                log_gabor = log_gabor_radial * angular_spread

                filtered = torch.fft.ifft2(x_fft * log_gabor.unsqueeze(0).unsqueeze(0))
                outputs.append(filtered.abs())

        return torch.cat(outputs, dim=1)

class MonogenicSignal(nn.Module):

    def __init__(self, epsilon: float = 1e-8):

        super().__init__()
        self.riesz = RieszTransform(epsilon=epsilon)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> dict:

        riesz_x, riesz_y = self.riesz(x)

        amplitude = torch.sqrt(x**2 + riesz_x**2 + riesz_y**2 + self.epsilon)

        orientation = torch.atan2(riesz_y, riesz_x + self.epsilon)

        riesz_magnitude = torch.sqrt(riesz_x**2 + riesz_y**2 + self.epsilon)
        phase = torch.atan2(riesz_magnitude, x + self.epsilon)

        return {
            'amplitude': amplitude,
            'phase': phase,
            'orientation': orientation,
            'riesz_x': riesz_x,
            'riesz_y': riesz_y
        }

class EnergyMap(nn.Module):

    def __init__(self, normalize: bool = True, smoothing_sigma: float = 1.0):

        super().__init__()
        self.monogenic = MonogenicSignal()
        self.normalize = normalize
        self.smoothing_sigma = smoothing_sigma

        if smoothing_sigma > 0:
            kernel_size = int(6 * smoothing_sigma) | 1
            self.register_buffer('smooth_kernel', self._create_gaussian_kernel(
                kernel_size, smoothing_sigma
            ))
        else:
            self.smooth_kernel = None

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:

        x = torch.arange(kernel_size) - kernel_size // 2
        x = x.float()
        gaussian_1d = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        return gaussian_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:

        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        mono_out = self.monogenic(x)

        energy = mono_out['amplitude']

        if self.smooth_kernel is not None:
            pad = self.smooth_kernel.shape[-1] // 2
            energy = F.conv2d(energy, self.smooth_kernel, padding=pad)

        if self.normalize:
            B = energy.shape[0]
            energy_flat = energy.view(B, -1)
            energy_min = energy_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            energy_max = energy_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            energy = (energy - energy_min) / (energy_max - energy_min + 1e-8)

        return energy, mono_out

class BoundaryDetector(nn.Module):

    def __init__(self, num_scales: int = 4, num_orientations: int = 6,
                 noise_threshold: float = 0.1):

        super().__init__()
        self.log_gabor = LogGaborFilter(num_scales, num_orientations)
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.noise_threshold = noise_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        responses = self.log_gabor(x)

        B, _, H, W = responses.shape
        responses = responses.view(B, self.num_scales, self.num_orientations, H, W)

        edge_strength = responses.max(dim=2)[0]

        edge_strength = edge_strength.sum(dim=1, keepdim=True)

        edge_max = edge_strength.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        edge_strength = edge_strength / (edge_max + 1e-8)
        edge_strength = torch.clamp(edge_strength - self.noise_threshold, min=0)
        edge_strength = edge_strength / (1 - self.noise_threshold + 1e-8)

        return edge_strength

if __name__ == "__main__":

    print("Testing Monogenic Signal Processing...")

    H, W = 128, 128
    x = torch.zeros(1, 1, H, W)
    x[:, :, 32:96, 32:96] = 1.0

    x = x + 0.1 * torch.randn_like(x)

    energy_extractor = EnergyMap(normalize=True)
    energy, mono = energy_extractor(x)

    print(f"Input shape: {x.shape}")
    print(f"Energy map shape: {energy.shape}")
    print(f"Energy range: [{energy.min():.3f}, {energy.max():.3f}]")
    print(f"Monogenic components: {list(mono.keys())}")

    boundary_detector = BoundaryDetector()
    boundaries = boundary_detector(x)

    print(f"Boundary map shape: {boundaries.shape}")
    print(f"Boundary range: [{boundaries.min():.3f}, {boundaries.max():.3f}]")

    print("\n✓ All tests passed!")
