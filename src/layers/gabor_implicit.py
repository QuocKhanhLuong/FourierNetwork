
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

class GaborBasis(nn.Module):

    def __init__(self, input_dim: int = 2, num_frequencies: int = 64,
                 sigma_range: Tuple[float, float] = (0.1, 2.0),
                 freq_range: Tuple[float, float] = (1.0, 10.0),
                 learnable: bool = True):

        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = num_frequencies * 2

        log_freqs = torch.linspace(
            math.log(freq_range[0]),
            math.log(freq_range[1]),
            num_frequencies
        )
        freqs = torch.exp(log_freqs)

        sigmas = torch.linspace(sigma_range[0], sigma_range[1], num_frequencies)

        orientations = torch.rand(num_frequencies) * 2 * math.pi

        phases = torch.rand(num_frequencies) * 2 * math.pi

        directions = torch.stack([
            torch.cos(orientations),
            torch.sin(orientations)
        ], dim=-1)

        if learnable:
            self.freqs = nn.Parameter(freqs)
            self.sigmas = nn.Parameter(sigmas)
            self.directions = nn.Parameter(directions)
            self.phases = nn.Parameter(phases)
        else:
            self.register_buffer('freqs', freqs)
            self.register_buffer('sigmas', sigmas)
            self.register_buffer('directions', directions)
            self.register_buffer('phases', phases)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:

        directions = F.normalize(self.directions, dim=-1)

        proj = torch.matmul(coords, directions.T)

        sigmas = torch.abs(self.sigmas) + 0.01
        gaussian = torch.exp(-proj**2 / (2 * sigmas**2 + 1e-8))

        freqs = torch.abs(self.freqs) + 0.1
        arg = 2 * math.pi * freqs * proj + self.phases

        cos_comp = gaussian * torch.cos(arg)
        sin_comp = gaussian * torch.sin(arg)

        gabor_features = torch.cat([cos_comp, sin_comp], dim=-1)

        return gabor_features

class FourierFeatures(nn.Module):

    def __init__(self, input_dim: int = 2, num_frequencies: int = 64,
                 scale: float = 10.0, learnable: bool = False):

        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = num_frequencies * 2

        B = torch.randn(input_dim, num_frequencies) * scale

        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:

        proj = 2 * math.pi * torch.matmul(coords, self.B)

        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

class SIRENLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):

        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):

        with torch.no_grad():
            if self.is_first:

                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:

                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))

class GaborNet(nn.Module):

    def __init__(self, coord_dim: int = 2, feature_dim: int = 256,
                 hidden_dim: int = 256, output_dim: int = 1,
                 num_layers: int = 4, num_frequencies: int = 64,
                 use_gabor: bool = True, omega_0: float = 30.0):

        super().__init__()

        if use_gabor:
            self.coord_encoder = GaborBasis(
                input_dim=coord_dim,
                num_frequencies=num_frequencies,
                learnable=True
            )
        else:
            self.coord_encoder = FourierFeatures(
                input_dim=coord_dim,
                num_frequencies=num_frequencies,
                learnable=False
            )

        coord_encoded_dim = self.coord_encoder.output_dim

        input_dim = coord_encoded_dim + feature_dim

        layers = []

        layers.append(SIRENLayer(input_dim, hidden_dim, omega_0, is_first=True))

        for _ in range(num_layers - 2):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, omega_0, is_first=False))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:

        coord_encoded = self.coord_encoder(coords)

        x = torch.cat([coord_encoded, features], dim=-1)

        output = self.network(x)

        return output

class ImplicitSegmentationHead(nn.Module):

    def __init__(self, feature_channels: int = 64, num_classes: int = 2,
                 hidden_dim: int = 256, num_layers: int = 4,
                 num_frequencies: int = 64, use_gabor: bool = True):

        super().__init__()

        self.feature_channels = feature_channels
        self.num_classes = num_classes

        self.feature_proj = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU()
        )

        self.implicit_decoder = GaborNet(
            coord_dim=2,
            feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers,
            num_frequencies=num_frequencies,
            use_gabor=use_gabor
        )

    def sample_features(self, feature_map: torch.Tensor,
                       coords: torch.Tensor) -> torch.Tensor:

        B, C, H, W = feature_map.shape
        N = coords.shape[1]

        grid = coords.view(B, 1, N, 2)

        sampled = F.grid_sample(
            feature_map, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        sampled = sampled.squeeze(2).permute(0, 2, 1)

        return sampled

    def forward(self, feature_map: torch.Tensor,
                coords: Optional[torch.Tensor] = None,
                output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:

        B, C, H_feat, W_feat = feature_map.shape
        device = feature_map.device

        feature_map = self.feature_proj(feature_map)

        if coords is None:
            if output_size is None:
                output_size = (H_feat * 4, W_feat * 4)

            H_out, W_out = output_size

            y = torch.linspace(-1, 1, H_out, device=device)
            x = torch.linspace(-1, 1, W_out, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([xx, yy], dim=-1)
            coords = coords.view(1, -1, 2).expand(B, -1, -1)

            reshape_output = True
        else:
            reshape_output = False
            H_out, W_out = None, None

        features = self.sample_features(feature_map, coords)

        logits = self.implicit_decoder(coords, features)

        if reshape_output:
            logits = logits.view(B, H_out, W_out, self.num_classes)
            logits = logits.permute(0, 3, 1, 2)

        return logits

class FiLMLayer(nn.Module):

    def __init__(self, feature_dim: int, modulation_dim: int):
        super().__init__()

        self.gamma_proj = nn.Linear(feature_dim, modulation_dim)
        self.beta_proj = nn.Linear(feature_dim, modulation_dim)

        nn.init.ones_(self.gamma_proj.weight.data[:modulation_dim, :modulation_dim // feature_dim + 1])
        nn.init.zeros_(self.gamma_proj.bias.data)
        nn.init.zeros_(self.beta_proj.weight.data)
        nn.init.zeros_(self.beta_proj.bias.data)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:

        gamma = self.gamma_proj(condition)
        beta = self.beta_proj(condition)

        return gamma * x + beta

class EnergyGatedGaborImplicit(nn.Module):

    def __init__(self, feature_dim: int, num_classes: int,
                 hidden_dim: int = 256, num_layers: int = 4,
                 num_frequencies: int = 64, use_gabor: bool = True,
                 omega_0: float = 30.0):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        if use_gabor:
            self.coord_encoder = GaborBasis(
                input_dim=2, num_frequencies=num_frequencies, learnable=True
            )
        else:
            self.coord_encoder = FourierFeatures(
                input_dim=2, num_frequencies=num_frequencies, learnable=False
            )

        coord_encoded_dim = self.coord_encoder.output_dim

        self.film_layers = nn.ModuleList([
            FiLMLayer(feature_dim, hidden_dim)
            for _ in range(num_layers - 1)
        ])

        self.input_proj = SIRENLayer(coord_encoded_dim, hidden_dim, omega_0, is_first=True)

        self.hidden_layers = nn.ModuleList([
            SIRENLayer(hidden_dim, hidden_dim, omega_0, is_first=False)
            for _ in range(num_layers - 2)
        ])

        self.output_proj = nn.Linear(hidden_dim, num_classes)

        self.gate_scale = nn.Parameter(torch.ones(1))
        self.gate_bias = nn.Parameter(torch.zeros(1))

    def forward(self, coords: torch.Tensor, features: torch.Tensor,
                energy: torch.Tensor) -> torch.Tensor:

        coord_enc = self.coord_encoder(coords)

        x = self.input_proj(coord_enc)

        for i, (hidden_layer, film_layer) in enumerate(
            zip(self.hidden_layers, self.film_layers[:-1])
        ):

            x = hidden_layer(x)

            x = film_layer(x, features)

        if len(self.film_layers) > 0:
            x = self.film_layers[-1](x, features)

        logits = self.output_proj(x)

        gate = torch.sigmoid(energy * self.gate_scale + self.gate_bias)

        gated_logits = logits * gate

        return gated_logits

class EnergyGatedImplicitHead(nn.Module):

    def __init__(self, feature_channels: int, num_classes: int,
                 hidden_dim: int = 256, num_layers: int = 4,
                 num_frequencies: int = 64):
        super().__init__()

        self.num_classes = num_classes

        self.feature_proj = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.GELU()
        )

        self.implicit_decoder = EnergyGatedGaborImplicit(
            feature_dim=hidden_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_frequencies=num_frequencies,
            use_gabor=True
        )

    def sample_at_coords(self, feature_map: torch.Tensor,
                        energy_map: torch.Tensor,
                        coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = feature_map.shape
        N = coords.shape[1]

        grid = coords.view(B, 1, N, 2)

        features = F.grid_sample(
            feature_map, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        ).squeeze(2).permute(0, 2, 1)

        energy = F.grid_sample(
            energy_map, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        ).squeeze(2).permute(0, 2, 1)

        return features, energy

    def forward(self, feature_map: torch.Tensor,
                energy_map: torch.Tensor,
                coords: Optional[torch.Tensor] = None,
                output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:

        B, C, H_feat, W_feat = feature_map.shape
        device = feature_map.device

        feature_map = self.feature_proj(feature_map)

        if coords is None:
            if output_size is None:
                output_size = (H_feat * 4, W_feat * 4)

            H_out, W_out = output_size

            y = torch.linspace(-1, 1, H_out, device=device)
            x = torch.linspace(-1, 1, W_out, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([xx, yy], dim=-1).view(1, -1, 2).expand(B, -1, -1)

            reshape_output = True
        else:
            reshape_output = False
            H_out, W_out = None, None

        features, energy = self.sample_at_coords(feature_map, energy_map, coords)

        logits = self.implicit_decoder(coords, features, energy)

        if reshape_output:
            logits = logits.view(B, H_out, W_out, self.num_classes)
            logits = logits.permute(0, 3, 1, 2)

        return logits

    print("Testing Gabor Implicit Modules...")

    print("\n[1] Testing GaborBasis...")
    gabor = GaborBasis(input_dim=2, num_frequencies=32)
    coords = torch.randn(4, 100, 2)
    encoded = gabor(coords)
    print(f"Input coords: {coords.shape}")
    print(f"Gabor encoded: {encoded.shape}")

    print("\n[2] Testing GaborNet...")
    net = GaborNet(coord_dim=2, feature_dim=64, hidden_dim=128,
                   output_dim=3, num_layers=3, num_frequencies=32)
    features = torch.randn(4, 100, 64)
    output = net(coords, features)
    print(f"GaborNet output: {output.shape}")

    print("\n[3] Testing ImplicitSegmentationHead...")
    head = ImplicitSegmentationHead(
        feature_channels=64, num_classes=3,
        hidden_dim=128, num_layers=3, num_frequencies=32
    )
    feature_map = torch.randn(2, 64, 32, 32)

    seg_output = head(feature_map, output_size=(128, 128))
    print(f"Feature map: {feature_map.shape}")
    print(f"Segmentation output (grid): {seg_output.shape}")

    custom_coords = torch.rand(2, 500, 2) * 2 - 1
    seg_points = head(feature_map, coords=custom_coords)
    print(f"Segmentation output (points): {seg_points.shape}")

    print("\n✓ All tests passed!")
