import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class ShearletController(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4)
        )
        
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data = torch.tensor([1.0, 1.0, 1.0, 0.0])

    def forward(self, features):
        params = self.net(features)
        freq = F.softplus(params[..., 0:1]) + 0.1
        scale_x = F.softplus(params[..., 1:2]) + 0.1
        scale_y = F.softplus(params[..., 2:3]) + 0.1
        shear = torch.tanh(params[..., 3:4]) * 2.0
        return freq, scale_x, scale_y, shear


class ShearletBasis(nn.Module):
    def __init__(self, num_orientations=8, base_freq=2.0):
        super().__init__()
        self.num_orientations = num_orientations
        self.base_freq = base_freq
        
        angles = torch.linspace(0, math.pi, num_orientations, endpoint=False)
        self.register_buffer('base_angles', angles)

    def forward(self, coords, freq, scale_x, scale_y, shear):
        B, N, _ = coords.shape
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        
        x_sheared = x + shear * y
        y_scaled = y
        
        x_norm = x_sheared * scale_x
        y_norm = y_scaled * scale_y
        
        envelope = torch.exp(-0.5 * (x_norm**2 + y_norm**2))
        
        oscillation = torch.cos(2 * math.pi * freq * x_sheared)
        
        shearlet = envelope * oscillation
        
        orientations = []
        for angle in self.base_angles:
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            x_rot_sheared = x_rot + shear * y_rot
            env = torch.exp(-0.5 * ((x_rot_sheared * scale_x)**2 + (y_rot * scale_y)**2))
            osc = torch.cos(2 * math.pi * freq * x_rot_sheared)
            orientations.append(env * osc)
        
        multi_orient = torch.cat(orientations, dim=-1)
        return multi_orient


class ShearletImplicitHead(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim=256, 
                 num_orientations=8, num_frequencies=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_orientations = num_orientations
        self.num_frequencies = num_frequencies
        
        self.controller = ShearletController(feature_dim, hidden_dim)
        
        self.freq_scales = nn.Parameter(torch.linspace(0.5, 4.0, num_frequencies))
        
        self.coord_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        shearlet_dim = num_orientations * num_frequencies
        fusion_dim = hidden_dim + shearlet_dim + feature_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.shearlet_basis = ShearletBasis(num_orientations)

    def forward(self, features, coords=None, energy_gate=None):
        B, C, H, W = features.shape
        
        if coords is None:
            y_coords = torch.linspace(-1, 1, H, device=features.device)
            x_coords = torch.linspace(-1, 1, W, device=features.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            coords = torch.stack([xx, yy], dim=-1).view(1, H*W, 2).expand(B, -1, -1)
        
        N = coords.shape[1]
        
        grid = coords.view(B, 1, N, 2)
        sampled_features = F.grid_sample(features, grid, mode='bilinear', 
                                          align_corners=True, padding_mode='border')
        sampled_features = sampled_features.squeeze(2).permute(0, 2, 1)
        
        freq, scale_x, scale_y, shear = self.controller(sampled_features)
        
        all_shearlets = []
        for i, f_scale in enumerate(self.freq_scales):
            shearlet = self.shearlet_basis(coords, freq * f_scale, scale_x, scale_y, shear)
            all_shearlets.append(shearlet)
        shearlet_features = torch.cat(all_shearlets, dim=-1)
        
        coord_encoded = self.coord_encoder(coords)
        
        fused = torch.cat([coord_encoded, shearlet_features, sampled_features], dim=-1)
        
        if energy_gate is not None:
            gate_grid = coords.view(B, 1, N, 2)
            gate_values = F.grid_sample(energy_gate, gate_grid, mode='bilinear',
                                        align_corners=True, padding_mode='border')
            gate_values = gate_values.squeeze(2).permute(0, 2, 1)
            fused = fused * gate_values
        
        logits = self.decoder(fused)
        
        logits = logits.permute(0, 2, 1).view(B, self.num_classes, H, W)
        
        return logits

    def forward_points(self, features, coords, energy_gate=None):
        B, C, H, W = features.shape
        N = coords.shape[1]
        
        grid = coords.view(B, 1, N, 2)
        sampled_features = F.grid_sample(features, grid, mode='bilinear',
                                          align_corners=True, padding_mode='border')
        sampled_features = sampled_features.squeeze(2).permute(0, 2, 1)
        
        freq, scale_x, scale_y, shear = self.controller(sampled_features)
        
        all_shearlets = []
        for f_scale in self.freq_scales:
            shearlet = self.shearlet_basis(coords, freq * f_scale, scale_x, scale_y, shear)
            all_shearlets.append(shearlet)
        shearlet_features = torch.cat(all_shearlets, dim=-1)
        
        coord_encoded = self.coord_encoder(coords)
        
        fused = torch.cat([coord_encoded, shearlet_features, sampled_features], dim=-1)
        
        if energy_gate is not None:
            gate_grid = coords.view(B, 1, N, 2)
            gate_values = F.grid_sample(energy_gate, gate_grid, mode='bilinear',
                                        align_corners=True, padding_mode='border')
            gate_values = gate_values.squeeze(2).permute(0, 2, 1)
            fused = fused * gate_values
        
        logits = self.decoder(fused)
        return logits


if __name__ == "__main__":
    B, C, H, W = 2, 64, 64, 64
    features = torch.randn(B, C, H, W)
    
    head = ShearletImplicitHead(feature_dim=C, num_classes=4)
    
    out = head(features)
    print(f"Grid output: {out.shape}")
    
    coords = torch.rand(B, 100, 2) * 2 - 1
    out_pts = head.forward_points(features, coords)
    print(f"Points output: {out_pts.shape}")
