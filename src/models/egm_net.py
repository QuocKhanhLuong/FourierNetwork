"""
Energy-Gated Gabor Mamba Network (EGM-Net).

A hybrid architecture for medical image segmentation combining physics-based
signal processing with deep learning for artifact-free boundary delineation.

Architecture Components:
    1. HRNetV2-Mamba Backbone: Dual-stream with high-res preservation
    2. Monogenic Signal Processing: Physics-based edge detection via Riesz transform
    3. RBF Constellation Head: Gaussian classifier for coarse segmentation
    4. Energy-Gated Gabor Implicit: FiLM-conditioned fine segmentation
    5. Energy-Gated Fusion: Selective boundary refinement

Key Innovations:
    - Dual-path decoding: Coarse (RBF) + Fine (Implicit) branches
    - Energy gating: Suppress artifacts in flat/homogeneous regions
    - Gabor basis: Localized oscillations prevent Gibbs ringing artifacts
    - Vector rotation augmentation: Maintains Riesz consistency under rotation

References:
    [1] Felsberg & Sommer, "The Monogenic Signal," IEEE TSP, 2001.
    [2] Wang et al., "Deep High-Resolution Representation Learning," CVPR 2020.
    [3] Liu et al., "VMamba: Visual State Space Model," arXiv, 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

# Handle imports for both package and standalone usage
try:
    from ..layers.monogenic import EnergyMap, MonogenicSignal
    from .hrnet_mamba import HRNetV2MambaBackbone, SpectralVSSBlock
    from ..layers.constellation_head import RBFConstellationHead
    from ..layers.gabor_implicit import EnergyGatedImplicitHead, GaborNet, ImplicitSegmentationHead
    from .mamba_block import VSSBlock, MambaBlockStack
except ImportError:
    try:
        # Standalone or different path structure fallback
        import sys
        sys.path.append("..")
        from layers.monogenic import EnergyMap, MonogenicSignal
        from models.hrnet_mamba import HRNetV2MambaBackbone, SpectralVSSBlock
        from layers.constellation_head import RBFConstellationHead
        from layers.gabor_implicit import EnergyGatedImplicitHead, GaborNet, ImplicitSegmentationHead
        from models.mamba_block import VSSBlock, MambaBlockStack
    except ImportError:
        # Testing fallback
        pass


# =============================================================================
# Energy-Gated Fusion Module
# =============================================================================

class EnergyGatedFusion(nn.Module):
    """
    Energy-Gated Fusion Module.
    
    Uses monogenic energy map to gate the implicit (fine) branch:
    - High energy (edges): Activate fine branch for sharp boundaries
    - Low energy (flat): Suppress fine branch to avoid artifacts
    
    Final output = Coarse + Energy × (Fine - Coarse)
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize Energy-Gated Fusion.
        
        Args:
            temperature: Softness of gating (lower = harder gating)
        """
        super().__init__()
        self.temperature = temperature
        self.gate_scale = nn.Parameter(torch.ones(1))
        self.gate_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, coarse: torch.Tensor, fine: torch.Tensor, 
                energy: torch.Tensor) -> torch.Tensor:
        """
        Fuse coarse and fine predictions using energy gating.
        
        Args:
            coarse: Coarse prediction (B, C, H, W)
            fine: Fine prediction (B, C, H, W) 
            energy: Energy map (B, 1, H, W) in [0, 1]
            
        Returns:
            Fused prediction (B, C, H, W)
        """
        # Resize energy to match prediction size
        if energy.shape[-2:] != coarse.shape[-2:]:
            energy = F.interpolate(energy, size=coarse.shape[-2:], 
                                   mode='bilinear', align_corners=True)
        
        # Resize fine to match coarse if needed
        if fine.shape[-2:] != coarse.shape[-2:]:
            fine = F.interpolate(fine, size=coarse.shape[-2:],
                                mode='bilinear', align_corners=True)
        
        # Apply learnable scaling and temperature
        gate = torch.sigmoid((energy * self.gate_scale + self.gate_bias) / self.temperature)
        
        # Blend: high energy → use fine, low energy → use coarse
        output = coarse + gate * (fine - coarse)
        
        return output


# =============================================================================
# EGM-Net: Main Model
# =============================================================================

class EGMNet(nn.Module):
    """
    Energy-Gated Gabor Mamba Network (EGM-Net).
    
    Full architecture with HRNetV2-Mamba backbone:
    
    1. Input: 3 channels (Intensity, Rx, Ry) or 1 channel (Intensity only)
    2. Backbone: HRNetV2-Mamba (dual-stream, maintains high-res throughout)
    3. Coarse Head: RBF Constellation (Gaussian classifier with PSK prototypes)
    4. Fine Head: Energy-Gated Gabor Implicit (FiLM-conditioned)
    5. Fusion: Energy-gated blending of coarse and fine
    
    The energy map (from Monogenic Signal) gates the fine branch:
    - High energy regions (boundaries): Use fine branch for sharp edges
    - Low energy regions (flat): Use coarse branch to avoid artifacts
    
    Args:
        in_channels: Number of input channels (1 or 3)
        num_classes: Number of segmentation classes
        img_size: Input image size
        base_channels: Base channel count for backbone
        num_stages: Number of backbone stages
        use_hrnet: Use HRNetV2 backbone (True) or simpler encoder (False)
        implicit_hidden: Hidden dim for implicit decoder
        implicit_layers: Number of implicit decoder layers
        num_frequencies: Number of Gabor frequencies
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 4,
                 img_size: int = 256, base_channels: int = 64,
                 num_stages: int = 4, use_hrnet: bool = True,
                 implicit_hidden: int = 256, implicit_layers: int = 4,
                 num_frequencies: int = 64):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_hrnet = use_hrnet
        
        # 1. Monogenic Energy Extractor (fixed, non-trainable)
        # For computing energy maps from input images
        self.energy_extractor = EnergyMap(normalize=True, smoothing_sigma=1.0)
        
        # 2. Backbone
        if use_hrnet:
            self.backbone = HRNetV2MambaBackbone(
                in_channels=in_channels,
                base_channels=base_channels,
                num_stages=num_stages,
                blocks_per_stage=2,
                mamba_depth=2,
                img_size=img_size
            )
            backbone_channels = self.backbone.out_channels
        else:
            # Fallback to simpler encoder for testing
            self.backbone = SimpleEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_stages=num_stages
            )
            backbone_channels = base_channels * (2 ** (num_stages - 1))
        
        self.backbone_channels = backbone_channels
        
        # 3. Coarse Head: RBF Constellation
        self.coarse_head = RBFConstellationHead(
            in_channels=backbone_channels,
            num_classes=num_classes,
            embedding_dim=2,
            init_gamma=1.0
        )
        
        # 4. Fine Head: Energy-Gated Gabor Implicit
        self.fine_head = EnergyGatedImplicitHead(
            feature_channels=backbone_channels,
            num_classes=num_classes,
            hidden_dim=implicit_hidden,
            num_layers=implicit_layers,
            num_frequencies=num_frequencies
        )
        
        # 5. Energy-Gated Fusion
        self.fusion = EnergyGatedFusion(temperature=1.0)
        
        # Store feature size for output upsampling
        if use_hrnet:
            self.feature_size = img_size // 4
        else:
            self.feature_size = img_size // (2 ** num_stages)
    
    def _compute_energy(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute energy map from input.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Tuple of (energy_map, monogenic_outputs)
        """
        with torch.no_grad():
            # Use first channel (intensity) for energy computation
            x_gray = x[:, 0:1] if x.shape[1] > 1 else x
            energy, mono_out = self.energy_extractor(x_gray)
        return energy, mono_out
    
    def forward(self, x: torch.Tensor, 
                output_size: Optional[Tuple[int, int]] = None,
                return_intermediates: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image (B, C, H, W)
               - If C=3: Expected [Intensity, Rx, Ry]
               - If C=1: Intensity only, energy computed online
            output_size: Optional output resolution (default: input size)
            return_intermediates: Whether to return coarse/fine/energy separately
            
        Returns:
            Dictionary containing:
            - 'output': Final fused segmentation (B, num_classes, H, W)
            - 'coarse': Coarse branch output (B, num_classes, H, W)
            - 'fine': Fine branch output (B, num_classes, H, W)
            - 'energy': Energy map (B, 1, H, W)
            - 'embeddings': 2D constellation embeddings (B, 2, H, W) [optional]
        """
        B, C, H, W = x.shape
        
        if output_size is None:
            output_size = (H, W)
        
        # 1. Compute or extract energy map
        if C >= 3:
            # If Riesz components are provided, compute energy from them
            intensity = x[:, 0:1]  # (B, 1, H, W)
            riesz_x = x[:, 1:2]    # (B, 1, H, W)  
            riesz_y = x[:, 2:3]    # (B, 1, H, W)
            
            # Energy = sqrt(I^2 + Rx^2 + Ry^2)
            energy = torch.sqrt(intensity**2 + riesz_x**2 + riesz_y**2 + 1e-8)
            # Normalize to [0, 1]
            energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        else:
            # Compute energy online using Monogenic Signal
            energy, _ = self._compute_energy(x)
        
        # 2. Backbone feature extraction
        if self.use_hrnet:
            backbone_out = self.backbone(x)
            features = backbone_out['features']
        else:
            features = self.backbone(x)
        
        # 3. Coarse Head (RBF Constellation)
        coarse_logits, embeddings = self.coarse_head(features)
        
        # Upsample coarse to output size
        coarse = F.interpolate(coarse_logits, size=output_size,
                              mode='bilinear', align_corners=True)
        
        # 4. Fine Head (Energy-Gated Gabor Implicit)
        # Resize energy to feature size for implicit head
        energy_for_fine = F.interpolate(energy, size=features.shape[-2:],
                                        mode='bilinear', align_corners=True)
        
        fine_logits = self.fine_head(features, energy_for_fine, output_size=output_size)
        
        # Ensure fine is same size as coarse
        if fine_logits.shape[-2:] != output_size:
            fine_logits = F.interpolate(fine_logits, size=output_size,
                                       mode='bilinear', align_corners=True)
        
        # 5. Energy-Gated Fusion
        output = self.fusion(coarse, fine_logits, energy)
        
        if return_intermediates:
            return {
                'output': output,
                'coarse': coarse,
                'fine': fine_logits,
                'energy': energy,
                'embeddings': embeddings
            }
        else:
            return {'output': output}
    
    def inference(self, x: torch.Tensor, 
                  output_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Simplified inference returning only final output.
        
        Args:
            x: Input image (B, C, H, W)
            output_size: Optional output resolution
            
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        return self.forward(x, output_size, return_intermediates=False)['output']
    
    def query_points(self, x: torch.Tensor, 
                     coords: torch.Tensor) -> torch.Tensor:
        """
        Query segmentation at arbitrary coordinates (implicit representation).
        
        This enables resolution-free inference - zoom into boundaries at any
        resolution without pixelation.
        
        Args:
            x: Input image (B, C, H, W)
            coords: Query coordinates (B, N, 2) normalized to [-1, 1]
            
        Returns:
            Segmentation logits at query points (B, N, num_classes)
        """
        B, C, H, W = x.shape
        
        # Compute energy
        if C >= 3:
            intensity = x[:, 0:1]
            riesz_x = x[:, 1:2]
            riesz_y = x[:, 2:3]
            energy = torch.sqrt(intensity**2 + riesz_x**2 + riesz_y**2 + 1e-8)
            energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        else:
            energy, _ = self._compute_energy(x)
        
        # Get backbone features
        if self.use_hrnet:
            backbone_out = self.backbone(x)
            features = backbone_out['features']
        else:
            features = self.backbone(x)
        
        # Sample features and energy at query coordinates
        N = coords.shape[1]
        grid = coords.view(B, 1, N, 2)
        
        # Project features
        features_proj = self.fine_head.feature_proj(features)
        
        # Sample at coordinates
        feat_sampled = F.grid_sample(
            features_proj, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        ).squeeze(2).permute(0, 2, 1)  # (B, N, C)
        
        energy_sampled = F.grid_sample(
            energy, grid, mode='bilinear',
            padding_mode='border', align_corners=True
        ).squeeze(2).permute(0, 2, 1)  # (B, N, 1)
        
        # Implicit decoding with energy gating
        point_logits = self.fine_head.implicit_decoder(coords, feat_sampled, energy_sampled)
        
        return point_logits


# =============================================================================
# Simple Encoder (Fallback for testing)
# =============================================================================

class SimpleEncoder(nn.Module):
    """Simple encoder for testing without full HRNet."""
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64,
                 num_stages: int = 4):
        super().__init__()
        
        layers = []
        channels = in_channels
        out_channels = base_channels
        
        for i in range(num_stages):
            layers.append(nn.Conv2d(channels, out_channels, 3, stride=2, padding=1))
            layers.append(nn.GroupNorm(min(32, out_channels), out_channels))
            layers.append(nn.GELU())
            channels = out_channels
            out_channels = min(out_channels * 2, 512)
        
        self.layers = nn.Sequential(*layers)
        self.out_channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# =============================================================================
# EGM-Net Lite (Lightweight version)
# =============================================================================

class EGMNetLite(nn.Module):
    """
    Lightweight version of EGM-Net for faster training/inference.
    
    Reduced channels and stages for lower memory/compute.
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 img_size: int = 256):
        super().__init__()
        
        self.model = EGMNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            base_channels=32,
            num_stages=3,
            use_hrnet=False,  # Use simpler encoder
            implicit_hidden=128,
            implicit_layers=3,
            num_frequencies=32
        )
    
    def forward(self, x, output_size=None):
        return self.model(x, output_size)
    
    def inference(self, x, output_size=None):
        return self.model.inference(x, output_size)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing EGM-Net (Energy-Gated Gabor Mamba Network)")
    print("=" * 60)
    
    # Test with simple encoder first (no HRNet dependencies)
    print("\n[1] Testing EGM-Net with Simple Encoder...")
    model = EGMNet(
        in_channels=3,
        num_classes=4,
        img_size=256,
        base_channels=64,
        num_stages=4,
        use_hrnet=False  # Use simple encoder for testing
    )
    
    # 3-channel input: Intensity + Rx + Ry
    x = torch.randn(2, 3, 256, 256)
    outputs = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {outputs['output'].shape}")
    print(f"Coarse: {outputs['coarse'].shape}")
    print(f"Fine: {outputs['fine'].shape}")
    print(f"Energy: {outputs['energy'].shape}")
    print(f"Embeddings: {outputs['embeddings'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test point query (resolution-free inference)
    print("\n[2] Testing Point Query (Resolution-Free)...")
    coords = torch.rand(2, 1000, 2) * 2 - 1  # Random points in [-1, 1]
    point_output = model.query_points(x, coords)
    print(f"Query coords: {coords.shape}")
    print(f"Point output: {point_output.shape}")
    
    # Test lite model
    print("\n[3] Testing EGM-Net Lite...")
    lite_model = EGMNetLite(in_channels=1, num_classes=3, img_size=256)
    x_lite = torch.randn(2, 1, 256, 256)
    lite_outputs = lite_model(x_lite)
    
    lite_params = sum(p.numel() for p in lite_model.parameters())
    print(f"Lite model parameters: {lite_params:,}")
    print(f"Lite output: {lite_outputs['output'].shape}")
    
    # Test single-channel input (energy computed online)
    print("\n[4] Testing Single-Channel Input (Auto Energy)...")
    x_single = torch.randn(2, 1, 256, 256)
    model_single = EGMNet(in_channels=1, num_classes=4, img_size=256, use_hrnet=False)
    outputs_single = model_single(x_single)
    print(f"Single-channel output: {outputs_single['output'].shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
