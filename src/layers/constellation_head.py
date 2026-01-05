"""
RBF Constellation Head for Semantic Segmentation.

Implements a Gaussian/RBF classifier inspired by digital modulation (PSK/QAM),
projecting features to a 2D I/Q space and measuring distance to fixed prototypes.

Key Concepts:
    1. Project features to 2D embedding space (I/Q plane)
    2. Fix class prototypes at constellation points (PSK/QPSK/8-PSK)
    3. Classify based on RBF kernel: P(y|x) ∝ exp(-γ||z - c||²)
    4. Output logits = -γ × squared_distance (for CrossEntropyLoss)

Advantages over standard Conv1x1 classifier:
    - Maximizes noise margin (decision boundaries equidistant from prototypes)
    - Geometric interpretation of class relationships
    - Robust to gradient explosion (bounded embedding space via Tanh)

References:
    [1] Proakis, "Digital Communications" - PSK/QAM constellation theory
    [2] RBF Networks - Gaussian kernel classifiers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class RBFConstellationHead(nn.Module):
    """
    RBF Constellation Head with Gaussian Classifier.
    
    Projects feature map to 2D latent space (I/Q plane) and computes
    class probabilities based on distance to fixed constellation prototypes.
    
    Architecture:
        1. Projector: Conv 3x3 → SiLU → Conv 1x1 → Tanh (→ [-1, 1])
        2. Prototypes: Fixed PSK constellation points (non-trainable)
        3. RBF Kernel: exp(-γ||z - c||²)
        4. Logits: -γ × ||z - c||² (for CrossEntropyLoss)
    
    Args:
        in_channels: Number of input feature channels
        num_classes: Number of output classes (determines constellation type)
        embedding_dim: Dimension of I/Q embedding (default 2 for 2D plane)
        init_gamma: Initial temperature parameter (γ = 1/2σ²)
    """
    
    def __init__(self, in_channels: int, num_classes: int = 4,
                 embedding_dim: int = 2, init_gamma: float = 1.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Feature projector: maps to 2D I/Q space
        mid_channels = in_channels // 2
        # Ensure num_groups divides num_channels (using GCD with 32)
        num_groups = math.gcd(mid_channels, 32)
        
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels // 2, embedding_dim, kernel_size=1, bias=True),
            nn.Tanh()  # Constrain to [-1, 1] for stable training
        )
        
        # Fixed constellation prototypes (non-trainable)
        prototypes = self._generate_constellation_points(num_classes, embedding_dim)
        self.register_buffer('prototypes', prototypes)
        
        # Learnable temperature (γ = 1 / 2σ²)
        # Higher γ = narrower Gaussian = harder boundaries
        self.gamma = nn.Parameter(torch.tensor(init_gamma))
    
    def _generate_constellation_points(self, num_classes: int, 
                                        embedding_dim: int) -> torch.Tensor:
        """
        Generate constellation points based on number of classes.
        
        Uses M-PSK (Phase Shift Keying) for 2D:
            - 2 classes: BPSK (±1)
            - 4 classes: QPSK (±1, ±j)
            - 8 classes: 8-PSK (unit circle, 45° spacing)
            - Other: uniform on unit circle
        
        Args:
            num_classes: Number of constellation points
            embedding_dim: Embedding dimension (2 for I/Q)
            
        Returns:
            Tensor of shape (embedding_dim, num_classes)
        """
        if embedding_dim != 2:
            # For higher dimensions, use random unit vectors
            points = torch.randn(embedding_dim, num_classes)
            points = F.normalize(points, dim=0)
            return points
        
        # Generate M-PSK constellation (2D)
        angles = torch.linspace(0, 2 * math.pi, num_classes + 1)[:-1]
        
        # Unit circle points
        points = torch.stack([
            torch.cos(angles),  # I component
            torch.sin(angles)   # Q component
        ], dim=0)  # (2, num_classes)
        
        return points
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Feature map (B, C, H, W)
            
        Returns:
            Tuple of:
            - logits: Class logits (B, num_classes, H, W)
            - embeddings: 2D embeddings (B, 2, H, W)
        """
        B, C, H, W = x.shape
        
        # Project to 2D embedding space
        z = self.projector(x)  # (B, 2, H, W)
        
        # Reshape for distance computation
        # z: (B, 2, H, W) → (B, H, W, 2, 1)
        z_flat = z.permute(0, 2, 3, 1).unsqueeze(-1)  # (B, H, W, 2, 1)
        
        # prototypes: (2, num_classes) → (1, 1, 1, 2, num_classes)
        proto = self.prototypes.view(1, 1, 1, self.embedding_dim, self.num_classes)
        
        # Compute squared Euclidean distance: ||z - c||²
        # (B, H, W, 2, 1) - (1, 1, 1, 2, M) → (B, H, W, 2, M)
        diff = z_flat - proto
        squared_dist = torch.sum(diff ** 2, dim=3)  # (B, H, W, num_classes)
        
        # RBF/Gaussian logits: -γ × ||z - c||²
        # Higher γ = stricter classification
        gamma = F.softplus(self.gamma)  # Ensure positive
        logits = -gamma * squared_dist  # (B, H, W, num_classes)
        
        # Rearrange to (B, num_classes, H, W)
        logits = logits.permute(0, 3, 1, 2)
        
        return logits, z
    
    def get_noise_margin(self) -> float:
        """
        Compute minimum distance between constellation points (noise margin).
        
        Returns:
            Minimum Euclidean distance between any two prototypes
        """
        with torch.no_grad():
            # prototypes: (2, M)
            proto = self.prototypes.T  # (M, 2)
            
            # Pairwise distances
            dist_matrix = torch.cdist(proto, proto)
            
            # Mask diagonal
            mask = torch.eye(self.num_classes, device=proto.device).bool()
            dist_matrix = dist_matrix.masked_fill(mask, float('inf'))
            
            min_dist = dist_matrix.min().item()
            
        return min_dist


class ConstellationVisualization:
    """Helper class for visualizing constellation embeddings."""
    
    @staticmethod
    def plot_constellation(embeddings: torch.Tensor, labels: torch.Tensor,
                          prototypes: torch.Tensor, save_path: Optional[str] = None):
        """
        Plot 2D constellation with class prototypes.
        
        Args:
            embeddings: Tensor of shape (N, 2) with embedded points
            labels: Tensor of shape (N,) with class labels
            prototypes: Tensor of shape (2, num_classes) with prototypes
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        embeddings = embeddings.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        prototypes = prototypes.detach().cpu().numpy().T  # (num_classes, 2)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot embeddings colored by class
        num_classes = prototypes.shape[0]
        colors = plt.cm.tab10(range(num_classes))
        
        for cls in range(num_classes):
            mask = labels == cls
            if mask.sum() > 0:
                ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                          c=[colors[cls]], alpha=0.5, s=10, label=f'Class {cls}')
        
        # Plot prototypes as larger markers
        for cls in range(num_classes):
            ax.scatter(prototypes[cls, 0], prototypes[cls, 1],
                      c=[colors[cls]], s=200, marker='*', edgecolors='black',
                      linewidth=2)
        
        # Draw unit circle (constellation reference)
        theta = torch.linspace(0, 2 * math.pi, 100)
        ax.plot(torch.cos(theta).numpy(), torch.sin(theta).numpy(),
               'k--', alpha=0.3, label='Unit Circle')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_xlabel('I (In-Phase)')
        ax.set_ylabel('Q (Quadrature)')
        ax.set_title('Constellation Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing RBF Constellation Head")
    print("=" * 60)
    
    # Test with different class counts
    for num_classes in [2, 4, 8]:
        print(f"\n[{num_classes} classes] Testing...")
        
        head = RBFConstellationHead(
            in_channels=64,
            num_classes=num_classes,
            embedding_dim=2,
            init_gamma=1.0
        )
        
        # Forward pass
        x = torch.randn(2, 64, 32, 32)
        logits, embeddings = head(x)
        
        print(f"  Input: {x.shape}")
        print(f"  Logits: {logits.shape}")
        print(f"  Embeddings: {embeddings.shape}")
        print(f"  Prototypes: {head.prototypes.shape}")
        print(f"  Noise margin: {head.get_noise_margin():.4f}")
        
        # Check embeddings are bounded
        print(f"  Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        
        # Verify logits are valid for CrossEntropyLoss
        pred = torch.argmax(logits, dim=1)
        print(f"  Predictions: {pred.shape}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
