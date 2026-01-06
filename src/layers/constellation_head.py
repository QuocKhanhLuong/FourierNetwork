
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class RBFConstellationHead(nn.Module):

    def __init__(self, in_channels: int, num_classes: int = 4,
                 embedding_dim: int = 2, init_gamma: float = 1.0):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        mid_channels = in_channels // 2

        num_groups = math.gcd(mid_channels, 32)

        self.projector = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, mid_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels // 2, embedding_dim, kernel_size=1, bias=True),
            nn.Tanh()
        )

        prototypes = self._generate_constellation_points(num_classes, embedding_dim)
        self.register_buffer('prototypes', prototypes)

        self.gamma = nn.Parameter(torch.tensor(init_gamma))

    def _generate_constellation_points(self, num_classes: int,
                                        embedding_dim: int) -> torch.Tensor:

        if embedding_dim != 2:

            points = torch.randn(embedding_dim, num_classes)
            points = F.normalize(points, dim=0)
            return points

        angles = torch.linspace(0, 2 * math.pi, num_classes + 1)[:-1]

        points = torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=0)

        return points

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C, H, W = x.shape

        z = self.projector(x)

        z_flat = z.permute(0, 2, 3, 1).unsqueeze(-1)

        proto = self.prototypes.view(1, 1, 1, self.embedding_dim, self.num_classes)

        diff = z_flat - proto
        squared_dist = torch.sum(diff ** 2, dim=3)

        gamma = F.softplus(self.gamma)
        logits = -gamma * squared_dist

        logits = logits.permute(0, 3, 1, 2)

        return logits, z

    def get_noise_margin(self) -> float:

        with torch.no_grad():

            proto = self.prototypes.T

            dist_matrix = torch.cdist(proto, proto)

            mask = torch.eye(self.num_classes, device=proto.device).bool()
            dist_matrix = dist_matrix.masked_fill(mask, float('inf'))

            min_dist = dist_matrix.min().item()

        return min_dist

class ConstellationVisualization:

    @staticmethod
    def plot_constellation(embeddings: torch.Tensor, labels: torch.Tensor,
                          prototypes: torch.Tensor, save_path: Optional[str] = None):

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return

        embeddings = embeddings.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        prototypes = prototypes.detach().cpu().numpy().T

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        num_classes = prototypes.shape[0]
        colors = plt.cm.tab10(range(num_classes))

        for cls in range(num_classes):
            mask = labels == cls
            if mask.sum() > 0:
                ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                          c=[colors[cls]], alpha=0.5, s=10, label=f'Class {cls}')

        for cls in range(num_classes):
            ax.scatter(prototypes[cls, 0], prototypes[cls, 1],
                      c=[colors[cls]], s=200, marker='*', edgecolors='black',
                      linewidth=2)

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

if __name__ == "__main__":
    print("=" * 60)
    print("Testing RBF Constellation Head")
    print("=" * 60)

    for num_classes in [2, 4, 8]:
        print(f"\n[{num_classes} classes] Testing...")

        head = RBFConstellationHead(
            in_channels=64,
            num_classes=num_classes,
            embedding_dim=2,
            init_gamma=1.0
        )

        x = torch.randn(2, 64, 32, 32)
        logits, embeddings = head(x)

        print(f"  Input: {x.shape}")
        print(f"  Logits: {logits.shape}")
        print(f"  Embeddings: {embeddings.shape}")
        print(f"  Prototypes: {head.prototypes.shape}")
        print(f"  Noise margin: {head.get_noise_margin():.4f}")

        print(f"  Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")

        pred = torch.argmax(logits, dim=1)
        print(f"  Predictions: {pred.shape}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
