
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:

    in_channels: int = 1
    out_channels: int = 3
    img_size: int = 256
    base_channels: int = 64
    num_stages: int = 4
    depth: int = 2

@dataclass
class LossConfig:

    spatial_weight: float = 1.0
    freq_weight: float = 0.1
    use_dice: bool = True
    use_focal: bool = True
    boundary_weight: float = 0.0

@dataclass
class TrainingConfig:

    model: ModelConfig = ModelConfig()

    loss: LossConfig = LossConfig()

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"

    num_epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4

    warmup_epochs: int = 10
    scheduler: str = "cosine"

    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 10

    device: str = "cuda"
    mixed_precision: bool = False

    log_interval: int = 100
    val_interval: int = 1

if __name__ == "__main__":
    config = TrainingConfig()
    print(config)
