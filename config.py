"""All settings of the Channel-DT10 case."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # Edit these paths for your machine.
    data_path: str = "data/channel_dt10.npy"
    output_root: str = "runs"
    coordinates_path: str = str(Path(__file__).resolve().with_name("channel_grid.json"))
    device: str = "cuda"

    # Data: 20 trajectories of 4000 fields on a 32 x 33 x 16 grid.
    dt: float = 0.05
    frames: int = 4000
    grid: tuple = (32, 33, 16)
    strides: tuple = (1, 2, 4, 8, 16, 32)
    train_ids: tuple = tuple(range(12))
    valid_ids: tuple = tuple(range(12, 16))
    test_ids: tuple = tuple(range(16, 20))

    # Predictors.
    width: int = 96
    value_channels: int = 96
    heads: int = 5
    iterations: int = 5

    # FA-CSF.
    fusion_width: int = 48
    fusion_value_channels: int = 32
    fusion_heads: int = 3
    projected_channels: int = 12

    # Stage 1 training.
    stage1_lr: float = 5e-4
    stage1_final_lr: float = 5e-6
    stage1_decay: float = 1e-2
    stage1_clip: float = 1.0
    stage1_batch: int = 10

    # Stage 2 training.
    stage2_lr: float = 1e-4
    stage2_decay: float = 1e-4
    stage2_batch: int = 4
    stage2_updates: int = 720
    stage2_every: int = 60  # Updates between VALID evaluations.
    window: int = 1024  # Predicted frames per training sample.
    segment: int = 32  # Frames per energy segment.

    # Implementation choices not given numerically in the paper.
    seed: int = 1234
    fusion_seed: int = 4314
    stats_chunk: int = 16
    spectrum_smoothing: float = 1e-4
    energy_smoothing: float = 1e-2
    covariance_floor: float = 1e-6
    max_lag: int = 1024
    short_origins: int = 32
    stage2_clip: float = 1.0

    def epochs(self, model, stride):
        """Stage 1 budget in epochs."""
        if model == "iff":
            return 48
        if model == "indep":
            return 18 if stride == self.strides[-1] else 6
        return 8  # Concat and FiLM train one predictor on all strides.


settings = Settings()
