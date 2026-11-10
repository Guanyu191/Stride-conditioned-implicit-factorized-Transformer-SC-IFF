"""Single-frame predictors: fixed-stride IFF and the SC-IFF variants.

Concat appends the normalized stride to the input, FiLM modulates the features
with a stride embedding, and Indep trains one unconditioned predictor per stride.
"""
import math
import torch
from torch import nn
from einops import rearrange
from attention import PAI


def stride_code(strides, maximum):
    """Encode a vector of strides (B,) as features (B,5)."""
    rho = strides.float() / maximum
    return torch.stack((rho, strides.float().log2() / math.log2(maximum), (math.pi * rho).sin(), (math.pi * rho).cos(), torch.ones_like(rho)), -1)


class Predictor(nn.Module):
    """Map a velocity field to the field `stride` stored intervals later."""

    def __init__(self, cfg, stats, kind):
        super().__init__()
        self.kind, self.dt, self.maximum = kind, cfg.dt, cfg.strides[-1]
        self.iterations = cfg.iterations
        width = cfg.width

        # TRAIN statistics: normalize the input and restore the physical rate.
        for key in ("state_mean", "state_std"):
            self.register_buffer(key, torch.tensor(stats[key], dtype=torch.float32).reshape(1, 1, 1, 1, 3))
        for key in ("rate_mean", "rate_std"):
            self.register_buffer(key, torch.tensor(stats[key], dtype=torch.float32).reshape(self.maximum, 1, -1, 1, 3))

        # Lifting and coordinate embedding. Conv1d over space acts pointwise.
        channels = 4 if kind == "concat" else 3
        self.lift = nn.Sequential(nn.Conv1d(channels, width // 2, 1, groups=1 if kind == "concat" else 3), nn.GELU(),
                                  nn.Conv1d(width // 2, width, 1, bias=False))
        self.fourier_matrix = nn.Parameter(torch.randn(3, width // 2) * 8, requires_grad=False)
        self.position_map = nn.Linear(width, width)

        # One PAI module shared by all iterations.
        self.pai = PAI(width, cfg.heads, cfg.value_channels)

        # Readout of the normalized rate, zero at initialization.
        self.output_norm = nn.GroupNorm(6, width)
        self.readout = nn.Sequential(nn.Conv1d(width, width, 1, groups=6, bias=False), nn.GELU(),
                                     nn.Conv1d(width, width // 2, 1, bias=False), nn.GELU(),
                                     nn.Conv1d(width // 2, 3, 1))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

        # FiLM: stride embedding and modulation layers, zero at initialization.
        if kind == "film":
            self.condition = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 8))
            self.film_pai, self.film_output = nn.Linear(8, 2 * width), nn.Linear(8, 2 * width)
            for layer in (self.film_pai, self.film_output):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state, strides, positions, features=False):
        """`state` is a physical field (B,X,Y,Z,3); `features=True` also returns the rate and latent."""
        batch, nx, ny, nz, _ = state.shape
        strides = torch.as_tensor(strides, device=state.device).expand(batch)

        # Normalize the input; Concat adds the stride channel.
        u = (state - self.state_mean) / self.state_std
        if self.kind == "concat":
            channel = (strides / self.maximum)[:, None, None, None, None].expand(batch, nx, ny, nz, 1)
            u = torch.cat((u, channel), -1)

        # Initial latent field: lifted velocity plus coordinate embedding.
        latent = self.lift(rearrange(u, "b x y z c -> b c (x y z)"))
        latent = rearrange(latent, "b c (x y z) -> b x y z c", x=nx, y=ny, z=nz)
        grid = torch.stack(torch.meshgrid(*[p.flatten() for p in positions], indexing="ij"), -1)
        phase = 2 * math.pi * (grid @ self.fourier_matrix)
        latent = latent + self.position_map(torch.cat((phase.sin(), phase.cos()), -1))

        # Shared implicit iterations.
        modulation = None
        if self.kind == "film":
            embedding = self.condition(stride_code(strides, self.maximum))
            modulation = self.film_pai(embedding).chunk(2, -1)
        for _ in range(self.iterations):
            latent = latent + self.pai(latent, positions, modulation) / self.iterations

        # Read out the normalized rate.
        output = self.output_norm(rearrange(latent, "b x y z c -> b c (x y z)"))
        if self.kind == "film":
            gamma, beta = self.film_output(embedding).chunk(2, -1)
            output = (1 + gamma[..., None]) * output + beta[..., None]
        rate = rearrange(self.readout(output), "b c (x y z) -> b x y z c", x=nx, y=ny, z=nz)

        # Restore the physical rate and advance the velocity.
        mean, std = self.rate_mean[strides.long() - 1], self.rate_std[strides.long() - 1]
        prediction = state + strides[:, None, None, None, None] * self.dt * (mean + std * rate)
        return (prediction, rate, latent) if features else prediction
