"""Parallel axial integration (PAI).

The three kernels act on the SAME value field, not sequentially on one another.
The initialization, gated MLP and RoPE frequency scale follow the original IFF
implementation; these details are not numerical hyperparameters in the paper.
"""
import torch
from torch import nn
from einops import rearrange


class AxisFeatures(nn.Module):
    """Average a field onto one axis and map it with a gated MLP."""

    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.expand = nn.Linear(width, 2 * width)
        self.contract = nn.Linear(width, width)

    def forward(self, field, axis):
        # (B, X, Y, Z, C) -> (B, N_axis, C).
        pooled = self.project(field).mean(tuple(d for d in (1, 2, 3) if d != axis))
        value, gate = self.expand(self.norm(pooled)).chunk(2, dim=-1)
        return self.contract(torch.nn.functional.gelu(value) * gate)


class AxialKernel(nn.Module):
    """A kernel matrix along one axis, from rotated queries and keys."""

    def __init__(self, width, heads, query_channels):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(width, heads * query_channels, bias=False)
        self.key = nn.Linear(width, heads * query_channels, bias=False)
        self.register_buffer("frequencies", 1 / (10000 ** (torch.arange(0, query_channels, 2).float() / query_channels)))
        nn.init.xavier_uniform_(self.query.weight, gain=0.02)
        nn.init.xavier_uniform_(self.key.weight, gain=0.02)

    def rotate(self, x, positions):
        """Rotary positional encoding along the axis."""
        phase = positions.flatten()[:, None] * 64 * self.frequencies[None]
        phase = torch.cat((phase, phase), -1)[None, None]
        first, second = x.chunk(2, dim=-1)
        return x * phase.cos() + torch.cat((-second, first), -1) * phase.sin()

    def forward(self, features, positions):
        # No softmax or 1/sqrt(Cq) in this integral kernel.
        q = rearrange(self.query(features), "b n (h c) -> b h n c", h=self.heads)
        k = rearrange(self.key(features), "b n (h c) -> b h n c", h=self.heads)
        return self.rotate(q, positions) @ self.rotate(k, positions).transpose(-1, -2)


class PAI(nn.Module):
    """Parallel axial integration; `film` optionally modulates the normalized field."""

    def __init__(self, width, heads, value_channels):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(width)
        self.features = nn.Linear(width, width, bias=False)
        self.values = nn.Linear(width, heads * value_channels, bias=False)
        self.axes = nn.ModuleList([AxisFeatures(width) for _ in range(3)])
        self.kernels = nn.ModuleList([AxialKernel(width, heads, 3 * value_channels) for _ in range(3)])
        self.spatial_norm = nn.InstanceNorm3d(3 * heads * value_channels)
        self.mix = nn.Sequential(nn.Linear(3 * heads * value_channels, width, bias=False), nn.GELU(), nn.Linear(width, width, bias=False))

    def forward(self, field, positions, film=None):
        u = self.norm(field)
        if film is not None:
            gamma, beta = film
            u = (1 + gamma[:, None, None, None]) * u + beta[:, None, None, None]
        z = self.features(u)
        v = rearrange(self.values(u), "b x y z (h c) -> b h x y z c", h=self.heads)
        kx, ky, kz = [kernel(axis(z, i + 1), pos) for i, (axis, kernel, pos) in enumerate(zip(self.axes, self.kernels, positions))]
        # Independent contractions along x, y and z, each (B,H,X,Y,Z,Cv).
        vx = torch.einsum("bhij,bhjklc->bhiklc", kx, v)
        vy = torch.einsum("bhij,bhkjlc->bhkilc", ky, v)
        vz = torch.einsum("bhij,bhkljc->bhklic", kz, v)
        combined = rearrange(torch.cat((vx, vy, vz), -1), "b h x y z c -> b (h c) x y z")
        combined = rearrange(self.spatial_norm(combined), "b c x y z -> b x y z c")
        return self.mix(combined)
