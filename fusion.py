"""FA-CSF: cross-stride correction on top of frozen SC-IFF predictors."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from attention import PAI
from model import stride_code


def adjacent_pair(query):
    """The training strides around a query. Query 1 only has stride 1."""
    if query == 1:
        return 0, 1
    upper = 1 << (query - 1).bit_length()
    return upper // 2, upper


class Fusion(nn.Module):
    def __init__(self, cfg, stats, parents):
        """`parents` holds the frozen predictors: {"shared": model} or {"1": model, "2": model, ...}."""
        super().__init__()
        self.cfg = cfg
        self.parents = nn.ModuleDict(parents)
        self.parents.eval().requires_grad_(False)
        self.parents.zero_grad(set_to_none=True)  # Free the gradients left by Stage 1.
        self.register_buffer("state_mean", torch.tensor(stats["state_mean"], dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(stats["state_std"], dtype=torch.float32))
        self.register_buffer("rate_std", torch.tensor(stats["rate_std"], dtype=torch.float32).reshape(cfg.strides[-1], 1, -1, 1, 3))

        # Project the latent features of the lower and upper predictors.
        width, projected = cfg.fusion_width, cfg.projected_channels
        self.project_lower = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, projected))
        self.project_upper = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, projected))

        # Spatial branch (factorized attention) and temporal branch (embedding of the query).
        route_channels = 2 * projected + 2 * 3 + 3
        self.spatial_lift = nn.Sequential(nn.Linear(route_channels, width), nn.GELU())
        self.spatial = PAI(width, cfg.fusion_heads, cfg.fusion_value_channels)
        self.time_embedding = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 8))
        self.temporal_lift = nn.Sequential(nn.Linear(route_channels + 8, width), nn.GELU())

        # Correction head, zero at initialization: FA-CSF starts as linear interpolation.
        self.output_norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, 3)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def train(self, mode=True):
        super().train(mode)
        self.parents.eval()  # The predictors stay in evaluation mode.
        return self

    def native(self, state, stride, positions):
        """Run one frozen predictor; returns (prediction, normalized rate, latent)."""
        parent = self.parents["shared"] if "shared" in self.parents else self.parents[str(stride)]
        fn = lambda x: parent(x, stride, positions, features=True)
        # The parameters are frozen, but gradients still flow through the input.
        return checkpoint(fn, state, use_reentrant=True) if torch.is_grad_enabled() else fn(state)

    def block(self, state, positions, count=None):
        """Predict the next `count` fields from `state`, by default a whole interval."""
        count = self.cfg.strides[-1] if count is None else count
        queries = range(1, count + 1)

        # Each predictor runs once per interval; all queries reuse its output.
        needed = sorted({s for h in queries for s in adjacent_pair(h) if s})
        cache = {s: self.native(state, s, positions) for s in needed}

        normalized = (state - self.state_mean) / self.state_std
        outputs, spatial_cache = [], {}
        for h in queries:
            # Base field: linear interpolation between adjacent strides.
            lower, upper = adjacent_pair(h)
            high, high_rate, high_latent = cache[upper]
            if lower:
                low, low_rate, low_latent = cache[lower]
                base = ((upper - h) * low + (h - lower) * high) / (upper - lower)
            else:
                base = high

            # Spatial branch: shared by all queries between the same two strides.
            pair = (lower, upper)
            if pair not in spatial_cache:
                high_feature = self.project_upper(high_latent)
                low_feature = self.project_lower(low_latent) if lower else torch.zeros_like(high_feature)
                low_rate = low_rate if lower else torch.zeros_like(high_rate)
                route = torch.cat((low_feature, high_feature, low_rate, high_rate, normalized), -1)
                spatial_cache[pair] = route, self.spatial(self.spatial_lift(route), positions)
            route, spatial = spatial_cache[pair]

            # Temporal branch: depends on the query h.
            strides = state.new_full((len(state),), h)
            embedding = self.time_embedding(stride_code(strides, self.cfg.strides[-1]))
            embedding = embedding[:, None, None, None].expand(*state.shape[:-1], 8)
            temporal = self.temporal_lift(torch.cat((route, embedding), -1))

            # Learned correction added to the base field.
            rate_correction = self.readout(self.output_norm(spatial + temporal))
            outputs.append(base + h * self.cfg.dt * self.rate_std[h - 1] * rate_correction)
        return torch.stack(outputs, 1)  # (B, queries, X, Y, Z, 3).

    def parameters_to_train(self):
        return [p for p in self.parameters() if p.requires_grad]
