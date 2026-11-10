"""Read the velocity array and build training samples.

The array has axes (trajectory, time, x, y, z, component). All normalization
statistics come from the TRAIN trajectories.
"""
import json
import numpy as np
import torch


def open_data(cfg):
    return np.load(cfg.data_path, mmap_mode="r")


def coordinates(cfg):
    """Return the x, y and z grid coordinates as column tensors."""
    with open(cfg.coordinates_path) as f:
        grid = json.load(f)
    return [torch.tensor(grid[axis], dtype=torch.float32, device=cfg.device)[:, None] for axis in "xyz"]


def update_moments(count, mean, moment2, values):
    """Merge one chunk into running moments; the full array does not fit in memory."""
    size = len(values)
    chunk_mean = values.mean(0)
    chunk_moment2 = ((values - chunk_mean) ** 2).sum(0)
    delta = chunk_mean - mean
    total = count + size
    return total, mean + delta * size / total, moment2 + chunk_moment2 + delta**2 * count * size / total


def fit_statistics(data, cfg):
    ny = cfg.grid[1]

    # Velocity statistics: input normalization and the fixed scale of
    # the statistical losses, which is the RMS fluctuation about the mean profile.
    count, mean, moment2 = 0, np.zeros(3), np.zeros(3)
    profile_count, profile_mean, profile_moment2 = 0, np.zeros((ny, 3)), np.zeros((ny, 3))
    for trajectory in cfg.train_ids:
        for start in range(0, cfg.frames, cfg.stats_chunk):
            u = np.array(data[trajectory, start:start + cfg.stats_chunk, ..., :3], dtype=np.float64)
            count, mean, moment2 = update_moments(count, mean, moment2, u.reshape(-1, 3))
            by_y = u.transpose(0, 1, 3, 2, 4).reshape(-1, ny, 3)
            profile_count, profile_mean, profile_moment2 = update_moments(profile_count, profile_mean, profile_moment2, by_y)

    # Velocity-change rate at every stride, as a function of y.
    rate_mean, rate_std = [], []
    for stride in range(1, cfg.strides[-1] + 1):
        n, mu, variance_sum = 0, np.zeros((ny, 3)), np.zeros((ny, 3))
        for trajectory in cfg.train_ids:
            for start in range(0, cfg.frames - stride, cfg.stats_chunk):
                stop = min(start + cfg.stats_chunk, cfg.frames - stride)
                u = np.array(data[trajectory, start:stop, ..., :3], dtype=np.float64)
                v = np.array(data[trajectory, start + stride:stop + stride, ..., :3], dtype=np.float64)
                rate = (v - u) / (stride * cfg.dt)
                n, mu, variance_sum = update_moments(n, mu, variance_sum, rate.transpose(0, 1, 3, 2, 4).reshape(-1, ny, 3))
        rate_mean.append(mu)
        rate_std.append(np.sqrt(variance_sum / n))
        print(f"TRAIN statistics: stride {stride}/{cfg.strides[-1]}", flush=True)

    return dict(state_mean=mean.tolist(), state_std=np.sqrt(moment2 / count).tolist(),
                rate_mean=np.asarray(rate_mean).tolist(), rate_std=np.asarray(rate_std).tolist(),
                loss_scale=np.sqrt((profile_moment2 / profile_count).mean(0)).tolist())


def origins(cfg, strides, ids):
    """Every legal (trajectory, start, stride), covering all phases."""
    return np.asarray([(t, n, s) for s in strides for t in ids for n in range(cfg.frames - 2 * s)], dtype=np.int64)


def two_step_batch(data, rows, device):
    """Stage 1 sample: the fields at n, n+s and n+2s, plus the strides."""
    values = [torch.as_tensor(np.stack([data[t, n + i * s, ..., :3] for t, n, s in rows]), device=device) for i in range(3)]
    return values, torch.as_tensor(rows[:, 2], device=device)


def trajectory_batch(data, cfg, rng):
    """Stage 2 sample: an initial field and the `cfg.window` true fields after it."""
    ts = rng.choice(cfg.train_ids, cfg.stage2_batch)
    ns = rng.randint(0, cfg.frames - cfg.window, cfg.stage2_batch)
    initial = np.stack([data[t, n, ..., :3] for t, n in zip(ts, ns)])
    truth = np.stack([data[t, n + 1:n + cfg.window + 1, ..., :3] for t, n in zip(ts, ns)])
    return torch.as_tensor(initial, device=cfg.device), torch.as_tensor(truth, device=cfg.device), list(zip(ts.tolist(), ns.tolist()))
