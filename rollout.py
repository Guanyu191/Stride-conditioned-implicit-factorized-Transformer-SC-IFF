"""Autoregressive prediction. Reference fields enter only at initialization."""
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def training_rollout(model, initial, positions, frames):
    """Stage 2 rollout: chain FA-CSF intervals, keeping gradients through the feedback."""
    state = initial.requires_grad_(True)
    blocks = []
    for _ in range(frames // model.cfg.strides[-1]):
        # Recompute each interval during backpropagation to save memory.
        block = checkpoint(lambda x: model.block(x, positions), state, use_reentrant=True)
        state = block[:, -1]  # Feed back the corrected last field; never detach.
        blocks.append(block)
    return torch.cat(blocks, 1)


@torch.no_grad()
def forecast(parents, initial, positions, frames, stride=1, fusion=None):
    """Free-running prediction; returns the predicted fields without the initial one.

    With `fusion`, FA-CSF generates one interval at a time. Otherwise `parents`
    maps each stride to a predictor and hierarchical prediction is used. `stride`
    sets the spacing of the output frames, which VALID uses for reindexed sequences.
    """
    if fusion is not None:
        result, state = [], initial
        for start in range(0, frames, fusion.cfg.strides[-1]):
            block = fusion.block(state, positions, min(fusion.cfg.strides[-1], frames - start))
            if not bool(torch.isfinite(block).all()):
                raise FloatingPointError("Free-running prediction diverged")
            result.append(block[0].cpu().numpy())
            state = block[:, -1]
        return np.concatenate(result)

    # Hierarchical prediction: fill coarse anchors first, then the gaps with smaller strides.
    fields = np.empty((frames + 1, *initial.shape[1:]), dtype=np.float32)
    fields[0] = initial[0].cpu().numpy()
    ready = np.zeros(frames + 1, bool)
    ready[0] = True
    for s in sorted(parents, reverse=True):
        for target in range(s, frames * stride + 1, s):
            index, source = target // stride, (target - s) // stride
            if ready[index]:
                continue
            state = torch.as_tensor(fields[source:source + 1], device=initial.device)
            value = parents[s](state, s, positions)
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("Free-running prediction diverged")
            fields[index] = value[0].cpu().numpy()
            ready[index] = True
    return fields[1:]
