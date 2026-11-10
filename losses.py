"""Training losses on physical velocities of shape (B, T, X, Y, Z, 3)."""
import torch


def relative_l2(pred, truth):
    return ((pred - truth).flatten(1).norm(dim=1) / truth.flatten(1).norm(dim=1)).mean()


def two_step_loss(model, values, strides, positions):
    """Stage 1 loss over two recursive steps: the mean of two relative norms, not squared."""
    initial, first_truth, second_truth = values
    first = model(initial, strides, positions)
    second = model(first, strides, positions)  # Backpropagates through both steps.
    return (relative_l2(first, first_truth) + relative_l2(second, second_truth)) / 2


def moments(u):
    """Mean profile, fluctuation and variance profile over time, x and z."""
    axes = (1, 2, 4)
    mean = u.mean(axes, keepdim=True)
    fluctuation = u - mean  # Prediction and reference each use their own mean.
    return mean, fluctuation, fluctuation.square().mean(axes)


def directional_spectrum(u, axis):
    """One-sided energy spectrum along x (axis=2) or z (axis=4) at each y."""
    n = u.shape[axis]
    coeff = torch.fft.rfft(u, dim=axis, norm="forward")
    weight = u.new_full((n // 2 + 1,), 2)
    weight[0] = weight[-1] = 1
    shape = [1] * u.ndim
    shape[axis] = len(weight)
    power = coeff.abs().square() * weight.reshape(shape)
    return power.mean((1, 4)) if axis == 2 else power.mean((1, 2)).transpose(1, 2)


def shape_loss(pred_fluctuation, truth_fluctuation, pred_var, truth_var, scale, cfg):
    """Compare the normalized spectra along x and z."""
    result = []
    for axis in (2, 4):
        p, t = directional_spectrum(pred_fluctuation, axis), directional_spectrum(truth_fluctuation, axis)
        epsilon = cfg.spectrum_smoothing * scale.square() / p.shape[1]
        p = (p + epsilon) / (pred_var[:, None] + p.shape[1] * epsilon)
        t = (t + epsilon) / (truth_var[:, None] + t.shape[1] * epsilon)
        reliable = truth_var / (truth_var + cfg.energy_smoothing * scale.square())
        band = p.new_ones(p.shape[1])
        band[len(band) // 2:] = 2
        band = band / band.mean()
        result.append((band[None, :, None, None] * reliable[:, None] * (p / t).log().square()).mean())
    return sum(result) / 2


def segment_energy(u, cfg):
    """Variance about the time/x/z mean profile of each segment; (B, segments, 3)."""
    values = u.reshape(len(u), -1, cfg.segment, *u.shape[2:])
    values = values - values.mean((2, 3, 5), keepdim=True)
    return values.square().mean((2, 3, 4, 5))


def trajectory_loss(pred, truth, scale, cfg):
    """Stage 2 objective: six terms, all with coefficient one."""
    pred_mean, pred_fluctuation, pred_var = moments(pred)
    truth_mean, truth_fluctuation, truth_var = moments(truth)
    axes = (1, 2, 4)
    pred_stress = (pred_fluctuation[..., 0] * pred_fluctuation[..., 1]).mean(axes)
    truth_stress = (truth_fluctuation[..., 0] * truth_fluctuation[..., 1]).mean(axes)
    epsilon = cfg.energy_smoothing * scale.square()
    terms = dict(
        trajectory=relative_l2(pred.flatten(0, 1), truth.flatten(0, 1)),
        mean=((pred_mean - truth_mean) / scale).square().mean(),
        variance=((pred_var - truth_var) / scale.square()).square().mean(),
        stress=((pred_stress - truth_stress) / (scale[0] * scale[1])).square().mean(),
        shape=shape_loss(pred_fluctuation, truth_fluctuation, pred_var, truth_var, scale, cfg),
        energy=((segment_energy(pred, cfg) + epsilon) / (segment_energy(truth, cfg) + epsilon)).log().square().mean(),
    )
    return sum(terms.values()), terms
