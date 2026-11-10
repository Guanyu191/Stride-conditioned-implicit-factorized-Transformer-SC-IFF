"""Flow statistics and model selection. These run on the CPU in float64."""
import numpy as np
import torch


def spectra(fields, cfg):
    """Streamwise and spanwise energy spectra of each frame, joined along the last axis."""
    result = []
    for start in range(0, len(fields), 16):
        u = torch.as_tensor(np.array(fields[start:start + 16]), dtype=torch.float64)
        parts = []
        for axis in (1, 3):
            power = 0.5 * torch.fft.rfft(u, dim=axis, norm="forward").abs().square().sum(-1)
            weight = torch.ones(cfg.grid[axis - 1] // 2 + 1, dtype=u.dtype)
            weight[1:-1] = 2
            average = power.mean((2, 3)) if axis == 1 else power.mean((1, 2))  # Arithmetic y average.
            parts.append((average * weight)[:, 1:].numpy())
        result.append(np.concatenate(parts, -1))
    return np.concatenate(result)


def own_acf(fields, cfg):
    """Temporal autocorrelation of a sequence about its own mean profile."""
    count = len(fields)
    nfft = 1 << (2 * count - 2).bit_length()
    numerator = np.zeros(count)
    means = fields.mean((0, 1, 3), dtype=np.float64)
    for y in range(cfg.grid[1]):  # Chunk over y to limit memory; every lag keeps all pairs.
        slab = np.array(fields[:, :, y], dtype=np.float64) - means[y]
        ft = np.fft.rfft(slab.reshape(count, -1), n=nfft, axis=0)
        numerator += np.fft.irfft((ft.real**2 + ft.imag**2).sum(1), n=nfft)[:count]
    covariance = numerator / np.arange(count, 0, -1)
    return covariance / covariance[0]


def inverse_covariance(reference, cfg):
    """Regularized inverse covariance of the log spectrum, used by `spectral_error`."""
    eigenvalues, vectors = np.linalg.eigh(np.cov(np.log(reference), rowvar=False, ddof=1))
    return (vectors / np.maximum(eigenvalues, eigenvalues[-1] * cfg.covariance_floor)) @ vectors.T


def spectral_error(pred, truth, inverse):
    """Covariance-weighted distance between time-averaged log spectra."""
    delta = np.log(pred.mean(0)) - np.log(truth.mean(0))
    return np.sqrt(max(0, float(delta @ inverse @ delta / len(delta))))


def acf_error(pred, truth, stride, cfg):
    """Relative error of the autocorrelation up to the maximum lag."""
    stop = min(len(pred), cfg.max_lag // stride + 1)
    p, t = pred[:stop], truth[:stop]
    integral = lambda a: ((a[1:] + a[:-1]) / 2).sum()
    return np.sqrt(integral((p - t) ** 2) / integral(t * t))


def choose(rows):
    """Keep the Pareto front, then take the smallest normalized worst error.

    Exact ties go to the smaller short VALID loss, then the earlier checkpoint.
    """
    values = np.array([[r["spectrum"], r["acf"]] for r in rows])
    front = [i for i, v in enumerate(values) if not any(np.all(w <= v) and np.any(w < v) for w in values)]
    scores = (values / values.min(0)).max(1)
    winner = min(front, key=lambda i: (scores[i], rows[i]["short_loss"], rows[i]["step"]))
    return rows[winner], dict(pareto=front, normalized_worst=scores.tolist())


def quadrature(y):
    """Trapezoidal weights on the nonuniform y grid."""
    y = np.asarray(y)
    weights = np.empty_like(y)
    weights[0], weights[-1] = (y[1] - y[0]) / 2, (y[-1] - y[-2]) / 2
    weights[1:-1] = (y[2:] - y[:-2]) / 2
    return weights / weights.sum()


def profiles(fields):
    """Mean velocity, RMS fluctuation and Reynolds shear stress as functions of y."""
    mean = fields.mean((0, 1, 3), dtype=np.float64)
    variance, stress = np.zeros_like(mean), np.zeros(len(mean))
    for part in np.array_split(fields, max(1, len(fields) // 16)):
        fluctuation = part.astype(np.float64) - mean[None, None, :, None]
        variance += (fluctuation**2).sum((0, 1, 3))
        stress += (fluctuation[..., 0] * fluctuation[..., 1]).sum((0, 1, 3))
    denominator = len(fields) * fields.shape[1] * fields.shape[3]
    return mean, np.sqrt(variance / denominator), stress / denominator


def field_histories(prediction, truth):
    """Per-frame relative L2 error and spatial correlation of the streamwise velocity."""
    errors, correlations = [], []
    for start in range(0, len(truth), 16):
        p, t = [np.asarray(u[start:start + 16], dtype=np.float64) for u in (prediction, truth)]
        errors.extend(np.linalg.norm((p - t).reshape(len(p), -1), axis=1) / np.linalg.norm(t.reshape(len(t), -1), axis=1))
        p, t = p[..., 0].reshape(len(p), -1), t[..., 0].reshape(len(t), -1)
        p, t = p - p.mean(1, keepdims=True), t - t.mean(1, keepdims=True)
        correlations.extend((p * t).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)))
    return np.asarray(errors), np.asarray(correlations)
