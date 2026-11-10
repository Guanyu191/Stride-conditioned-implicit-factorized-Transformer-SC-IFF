"""VALID selects parameters; TEST is evaluated only after selection."""
import json
from pathlib import Path
import numpy as np
import torch
from data import two_step_batch
from losses import two_step_loss, relative_l2
from rollout import forecast
import metrics


def references(data, cfg, directory):
    """Inverse covariances of the x and z spectra, estimated once from VALID fields."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    full = np.concatenate([metrics.spectra(np.asarray(data[t, 1:, ..., :3]), cfg) for t in cfg.valid_ids])
    cut = cfg.grid[0] // 2
    return directory, [metrics.inverse_covariance(full[:, :cut], cfg), metrics.inverse_covariance(full[:, cut:], cfg)]


def reference_phase(data, cfg, directory, trajectory, stride, phase):
    """Spectrum and autocorrelation of one reindexed VALID sequence, cached on disk."""
    path = directory / f"t{trajectory}_s{stride}_p{phase}.npz"
    if not path.exists():
        field = np.asarray(data[trajectory, phase + stride::stride, ..., :3])
        np.savez(path, spectrum=metrics.spectra(field, cfg), acf=metrics.own_acf(field, cfg))
    with np.load(path) as saved:
        return saved["spectrum"], saved["acf"]


def scores(prediction, reference_spectrum, reference_acf, inverse, stride, cfg):
    """The two selection scores of one predicted sequence."""
    spectrum = metrics.spectra(prediction, cfg)
    cut = cfg.grid[0] // 2
    dx = metrics.spectral_error(spectrum[:, :cut], reference_spectrum[:, :cut], inverse[0])
    dz = metrics.spectral_error(spectrum[:, cut:], reference_spectrum[:, cut:], inverse[1])
    return float(np.sqrt((dx**2 + dz**2) / 2)), float(metrics.acf_error(metrics.own_acf(prediction, cfg), reference_acf, stride, cfg))


@torch.no_grad()
def short_validation(model, data, cfg, positions, strides, fusion=None):
    """Two-step VALID loss from evenly spaced origins, used only to break exact ties."""
    total, count = 0.0, 0
    for s in strides:
        starts = np.unique(np.linspace(0, cfg.frames - 2 * s - 1, cfg.short_origins, dtype=int))
        rows = np.array([(t, n, s) for t in cfg.valid_ids for n in starts])
        for start in range(0, len(rows), cfg.stage1_batch):
            values, hs = two_step_batch(data, rows[start:start + cfg.stage1_batch], cfg.device)
            if fusion is None:
                loss = two_step_loss(model, values, hs, positions)
            else:
                first = fusion.block(values[0], positions, 1)[:, 0]
                second = fusion.block(first, positions, 1)[:, 0]
                loss = (relative_l2(first, values[1]) + relative_l2(second, values[2])) / 2
            total += float(loss) * len(hs)
            count += len(hs)
    return total / count


def validate(parents, data, cfg, positions, reference, stride=1, fusion=None):
    """Score a candidate on complete VALID sequences, over all phases of the stride."""
    directory, inverse = reference
    rows = []
    for trajectory in cfg.valid_ids:
        for phase in range(stride):
            initial = torch.tensor(np.array(data[trajectory, phase:phase + 1, ..., :3]), device=cfg.device)
            frames = (cfg.frames - 1 - phase) // stride
            prediction = forecast(parents, initial, positions, frames, stride, fusion)
            spectrum, acf = reference_phase(data, cfg, directory, trajectory, stride, phase)
            d, a = scores(prediction, spectrum, acf, inverse, stride, cfg)
            rows.append(dict(trajectory=trajectory, phase=phase, spectrum=d, acf=a))
    # Average the phases of each trajectory first, then average the trajectories.
    by_trajectory = [[np.mean([r[key] for r in rows if r["trajectory"] == t]) for key in ("spectrum", "acf")] for t in cfg.valid_ids]
    average = np.asarray(by_trajectory).mean(0)
    return dict(spectrum=float(average[0]), acf=float(average[1]), phases=rows)


def test(parents, data, cfg, positions, reference, output, fusion=None):
    """Predict each TEST trajectory from its first field; save predictions and statistics."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    y = positions[1].flatten().cpu().numpy()
    weights = metrics.quadrature(y)
    cut = cfg.grid[0] // 2
    rows = []
    for trajectory in cfg.test_ids:
        initial = torch.tensor(np.array(data[trajectory, :1, ..., :3]), device=cfg.device)
        prediction = forecast(parents, initial, positions, cfg.frames - 1, fusion=fusion)
        truth = np.asarray(data[trajectory, 1:, ..., :3])
        np.save(output / f"trajectory_{trajectory}.npy", prediction)

        # Selection scores and field errors.
        reference_spectrum = metrics.spectra(truth, cfg)
        d, a = scores(prediction, reference_spectrum, metrics.own_acf(truth, cfg), reference[1], 1, cfg)
        arrays = dict(spectrum_prediction=metrics.spectra(prediction, cfg), spectrum_reference=reference_spectrum,
                      dt=cfg.dt, y=y, spectrum_cut=cut)
        arrays["field_relative_l2"], arrays["streamwise_pearson"] = metrics.field_histories(prediction, truth)
        row = dict(trajectory=trajectory, spectrum=d, acf=a, mean_field_relative_l2=float(arrays["field_relative_l2"].mean()))

        # Relative errors of the time-averaged spectra along x and z.
        p_mean, t_mean = arrays["spectrum_prediction"].mean(0), arrays["spectrum_reference"].mean(0)
        for name, part in (("spectrum_x_relative_l2", slice(None, cut)), ("spectrum_z_relative_l2", slice(cut, None))):
            row[name] = float(np.linalg.norm(p_mean[part] - t_mean[part]) / np.linalg.norm(t_mean[part]))

        # Relative errors of the wall-normal profiles, with quadrature weights in y.
        for name, p, t in zip(("mean", "rms", "stress"), metrics.profiles(prediction), metrics.profiles(truth)):
            arrays[name + "_prediction"], arrays[name + "_reference"] = p, t
            if name == "mean":
                p, t = p[:, 0], t[:, 0]  # The paper reports the mean streamwise velocity.
            w = weights[:, None] if p.ndim == 2 else weights
            row[name] = float(np.sqrt(((p - t)**2 * w).sum() / (t**2 * w).sum()))
        np.savez(output / f"statistics_{trajectory}.npz", **arrays)

        # Four x-z slices at the centre of the channel, for a quick visual check.
        frames = np.unique(np.linspace(0, len(truth) - 1, 4, dtype=int))
        center = cfg.grid[1] // 2
        np.savez(output / f"slices_{trajectory}.npz", source_frames=frames + 1, dt=cfg.dt,
                 prediction=prediction[frames, :, center, :, 0], reference=truth[frames, :, center, :, 0])
        rows.append(row)
        print(json.dumps(dict(test=row)), flush=True)
    summary = dict(trajectories=rows, mean={k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "trajectory"})
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
