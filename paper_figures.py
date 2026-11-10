"""Paper x-y slices from the saved TEST predictions. Usage: python paper_figures.py runs/film"""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
from config import settings
from data import open_data, coordinates
from plot import save
import matplotlib.pyplot as plt


# Source indices start at zero. Saved predictions omit the initial field, so index
# n-1 is the prediction for source frame n. The last paper label is approximate.
FRAMES = (100, 500, 2000, 3999)
LABELS = (100, 500, 2000, 4000)


def channel_slices(prediction, truth, positions, frames, labels, output):
    """fDNS, prediction and error on the plane z = 2*pi/3, for one trajectory."""
    x, y, z = [p.flatten().cpu().numpy() for p in positions]
    plane = len(z) // 2
    p, t = prediction[:, :, :, plane, 0], truth[:, :, :, plane, 0]
    low, high = t.min(), t.max()
    error = p - t
    limit = np.abs(error).max()
    fig, axes = plt.subplots(3, len(frames), figsize=(12, 7))
    for row, values in enumerate((t, p, error)):
        for col, field in enumerate(values):
            ax = axes[row, col]
            vmin, vmax = (-limit, limit) if row == 2 else (low, high)
            # pcolormesh keeps the physical nonuniform wall-normal grid.
            im = ax.pcolormesh(x, y, field.T, shading="nearest", cmap="RdBu_r", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set(xlabel=r"$x/\delta$", ylabel=r"$y/\delta$", title=f"{('fDNS', 'Prediction', 'Error')[row]}, snapshot {labels[col]}")
    save(fig, output / "channel_xy_slices")
    np.savez(output / "channel_xy_slices.npz", source_frames=frames, paper_labels=labels, x=x, y=y, z=z[plane], prediction=p, reference=t)


def main(run, cfg):
    data = open_data(cfg)
    positions = coordinates(cfg)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42})
    frames = np.asarray(FRAMES)
    for test in sorted(run.glob("test_*")):
        for trajectory in cfg.test_ids:
            prediction = np.load(test / f"trajectory_{trajectory}.npy", mmap_mode="r")
            p, t = np.array(prediction[frames - 1]), np.array(data[trajectory, frames, ..., :3])
            output = run / "paper_figures" / test.name / f"trajectory_{trajectory}"
            output.mkdir(parents=True, exist_ok=True)
            channel_slices(p, t, positions, FRAMES, LABELS, output)
    print("Paper figures written from the saved TEST predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run, replace(settings, device="cpu"))
