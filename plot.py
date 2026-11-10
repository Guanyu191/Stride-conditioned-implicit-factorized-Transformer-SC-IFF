"""Plot the saved results of one run. Usage: python plot.py runs/film"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter


def save(fig, destination):
    """Save a figure as PNG and PDF, with readable ticks on short log axes."""
    for axis in fig.axes:
        if axis.get_xscale() == "log":
            axis.xaxis.set_major_locator(LogLocator(base=2))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            axis.xaxis.set_minor_formatter(NullFormatter())
        if axis.get_yscale() == "log":
            axis.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def plot_loss(path, destination):
    rows = json.loads(path.read_text())
    key = "train_loss" if "train_loss" in rows[0] else "loss"
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot([r["step"] for r in rows], [r[key] for r in rows], color="black")
    ax.set(xlabel="Epoch" if key == "train_loss" else "Update", ylabel="Training loss")
    save(fig, destination)


def plot_spectrum(values, destination):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    cut = int(values["spectrum_cut"])
    for direction, ax in enumerate(axes):
        for role, color in (("reference", "#c62828"), ("prediction", "#1565c0")):
            energy = values["spectrum_" + role].mean(0)
            energy = energy[:cut] if direction == 0 else energy[cut:]
            ax.loglog(np.arange(1, len(energy) + 1), energy, color=color, label=role)
        ax.set(xlabel=r"$k_x L_x/(2\pi)$" if direction == 0 else r"$k_z L_z/(2\pi)$", ylabel="Mean absolute energy")
        ax.legend(frameon=False)
    save(fig, destination)


def plot_field_errors(values, destination):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    time = np.arange(1, len(values["field_relative_l2"]) + 1) * float(values["dt"])
    for axis, name, label in zip(axes, ("field_relative_l2", "streamwise_pearson"), ("Velocity field relative L2", "Streamwise Pearson correlation")):
        axis.plot(time, values[name], color="#1565c0")
        axis.set(xlabel="Physical time from seed", ylabel=label)
    save(fig, destination)


def plot_profiles(values, destination):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for role, color in (("reference", "#c62828"), ("prediction", "#1565c0")):
        axes[0].plot(values["y"], values["mean_" + role][:, 0], color=color, label=role)
        for component, style in enumerate(("-", "--", ":")):
            axes[1].plot(values["y"], values["rms_" + role][:, component], color=color, linestyle=style, label=role + " " + "uvw"[component])
        axes[2].plot(values["y"], values["stress_" + role], color=color, label=role)
    for ax, label in zip(axes, (r"$\overline{u}$", "Fluctuation RMS", r"$\overline{u'v'}$")):
        ax.set(xlabel=r"$y$", ylabel=label)
        ax.legend(frameon=False, fontsize=8)
    save(fig, destination)


def plot_slices(values, destination):
    """Reference, prediction and error at four frames, on shared color limits."""
    count = len(values["source_frames"])
    fig, axes = plt.subplots(count, 3, figsize=(8, 2 * count))
    low, high = values["reference"].min(), values["reference"].max()
    limit = np.abs(values["prediction"] - values["reference"]).max()
    for i, frame in enumerate(values["source_frames"]):
        fields = (values["reference"][i], values["prediction"][i], values["prediction"][i] - values["reference"][i])
        for j, field in enumerate(fields):
            vmin, vmax = (-limit, limit) if j == 2 else (low, high)
            im = axes[i, j].imshow(field.T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
            axes[i, j].set(xlabel="x grid index", ylabel="z grid index", title=f"{('Reference', 'Prediction', 'Error')[j]}, frame {frame}")
            fig.colorbar(im, ax=axes[i, j], shrink=0.85)
    save(fig, destination)


def plot_run(root):
    root = Path(root)
    output = root / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                         "pdf.fonttype": 42, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    for path in root.glob("stage*/history.json"):
        plot_loss(path, output / (path.parent.name + "_loss"))
    for path in root.glob("test*/statistics_*.npz"):
        stem = path.parent.name + "_" + path.stem
        with np.load(path) as values:
            plot_spectrum(values, output / (stem + "_spectrum"))
            plot_field_errors(values, output / (stem + "_field_errors"))
            plot_profiles(values, output / (stem + "_profiles"))
    for path in root.glob("test*/slices_*.npz"):
        with np.load(path) as values:
            plot_slices(values, output / (path.parent.name + "_" + path.stem))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    plot_run(parser.parse_args().run)
