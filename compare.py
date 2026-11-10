"""Compare IFF, the three SC-IFF variants and their FA-CSF versions on the TEST trajectories."""
import csv
import json
from pathlib import Path
import numpy as np
from config import settings
from plot import save
import matplotlib.pyplot as plt

MODELS = (("iff", "IFF"), ("concat", "SC-IFF-Concat"), ("indep", "SC-IFF-Indep"), ("film", "SC-IFF-FiLM"))
COLORS = ("#c62828", "#1565c0", "#202020", "#2e7d32", "#7b1fa2", "#ef6c00", "#00838f", "#6d4c41")


def average(folder, name, cfg):
    """Mean of one saved array over the TEST trajectories."""
    arrays = []
    for trajectory in cfg.test_ids:
        with np.load(folder / f"statistics_{trajectory}.npz") as values:
            arrays.append(values[name])
    return np.mean(arrays, axis=0)


def compare(cfg):
    root = Path(cfg.output_root)
    methods = [(label, root / model / "test_native") for model, label in MODELS]
    methods += [(label + " + FA-CSF", root / model / "test_fa_csf") for model, label in MODELS[1:]]
    output = root / "comparison"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10, "pdf.fonttype": 42})

    # Error table, averaged over the TEST trajectories.
    summaries = [dict(model=title, **json.loads((folder / "summary.json").read_text())["mean"]) for title, folder in methods]
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    # Time-averaged energy spectra.
    reference_folder = methods[0][1]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    spectra = [("fDNS", average(reference_folder, "spectrum_reference", cfg).mean(0))]
    spectra += [(title, average(folder, "spectrum_prediction", cfg).mean(0)) for title, folder in methods]
    cut = cfg.grid[0] // 2
    for direction, axis in enumerate(axes):
        for color, (title, energy) in zip(COLORS, spectra):
            energy = energy[:cut] if direction == 0 else energy[cut:]
            axis.loglog(np.arange(1, len(energy) + 1), energy, color=color, label=title)
        axis.set(xlabel="Streamwise mode" if direction == 0 else "Spanwise mode", ylabel="Mean absolute energy")
        axis.legend(frameon=False, fontsize=8)
    save(fig, output / "mean_spectra")

    # Field errors over time.
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    time = np.arange(1, cfg.frames) * cfg.dt
    for color, (title, folder) in zip(COLORS[1:], methods):
        axes[0].plot(time, average(folder, "field_relative_l2", cfg), color=color, label=title)
        axes[1].plot(time, average(folder, "streamwise_pearson", cfg), color=color, label=title)
    for axis, label in zip(axes, ("Mean field relative L2", "Mean streamwise Pearson correlation")):
        axis.set(xlabel="Physical time from seed", ylabel=label)
        axis.legend(frameon=False, fontsize=8)
    save(fig, output / "field_errors")

    # Mean velocity and Reynolds shear stress profiles.
    with np.load(reference_folder / f"statistics_{cfg.test_ids[0]}.npz") as values:
        y = values["y"]
    curves = [("fDNS", reference_folder, "reference")] + [(title, folder, "prediction") for title, folder in methods]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for color, (title, folder, role) in zip(COLORS, curves):
        axes[0].plot(y, average(folder, "mean_" + role, cfg)[:, 0], color=color, label=title)
        axes[1].plot(y, average(folder, "stress_" + role, cfg), color=color, label=title)
    for axis, label in zip(axes, (r"$\overline{u}$", r"$\overline{u'v'}$")):
        axis.set(xlabel=r"$y$", ylabel=label)
        axis.legend(frameon=False, fontsize=8)
    save(fig, output / "mean_and_stress")

    # RMS velocity fluctuation profiles.
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for color, (title, folder, role) in zip(COLORS, curves):
        rms = average(folder, "rms_" + role, cfg)
        for component, axis in enumerate(axes):
            axis.plot(y, rms[:, component], color=color, label=title)
    for component, axis in enumerate(axes):
        axis.set(xlabel=r"$y$", ylabel="uvw"[component] + " fluctuation RMS")
    axes[1].legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    save(fig, output / "rms_profiles")
    print(f"Comparison written to {output}")


if __name__ == "__main__":
    compare(settings)
