"""Figures: confusion matrices, training curves, sample step signals."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C


def confusion(cm, names, title, path):
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * n, 0.8 + 0.7 * n))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def training_curves(histories, path):
    """histories: {label: history dict}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, target in zip(axes, ("class", "terrain")):
        for label, h in histories.items():
            if not label.endswith(target):
                continue
            ep = np.arange(1, len(h["train_acc"]) + 1)
            line, = ax.plot(ep, h["val_acc"], label=f"{label.rsplit('/', 1)[0]} val")
            ax.plot(ep, h["train_acc"], "--", color=line.get_color(), alpha=0.5)
        ax.set_title(f"{target} classifier (dashed = train)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def step_signals(ds, path):
    """One representative step per terrain: 5 normal taxels + shear."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    t = np.arange(C.N_TIMESTEPS) / 217.0 * 1000
    for k, ax in enumerate(axes.flat):
        i = np.where(ds.terrain == k)[0][0]
        for ch in range(C.N_CHANNELS):
            ax.plot(t, ds.raw[i, ch], lw=1.2 if ch < 5 else 2, ls="-" if ch < 5 else "--",
                    label=f"Ch{ch + 1}" + (" (shear)" if ch == 5 else ""))
        ax.set_title(f"{C.TERRAINS[k]} [{C.TERRAIN_TO_CLASS[C.TERRAINS[k]]}]")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("time in step (ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
