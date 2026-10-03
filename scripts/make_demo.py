"""Generate small, fast-loading demo GIFs for README + preview.html.

- demo_autotune.gif: closed-loop walk from high-voltage corner to the
  single-electron regime on a synthetic charge stability diagram.
- demo_rabi.gif: Rabi chevron building up row by row (Hamiltonian learning).

Keep under ~5 MB: small figure size, limited frames, Pillow writer.
Deterministic (seed 7).
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from qdotml.physics import generate_csd
from qdotml.data import build_dataset
from qdotml.models import train_line_detector
from qdotml.tuning import autotune_run
from qdotml.hamiltonian import rabi_chevron

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def make_autotune_gif():
    rng = np.random.default_rng(7)
    ds = build_dataset(seed=7, grid=96, n_devices=8, patches_per=40, patch_size=28)
    clf = train_line_detector(ds["X"], ds["y_line"], seed=7)
    csd = generate_csd(rng, grid=96, noise=1.0, device=ds["devices"][0])
    out = autotune_run(csd["signal"], csd["total"], clf, patch=28, target=1,
                       max_steps=120, verify=True,
                       rng=np.random.default_rng(123))
    traj = out["traj"]
    # thin trajectory to ~28 frames
    idx = np.linspace(0, len(traj) - 1, min(28, len(traj))).astype(int)
    frames = [traj[i] for i in idx]
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=110)
    fig.tight_layout()
    im = ax.imshow(csd["signal"], origin="lower", cmap="viridis")
    ax.set_title("Autotuning walk to 1 electron")
    ax.set_xlabel("Gate 1"); ax.set_ylabel("Gate 2")
    (line,) = ax.plot([], [], "w-o", ms=3, lw=1.5)
    dot, = ax.plot([], [], "ro", ms=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="sensor")

    xs = [j + 14 for (i, j) in frames]
    ys = [i + 14 for (i, j) in frames]

    def upd(k):
        line.set_data(xs[:k + 1], ys[:k + 1])
        dot.set_data([xs[k]], [ys[k]])
        ax.set_title(f"Autotuning walk to 1 electron  (step {k + 1}/{len(frames)})")
        return line, dot

    ani = FuncAnimation(fig, upd, frames=len(frames), interval=160, blit=False)
    out_p = RES / "demo_autotune.gif"
    ani.save(out_p, writer="pillow", fps=6)
    plt.close(fig)
    return out_p


def make_rabi_gif():
    B, t, P, _ = rabi_chevron(15, 2.0, 120, seed=0)
    nB = len(B)
    idx = np.linspace(0, nB - 1, 24).astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=110)
    fig.tight_layout()
    im = ax.imshow(np.full_like(P, 0.5), origin="lower", aspect="auto",
                   vmin=0, vmax=1, cmap="magma",
                   extent=[t[0], t[-1], B[0], B[-1]])
    ax.set_xlabel("burst time (ns)")
    ax.set_ylabel("magnetic field (T)")
    ax.set_title("Rabi chevron measurement")
    fig.colorbar(im, ax=ax, shrink=0.8, label="P")

    def upd(k):
        shown = np.full_like(P, np.nan)
        shown[:idx[k] + 1] = P[:idx[k] + 1]
        im.set_data(shown)
        ax.set_title(f"Rabi chevron measurement  ({idx[k] + 1}/{nB} rows)")
        return (im,)

    ani = FuncAnimation(fig, upd, frames=len(idx), interval=180, blit=False)
    out_p = RES / "demo_rabi.gif"
    ani.save(out_p, writer="pillow", fps=6)
    plt.close(fig)
    return out_p


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    a = make_autotune_gif()
    b = make_rabi_gif()
    for p in (a, b):
        print(p, round(p.stat().st_size / 1e6, 2), "MB")
