"""Dataset construction: full diagrams -> labelled patches, group-safe splits."""
from __future__ import annotations
import numpy as np
from .physics import generate_csd, sample_device


PATCH_LABELS = ["no-line", "line"]  # W1/W2 patch classifier (Yon et al. 2024 style)
STATE_LABELS = ["barrier", "single", "double"]  # W1 configuration classifier


def state_of(n1tot: int) -> int:
    if n1tot <= 0:
        return 0
    if n1tot == 1:
        return 1
    return 2


def extract_patches(csd: dict, rng, n_patches=40, patch=28):
    sig = csd["signal"]; lines = csd["lines"]; tot = csd["total"]
    H, W = sig.shape
    X, y_line, y_state, meta = [], [], [], []
    for _ in range(n_patches):
        i = rng.integers(0, H - patch); j = rng.integers(0, W - patch)
        p = sig[i:i+patch, j:j+patch]
        lm = lines[i:i+patch, j:j+patch]
        frac = lm.mean()
        # lines are 1-px thin (~2-3% of patch); 0.015 captures single crossings
        y_line.append(1 if frac > 0.015 else 0)
        center_tot = int(tot[i+patch//2, j+patch//2])
        y_state.append(state_of(center_tot))
        # gradient-magnitude feature kept raw; models flatten
        X.append(p.astype(np.float32))
        meta.append({"i": int(i), "j": int(j), "line_frac": float(frac),
                     "total": center_tot})
    return np.stack(X), np.array(y_line), np.array(y_state), meta


def build_dataset(seed=7, grid=96, n_devices=24, patches_per=40, patch_size=28,
                  noise=1.0):
    rng = np.random.default_rng(seed)
    Xs, Yl, Ys, Dev, Metas = [], [], [], [], []
    devices = []
    for d in range(n_devices):
        dev = sample_device(rng)
        devices.append(dev)
        csd = generate_csd(rng, grid=grid, noise=noise, device=dev)
        X, yl, ys, meta = extract_patches(csd, rng, n_patches=patches_per,
                                          patch=patch_size)
        Xs.append(X); Yl.append(yl); Ys.append(ys)
        Dev += [d] * len(yl)
        for m in meta:
            m["device"] = d
            Metas.append(m)
    return {"X": np.concatenate(Xs), "y_line": np.concatenate(Yl),
            "y_state": np.concatenate(Ys), "device": np.array(Dev),
            "metas": Metas, "devices": devices,
            "patch_size": patch_size}
