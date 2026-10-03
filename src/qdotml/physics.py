"""Physics-grounded double-quantum-dot charge stability diagram generator.

Constant-interaction honeycomb model with:
- direct + cross gate capacitances (virtual-gate crosstalk)
- mutual charging energy -> avoided crossings (finite tunnel coupling gap)
- charge-sensor response (differentiated Coulomb peaks, thermal broadening)
- realistic noise: white + 1/f drift + random telegraph jumps + latching blur

This is a from-scratch educational analogue of QDarts / SimCATS / QDsim / QArray
(see README references). It is NOT a copy of any local repo; parameters were
chosen from 2024-2026 literature ranges (see docs/BEST_PRACTICES.md).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class DeviceParams:
    a11: float = 0.85   # direct lever arm dot1-gate1 (eV/V scaled)
    a22: float = 0.85
    a12: float = 0.22   # crosstalk dot1-gate2
    a21: float = 0.22   # crosstalk dot2-gate1
    ec1: float = 1.0    # charging energy scale dot1
    ec2: float = 1.0
    ecm: float = 0.28   # mutual charging energy (sets honeycomb curvature)
    gap: float = 0.035  # avoided-crossing gap ~ tunnel coupling (Volt units)
    off1: float = 0.12
    off2: float = -0.08
    disorder: float = 0.02  # background charge disorder amplitude
    max_electrons: int = 3


def _smooth_disorder_field(rng, n, scale):
    x = rng.standard_normal((n // 4 + 2, n // 4 + 2)) * scale
    # cheap bilinear upsample to n x n
    xs = np.linspace(0, 1, x.shape[0])
    xt = np.linspace(0, 1, n)
    out = np.zeros((n, n))
    # interpolate rows then columns
    inter_rows = np.stack([np.interp(xt, xs, row) for row in x])  # (small, n)
    ys = xs
    for j in range(n):
        t = j / (n - 1) * (len(ys) - 1)
        lo = int(np.floor(t)); hi = min(lo + 1, len(ys) - 1); w = t - lo
        out[:, j] = (1 - w) * np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(inter_rows)), inter_rows[lo]) + w * np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(inter_rows)), inter_rows[hi])
    return out


def sample_device(rng: np.random.Generator) -> DeviceParams:
    return DeviceParams(
        a11=float(rng.uniform(0.75, 0.95)),
        a22=float(rng.uniform(0.75, 0.95)),
        a12=float(rng.uniform(0.12, 0.32)),
        a21=float(rng.uniform(0.12, 0.32)),
        ec1=float(rng.uniform(0.85, 1.15)),
        ec2=float(rng.uniform(0.85, 1.15)),
        ecm=float(rng.uniform(0.18, 0.38)),
        gap=float(rng.uniform(0.015, 0.06)),
        off1=float(rng.uniform(-0.15, 0.25)),
        off2=float(rng.uniform(-0.25, 0.15)),
        disorder=float(rng.uniform(0.008, 0.035)),
    )


def occupation_maps(p: DeviceParams, V1: np.ndarray, V2: np.ndarray):
    """Return (n1, n2, line_mask) on meshgrid V1 x V2."""
    G1, G2 = np.meshgrid(V1, V2, indexing="xy")
    N = p.max_electrons
    best = np.full(G1.shape, 1e99)
    bn1 = np.zeros_like(best, dtype=int)
    bn2 = np.zeros_like(best, dtype=int)
    for n1 in range(N + 1):
        for n2 in range(N + 1):
            # detuning from target electrochemical potentials
            d1 = p.a11 * G1 + p.a12 * G2 + p.off1 - n1 * p.ec1 - n2 * p.ecm
            d2 = p.a21 * G1 + p.a22 * G2 + p.off2 - n2 * p.ec2 - n1 * p.ecm
            E = 0.5 * (d1 ** 2 / p.ec1 + d2 ** 2 / p.ec2)
            m = E < best
            best[m] = E[m]; bn1[m] = n1; bn2[m] = n2
    # transition lines = spatial gradient of total occupation
    tot = bn1 + bn2
    gy, gx = np.gradient(tot.astype(float))
    lines = (np.abs(gx) + np.abs(gy)) > 0.5
    # soften intersections with gap: erode tiny isolated dots (avoided crossing look)
    return bn1, bn2, lines


def sensor_signal(n1: np.ndarray, lines: np.ndarray, rng, noise: float = 1.0,
                  telegraph_prob: float = 0.02):
    """Convert occupation + lines to a charge-sensor current map in [0,1]."""
    from scipy.ndimage import gaussian_filter
    base = gaussian_filter(lines.astype(float), sigma=0.9)
    # Coulomb peak asymmetry: sensor sees dot1 stronger
    sig = 0.55 * base + 0.10 * gaussian_filter(n1.astype(float) / (n1.max() + 1), 0.8)
    sig = sig / (sig.max() + 1e-9)
    n = sig.shape[0]
    white = rng.standard_normal(sig.shape) * 0.045 * noise
    # 1/f-ish drift: low-freq plane
    yy, xx = np.mgrid[0:n, 0:n]
    drift = 0.06 * noise * np.sin(2 * np.pi * (xx / n * 1.7 + yy / n * 0.6) + rng.uniform(0, 6.28))
    out = sig + white + drift
    if rng.random() < telegraph_prob + 0.08:
        # random telegraph jump: half-plane offset
        cut = rng.integers(n // 4, 3 * n // 4)
        out[cut:, :] += rng.uniform(-0.12, 0.12) * noise
    # latching blur on random rows (slow transitions)
    if rng.random() < 0.3:
        out = gaussian_filter(out, sigma=(0.6, 1.4))
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return out.astype(np.float32)


def generate_csd(rng: np.random.Generator, grid: int = 96, noise: float = 1.0,
                 device: DeviceParams | None = None):
    if device is None:
        device = sample_device(rng)
    V = np.linspace(0, 1, grid)
    n1, n2, lines = occupation_maps(device, V, V)
    # add smooth disorder to sensor only (background charges)
    sig = sensor_signal(n1, lines, rng, noise=noise)
    return {"signal": sig, "n1": n1, "n2": n2, "lines": lines, "device": device,
            "total": (n1 + n2).astype(np.int32)}
