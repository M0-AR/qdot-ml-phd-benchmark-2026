"""Hamiltonian learning fine-tuning (W4): DQD Hubbard + EDSR Rabi chevron fit.

Model: detuning eps (ueV), tunnel coupling tc, g-factor, T2*.
Resonance f(eps) = sqrt(eps^2 + (2tc)^2) * k + g muB B / h (simplified, ueV->GHz).
Rabi chevron P(t,B) = sin^2(pi f(B) t) exp(-(t/T2*)^2) + noise.
Fit recovers (tc, g, T2*) via least squares on peak positions + decay.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit


MU_B_H = 13.996  # GHz/T (Bohr magneton / Planck)


def rabi_chevron(tc_uev, g, T2_ns, B0=0.5, nB=24, nt=60, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    B = np.linspace(B0 - 0.15, B0 + 0.15, nB)
    t = np.linspace(0, 200, nt)  # ns
    # Rabi frequency in 1/ns (= GHz): few x10 MHz, slope d f/dB ∝ g,
    # intercept ∝ tc (exchange). Period ~20-50 ns so 0-200 ns resolves it.
    # f_i = 0.030 + 0.020*g*(B_i-B0) + 0.015*(tc/30)
    P = np.zeros((nB, nt))
    f = np.zeros(nB)
    for i in range(nB):
        fi = 0.030 + 0.020 * g * (B[i] - B0) + 0.015 * (tc_uev / 30.0)
        f[i] = fi
        P[i] = np.sin(np.pi * fi * t) ** 2 * np.exp(-(t / T2_ns) ** 2)
    P += rng.standard_normal(P.shape) * noise
    return B, t, np.clip(P, 0, 1), f


def fit_hamiltonian(B, t, P):
    """Recover g-slope from f(B) + T2* from envelope. Returns g-like slope."""
    nB = len(B)
    # peak freq per B via FFT
    freqs = []
    for i in range(nB):
        tr = P[i] - P[i].mean()
        spec = np.abs(np.fft.rfft(tr))
        fq = np.fft.rfftfreq(len(tr), d=(t[1] - t[0]))
        # Rabi band for our toy model
        m = (fq > 0.005) & (fq < 0.15)
        freqs.append(float(fq[m][np.argmax(spec[m])]) if m.sum() else 0.0)
    freqs = np.array(freqs)
    # linear fit f = slope*B + intercept; slope = 0.020*g -> g_hat = slope/0.020
    A = np.stack([B, np.ones_like(B)], axis=1)
    coef, *_ = np.linalg.lstsq(A, freqs, rcond=None)
    slope_hat, J_hat = float(coef[0]), float(coef[1])
    g_hat = float(slope_hat / 0.020)
    # T2* from central-field trace: fix f from FFT, fit decaying sinusoid.
    # P(t) = a*sin(pi*f*t)^2*exp(-(t/T2)^2)+c captures Rabi + decoherence.
    mid = nB // 2
    f0 = float(freqs[mid]) if freqs[mid] > 0 else float(np.median(freqs))
    y = P[mid]
    def model2(tt, T2, a, c):
        return a * (np.sin(np.pi * f0 * tt) ** 2) * np.exp(-((tt / T2) ** 2)) + c
    try:
        popt, _ = curve_fit(model2, t, y, p0=[120, float(y.max() - y.min()), float(y.min())],
                            maxfev=12000,
                            bounds=([20, 0, -np.inf], [600, 2.0, np.inf]))
        T2_hat = float(popt[0])
    except Exception:
        T2_hat = float("nan")
    return {"g_hat": g_hat, "J_hat": J_hat, "T2_hat": T2_hat,
            "freqs": freqs}
