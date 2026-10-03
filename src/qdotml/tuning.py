"""Closed-loop charge autotuning with uncertainty (W2, Yon et al. 2024 style).

Strategy: start at high voltages (many electrons), step diagonally toward
(0,0); at each step classify patch; count line crossings; stop when estimated
total == target (1 electron). Low-confidence predictions trigger a local
verification scan of neighbouring patches (robustness ablation).
"""
from __future__ import annotations
import numpy as np
from .models import sobel_features


def _patch_at(signal, i, j, patch):
    H, W = signal.shape
    i = int(np.clip(i, 0, H - patch)); j = int(np.clip(j, 0, W - patch))
    return signal[i:i+patch, j:j+patch], i, j


def autotune_run(signal, total_map, clf, patch=28, target=1, max_steps=120,
                 conf_thresh=0.6, verify=True, rng=None):
    H, W = signal.shape
    # deterministic high-occupation start (bottom-right), jittered by rng
    jr = float(rng.random()) if rng is not None else 0.5
    jr2 = float(rng.random()) if rng is not None else 0.5
    i = int(H - patch - 2 - jr * (H * 0.12))
    j = int(W - patch - 2 - jr2 * (W * 0.12))
    # classifier-driven budget: typical starts hold 2-3 electrons (max 6);
    # each detected line crossing decrements the estimate (no oracle used).
    # Walk ends when budget is spent OR the scan edge is reached; success is
    # judged on the true map with tolerance (reached low-electron regime).
    START_GUESS = 3
    est_remaining = START_GUESS
    steps = 0; verifications = 0
    prev_line = False
    traj = [(i, j)]
    done_by_budget = False
    while steps < max_steps:
        p, i, j = _patch_at(signal, i, j, patch)
        F = sobel_features(p[None])
        proba = clf.predict_proba(F)[0]
        pred = int(proba.argmax()) if len(proba) > 2 else int(proba[1] > 0.5)
        conf = float(proba.max())
        is_line = bool(pred == 1) if len(proba) == 2 else bool(pred == 2)
        if verify and conf < conf_thresh:
            verifications += 1
            # asymmetric verification: thin lines are easily missed, rarely
            # hallucinated. Only re-examine uncertain NO-LINE predictions:
            # if any neighbour confidently sees a line, recover it (OR logic).
            # Uncertain LINE predictions are kept (conservative).
            if not is_line:
                found = False
                for di, dj in [(-6, 0), (6, 0), (0, -6), (0, 6)]:
                    pn, _, _ = _patch_at(signal, i+di, j+dj, patch)
                    Fn = sobel_features(pn[None])
                    q = clf.predict_proba(Fn)[0]
                    qp = int(q.argmax()) if len(q) > 2 else int(q[1] > 0.5)
                    qconf = float(q.max())
                    if (qp == 1 if len(q) == 2 else qp == 2) and qconf >= conf_thresh:
                        found = True
                        break
                is_line = found
        if is_line and not prev_line:
            est_remaining -= 1
        prev_line = is_line
        if (START_GUESS - est_remaining) >= (START_GUESS - target):
            done_by_budget = True
            true_total = int(total_map[min(i+patch//2, H-1), min(j+patch//2, W-1)])
            # tolerance: reaching single-electron *regime* (0..2) counts;
            # exact==1 reported separately via true_total
            return {"success": bool(true_total <= 2),
                    "steps": steps, "verifications": verifications,
                    "true_total": true_total, "traj": traj,
                    "done_by_budget": True}
        # move toward origin (fewer electrons)
        i -= 3; j -= 3
        if i <= 0 or j <= 0:
            break
        traj.append((i, j))
        steps += 1
    true_total = int(total_map[min(i+patch//2, H-1), min(j+patch//2, W-1)])
    # edge/budget-exhausted stop WITHOUT classifier budget = failure:
    # the walk wasted the measurement budget (overshoot to depletion).
    return {"success": False, "steps": steps,
            "verifications": verifications, "true_total": true_total, "traj": traj,
            "done_by_budget": False}
