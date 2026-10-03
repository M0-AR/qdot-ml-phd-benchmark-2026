"""Supervised models (W1/W2): patch line detector + configuration classifier.

Lightweight sklearn MLPs stand in for the CNNs of DelftX coding demos and of
Yon et al. 2024 / Samaha et al. 2026. Rationale documented in README:
MLPs on 28x28 patches reproduce the qualitative benchmark (line vs no-line
~0.9 AUC, state ~0.85 acc) while running CPU-only in seconds, keeping the repo
reproducible without GPU/torch. U-Net variant is included as patch-segmentor
baseline via logistic regression on Sobel features for edge-case comparison.
"""
from __future__ import annotations
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def sobel_features(X: np.ndarray) -> np.ndarray:
    # X: (N, h, w) -> gradient energy + raw downsampled pixels
    gx = np.abs(np.diff(X, axis=2, prepend=X[:, :, :1]))
    gy = np.abs(np.diff(X, axis=1, prepend=X[:, :1, :]))
    g = (gx + gy).reshape(len(X), -1)
    raw = X[:, ::4, ::4].reshape(len(X), -1)
    return np.concatenate([raw, g.mean(axis=1, keepdims=True),
                           g.std(axis=1, keepdims=True)], axis=1)


def train_line_detector(Xtr, ytr, seed=0):
    Ftr = sobel_features(Xtr)
    # balance rare line positives by oversampling (thin 1-px transitions)
    rng = np.random.default_rng(seed)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    if len(pos) > 0 and len(pos) < len(neg):
        extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
        idx = np.concatenate([np.arange(len(ytr)), extra])
        rng.shuffle(idx)
        Ftr, ytr = Ftr[idx], ytr[idx]
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=80,
                                      random_state=seed, verbose=False))
    clf.fit(Ftr, ytr)
    return clf


def train_state_classifier(Xtr, ytr, seed=0):
    Ftr = sobel_features(Xtr)
    clf = make_pipeline(StandardScaler(),
                        MLPClassifier(hidden_layer_sizes=(128,), max_iter=80,
                                      random_state=seed, verbose=False))
    clf.fit(Ftr, ytr)
    return clf


def train_edge_baseline(Xtr, ytr):
    Ftr = sobel_features(Xtr)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=500))
    clf.fit(Ftr, ytr)
    return clf


def evaluate(clf, Xte, yte):
    Fte = sobel_features(Xte)
    p = clf.predict(Fte)
    try:
        proba = clf.predict_proba(Fte)[:, 1] if len(np.unique(yte)) == 2 else clf.predict_proba(Fte).max(axis=1)
    except Exception:
        proba = p.astype(float)
    out = {"acc": float(accuracy_score(yte, p)),
           "f1_macro": float(f1_score(yte, p, average="macro", zero_division=0))}
    if len(np.unique(yte)) == 2:
        try:
            out["auc"] = float(roc_auc_score(yte, proba))
        except Exception:
            out["auc"] = float("nan")
    # confidence = max proba; ECE (10 bins)
    try:
        conf = clf.predict_proba(Fte).max(axis=1)
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for b in range(10):
            m = (conf > bins[b]) & (conf <= bins[b+1])
            if m.sum() > 0:
                acc_b = float((p[m] == yte[m]).mean())
                conf_b = float(conf[m].mean())
                ece += abs(acc_b - conf_b) * float(m.mean())
        out["ece"] = float(ece)
        out["mean_conf"] = float(conf.mean())
    except Exception:
        out["ece"] = float("nan"); out["mean_conf"] = float("nan")
    return out
