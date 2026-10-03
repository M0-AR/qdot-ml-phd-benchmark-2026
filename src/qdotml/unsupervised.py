"""Unsupervised analysis (W3): kernel-PCA + clustering + hidden-pattern mining."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def run_kpca_clustering(X, y_true, gamma=0.02, n_clusters=6, seed=0):
    N = len(X)
    # physics-aware patch features: downsampled pixels + gradient energy +
    # mean/std (charge count proxies) so clusters can discover charge regimes.
    gx = np.abs(np.diff(X, axis=2, prepend=X[:, :, :1]))
    gy = np.abs(np.diff(X, axis=1, prepend=X[:, :1, :]))
    gmean = (gx + gy).reshape(N, -1).mean(axis=1, keepdims=True)
    gstd = (gx + gy).reshape(N, -1).std(axis=1, keepdims=True)
    pmean = X.reshape(N, -1).mean(axis=1, keepdims=True)
    pstd = X.reshape(N, -1).std(axis=1, keepdims=True)
    flat = np.concatenate([X[:, ::4, ::4].reshape(N, -1),
                           gmean, gstd, pmean, pstd], axis=1)
    # standardise
    flat = (flat - flat.mean()) / (flat.std() + 1e-9)
    kpca = KernelPCA(n_components=12, kernel="rbf", gamma=gamma,
                     random_state=seed)
    Z = kpca.fit_transform(flat)
    # standardise embedding before density/distance clustering
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(Zs)
    # adaptive eps: median pairwise distance on a subsample
    sub = Zs[:400] if len(Zs) > 400 else Zs
    from scipy.spatial.distance import pdist
    med = float(np.median(pdist(sub))) if len(sub) > 10 else 1.0
    db = DBSCAN(eps=0.45 * med, min_samples=8).fit(Zs)
    return {
        "Z": Z,
        "kmeans": km.labels_,
        "dbscan": db.labels_,
        "ari_kmeans": float(adjusted_rand_score(y_true, km.labels_)),
        "nmi_kmeans": float(normalized_mutual_info_score(y_true, km.labels_)),
        "n_dbscan_clusters": int(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)),
        "dbscan_noise_frac": float((db.labels_ == -1).mean()),
    }


def hidden_pattern_stats(devices, per_device_success):
    """Test the hypothesised hidden pattern.

    H1: cross-capacitance asymmetry A=|a12-a21|/(a11+a22) predicts tuning difficulty.
    H2: disorder amplitude predicts line-detector failure.
    Returns Pearson r values (computed by caller with numpy).
    """
    A = np.array([abs(d.a12 - d.a21) / (d.a11 + d.a22) for d in devices])
    dis = np.array([d.disorder for d in devices])
    s = np.array(per_device_success)
    def pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a*b).mean() / (a.std()*b.std() + 1e-12))
    return {"r_asym_vs_success": pearson(A, s),
            "r_disorder_vs_success": pearson(dis, s),
            "asymmetry": A, "disorder": dis}
