"""End-to-end benchmark: W1 supervised -> W2 autotune -> W3 unsupervised+hidden patterns -> W4 Hamiltonian.
Saves results/benchmark.json + figures. Deterministic via configs/experiment.yaml seed.
"""
import json, pathlib, yaml, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qdotml.data import build_dataset
from qdotml.models import train_line_detector, train_state_classifier, train_edge_baseline, evaluate
from qdotml.benchmarks import group_cv_indices
from qdotml.physics import generate_csd
from qdotml.tuning import autotune_run
from qdotml.unsupervised import run_kpca_clustering, hidden_pattern_stats
from qdotml.hamiltonian import rabi_chevron, fit_hamiltonian
from qdotml.live_verify import check as live_check

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "experiment.yaml"))
    seed = cfg["seed"]
    t0 = time.time()
    print("== live public-data check ==")
    try:
        live = live_check()
    except Exception as e:
        live = {"status": f"error: {e}"}
    (RES / "live_manifest.json").write_text(json.dumps(live, indent=2))
    print("live status:", live.get("status"))

    print("== W1: supervised dataset ==")
    ds = build_dataset(seed=seed, grid=cfg["grid"], n_devices=cfg["n_devices_train"],
                       patches_per=cfg["patches_per_diagram"], patch_size=cfg["patch_size"])
    X, yl, ys, dev = ds["X"], ds["y_line"], ds["y_state"], ds["device"]
    print(f"patches={len(X)} line_pos_rate={yl.mean():.3f}")

    # group CV (device-level, no leakage) for line detector
    line_scores, state_scores, edge_scores = [], [], []
    for tr, te in group_cv_indices(dev, n_splits=4):
        clf = train_line_detector(X[tr], yl[tr], seed=seed)
        line_scores.append(evaluate(clf, X[te], yl[te]))
        cls = train_state_classifier(X[tr], ys[tr], seed=seed)
        state_scores.append(evaluate(cls, X[te], ys[te]))
        eb = train_edge_baseline(X[tr], yl[tr])
        edge_scores.append(evaluate(eb, X[te], yl[te]))
    def agg(scores):
        return {k: float(np.nanmean([s[k] for s in scores])) for k in scores[0]}
    w1 = {"line_mlp": agg(line_scores), "state_mlp": agg(state_scores),
          "line_logreg_baseline": agg(edge_scores), "folds": len(line_scores)}

    # train final clf on all for W2
    final_clf = train_line_detector(X, yl, seed=seed)

    print("== W2: closed-loop autotuning ==")
    rng = np.random.default_rng(seed + 1)
    runs = []
    runs_shifted = []
    per_dev_succ = {d: [] for d in range(cfg["n_devices_train"])}
    for r in range(cfg["tuning_runs"]):
        d = int(r % cfg["n_devices_train"])
        csd = generate_csd(rng, grid=cfg["grid"], noise=1.0, device=ds["devices"][d])
        csd_noisy = generate_csd(rng, grid=cfg["grid"], noise=2.0, device=ds["devices"][d])
        # fair ablation: identical start for verify True/False (paired runs)
        start_seed = int(rng.integers(1_000_000_000))
        for verify in [True, False]:
            out = autotune_run(csd["signal"], csd["total"], final_clf,
                               patch=cfg["patch_size"], target=1,
                               max_steps=cfg["tuning_max_steps"],
                               verify=verify,
                               rng=np.random.default_rng(start_seed))
            out.update({"run": r, "device": d, "verify": verify})
            runs.append(out)
            if verify:
                per_dev_succ[d].append(1.0 if out["success"] else 0.0)
            # distribution-shift stress: same start, 2x noise (unseen regime)
            outs = autotune_run(csd_noisy["signal"], csd_noisy["total"], final_clf,
                                patch=cfg["patch_size"], target=1,
                                max_steps=cfg["tuning_max_steps"],
                                verify=verify,
                                rng=np.random.default_rng(start_seed))
            outs.update({"run": r, "device": d, "verify": verify})
            runs_shifted.append(outs)
    succ_v = np.mean([o["success"] for o in runs if o["verify"]])
    succ_nv = np.mean([o["success"] for o in runs if not o["verify"]])
    steps_v = np.mean([o["steps"] for o in runs if o["verify"] and o["success"]]) if succ_v > 0 else float("nan")
    sh_v = float(np.mean([o["success"] for o in runs_shifted if o["verify"]]))
    sh_nv = float(np.mean([o["success"] for o in runs_shifted if not o["verify"]]))
    w2 = {"success_verify": float(succ_v), "success_no_verify": float(succ_nv),
          "delta_uncertainty": float(succ_v - succ_nv),
          "mean_steps_success": float(steps_v),
          "mean_verifications": float(np.mean([o["verifications"] for o in runs if o["verify"]])),
          "shifted_noise2_success_verify": sh_v,
          "shifted_noise2_success_no_verify": sh_nv,
          "shifted_delta_uncertainty": float(sh_v - sh_nv),
          "shifted_mean_verifications": float(np.mean([o["verifications"] for o in runs_shifted if o["verify"]]))}

    print("== W3: unsupervised + hidden patterns ==")
    # use state labels as ground truth for clustering quality
    cl = run_kpca_clustering(X[:1200], ys[:1200], gamma=cfg["khuizen_pca_gamma"],
                             n_clusters=cfg["n_clusters"], seed=seed)
    per_dev_rate = [float(np.mean(per_dev_succ[d])) if per_dev_succ[d] else 0.0
                    for d in range(cfg["n_devices_train"])]
    hp = hidden_pattern_stats(ds["devices"], per_dev_rate)
    w3 = {"ari_kmeans": cl["ari_kmeans"], "nmi_kmeans": cl["nmi_kmeans"],
          "dbscan_clusters": cl["n_dbscan_clusters"],
          "dbscan_noise_frac": cl["dbscan_noise_frac"],
          "r_asym_vs_success": hp["r_asym_vs_success"],
          "r_disorder_vs_success": hp["r_disorder_vs_success"]}
    # scatter hidden pattern
    plt.figure(figsize=(5, 4))
    plt.scatter(hp["asymmetry"], per_dev_rate)
    plt.xlabel("cross-capacitance asymmetry A=|a12-a21|/(a11+a22)")
    plt.ylabel("per-device autotune success")
    plt.title("Hidden pattern: crosstalk asymmetry vs tunability")
    plt.tight_layout(); plt.savefig(RES / "hidden_pattern_asymmetry.png", dpi=150)

    # example CSD
    ex = generate_csd(np.random.default_rng(seed + 99), grid=cfg["grid"], noise=1.0)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(ex["signal"], origin="lower", cmap="viridis"); ax[0].set_title("sensor signal (synthetic CSD)")
    ax[1].imshow(ex["lines"], origin="lower", cmap="gray"); ax[1].set_title("ground-truth transitions")
    plt.tight_layout(); plt.savefig(RES / "example_csd.png", dpi=150)

    # kpca scatter
    plt.figure(figsize=(5, 4))
    Z = cl["Z"]
    plt.scatter(Z[:, 0], Z[:, 1], c=cl["kmeans"], s=8, cmap="tab10", alpha=0.7)
    plt.xlabel("kPCA-1"); plt.ylabel("kPCA-2"); plt.title("kernel-PCA of CSD patches (coloured: k-means)")
    plt.tight_layout(); plt.savefig(RES / "kpca_clusters.png", dpi=150)

    print("== W4: Hamiltonian learning ==")
    g_err, t2_err, j_list = [], [], []
    rng2 = np.random.default_rng(seed + 2)
    for f in range(cfg["hamiltonian_fits"]):
        tc = float(rng2.uniform(5, 30)); g = float(rng2.uniform(1.8, 2.2))
        T2 = float(rng2.uniform(60, 200))
        B, t, P, ftrue = rabi_chevron(tc, g, T2, seed=int(rng2.integers(1e9)))
        hat = fit_hamiltonian(B, t, P)
        g_err.append(abs(hat["g_hat"] - g)); j_list.append((hat["J_hat"], tc))
        if np.isfinite(hat["T2_hat"]):
            t2_err.append(abs(hat["T2_hat"] - T2) / T2)
    w4 = {"mae_g": float(np.mean(g_err)), "mare_T2": float(np.mean(t2_err)) if t2_err else float("nan"),
          "n_fits": cfg["hamiltonian_fits"]}

    bench = {"config": cfg, "W1_supervised": w1, "W2_autotune": w2,
             "W3_unsupervised": w3, "W4_hamiltonian": w4,
             "elapsed_s": time.time() - t0,
             "live": live.get("status"),
             "notes": "Synthetic physics-grounded CSDs (QDarts/SimCATS-class model, from scratch). "
                      "Public experimental sets (QDSD/QFlow) referenced via live_manifest; "
                      "no financial market data involved — device tuning is verified on device data."}
    (RES / "benchmark.json").write_text(json.dumps(bench, indent=2, default=float))
    print(json.dumps(bench, indent=2))
    print("saved results/benchmark.json + pngs")

if __name__ == "__main__":
    main()
